--!strict

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close

local function random_matrix(R: typeof(lib.f64), n: number): lib.Matrix
	local A = R.matrix(n, n)
	for i = 1, n do
		for j = 1, n do
			R.matset(A, i, j, 2 * math.random() - 1)
		end
	end
	return A
end

local function frob(R: typeof(lib.f64), A: lib.Matrix): number
	return R.norm(A.data)
end

for _, case in { { name = "f32", R = lib.f32, eps = h.eps32 }, { name = "f64", R = lib.f64, eps = h.eps64 } } do
	local R: typeof(lib.f64), eps, name = case.R, case.eps, case.name
	local tol = 8 * eps

	test(`{name} construction and access`, function()
		local x = R.from({ 1, 2, 3 })
		check(R.length(x) == 3)
		check(R.get(x, 2) == 2)
		R.set(x, 2, 5)
		local t = R.totable(x)
		check(t[1] == 1 and t[2] == 5 and t[3] == 3)
		check(R.length(R.create(4)) == 4 and R.get(R.create(4), 4) == 0)
		check(R.get(R.fill(R.create(2), 7), 2) == 7)
	end)

	test(`{name} elementwise kernels`, function()
		local x, y = R.from({ 1, 2, 3 }), R.from({ 4, 5, 6 })
		local t = R.totable(R.add(nil, x, y))
		check(t[1] == 5 and t[3] == 9, "add")
		t = R.totable(R.sub(nil, x, y))
		check(t[1] == -3, "sub")
		t = R.totable(R.mul(nil, x, y))
		check(t[2] == 10, "mul")
		t = R.totable(R.scale(nil, 2, x))
		check(t[3] == 6, "scale")
		t = R.totable(R.axpy(nil, 2, x, y))
		check(t[1] == 6 and t[3] == 12, "axpy")
		local o = R.create(3)
		check(R.add(o, x, y) == o, "writes into out and returns it")
		check(R.axpy(x, 1, x, x) == x and R.get(x, 2) == 4, "out may alias an input")
		check(not pcall(R.add, nil, x, R.create(2)), "length mismatch is an error")
	end)

	test(`{name} reductions`, function()
		local x, y = R.from({ 1, -2, 3 }), R.from({ 4, 5, 6 })
		check(R.dot(x, y) == 12, "dot")
		check(R.sum(x) == 2, "sum")
		check(R.amax(x) == 3, "amax")
		close(R.norm(x), math.sqrt(14), tol, "norm")
		check(R.norm(R.create(0)) == 0, "empty norm")
		local big = if name == "f64" then 1e200 else 1e30
		close(R.norm(R.from({ big, big })) / big, math.sqrt(2), tol, "norm without overflow")
		-- (1e-40 is subnormal in single precision; 1e-37 where subnormals flush to zero.)
		local small = if name == "f64" then 1e-300 elseif h.subnormals then 1e-40 else 1e-37
		close(R.norm(R.from({ 3 * small, 4 * small })) / small, 5, 1e-6, "norm without underflow")
		check(R.norm(R.from({ 1, 0 / 0 })) ~= R.norm(R.from({ 1, 0 / 0 })), "norm propagates NaN")
	end)

	test(`{name} matrix basics`, function()
		local A = R.matfrom({ { 1, 2, 3 }, { 4, 5, 6 } })
		check(A.rows == 2 and A.cols == 3)
		check(R.matget(A, 2, 1) == 4 and R.matget(A, 1, 3) == 3, "matget")
		check(R.get(A.data, 2) == 4, "column-major storage")
		local T = R.transpose(nil, A)
		check(T.rows == 3 and R.matget(T, 3, 2) == 6, "transpose")
		local y = R.totable(R.matvec(nil, A, R.from({ 1, 1, 1 })))
		check(y[1] == 6 and y[2] == 15, "matvec")
		local C = R.mattotable(R.matmul(nil, A, T))
		check(C[1][1] == 14 and C[1][2] == 32 and C[2][2] == 77, "matmul")
		local I = R.identity(3)
		check(R.matget(I, 2, 2) == 1 and R.matget(I, 1, 2) == 0, "identity")
		check(not pcall(R.matmul, nil, A, A), "shape mismatch is an error")
	end)

	test(`{name} LU and solve`, function()
		local A = R.matfrom({ { 0, 1 }, { 2, 3 } })
		local x = R.totable(R.solve(nil, A, R.from({ 1, 8 })))
		close(x[1], 2.5, tol, "needs pivoting: x1")
		close(x[2], 1, tol, "needs pivoting: x2")
		close(R.det(A), -2, tol, "det")
		close(R.det(R.matfrom({ { 2, 0, 0 }, { 0, 3, 0 }, { 0, 0, 4 } })), 24, tol, "det diagonal")

		local S = R.matfrom({ { 1, 2 }, { 2, 4 } })
		check(R.lu_copy(S).singular, "singular matrix detected")
		check(R.det(S) == 0, "singular det is 0")
		check(not pcall(function()
			R.solve(nil, S, R.from({ 1, 1 }))
		end), "solving a singular system is an error")

		math.randomseed(1)
		for _, n in { 1, 5, 40 } do
			local M = random_matrix(R, n)
			local b = R.create(n)
			for i = 1, n do
				R.set(b, i, 2 * math.random() - 1)
			end
			local F = R.lu_copy(M)
			local sol = R.lu_solve(nil, F, b)
			local r = R.sub(nil, R.matvec(nil, M, sol), b)
			local bound = 50 * n * eps * frob(R, M) * R.norm(sol)
			check(R.norm(r) <= bound, `residual n={n}: {R.norm(r)} > {bound}`)
		end
	end)
end

return nil
