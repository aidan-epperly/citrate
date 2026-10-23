--!strict

-- solve accepts a matrix or one of its factorizations, told apart by its `form` argument.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

type Case = { name: string, M: any, complex: boolean, eps: number }

local cases: { Case } = {
	{ name = "f32", M = lib.f32, complex = false, eps = h.eps32 },
	{ name = "f64", M = lib.f64, complex = false, eps = h.eps64 },
	{ name = "c32", M = lib.c32, complex = true, eps = h.eps32 },
	{ name = "c64", M = lib.c64, complex = true, eps = h.eps64 },
}

local function random_values(n: number): { number }
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = 2 * math.random() - 1
	end
	return t
end

local function random_vector(case: Case, n: number): buffer
	if case.complex then
		return case.M.from(random_values(n), random_values(n))
	end
	return case.M.from(random_values(n))
end

local function random_matrix(case: Case, m: number, n: number): lib.Matrix
	local A = case.M.matrix(m, n)
	A.data = random_vector(case, m * n)
	return A
end

local function fails(f: () -> ()): boolean
	return not pcall(f)
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps
	local function near(x: buffer, y: buffer): boolean
		return M.norm(M.sub(nil, x, y)) <= 1e4 * eps * M.norm(y)
	end

	test(`{name} solve with each form`, function()
		math.randomseed(21)
		local A = random_matrix(case, 6, 6)
		local A0 = M.matcopy(nil, A)
		local b = random_vector(case, 6)
		local x = M.solve(nil, A, b)
		check(near(M.solve(nil, A, b, "matrix"), x), "form \"matrix\" is the default")

		local Fl = M.lu_copy(A)
		local lu0 = M.matcopy(nil, Fl.lu)
		check(near(M.solve(nil, Fl, b, "lu"), x), "form \"lu\"")
		check(M.norm(M.sub(nil, Fl.lu.data, lu0.data)) == 0, "the LU factorization is reusable")

		local Fq = M.qr_copy(A)
		check(near(M.solve(nil, Fq, b, "qr"), x), "form \"qr\"")
		check(M.norm(M.sub(nil, A.data, A0.data)) == 0, "A is never modified")

		local o = M.create(6)
		check(M.solve(o, Fl, b, "lu") == o, "writes into out")
	end)

	test(`{name} solve on rectangular matrices`, function()
		math.randomseed(22)
		local A = random_matrix(case, 12, 5)
		local b = random_vector(case, 12)
		local expected = M.qr_solve(nil, M.qr_copy(A), b)
		check(near(M.solve(nil, A, b), expected), "tall matrix uses QR")
		check(near(M.solve(nil, M.qr_copy(A), b, "qr"), expected), "tall QR factorization")

		local W = random_matrix(case, 3, 5)
		local W0 = M.matcopy(nil, W)
		local w = M.matvec(nil, W, random_vector(case, 5))
		local x = M.solve(nil, W, w)
		check(near(x, M.qrp_solve(nil, M.qrp_copy(W), w)), "wide matrix uses pivoted QR")
		check(M.norm(M.sub(nil, M.matvec(nil, W, x), w)) <= 1e3 * eps * M.norm(w), "wide: A x = b")
		check(M.norm(M.sub(nil, W.data, W0.data)) == 0, "wide: A is not modified")
	end)

	test(`{name} solve rejects a mismatched form`, function()
		local A = random_matrix(case, 4, 4)
		local b = M.create(4)
		check(fails(function()
			M.solve(nil, A, b, "lu")
		end), "matrix passed as \"lu\"")
		check(fails(function()
			M.solve(nil, M.lu_copy(A), b)
		end), "LU passed without a form")
		check(fails(function()
			M.solve(nil, M.lu_copy(A), b, "qr")
		end), "LU passed as \"qr\"")
		check(fails(function()
			M.solve(nil, M.qr_copy(A), b, "lu")
		end), "QR passed as \"lu\"")
		check(fails(function()
			M.solve(nil, A, b, "cholesky" :: any)
		end), "unknown form")
	end)
	test(`{name} solve with a matrix of right-hand sides`, function()
		math.randomseed(23)
		local function columns_match(F: any, form: string?, B: lib.Matrix, X: lib.Matrix): boolean
			local ok = true
			for j = 1, B.cols do
				local col = M.create(B.rows)
				buffer.copy(col, 0, B.data, buffer.len(col) * (j - 1), buffer.len(col))
				local x = M.solve(nil, F, col, form)
				local got = M.create(X.rows)
				buffer.copy(got, 0, X.data, buffer.len(got) * (j - 1), buffer.len(got))
				ok = ok and M.norm(M.sub(nil, got, x)) == 0
			end
			return ok
		end
		local A = random_matrix(case, 5, 5)
		local A0 = M.matcopy(nil, A)
		local B = random_matrix(case, 5, 3)
		local B0 = M.matcopy(nil, B)
		local X = M.solve(nil, A, B)
		check(X.rows == 5 and X.cols == 3, "shape")
		local r = M.norm(M.sub(nil, M.matmul(nil, A, X).data, B.data))
		check(r <= 1e3 * eps * M.norm(A.data) * M.norm(X.data), `|A X - B| = {r}`)
		check(columns_match(A, nil, B, X), "each column is solve on that column")
		check(M.norm(M.sub(nil, A.data, A0.data)) == 0 and M.norm(M.sub(nil, B.data, B0.data)) == 0, "A and B are not modified")

		-- Every form.
		local H = M.matmul(nil, A, if case.complex then M.adjoint(nil, A) else M.transpose(nil, A))
		H.data = M.add(nil, H.data, M.identity(5).data)
		local forms: { { any } } = {
			{ M.lu_copy(A), "lu" },
			{ M.qr_copy(A), "qr" },
			{ M.qrp_copy(A), "qrp" },
			{ M.chol_copy(H), "chol" },
			{ M.svd_copy(A), "svd" },
		}
		for _, pair in forms do
			local Y = M.solve(nil, pair[1], B, pair[2])
			check(columns_match(pair[1], pair[2], B, Y), `form "{pair[2]}"`)
		end

		-- Tall (least squares) and wide (basic solution) matrices.
		local T = random_matrix(case, 8, 4)
		local C = random_matrix(case, 8, 2)
		local XT = M.solve(nil, T, C)
		check(XT.rows == 4 and XT.cols == 2 and columns_match(T, nil, C, XT), "tall")
		local W = random_matrix(case, 3, 6)
		local D = random_matrix(case, 3, 2)
		local XW = M.solve(nil, W, D)
		check(XW.rows == 6 and columns_match(W, nil, D, XW), "wide")

		local O = M.matrix(5, 3)
		check(M.solve(O, A, B) == O and M.norm(M.sub(nil, O.data, X.data)) == 0, "writes into out")
		check(fails(function()
			M.solve(M.matrix(4, 3), A, B)
		end), "out of the wrong shape")
		local Sq = M.matcopy(nil, A)
		check(fails(function()
			M.solve(Sq, A, Sq)
		end), "out cannot be B")
		local E = M.solve(nil, A, M.matrix(5, 0))
		check(E.rows == 5 and E.cols == 0, "no columns")
	end)
end

-- The overloads keep their types: a vector gives a vector and a matrix gives a matrix.
test("solve is typed by its right-hand side", function()
	local A = lib.f64.matfrom({ { 2, 0 }, { 0, 4 } })
	local x: buffer = lib.f64.solve(nil, A, lib.f64.from({ 2, 4 }))
	local X: lib.Matrix = lib.f64.solve(nil, A, lib.f64.identity(2))
	check(lib.f64.get(x, 2) == 1 and lib.f64.matget(X, 2, 2) == 0.25, "values")
end)

return nil
