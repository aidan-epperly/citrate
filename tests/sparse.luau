--!strict

-- Sparse matrices against their dense equivalents, and the Krylov solvers on model
-- problems.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local krylov = lib.krylov

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

local function values(case: Case, re: { number }, im: { number }?): buffer
	if case.complex then
		return case.M.from(re, im or table.create(#re, 0))
	end
	return case.M.from(re)
end

local function random_vector(case: Case, n: number): buffer
	return values(case, random_values(n), if case.complex then random_values(n) else nil)
end

local function near(M: any, x: buffer, y: buffer, tol: number): boolean
	return M.norm(M.sub(nil, x, y)) <= tol * math.max(M.norm(y), 1e-300)
end

-- Triplets of the 5-point Laplacian on a k by k grid (n = k^2), plus `shift` on the
-- diagonal and a first-order convection term of strength `conv` (nonsymmetric if nonzero).
local function laplacian(k: number, shift: number, conv: number): (number, { number }, { number }, { number })
	local ii, jj, vv = {}, {}, {}
	local function add(i: number, j: number, v: number)
		table.insert(ii, i)
		table.insert(jj, j)
		table.insert(vv, v)
	end
	for gy = 0, k - 1 do
		for gx = 0, k - 1 do
			local p = gy * k + gx + 1
			add(p, p, 4 + shift)
			if gx > 0 then
				add(p, p - 1, -1 - conv)
			end
			if gx < k - 1 then
				add(p, p + 1, -1 + conv)
			end
			if gy > 0 then
				add(p, p - k, -1)
			end
			if gy < k - 1 then
				add(p, p + k, -1)
			end
		end
	end
	return k * k, ii, jj, vv
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps
	local tol = 1e3 * eps

	test(`{name} sparse construction and access`, function()
		-- Duplicates are summed; the order of the triplets does not matter.
		local S = M.sparse(3, 4, { 2, 1, 3, 1, 2 }, { 1, 1, 4, 1, 3 }, values(case, { 5, 1, 7, 2, -1 }))
		check(M.sparse_nnz(S) == 4, "duplicates merged")
		local want = values(case, { 3, 5, 0, 0, 0, 0, 0, -1, 0, 0, 0, 7 })
		check(M.norm(M.sub(nil, M.sparse_matrix(S).data, want)) == 0, "dense form")
		local g = M.sparse_get(S, 1, 1)
		check((if typeof(g) == "vector" then (g :: vector).x else g :: number) == 3, "sparse_get stored entry")
		local z = M.sparse_get(S, 3, 1)
		check((if typeof(z) == "vector" then (z :: vector).x else z :: number) == 0, "sparse_get missing entry")
		check(not pcall(function()
			M.sparse(2, 2, { 3 }, { 1 }, values(case, { 1 }))
		end), "position outside the matrix")

		math.randomseed(101)
		local A = M.matrix(6, 5)
		A.data = random_vector(case, 30)
		-- Zero out about half the entries.
		local re, im = M.mattotable(A)
		for i = 1, 6 do
			for j = 1, 5 do
				if math.random() < 0.5 then
					re[i][j] = 0
					if im then
						im[i][j] = 0
					end
				end
			end
		end
		A = if im then M.matfrom(re, im) else M.matfrom(re)
		local Sp = M.sparse_from_matrix(A)
		check(M.norm(M.sub(nil, M.sparse_matrix(Sp).data, A.data)) == 0, "from_matrix round trip")
		local x, y = random_vector(case, 5), random_vector(case, 6)
		check(near(M, M.sparse_matvec(nil, Sp, x), M.matvec(nil, A, x), tol), "matvec")
		local AH = if case.complex then M.adjoint(nil, A) else M.transpose(nil, A)
		check(near(M, M.sparse_matvec_adjoint(nil, Sp, y), M.matvec(nil, AH, y), tol), "adjoint matvec")
		check(M.norm(M.sub(nil, M.sparse_matrix(M.sparse_adjoint(Sp)).data, AH.data)) == 0, "sparse_adjoint")
		check(M.norm(M.sub(nil, M.sparse_matrix(M.sparse_transpose(Sp)).data, M.transpose(nil, A).data)) == 0, "sparse_transpose")
		local B = M.matrix(5, 3)
		B.data = random_vector(case, 15)
		check(near(M, M.sparse_matmul(nil, Sp, B).data, M.matmul(nil, A, B).data, tol), "sparse times dense")
		local twice = M.sparse_add(Sp, Sp)
		check(near(M, M.sparse_matrix(twice).data, M.scale(nil, 2, A.data), 0), "sparse_add")
		local sq = M.sparse_from_matrix(M.identity(4))
		check(M.norm(M.sub(nil, M.sparse_diag(sq), values(case, { 1, 1, 1, 1 }))) == 0, "sparse_diag")
	end)

	test(`{name} CG on the 2D Laplacian`, function()
		math.randomseed(102)
		local n, ii, jj, vv = laplacian(10, 0, 0)
		local S = M.sparse(n, n, ii, jj, values(case, vv))
		local A: lib.Operator<buffer> = function(out, x)
			return M.sparse_matvec(out, S, x)
		end
		local b = random_vector(case, n)
		local want = M.solve(nil, M.sparse_matrix(S), b)
		local rtol = if eps > 1e-10 then 1e-4 else 1e-10
		local x, info = krylov.cg(M.space, A, b, { tol = rtol })
		check(info.converged and info.residual <= rtol, `CG converged ({info.residual} after {info.iterations})`)
		check(near(M, x, want, 100 * rtol), "CG agrees with the dense solve")
		local _, pinfo = krylov.cg(M.space, A, b, { tol = rtol, precond = M.sparse_jacobi(S) })
		check(pinfo.converged and pinfo.iterations <= info.iterations, "Jacobi-preconditioned CG")
		-- x0 is used and overwritten.
		local x0 = M.copy(nil, want)
		local x2, info2 = krylov.cg(M.space, A, b, { tol = rtol, x0 = x0 })
		check(x2 == x0 and info2.iterations <= 2, "a good x0 needs almost no iterations")
		local _, capped = krylov.cg(M.space, A, b, { tol = 1e-30, maxiter = 3 })
		check(not capped.converged and capped.iterations == 3, "maxiter stops it")
		local zx, zinfo = krylov.cg(M.space, A, M.create(n), {})
		check(zinfo.converged and M.norm(zx) == 0, "zero right-hand side")
	end)

	test(`{name} MINRES on an indefinite Laplacian`, function()
		math.randomseed(103)
		-- Shifting by -3 puts eigenvalues on both sides of zero (and away from it).
		local n, ii, jj, vv = laplacian(8, -3.05, 0)
		local S = M.sparse(n, n, ii, jj, values(case, vv))
		local A: lib.Operator<buffer> = function(out, x)
			return M.sparse_matvec(out, S, x)
		end
		local b = random_vector(case, n)
		local rtol = if eps > 1e-10 then 1e-3 else 1e-9
		local x, info = krylov.minres(M.space, A, b, { tol = rtol })
		check(info.converged, `MINRES converged ({info.residual} after {info.iterations})`)
		check(near(M, x, M.solve(nil, M.sparse_matrix(S), b), 1e3 * rtol), "MINRES agrees with the dense solve")
	end)

	test(`{name} GMRES and BiCGSTAB on convection-diffusion`, function()
		math.randomseed(104)
		local n, ii, jj, vv = laplacian(10, 0.5, 0.4)
		-- Complex types get a complex shift too, so the matrix is genuinely complex.
		local im = nil
		if case.complex then
			im = table.create(#vv, 0)
			for k = 1, #vv do
				if ii[k] == jj[k] then
					im[k] = 0.7
				end
			end
		end
		local S = M.sparse(n, n, ii, jj, values(case, vv, im))
		local A: lib.Operator<buffer> = function(out, x)
			return M.sparse_matvec(out, S, x)
		end
		local b = random_vector(case, n)
		local want = M.solve(nil, M.sparse_matrix(S), b)
		local rtol = if eps > 1e-10 then 1e-4 else 1e-10
		for _, restart in { 10, 50 } do
			local x, info = krylov.gmres(M.space, A, b, { tol = rtol, restart = restart })
			check(info.converged, `GMRES({restart}) converged ({info.residual} after {info.iterations})`)
			check(near(M, x, want, 100 * rtol), `GMRES({restart}) agrees with the dense solve`)
		end
		local _, pinfo = krylov.gmres(M.space, A, b, { tol = rtol, precond = M.sparse_jacobi(S) })
		check(pinfo.converged, "preconditioned GMRES converged")
		local x, info = krylov.bicgstab(M.space, A, b, { tol = rtol })
		check(info.converged, `BiCGSTAB converged ({info.residual} after {info.iterations})`)
		if case.complex then
			-- Regression bound (seeded data): the optimal omega = <t, s> / <t, t> takes 10
			-- (c32) or 20 (c64) iterations here; the conjugated one takes 13 or 28.
			local bound = if name == "c32" then 12 else 24
			check(info.iterations <= bound, `BiCGSTAB iterations {info.iterations} <= {bound}`)
		end
		check(near(M, x, want, 100 * rtol), "BiCGSTAB agrees with the dense solve")
		local _, binfo = krylov.bicgstab(M.space, A, b, { tol = rtol, precond = M.sparse_jacobi(S) })
		check(binfo.converged, "preconditioned BiCGSTAB converged")
	end)
	test(`{name} Krylov solvers report failures and use x0`, function()
		math.randomseed(105)
		local function operator(S: any): lib.Operator<buffer>
			return function(out, x)
				return M.sparse_matvec(out, S, x)
			end
		end
		local function true_residual(A: lib.Operator<buffer>, x: buffer, b: buffer): number
			return M.norm(M.sub(nil, b, A(nil :: any, x))) / M.norm(b)
		end
		-- MINRES needs a Hermitian matrix; GMRES and BiCGSTAB get a nonsymmetric one.
		local n1, i1, j1, v1 = laplacian(6, -3.05, 0)
		local n2, i2, j2, v2 = laplacian(6, 0.5, 0.4)
		local problems = {
			{ "MINRES", krylov.minres, M.sparse(n1, n1, i1, j1, values(case, v1)), n1 },
			{ "GMRES", krylov.gmres, M.sparse(n2, n2, i2, j2, values(case, v2)), n2 },
			{ "BiCGSTAB", krylov.bicgstab, M.sparse(n2, n2, i2, j2, values(case, v2)), n2 },
		}
		for _, problem in problems do
			local label, solve, S, n = problem[1], problem[2] :: any, problem[3], problem[4]
			local A = operator(S)
			local b = random_vector(case, n)
			local x, capped = solve(M.space, A, b, { tol = 1e-30, maxiter = 3 })
			check(not capped.converged and capped.iterations == 3, `{label}: maxiter stops it ({capped.iterations})`)
			close(capped.residual, true_residual(A, x, b), 1e-6, `{label}: reported residual is the true one`)
			local want = M.solve(nil, M.sparse_matrix(S), b)
			local x0 = M.copy(nil, want)
			local tol = if eps > 1e-10 then 1e-3 else 1e-9
			local x2, good = solve(M.space, A, b, { tol = tol, x0 = x0 })
			check(x2 == x0 and good.converged and good.iterations <= 2, `{label}: a good x0 needs almost no iterations ({good.iterations})`)
			local zx, zinfo = solve(M.space, A, M.create(n), {})
			check(zinfo.converged and M.norm(zx) == 0, `{label}: zero right-hand side`)
		end

		-- Breakdowns end the iteration with an honest report instead of NaN.
		local K = M.sparse_from_matrix(M.matfrom({ { 0, 1 }, { -1, 0 } }, if case.complex then { { 0, 0 }, { 0, 0 } } else nil))
		local KA = operator(K)
		local kb = values(case, { 1, 0 })
		-- r^H A r = 0 for a skew-symmetric A and a real r.
		local kx, kinfo = krylov.bicgstab(M.space, KA, kb, {})
		check(not kinfo.converged and kinfo.residual == true_residual(KA, kx, kb) and kinfo.residual == kinfo.residual, "BiCGSTAB breakdown is reported")
		local D = M.sparse_from_matrix(M.matfrom({ { 1, 0 }, { 0, -1 } }, if case.complex then { { 0, 0 }, { 0, 0 } } else nil))
		local DA = operator(D)
		local db = values(case, { 1, 1 })
		-- p^H A p = 0 on the first step: CG stops because A is not positive definite.
		local dx, dinfo = krylov.cg(M.space, DA, db, {})
		check(not dinfo.converged and dinfo.residual == true_residual(DA, dx, db), "CG stops on an indefinite matrix")
	end)
end

test("CG and MINRES on vectors (matrix-free)", function()
	-- A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]] as a function of a vector.
	local A: lib.Operator<vector> = function(_: vector, x: vector): vector
		return vector.create(4 * x.x + x.y, x.x + 3 * x.y + x.z, x.y + 2 * x.z)
	end
	local b = vector.create(1, 2, 3)
	local x, info = krylov.cg(lib.vec.space, A, b, { tol = 1e-6 })
	check(typeof(x) == "vector" and info.converged and info.iterations <= 3, `CG on vectors ({info.iterations} iterations)`)
	local r = A(x, x)
	close(r.x, 1, 1e-5)
	close(r.y, 2, 1e-5)
	close(r.z, 3, 1e-5, "A x = b")
	local y, minfo = krylov.minres(lib.vec.space, A, b, { tol = 1e-6 })
	check(typeof(y) == "vector" and minfo.converged, "MINRES on vectors")
end)

test("GMRES on a real space rejects a complex coefficient", function()
	check(not pcall(function()
		lib.f64.space.caxpy(lib.f64.create(2), 1, 1, lib.f64.create(2), lib.f64.create(2))
	end), "imaginary coefficient on a real space")
end)

return nil
