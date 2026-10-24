--!strict

-- Triangular, tridiagonal, banded, circulant and Toeplitz matrices, each checked against the
-- same matrix stored densely.

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

local function random_matrix(case: Case, n: number): lib.Matrix
	local A = case.M.matrix(n, n)
	A.data = random_vector(case, n * n)
	return A
end

-- |x - y| <= tol |y|
local function near(M: any, x: buffer, y: buffer, tol: number): boolean
	return M.norm(M.sub(nil, x, y)) <= tol * math.max(M.norm(y), 1e-300)
end

-- Overwrite entries of a dense matrix that fail keep(i, j) (1-based) with `value` (and the
-- diagonal with `diag` if given), returning a new matrix.
local function rewrite(case: Case, A: lib.Matrix, keep: (number, number) -> boolean, value: number, diag: number?): lib.Matrix
	local re, im = case.M.mattotable(A)
	for i = 1, A.rows do
		for j = 1, A.cols do
			local v: number? = nil
			if diag and i == j then
				v = diag
			elseif not keep(i, j) then
				v = value
			end
			if v then
				re[i][j] = v
				if im then
					im[i][j] = if v == v then 0 else v
				end
			end
		end
	end
	return if im then case.M.matfrom(re, im) else case.M.matfrom(re)
end

local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

-- Sets entry i of an array to the real number v, whatever the module's scalar format.
local function set_real(case: Case, x: buffer, i: number, v: number)
	if case.name == "c32" then
		case.M.set(x, i, vector.create(v, 0, 0))
	elseif case.complex then
		case.M.set(x, i, v, 0)
	else
		case.M.set(x, i, v)
	end
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps
	local tol = 1e3 * eps

	test(`{name} triangular solve and multiply`, function()
		math.randomseed(91)
		local n = 7
		local A = random_matrix(case, n)
		-- Keep the triangles well conditioned.
		A.data = M.axpy(nil, 4, M.identity(n).data, A.data)
		local b = random_vector(case, n)
		for _, uplo in { "upper", "lower" } do
			local up = uplo == "upper"
			local function inside(i: number, j: number): boolean
				return if up then i <= j else i >= j
			end
			for _, unit in { false, true } do
				-- The dense triangle, and A with NaN everywhere the routine must not read.
				local T = rewrite(case, A, inside, 0, if unit then 1 else nil)
				local messy = rewrite(case, A, inside, 0 / 0, if unit then 0 / 0 else nil)
				local label = `{uplo}{if unit then ", unit" else ""}`
				local x = M.triangular_solve(nil, messy, b, uplo :: lib.Uplo, unit)
				check(near(M, x, M.solve(nil, T, b), tol), `{label}: solve`)
				local y = M.triangular_matvec(nil, messy, b, uplo :: lib.Uplo, unit)
				check(near(M, y, M.matvec(nil, T, b), tol), `{label}: matvec`)
				local z = M.copy(nil, b)
				check(M.triangular_solve(z, messy, z, uplo :: lib.Uplo, unit) == z and near(M, z, x, 0), `{label}: in place`)
			end
		end
	end)

	test(`{name} tridiagonal`, function()
		math.randomseed(92)
		for _, n in { 1, 2, 3, 8, 9 } do
			-- Even n: a zero diagonal, so the LU must pivot (with odd n that matrix would be
			-- singular). Odd n: a random diagonal.
			local diag = if n % 2 == 0 then M.create(n) else random_vector(case, n)
			local T = M.tridiagonal(random_vector(case, n - 1), diag, random_vector(case, n - 1))
			local A = M.tridiag_matrix(T)
			local b = random_vector(case, n)
			check(near(M, M.tridiag_matvec(nil, T, b), M.matvec(nil, A, b), tol), `n={n}: matvec`)
			local d0 = M.copy(nil, T.diag)
			local F = M.tridiag_lu_copy(T)
			check(M.norm(M.sub(nil, T.diag, d0)) == 0, `n={n}: tridiag_lu_copy leaves T alone`)
			check(near(M, M.tridiag_lu_solve(nil, F, b), M.solve(nil, A, b), tol), `n={n}: LU solve`)
			local Tc = M.tridiag_copy(T)
			local G = M.tridiag_lu(T)
			check(G.lu == T, `n={n}: tridiag_lu is in place`)
			check(Tc.diag ~= T.diag and M.norm(M.sub(nil, Tc.diag, d0)) == 0, `n={n}: tridiag_copy is independent`)
			check(near(M, M.tridiag_matvec(nil, Tc, b), M.matvec(nil, A, b), tol), `n={n}: tridiag_copy is the same matrix`)
		end
		check(M.tridiag_lu(M.tridiagonal(M.create(2), M.create(3), M.create(2))).singular, "zero matrix is singular")

		-- Hermitian positive definite: diagonal 4, random subdiagonal, superdiagonal its conjugate.
		local n = 10
		local lower = random_vector(case, n - 1)
		local diag = M.scale(nil, 4, M.add(nil, M.create(n), M.from(table.create(n, 1))))
		local upper = if case.complex then M.conj(nil, lower) else M.copy(nil, lower)
		local P = M.tridiagonal(lower, diag, upper)
		local A = M.tridiag_matrix(P)
		local b = random_vector(case, n)
		local F = M.tridiag_ldl_copy(P)
		check(F.definite, "LDL: positive definite")
		check(near(M, M.tridiag_ldl_solve(nil, F, b), M.solve(nil, A, b), tol), "LDL solve")
		local neg = M.tridiagonal(M.create(1), M.from({ 1, -1 }), M.create(1))
		check(not M.tridiag_ldl(neg).definite, "LDL: indefinite detected")
	end)

	test(`{name} tridiagonal eigenproblem`, function()
		math.randomseed(97)
		for _, n in { 1, 2, 7, 30 } do
			-- Symmetric (real) or Hermitian (complex): upper = conj(lower), real diagonal.
			local lower = random_vector(case, n - 1)
			local diag = M.from(random_values(n))
			local upper = if case.complex then M.conj(nil, lower) else M.copy(nil, lower)
			local T = M.tridiagonal(lower, diag, upper)
			local A = M.tridiag_matrix(T)
			local F = M.tridiag_eigh(T)
			-- Eigenvalues are real arrays of the same precision.
			local Rm = if not case.complex then M elseif eps > 1e-10 then lib.f32 else lib.f64
			local values = Rm.totable(F.values)
			local dense = Rm.totable(M.eigh_copy(A, false).values)
			local agree = true
			for i = 1, n do
				agree = agree and math.abs(values[i] - dense[i]) <= tol * math.max(1, math.abs(dense[n]))
			end
			check(agree, `n={n}: values agree with eigh`)
			local V = F.vectors :: lib.Matrix
			local rows = {}
			for i = 1, n do
				rows[i] = table.create(n, 0)
				rows[i][i] = values[i]
			end
			local zeros = {}
			for i = 1, n do
				zeros[i] = table.create(n, 0)
			end
			local D = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
			local res = M.norm(M.sub(nil, M.matmul(nil, A, V).data, M.matmul(nil, V, D).data))
			check(res <= 10 * tol * math.max(1, M.norm(A.data)), `n={n}: A V = V L ({res})`)
			local Vh = if case.complex then M.adjoint(nil, V) else M.transpose(nil, V)
			check(M.norm(M.sub(nil, M.matmul(nil, Vh, V).data, M.identity(n).data)) <= 10 * tol, `n={n}: orthonormal vectors`)
		end
	end)

	test(`{name} banded`, function()
		math.randomseed(93)
		local n, kl, ku = 12, 2, 3
		local Full = random_matrix(case, n)
		local B = M.band_from_matrix(Full, kl, ku)
		local A = M.band_matrix(B)
		local inside = function(i: number, j: number): boolean
			return i - j <= kl and j - i <= ku
		end
		check(M.norm(M.sub(nil, A.data, rewrite(case, Full, inside, 0).data)) == 0, "band_from_matrix keeps exactly the band")
		local b = random_vector(case, n)
		check(near(M, M.band_matvec(nil, B, b), M.matvec(nil, A, b), tol), "matvec")
		-- Zero the diagonal so the LU has to pivot.
		local Z = rewrite(case, A, inside, 0, 0)
		local BZ = M.band_from_matrix(Z, kl, ku)
		local before = M.copy(nil, BZ.data)
		local F = M.band_lu_copy(BZ)
		check(M.norm(M.sub(nil, BZ.data, before)) == 0, "band_lu_copy leaves B alone")
		check(near(M, M.band_lu_solve(nil, F, b), M.solve(nil, Z, b), tol), "LU solve with pivoting")
		local BC = M.band_copy(BZ)
		check(M.band_lu(BZ).lu == BZ, "band_lu is in place")
		check(BC.data ~= BZ.data and M.norm(M.sub(nil, BC.data, before)) == 0, "band_copy is independent")
		check(BC.n == n and BC.kl == kl and BC.ku == ku, "band_copy keeps the shape")
		check(M.band_lu(M.banded(4, 1, 1)).singular, "zero band matrix is singular")
		check(not pcall(function()
			if name == "c32" then
				M.band_set(B, 1, 6, vector.create(1, 0, 0))
			else
				M.band_set(B, 1, 6, 1, 0)
			end
		end), "band_set outside the band is an error")
		local g = M.band_get(B, 7, 1)
		check((if type(g) == "vector" then (g :: vector).x else g :: number) == 0, "band_get outside the band is 0")

		-- Hermitian positive definite band: lower band random, diagonal 8, upper its adjoint.
		local L = rewrite(case, random_matrix(case, n), function(i, j)
			return i - j >= 1 and i - j <= kl
		end, 0, 0)
		local H = M.add(nil, L.data, adjoint(case, L).data)
		local Hm = M.matrix(n, n)
		Hm.data = M.axpy(nil, 8, M.identity(n).data, H)
		local HB = M.band_from_matrix(rewrite(case, Hm, function(i, j)
			return i >= j
		end, 0 / 0), kl, kl)
		local C = M.band_chol_copy(HB)
		check(C.definite, "band Cholesky: positive definite")
		check(near(M, M.band_chol_solve(nil, C, b), M.solve(nil, Hm, b), tol), "band Cholesky solve (lower band only read)")
		local bad = M.band_from_matrix(M.identity(3), 1, 1)
		bad.data = M.scale(nil, -1, bad.data)
		check(not M.band_chol(bad).definite, "band Cholesky: indefinite detected")
	end)

	test(`{name} circulant`, function()
		math.randomseed(94)
		for _, n in { 1, 2, 5, 8, 13 } do
			local c = random_vector(case, n)
			-- Diagonally dominant, so certainly nonsingular.
			set_real(case, c, 1, n + 1)
			local C = M.circulant(c)
			local A = M.circulant_matrix(c)
			local b = random_vector(case, n)
			check(near(M, M.circulant_matvec(nil, C, b), M.matvec(nil, A, b), tol), `n={n}: matvec`)
			check(near(M, M.circulant_solve(nil, C, b), M.solve(nil, A, b), tol), `n={n}: solve`)
		end
		local ones = if case.complex then M.from({ 1, 1, 1, 1 }, { 0, 0, 0, 0 }) else M.from({ 1, 1, 1, 1 })
		check(not pcall(function()
			M.circulant_solve(nil, M.circulant(ones), M.create(4))
		end), "singular circulant is an error")
	end)

	test(`{name} Toeplitz`, function()
		math.randomseed(95)
		for _, n in { 1, 2, 5, 17, 40 } do
			local col, row = random_vector(case, n), random_vector(case, n)
			set_real(case, col, 1, 2 * n)
			local T = M.toeplitz(col, row)
			local A = M.toeplitz_matrix(T)
			local b = random_vector(case, n)
			check(near(M, M.toeplitz_matvec(nil, T, b), M.matvec(nil, A, b), tol), `n={n}: matvec`)
			check(near(M, M.toeplitz_solve(nil, T, b), M.solve(nil, A, b), 10 * tol), `n={n}: Levinson solve`)
		end
		-- Default first row: symmetric (real) or Hermitian (complex, real diagonal).
		local col = random_vector(case, 6)
		set_real(case, col, 1, 3)
		local S = M.toeplitz_matrix(M.toeplitz(col))
		check(M.norm(M.sub(nil, S.data, adjoint(case, S).data)) == 0, "default row makes T symmetric or Hermitian")
		check(not pcall(function()
			local z = if case.complex then M.from({ 0, 1 }, { 0, 0 }) else M.from({ 0, 1 })
			M.toeplitz_solve(nil, M.toeplitz(z), M.create(2))
		end), "a singular leading submatrix is reported")
	end)
	test(`{name} outputs that alias an input are rejected`, function()
		math.randomseed(96)
		local n = 4
		local A = random_matrix(case, n)
		local x = random_vector(case, n)
		local function rejects(f: () -> ()): boolean
			return not pcall(f)
		end
		check(rejects(function()
			M.matvec(x, A, x)
		end), "matvec")
		check(rejects(function()
			M.matmul(A, A, M.identity(n))
		end) and rejects(function()
			M.matmul(A, M.identity(n), A)
		end), "matmul")
		check(rejects(function()
			M.transpose(A, A)
		end), "transpose")
		check(rejects(function()
			M.triangular_matvec(x, A, x, "upper")
		end), "triangular_matvec")
		local T = M.tridiagonal(random_vector(case, n - 1), random_vector(case, n), random_vector(case, n - 1))
		check(rejects(function()
			M.tridiag_matvec(x, T, x)
		end), "tridiag_matvec")
		local B = M.band_from_matrix(A, 1, 2)
		check(rejects(function()
			M.band_matvec(x, B, x)
		end), "band_matvec")
		local S = M.sparse_from_matrix(A)
		check(rejects(function()
			M.sparse_matvec(x, S, x)
		end) and rejects(function()
			M.sparse_matvec_adjoint(x, S, x)
		end), "sparse_matvec and its adjoint")
		local D = M.matcopy(nil, A)
		check(rejects(function()
			M.sparse_matmul(D, S, D)
		end), "sparse_matmul")
		-- The same calls with a fresh output work, so the failures above are the alias checks.
		check(pcall(function()
			M.matvec(nil, A, x)
			M.matmul(nil, A, M.identity(n))
			M.transpose(nil, A)
			M.triangular_matvec(nil, A, x, "upper")
			M.tridiag_matvec(nil, T, x)
			M.band_matvec(nil, B, x)
			M.sparse_matvec(nil, S, x)
			M.sparse_matvec_adjoint(nil, S, x)
			M.sparse_matmul(nil, S, D)
		end), "fresh outputs work")
	end)
end

return nil
