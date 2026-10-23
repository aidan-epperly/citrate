--!strict

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

local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

-- A well-conditioned Hermitian (symmetric if real) positive definite matrix: B^H B + n I.
local function positive_definite(case: Case, n: number): lib.Matrix
	local M = case.M
	local B = random_matrix(case, n, n)
	local A = M.matmul(nil, adjoint(case, B), B)
	A.data = M.axpy(nil, n, M.identity(n).data, A.data)
	return A
end

-- A copy of A with every strictly upper entry set to `value`, and, if `diag_im` is given,
-- the imaginary parts of the diagonal set to it.
local function scribble(case: Case, A: lib.Matrix, value: number, diag_im: number?): lib.Matrix
	local re, im = case.M.mattotable(A)
	for i = 1, A.rows do
		for j = i + 1, A.cols do
			re[i][j] = value
			if im then
				im[i][j] = value
			end
		end
		if im and diag_im then
			im[i][i] = diag_im
		end
	end
	return if im then case.M.matfrom(re, im) else case.M.matfrom(re)
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps

	test(`{name} Cholesky factors in place`, function()
		math.randomseed(41)
		for _, n in { 1, 5, 30 } do
			local A = positive_definite(case, n)
			local A0 = M.matcopy(nil, A)
			local F = M.chol(A)
			check(F.definite and F.chol == A, `n={n}: definite, returns the input matrix`)
			local L = M.chol_l(nil, F)
			local err = M.norm(M.sub(nil, M.matmul(nil, L, adjoint(case, L)).data, A0.data)) / M.norm(A0.data)
			check(err <= 30 * n * eps, `n={n}: |L L^H - A| / |A| = {err}`)

			local re, im = M.mattotable(F.chol)
			local are, aim = M.mattotable(A0)
			local upper_kept, diag_ok = true, true
			for i = 1, n do
				diag_ok = diag_ok and re[i][i] > 0 and (im == nil or im[i][i] == 0)
				for j = i + 1, n do
					upper_kept = upper_kept and re[i][j] == are[i][j] and (im == nil or im[i][j] == aim[i][j])
				end
			end
			check(diag_ok, `n={n}: diagonal of L is real and positive`)
			check(upper_kept, `n={n}: strictly upper triangle of A is untouched`)
		end
	end)

	test(`{name} Cholesky reads only the lower triangle`, function()
		math.randomseed(42)
		local A = positive_definite(case, 6)
		local clean = M.chol_l(nil, M.chol_copy(A))
		local messy = M.chol_l(nil, M.chol(scribble(case, A, 0 / 0, if case.complex then 123 else nil)))
		check(M.norm(M.sub(nil, clean.data, messy.data)) == 0, "same L with NaN above the diagonal")
	end)

	test(`{name} Cholesky solve`, function()
		math.randomseed(43)
		local n = 12
		local A = positive_definite(case, n)
		local b = random_vector(case, n)
		local b0 = M.copy(nil, b)
		local F = M.chol_copy(A)
		local x = M.chol_solve(nil, F, b)
		check(M.norm(M.sub(nil, b, b0)) == 0, "b is not modified")
		local r = M.norm(M.sub(nil, M.matvec(nil, A, x), b))
		check(r <= 30 * n * eps * M.norm(A.data) * M.norm(x), `residual {r}`)
		local x_lu = M.solve(nil, A, b)
		check(M.norm(M.sub(nil, x, x_lu)) <= 1e3 * eps * M.norm(x_lu), "agrees with LU")
		check(M.norm(M.sub(nil, M.solve(nil, F, b, "chol"), x)) == 0, "form \"chol\"")
		local y = M.copy(nil, b)
		check(M.chol_solve(y, F, y) == y and M.norm(M.sub(nil, y, x)) == 0, "solves in place when out is b")

		local A0 = M.matcopy(nil, A)
		M.chol_copy(A)
		check(M.norm(M.sub(nil, A.data, A0.data)) == 0, "chol_copy leaves A alone")
	end)

	test(`{name} Cholesky detects matrices that are not positive definite`, function()
		local indefinite = M.matfrom({ { 2, 1 }, { 1, -3 } })
		local F = M.chol(indefinite)
		check(not F.definite, "indefinite")
		check(not pcall(function()
			M.chol_solve(nil, F, M.create(2))
		end), "chol_solve refuses")
		check(not pcall(function()
			M.solve(nil, F, M.create(2), "chol")
		end), "solve refuses")
		check(not M.chol(M.matrix(3, 3)).definite, "zero matrix")
		check(not M.chol(M.matfrom({ { 1, 0 }, { 0, 0 / 0 } })).definite, "NaN on the diagonal")
		check(not M.chol(M.matfrom({ { 1, 2 }, { 2, 1 } })).definite, "positive diagonal but indefinite")
		check(not pcall(function()
			M.solve(nil, M.lu_copy(M.identity(2)), M.create(2), "chol")
		end), "LU passed as \"chol\"")
	end)
end

return nil
