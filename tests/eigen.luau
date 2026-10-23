--!strict

-- Hessenberg reduction and the symmetric/Hermitian eigensolver.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close

-- R is the real module of the same precision, which holds eigenvalues.
type Case = { name: string, M: any, R: any, complex: boolean, eps: number }

local cases: { Case } = {
	{ name = "f32", M = lib.f32, R = lib.f32, complex = false, eps = h.eps32 },
	{ name = "f64", M = lib.f64, R = lib.f64, complex = false, eps = h.eps64 },
	{ name = "c32", M = lib.c32, R = lib.f32, complex = true, eps = h.eps32 },
	{ name = "c64", M = lib.c64, R = lib.f64, complex = true, eps = h.eps64 },
}

local function random_values(n: number): { number }
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = 2 * math.random() - 1
	end
	return t
end

local function random_matrix(case: Case, m: number, n: number): lib.Matrix
	local A = case.M.matrix(m, n)
	A.data = if case.complex
		then case.M.from(random_values(m * n), random_values(m * n))
		else case.M.from(random_values(m * n))
	return A
end

local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

local function hermitian(case: Case, n: number): lib.Matrix
	local B = random_matrix(case, n, n)
	local A = adjoint(case, B)
	A.data = case.M.add(nil, A.data, B.data)
	return A
end

local function from_rows(case: Case, rows: { { number } }): lib.Matrix
	if case.complex then
		local zeros = {}
		for i, row in rows do
			zeros[i] = table.create(#row, 0)
		end
		return case.M.matfrom(rows, zeros)
	end
	return case.M.matfrom(rows)
end

-- The diagonal matrix with real entries `values`.
local function diagonal(case: Case, values: { number }): lib.Matrix
	local n = #values
	local rows = {}
	for i = 1, n do
		rows[i] = table.create(n, 0)
		rows[i][i] = values[i]
	end
	return from_rows(case, rows)
end

-- A copy of A with NaN in every strictly upper entry and, for complex matrices,
-- `diag_im` in the imaginary parts of the diagonal.
local function scribble(case: Case, A: lib.Matrix, diag_im: number): lib.Matrix
	local re, im = case.M.mattotable(A)
	for i = 1, A.rows do
		for j = i + 1, A.cols do
			re[i][j] = 0 / 0
			if im then
				im[i][j] = 0 / 0
			end
		end
		if im then
			im[i][i] = diag_im
		end
	end
	return if im then case.M.matfrom(re, im) else case.M.matfrom(re)
end

local function frob(M: any, A: lib.Matrix): number
	return M.norm(A.data)
end

local function diff(M: any, A: lib.Matrix, B: lib.Matrix): number
	return M.norm(M.sub(nil, A.data, B.data))
end

for _, case in cases do
	local M, R, name, eps = case.M, case.R, case.name, case.eps

	test(`{name} Hessenberg reduction`, function()
		math.randomseed(51)
		for _, n in { 1, 2, 3, 7, 25 } do
			local A = random_matrix(case, n, n)
			local A0 = M.matcopy(nil, A)
			local F = M.hess(A)
			check(F.hess == A, `n={n}: in place`)
			local H, Q = M.hess_h(nil, F), M.hess_q(nil, F)
			local tol = 30 * n * eps
			local recon = diff(M, M.matmul(nil, M.matmul(nil, Q, H), adjoint(case, Q)), A0) / frob(M, A0)
			check(recon <= tol, `n={n}: |Q H Q^H - A| / |A| = {recon}`)
			local orth = diff(M, M.matmul(nil, adjoint(case, Q), Q), M.identity(n))
			check(orth <= tol, `n={n}: |Q^H Q - I| = {orth}`)
			-- Q^H A Q reproduces H, including the zeros below the subdiagonal.
			local back = diff(M, M.matmul(nil, M.matmul(nil, adjoint(case, Q), A0), Q), H) / frob(M, A0)
			check(back <= tol, `n={n}: |Q^H A Q - H| / |A| = {back}`)
		end
		local A = random_matrix(case, 5, 5)
		local A0 = M.matcopy(nil, A)
		M.hess_copy(A)
		check(diff(M, A, A0) == 0, "hess_copy leaves A alone")
	end)

	test(`{name} eigh on known matrices`, function()
		local function values(A: lib.Matrix): { number }
			return R.totable(M.eigh_copy(A, false).values)
		end
		local v = values(diagonal(case, { 3, 1, 2 }))
		check(v[1] == 1 and v[2] == 2 and v[3] == 3, "diagonal matrix, sorted")
		v = values(from_rows(case, { { 2, 1 }, { 1, 2 } }))
		close(v[1], 1, 4 * eps)
		close(v[2], 3, 4 * eps, "[[2, 1], [1, 2]]")
		-- The Rayleigh-quotient shift never converges on this one; Wilkinson's does.
		v = values(from_rows(case, { { 0, 1 }, { 1, 0 } }))
		close(v[1], -1, 4 * eps)
		close(v[2], 1, 4 * eps, "[[0, 1], [1, 0]]")
		v = values(M.identity(4))
		check(v[1] == 1 and v[4] == 1, "identity")
		v = values(M.matrix(3, 3))
		check(v[1] == 0 and v[3] == 0, "zero matrix")
		v = values(from_rows(case, { { -7 } }))
		check(#v == 1 and v[1] == -7, "1 by 1")
		local E = M.eigh(M.matrix(0, 0))
		check(buffer.len(E.values) == 0, "0 by 0")

		local F = M.eigh_copy(M.identity(4))
		local V = F.vectors :: lib.Matrix
		check(diff(M, M.matmul(nil, adjoint(case, V), V), M.identity(4)) <= 10 * eps, "identity: orthonormal vectors")
	end)

	test(`{name} eigh on random matrices`, function()
		math.randomseed(52)
		for _, n in { 2, 5, 30 } do
			local A = hermitian(case, n)
			local A0 = M.matcopy(nil, A)
			local F = M.eigh(A)
			check(F.vectors == A, `n={n}: vectors overwrite A`)
			local V = F.vectors :: lib.Matrix
			local lambda = R.totable(F.values)
			local ascending, trace, sum = true, 0, 0
			for i = 1, n do
				if i > 1 and lambda[i] < lambda[i - 1] then
					ascending = false
				end
				sum += lambda[i]
			end
			local re = M.mattotable(A0)
			for i = 1, n do
				trace += re[i][i]
			end
			check(ascending, `n={n}: ascending`)
			close(sum, trace, 30 * n * eps * frob(M, A0), `n={n}: sum of eigenvalues = trace`)
			local tol = 30 * n * eps
			local resid = diff(M, M.matmul(nil, A0, V), M.matmul(nil, V, diagonal(case, lambda))) / frob(M, A0)
			check(resid <= tol, `n={n}: |A V - V L| / |A| = {resid}`)
			local orth = diff(M, M.matmul(nil, adjoint(case, V), V), M.identity(n))
			check(orth <= tol, `n={n}: |V^H V - I| = {orth}`)
			local only = R.totable(M.eigh_copy(A0, false).values)
			local same = true
			for i = 1, n do
				same = same and math.abs(only[i] - lambda[i]) <= tol * frob(M, A0)
			end
			check(same, `n={n}: values without vectors agree`)
		end
	end)

	test(`{name} eigh with close eigenvalues`, function()
		-- Wilkinson's W21+: its largest eigenvalues come in pairs that agree to many digits.
		local n = 21
		local rows = {}
		for i = 1, n do
			rows[i] = table.create(n, 0)
			rows[i][i] = math.abs(11 - i)
			if i > 1 then
				rows[i][i - 1], rows[i - 1][i] = 1, 1
			end
		end
		local A0 = from_rows(case, rows)
		local F = M.eigh_copy(A0)
		local V = F.vectors :: lib.Matrix
		local lambda = R.totable(F.values)
		local tol = 30 * n * eps
		local resid = diff(M, M.matmul(nil, A0, V), M.matmul(nil, V, diagonal(case, lambda))) / frob(M, A0)
		check(resid <= tol, `W21+: |A V - V L| / |A| = {resid}`)
		check(diff(M, M.matmul(nil, adjoint(case, V), V), M.identity(n)) <= tol, "W21+: orthonormal vectors")
		check(math.abs(lambda[n] - lambda[n - 1]) < 1e-6, "W21+: top pair is nearly equal")
	end)

	test(`{name} eigh reads only the lower triangle`, function()
		math.randomseed(53)
		local A = hermitian(case, 6)
		local clean = M.eigh_copy(A)
		local messy = M.eigh(scribble(case, A, 99))
		check(R.norm(R.sub(nil, clean.values, messy.values)) == 0, "same eigenvalues")
		check(diff(M, clean.vectors :: lib.Matrix, messy.vectors :: lib.Matrix) == 0, "same eigenvectors")
		local A0 = M.matcopy(nil, A)
		M.eigh_copy(A)
		check(diff(M, A, A0) == 0, "eigh_copy leaves A alone")
	end)
end

return nil
