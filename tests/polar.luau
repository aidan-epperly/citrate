--!strict

-- Polar decomposition A = U H by QDWH.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

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

local function diff(M: any, A: lib.Matrix, B: lib.Matrix): number
	return M.norm(M.sub(nil, A.data, B.data))
end

-- A random matrix with orthonormal columns.
local function orthonormal(case: Case, m: number, n: number): lib.Matrix
	return case.M.qr_q(nil, case.M.qr(random_matrix(case, m, n)))
end

-- Q1 diag(s) Q2^H with the given singular values.
local function with_singular_values(case: Case, m: number, n: number, s: { number }): lib.Matrix
	local M = case.M
	local k = #s
	local rows = {}
	for i = 1, k do
		rows[i] = table.create(k, 0)
		rows[i][i] = s[i]
	end
	local zeros = {}
	for i = 1, k do
		zeros[i] = table.create(k, 0)
	end
	local D = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
	return M.matmul(nil, M.matmul(nil, orthonormal(case, m, k), D), adjoint(case, orthonormal(case, n, k)))
end

for _, case in cases do
	local M, R, name, eps = case.M, case.R, case.name, case.eps
	local tol = 100 * eps

	-- |U H - A| / |A|, the orthonormality of U (columns, or rows if wide), and whether H
	-- is Hermitian positive semidefinite.
	local function quality(A: lib.Matrix, F: lib.Polar): (number, number, boolean)
		local m, n = A.rows, A.cols
		local recon = diff(M, M.matmul(nil, F.u, F.h), A) / M.norm(A.data)
		local UU = if m >= n then M.matmul(nil, adjoint(case, F.u), F.u) else M.matmul(nil, F.u, adjoint(case, F.u))
		local orth = diff(M, UU, M.identity(math.min(m, n)))
		local herm = diff(M, F.h, adjoint(case, F.h)) == 0
		local values = R.totable(M.eigh_copy(F.h, false).values)
		local psd = herm and values[1] >= -tol * math.max(values[#values], 1)
		return recon, orth, psd
	end

	test(`{name} polar decomposition`, function()
		math.randomseed(301)
		for _, shape in { { 1, 1 }, { 5, 5 }, { 9, 4 }, { 4, 9 }, { 20, 20 } } do
			local m, n = shape[1], shape[2]
			local A = random_matrix(case, m, n)
			local A0 = M.matcopy(nil, A)
			local F = M.polar(A)
			local recon, orth, psd = quality(A0, F)
			check(F.u == A, `{m}x{n}: U overwrites A`)
			check(F.h.rows == n and F.h.cols == n, `{m}x{n}: H is cols by cols`)
			check(recon <= tol and orth <= tol, `{m}x{n}: |U H - A| / |A| = {recon}, orthonormality {orth}`)
			check(psd, `{m}x{n}: H is Hermitian positive semidefinite`)
			check(F.iterations <= 6, `{m}x{n}: {F.iterations} iterations`)
		end
		local A = random_matrix(case, 6, 4)
		local A0 = M.matcopy(nil, A)
		local F = M.polar_copy(A)
		check(diff(M, A, A0) == 0 and F.u ~= A, "polar_copy leaves A alone")
	end)

	test(`{name} polar agrees with the SVD`, function()
		math.randomseed(302)
		-- A = W S V^H gives U = W V^H and H = V S V^H.
		local A = random_matrix(case, 7, 5)
		local S = M.svd_copy(A)
		local W, V = S.u :: lib.Matrix, S.v :: lib.Matrix
		local s = R.totable(S.values)
		local rows, zeros = {}, {}
		for i = 1, 5 do
			rows[i], zeros[i] = table.create(5, 0), table.create(5, 0)
			rows[i][i] = s[i]
		end
		local D = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
		local F = M.polar_copy(A)
		check(diff(M, F.u, M.matmul(nil, W, adjoint(case, V))) <= 1e3 * eps, "U = W V^H")
		local Hs = M.matmul(nil, M.matmul(nil, V, D), adjoint(case, V))
		check(diff(M, F.h, Hs) <= 1e3 * eps * M.norm(A.data), "H = V S V^H")
		-- U is the closest matrix with orthonormal columns to A.
		local closest = true
		local dU = diff(M, A, F.u)
		for _ = 1, 10 do
			closest = closest and diff(M, A, orthonormal(case, 7, 5)) >= dU
		end
		check(closest, "U is closer to A than random orthonormal matrices")
	end)

	test(`{name} polar on hard inputs`, function()
		math.randomseed(303)
		-- Condition number 1 / (100 eps): still at most 6 iterations, backward stable.
		local s = {}
		for i = 1, 8 do
			s[i] = (100 * eps) ^ ((i - 1) / 7)
		end
		local A = with_singular_values(case, 8, 8, s)
		local F = M.polar_copy(A)
		local recon, orth, psd = quality(A, F)
		check(F.iterations <= 6, `ill-conditioned: {F.iterations} iterations`)
		check(recon <= tol and orth <= tol and psd, `ill-conditioned: {recon}, {orth}`)

		-- Already orthonormal: U = A and H = I.
		local Q = orthonormal(case, 6, 4)
		local G = M.polar_copy(Q)
		check(diff(M, G.u, Q) <= tol and diff(M, G.h, M.identity(4)) <= tol, "orthonormal input: U = A, H = I")

		-- Numerically rank deficient (rank 2): rounding makes the zero singular values
		-- tiny but nonzero, and U still comes out orthonormal.
		local B = M.matmul(nil, random_matrix(case, 7, 2), random_matrix(case, 2, 5))
		local Fb = M.polar_copy(B)
		local rb, ob, pb = quality(B, Fb)
		check(rb <= tol and ob <= tol and pb, `rank 2: {rb}, {ob}`)

		-- Exactly rank deficient: U is the canonical partial isometry, U^H U a projector.
		local Z = M.matrix(3, 2)
		M.matset(Z, 1, 1, if name == "c32" then vector.create(1, 0, 0) else 1, 0)
		M.matset(Z, 2, 1, if name == "c32" then vector.create(2, 0, 0) else 2, 0)
		local Fz = M.polar_copy(Z)
		local P = M.matmul(nil, adjoint(case, Fz.u), Fz.u)
		check(diff(M, M.matmul(nil, P, P), P) <= tol, "zero column: U^H U is a projector")
		check(diff(M, M.matmul(nil, Fz.u, Fz.h), Z) <= tol * M.norm(Z.data), "zero column: U H = A")

		local Zero = M.polar(M.matrix(4, 3))
		check(M.norm(Zero.h.data) == 0 and diff(M, M.matmul(nil, adjoint(case, Zero.u), Zero.u), M.identity(3)) == 0, "zero matrix: H = 0, U orthonormal")
		local N = random_matrix(case, 3, 3)
		M.matset(N, 2, 2, if name == "c32" then vector.create(0 / 0, 0, 0) else 0 / 0, 0)
		local Fn = M.polar(N)
		check(h.all_nan(Fn.u.data, eps == h.eps32) and h.all_nan(Fn.h.data, eps == h.eps32), "NaN gives NaN")
	end)
end

return nil
