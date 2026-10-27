--!strict

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close

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

-- A random m by n matrix of rank r.
local function low_rank(case: Case, m: number, n: number, r: number): lib.Matrix
	return case.M.matmul(nil, random_matrix(case, m, r), random_matrix(case, r, n))
end

local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

local function permute_columns(case: Case, A: lib.Matrix, perm: { number }): lib.Matrix
	local re, im = case.M.mattotable(A)
	local function pick(t: { { number } }): { { number } }
		local out = {}
		for i, row in t do
			out[i] = {}
			for j, p in perm do
				out[i][j] = row[p]
			end
		end
		return out
	end
	return if im then case.M.matfrom(pick(re), pick(im)) else case.M.matfrom(pick(re))
end

-- |R_kk| for k = 1, ..., min(rows, cols).
local function diagonal(case: Case, F: lib.QRP): { number }
	local re, im = case.M.mattotable(F.qr)
	local d = {}
	for k = 1, math.min(F.qr.rows, F.qr.cols) do
		d[k] = if im then lib.complex64.abs(re[k][k], im[k][k]) else math.abs(re[k][k])
	end
	return d
end

local function is_permutation(perm: { number }, n: number): boolean
	local seen = {}
	for _, p in perm do
		if p < 1 or p > n or seen[p] then
			return false
		end
		seen[p] = true
	end
	return #perm == n
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps

	test(`{name} QRP factors A P = Q R in place`, function()
		math.randomseed(31)
		for _, shape in { { 9, 5 }, { 6, 6 }, { 4, 8 }, { 1, 1 }, { 1, 5 }, { 7, 1 }, { 30, 20 } } do
			local m, n = shape[1], shape[2]
			local A = random_matrix(case, m, n)
			local A0 = M.matcopy(nil, A)
			local F = M.qrp(A)
			check(F.qr == A, "returns the input matrix")
			check(is_permutation(h.ints(F.perm), n), "perm is a permutation")
			local Q, R = M.qr_q(nil, F), M.qr_r(nil, F)
			local AP = permute_columns(case, A0, h.ints(F.perm))
			local tol = 30 * math.max(m, n) * eps
			local err = M.norm(M.sub(nil, M.matmul(nil, Q, R).data, AP.data)) / M.norm(A0.data)
			check(err <= tol, `{m}x{n}: |QR - AP| / |A| = {err}`)
			local k = math.min(m, n)
			local orth = M.norm(M.sub(nil, M.matmul(nil, adjoint(case, Q), Q).data, M.identity(k).data))
			check(orth <= tol, `{m}x{n}: |Q^H Q - I| = {orth}`)
			local d = diagonal(case, F)
			for i = 2, #d do
				check(d[i] <= d[i - 1] * (1 + 10 * eps), `{m}x{n}: |R_kk| is nonincreasing at {i}`)
			end
		end
	end)

	test(`{name} QRP reveals the rank`, function()
		math.randomseed(32)
		for _, r in { 1, 3, 5 } do
			local F = M.qrp(low_rank(case, 12, 8, r))
			check(M.qrp_rank(F) == r, `rank {r} detected (got {M.qrp_rank(F)})`)
		end
		check(M.qrp_rank(M.qrp(random_matrix(case, 10, 6))) == 6, "full column rank")
		check(M.qrp_rank(M.qrp(random_matrix(case, 4, 9))) == 4, "full row rank")
		check(M.qrp_rank(M.qrp(M.matrix(5, 3))) == 0, "zero matrix has rank 0")

		-- Nearly rank 1: every column is a multiple of one vector plus a small
		-- perturbation, so the partial norms collapse after the first step and must be
		-- recomputed rather than downdated.
		local m, n = 20, 6
		local small = if name == "f32" or name == "c32" then 1e-3 else 1e-9
		local A = M.matmul(nil, random_matrix(case, m, 1), random_matrix(case, 1, n))
		A.data = M.axpy(nil, small, random_vector(case, m * n), A.data)
		local A0 = M.matcopy(nil, A)
		local F = M.qrp(A)
		check(M.qrp_rank(F) == n, "perturbation is above the default tolerance")
		check(M.qrp_rank(F, 100 * small) == 1, "rank 1 at a tolerance above the perturbation")
		local d = diagonal(case, F)
		for i = 3, #d do
			check(d[i] <= d[i - 1] * (1 + 1e3 * eps), `nearly rank 1: |R_kk| is nonincreasing at {i}`)
		end
		local err = M.norm(M.sub(nil, M.matmul(nil, M.qr_q(nil, F), M.qr_r(nil, F)).data,
			permute_columns(case, A0, h.ints(F.perm)).data)) / M.norm(A0.data)
		check(err <= 30 * m * eps, `nearly rank 1: |QR - AP| / |A| = {err}`)
	end)

	test(`{name} QRP least squares`, function()
		math.randomseed(33)
		-- Full column rank: the same answer as plain QR.
		local A = random_matrix(case, 15, 6)
		local b = random_vector(case, 15)
		local x_qr = M.qr_solve(nil, M.qr_copy(A), b)
		local x, resid, r = M.qrp_solve(nil, M.qrp_copy(A), b)
		check(r == 6, "full rank")
		check(M.norm(M.sub(nil, x, x_qr)) <= 1e3 * eps * M.norm(x_qr), "matches plain QR")
		close(resid, M.norm(M.sub(nil, M.matvec(nil, A, x), b)), 100 * 15 * eps, "residual norm")

		-- Rank deficient: a least-squares solution using r of the columns.
		local B = low_rank(case, 15, 7, 3)
		local c = random_vector(case, 15)
		local y, resid2, r2 = M.qrp_solve(nil, M.qrp_copy(B), c)
		check(r2 == 3, "rank 3")
		local zeros = 0
		local yre, yim = M.totable(y)
		for i = 1, 7 do
			if yre[i] == 0 and (yim == nil or yim[i] == 0) then
				zeros += 1
			end
		end
		check(zeros == 4, `basic solution has n - r zeros (got {zeros})`)
		local res = M.sub(nil, M.matvec(nil, B, y), c)
		close(resid2, M.norm(res), 1e3 * eps, "residual norm, rank deficient")
		local g = M.matvec(nil, adjoint(case, B), res)
		local bound = 1e3 * 15 * eps * M.norm(B.data) * (M.norm(B.data) * M.norm(y) + M.norm(c))
		check(M.norm(g) <= bound, `normal equations, rank deficient: {M.norm(g)} > {bound}`)

		-- Wide, consistent: an exact solution.
		local W = random_matrix(case, 4, 9)
		local x0 = random_vector(case, 9)
		local w = M.matvec(nil, W, x0)
		local z, resid3, r3 = M.qrp_solve(nil, M.qrp_copy(W), w)
		check(r3 == 4, "wide: full row rank")
		check(M.norm(M.sub(nil, M.matvec(nil, W, z), w)) <= 1e3 * eps * M.norm(w), "wide: A x = b")
		check(resid3 <= 1e3 * eps * M.norm(w), "wide: residual ~0")

		-- Zero matrix: x = 0 and the residual is |b|.
		local x4, resid4, r4 = M.qrp_solve(nil, M.qrp(M.matrix(5, 3)), M.create(5))
		check(r4 == 0 and M.norm(x4) == 0 and resid4 == 0, "zero matrix and zero b")
		local b5 = random_vector(case, 5)
		local x5, resid5 = M.qrp_solve(nil, M.qrp(M.matrix(5, 3)), b5)
		check(M.norm(x5) == 0, "zero matrix: x = 0")
		close(resid5, M.norm(b5), 10 * eps, "zero matrix: residual is |b|")
	end)

	test(`{name} QRP copies and solve forms`, function()
		math.randomseed(34)
		local A = random_matrix(case, 6, 6)
		local A0 = M.matcopy(nil, A)
		local F = M.qrp_copy(A)
		check(M.norm(M.sub(nil, A.data, A0.data)) == 0 and F.qr ~= A, "qrp_copy leaves A alone")
		local b = random_vector(case, 6)
		local x = M.solve(nil, A, b)
		check(M.norm(M.sub(nil, M.solve(nil, F, b, "qrp"), x)) <= 1e4 * eps * M.norm(x), "form \"qrp\"")
		check(not pcall(function()
			M.solve(nil, F, b, "qr")
		end), "pivoted factorization passed as \"qr\"")
		check(not pcall(function()
			M.solve(nil, M.qr_copy(A), b, "qrp")
		end), "plain QR passed as \"qrp\"")
	end)
end

return nil
