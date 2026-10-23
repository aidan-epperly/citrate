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

-- Conjugate transpose (plain transpose for real types).
local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

local function diff_norm(M: any, A: lib.Matrix, B: lib.Matrix): number
	return M.norm(M.sub(nil, A.data, B.data))
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps

	test(`{name} QR factors in place and reproduces A`, function()
		math.randomseed(3)
		for _, shape in { { 8, 5 }, { 6, 6 }, { 4, 7 }, { 1, 1 }, { 1, 4 }, { 5, 1 }, { 40, 25 } } do
			local m, n = shape[1], shape[2]
			local k = math.min(m, n)
			local A = random_matrix(case, m, n)
			local A0 = M.matcopy(nil, A)
			local F = M.qr(A)
			check(F.qr == A, "returns the input matrix")
			check(m == 1 or diff_norm(M, A, A0) > 0, "A was overwritten")

			local Q, R = M.qr_q(nil, F), M.qr_r(nil, F)
			check(Q.rows == m and Q.cols == k and R.rows == k and R.cols == n, `shapes {m}x{n}`)
			local tol = 30 * math.max(m, n) * eps
			local err = diff_norm(M, M.matmul(nil, Q, R), A0) / M.norm(A0.data)
			check(err <= tol, `{m}x{n}: |QR - A| / |A| = {err}`)
			local orth = diff_norm(M, M.matmul(nil, adjoint(case, Q), Q), M.identity(k))
			check(orth <= tol, `{m}x{n}: |Q^H Q - I| = {orth}`)

			local re, im = M.mattotable(R)
			local below = 0
			for j = 1, n do
				for i = j + 1, k do
					below += math.abs(re[i][j]) + (if im then math.abs(im[i][j]) else 0)
				end
			end
			check(below == 0, "R is upper triangular")
			if im then
				for i = 1, k do
					check(im[i][i] == 0, "diagonal of R is real")
				end
			end
		end
	end)

	test(`{name} applying Q`, function()
		math.randomseed(4)
		local m, n = 9, 4
		local A = random_matrix(case, m, n)
		local F = M.qr(A)
		local x = random_vector(case, m)
		local x0 = M.copy(nil, x)
		local y = M.qr_apply_qh(F, x)
		check(y == x, "applies in place and returns x")
		close(M.norm(y), M.norm(x0), 10 * eps, "Q^H preserves the norm")
		M.qr_apply_q(F, y)
		check(M.norm(M.sub(nil, y, x0)) <= 30 * m * eps * M.norm(x0), "Q Q^H x = x")
		local Qh = adjoint(case, M.qr_q(nil, F))
		local thin = M.matvec(nil, Qh, x0)
		local applied = M.qr_apply_qh(F, M.copy(nil, x0))
		local head = M.create(n)
		buffer.copy(head, 0, applied, 0, buffer.len(head))
		check(M.norm(M.sub(nil, head, thin)) <= 30 * m * eps * M.norm(x0), "matches the explicit thin Q")
	end)

	test(`{name} least squares`, function()
		math.randomseed(5)
		local m, n = 30, 8
		local A = random_matrix(case, m, n)
		local A0 = M.matcopy(nil, A)
		local b = random_vector(case, m)
		local b0 = M.copy(nil, b)
		local F = M.qr(A)
		local x, resid = M.qr_solve(nil, F, b)
		check(M.norm(M.sub(nil, b, b0)) == 0, "b is not modified")
		local r = M.sub(nil, M.matvec(nil, A0, x), b)
		close(resid, M.norm(r), 100 * m * eps, "returned residual norm")
		local g = M.matvec(nil, adjoint(case, A0), r)
		local bound = 100 * m * eps * M.norm(A0.data) * (M.norm(A0.data) * M.norm(x) + M.norm(b))
		check(M.norm(g) <= bound, `normal equations: |A^H r| = {M.norm(g)} > {bound}`)

		-- A consistent system is solved exactly, with zero residual.
		local x_true = random_vector(case, n)
		local c = M.matvec(nil, A0, x_true)
		local x2, resid2 = M.qr_solve(nil, F, c)
		check(M.norm(M.sub(nil, x2, x_true)) <= 1000 * eps * M.norm(x_true), "consistent system")
		check(resid2 <= 100 * m * eps * M.norm(c), "consistent residual is ~0")

		-- Square systems agree with LU.
		local S = random_matrix(case, 6, 6)
		local s = random_vector(case, 6)
		local via_lu = M.solve(nil, S, s)
		local via_qr = M.qr_solve(nil, M.qr_copy(S), s)
		check(M.norm(M.sub(nil, via_lu, via_qr)) <= 1e4 * eps * M.norm(via_lu), "square: QR agrees with LU")
	end)

	test(`{name} degenerate inputs`, function()
		local Z = M.matrix(4, 3)
		local F = M.qr(Z)
		check(M.norm(F.qr.data) == 0 and M.norm(F.tau) == 0, "zero matrix: R = 0, all reflectors are I")
		check(diff_norm(M, M.matmul(nil, adjoint(case, M.qr_q(nil, F)), M.qr_q(nil, F)), M.identity(3)) == 0, "zero matrix: Q = I")
		check(not pcall(M.qr_solve, nil, F, M.create(4)), "rank deficient system is an error")
		check(not pcall(M.qr_solve, nil, M.qr(M.matrix(2, 3)), M.create(2)), "wide least squares is an error")
	end)
end

return nil
