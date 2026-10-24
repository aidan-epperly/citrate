--!strict

-- Updating QR and Cholesky factorizations, and the randomized SVD.

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

local function random_vector(case: Case, n: number): buffer
	return if case.complex then case.M.from(random_values(n), random_values(n)) else case.M.from(random_values(n))
end

local function random_matrix(case: Case, m: number, n: number): lib.Matrix
	local A = case.M.matrix(m, n)
	A.data = random_vector(case, m * n)
	return A
end

local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

local function diff(M: any, A: lib.Matrix, B: lib.Matrix): number
	return M.norm(M.sub(nil, A.data, B.data))
end

for _, case in cases do
	local M, R, name, eps = case.M, case.R, case.name, case.eps
	local tol = 100 * eps

	-- Q unitary, R upper triangular and Q R = A.
	local function valid(F: lib.QRFull, A: lib.Matrix): (boolean, string)
		local m, n = A.rows, A.cols
		local Q, Rm = F.q, F.r
		if Q.rows ~= m or Q.cols ~= m or Rm.rows ~= m or Rm.cols ~= n then
			return false, "shape"
		end
		local re, im = M.mattotable(Rm)
		for i = 1, m do
			for j = 1, math.min(i - 1, n) do
				if re[i][j] ~= 0 or (im and im[i][j] ~= 0) then
					return false, "R is not upper triangular"
				end
			end
		end
		local orth = diff(M, M.matmul(nil, adjoint(case, Q), Q), M.identity(m))
		local recon = diff(M, M.matmul(nil, Q, Rm), A) / math.max(M.norm(A.data), 1)
		return orth <= 30 * tol and recon <= 30 * tol, `orth {orth}, recon {recon}`
	end

	test(`{name} QR updates`, function()
		math.randomseed(701)
		for _, shape in { { 7, 4 }, { 5, 5 }, { 3, 6 } } do
			local m, n = shape[1], shape[2]
			local A = random_matrix(case, m, n)
			local A0 = M.matcopy(nil, A)
			local F = M.qr_full(A)
			local ok, why = valid(F, A)
			check(ok and diff(M, A, A0) == 0, `{m}x{n}: qr_full ({why})`)
			-- Rank-one update.
			local u, v = random_vector(case, m), random_vector(case, n)
			M.qr_update(F, u, v)
			local A1 = M.ger(M.matcopy(nil, A), 1, u, v)
			ok, why = valid(F, A1)
			check(ok, `{m}x{n}: rank-one update ({why})`)
			-- Delete column 2, then insert a new column at 1 and at the end.
			M.qr_delete_col(F, 2)
			local keep = M.hstack({ M.submatrix(nil, A1, 1, m, 1, 1), M.submatrix(nil, A1, 1, m, 3, n) })
			ok, why = valid(F, keep)
			check(ok, `{m}x{n}: delete a column ({why})`)
			local x = random_vector(case, m)
			M.qr_insert_col(F, 1, x)
			local X = M.matrix(m, 1)
			X.data = M.copy(nil, x)
			local inserted = M.hstack({ X, keep })
			ok, why = valid(F, inserted)
			check(ok, `{m}x{n}: insert a column at the front ({why})`)
			local y = random_vector(case, m)
			M.qr_insert_col(F, inserted.cols + 1, y)
			local Y = M.matrix(m, 1)
			Y.data = M.copy(nil, y)
			ok, why = valid(F, M.hstack({ inserted, Y }))
			check(ok, `{m}x{n}: append a column ({why})`)
		end
		local F = M.qr_full(random_matrix(case, 4, 3))
		check(not pcall(function()
			M.qr_delete_col(F, 4)
		end) and not pcall(function()
			M.qr_insert_col(F, 5, random_vector(case, 4))
		end), "positions out of range are errors")
	end)

	test(`{name} Cholesky updates`, function()
		math.randomseed(702)
		local n = 8
		local G = random_matrix(case, n, n)
		local A = M.matmul(nil, adjoint(case, G), G)
		A.data = M.axpy(nil, 1, M.identity(n).data, A.data)
		local F = M.chol_copy(A)
		local x = random_vector(case, n)
		local x0 = M.copy(nil, x)
		local function LLh(Fc: lib.Cholesky): lib.Matrix
			local L = M.chol_l(nil, Fc)
			return M.matmul(nil, L, adjoint(case, L))
		end
		M.chol_update(F, x)
		local Ax = M.ger(M.matcopy(nil, A), 1, x, x)
		check(diff(M, LLh(F), Ax) <= 30 * tol * M.norm(Ax.data), "update: L L^H = A + x x^H")
		check(M.norm(M.sub(nil, x, x0)) == 0, "x is not modified")
		local d = M.diag(nil, M.chol_l(nil, F))
		local positive = true
		for i = 1, n do
			local v = M.get(d, i)
			local re, im = if typeof(v) == "vector" then v.x else v, if typeof(v) == "vector" then v.y else select(2, M.get(d, i))
			positive = positive and re > 0 and (im == nil or im == 0)
		end
		check(positive, "update: real positive diagonal")
		M.chol_downdate(F, x)
		check(diff(M, LLh(F), A) <= 100 * tol * M.norm(A.data), "downdate undoes the update")
		-- A downdate that would lose definiteness is an error and changes nothing.
		local big = M.scale(nil, 10, M.copy(nil, x))
		local before = M.matcopy(nil, F.chol)
		check(not pcall(function()
			M.chol_downdate(F, big)
		end), "an indefinite downdate is an error")
		check(diff(M, F.chol, before) == 0, "and leaves F unchanged")
	end)

	test(`{name} randomized SVD`, function()
		math.randomseed(703)
		-- Exactly rank 5: recovered to working precision.
		local m, n, r = 40, 30, 5
		local A = M.matmul(nil, random_matrix(case, m, r), random_matrix(case, r, n))
		local S = M.rsvd(A, r, { seed = 7 })
		local U, V = S.u :: lib.Matrix, S.v :: lib.Matrix
		check(U.rows == m and U.cols == r and V.rows == n and V.cols == r and R.length(S.values) == r, "shapes")
		local exact = R.totable(M.svd_copy(A, false).values)
		local got = R.totable(S.values)
		local agree = true
		for i = 1, r do
			agree = agree and math.abs(got[i] - exact[i]) <= 1e3 * eps * exact[1]
		end
		check(agree, "rank 5: the singular values")
		local rows = {}
		for i = 1, r do
			rows[i] = table.create(r, 0)
			rows[i][i] = got[i]
		end
		local zeros = {}
		for i = 1, r do
			zeros[i] = table.create(r, 0)
		end
		local D = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
		local recon = diff(M, M.matmul(nil, M.matmul(nil, U, D), adjoint(case, V)), A) / M.norm(A.data)
		check(recon <= 1e3 * eps, `rank 5: U S V^H = A ({recon})`)
		check(diff(M, M.matmul(nil, adjoint(case, U), U), M.identity(r)) <= 1e3 * eps, "U has orthonormal columns")
		-- Decaying singular values 2^-i: the rank-k error is close to s_(k+1).
		local k = 6
		local Q1 = M.qr_q(nil, M.qr(random_matrix(case, m, n)))
		local Q2 = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
		local srows = {}
		local szeros = {}
		for i = 1, n do
			srows[i] = table.create(n, 0)
			szeros[i] = table.create(n, 0)
			srows[i][i] = 2 ^ -(i - 1)
		end
		local Sd = if case.complex then M.matfrom(srows, szeros) else M.matfrom(srows)
		local B = M.matmul(nil, M.matmul(nil, Q1, Sd), adjoint(case, Q2))
		local T = M.rsvd(B, k, { seed = 9 })
		local tv = R.totable(T.values)
		local trows = {}
		local tzeros = {}
		for i = 1, k do
			trows[i] = table.create(k, 0)
			tzeros[i] = table.create(k, 0)
			trows[i][i] = tv[i]
		end
		local Dt = if case.complex then M.matfrom(trows, tzeros) else M.matfrom(trows)
		local err = M.matnorm(M.matrix(1, 1), 1)
		local E = M.matrix(m, n)
		E.data = M.sub(nil, B.data, M.matmul(nil, M.matmul(nil, T.u :: lib.Matrix, Dt), adjoint(case, T.v :: lib.Matrix)).data)
		err = M.matnorm(E, 2)
		check(err <= 1.5 * 2 ^ -k, `decaying spectrum: |A - A_k|_2 = {err} vs s_(k+1) = {2 ^ -k}`)
		local T2 = M.rsvd(B, k, { seed = 9 })
		check(R.norm(R.sub(nil, T2.values, T.values)) == 0, "a seed makes it reproducible")
	end)
end

return nil
