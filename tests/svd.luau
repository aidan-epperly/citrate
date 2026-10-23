--!strict

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close

-- R is the real module of the same precision, which holds singular values.
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

local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

local function diagonal(case: Case, values: { number }): lib.Matrix
	local n = #values
	local rows = {}
	for i = 1, n do
		rows[i] = table.create(n, 0)
		rows[i][i] = values[i]
	end
	return from_rows(case, rows)
end

local function diff(M: any, A: lib.Matrix, B: lib.Matrix): number
	return M.norm(M.sub(nil, A.data, B.data))
end

for _, case in cases do
	local M, R, name, eps = case.M, case.R, case.name, case.eps

	test(`{name} SVD reproduces A`, function()
		math.randomseed(71)
		for _, shape in { { 8, 5 }, { 5, 8 }, { 6, 6 }, { 1, 1 }, { 1, 5 }, { 5, 1 }, { 30, 20 }, { 20, 30 } } do
			local m, n = shape[1], shape[2]
			local k = math.min(m, n)
			local A = random_matrix(case, m, n)
			local A0 = M.matcopy(nil, A)
			local F = M.svd(A)
			local U, V = F.u :: lib.Matrix, F.v :: lib.Matrix
			local s = R.totable(F.values)
			check(#s == k and U.rows == m and U.cols == k and V.rows == n and V.cols == k, `{m}x{n}: shapes`)
			if m >= n then
				check(U == A, `{m}x{n}: U overwrites A`)
			else
				check(diff(M, A, A0) == 0, `{m}x{n}: a wide A is not modified`)
			end
			local ordered = true
			for i = 1, k do
				ordered = ordered and s[i] >= 0 and (i == 1 or s[i] <= s[i - 1])
			end
			check(ordered, `{m}x{n}: nonnegative and descending`)
			local tol = 30 * math.max(m, n) * eps
			local recon = M.matmul(nil, M.matmul(nil, U, diagonal(case, s)), adjoint(case, V))
			local err = diff(M, recon, A0) / M.norm(A0.data)
			check(err <= tol, `{m}x{n}: |U S V^H - A| / |A| = {err}`)
			check(diff(M, M.matmul(nil, adjoint(case, U), U), M.identity(k)) <= tol, `{m}x{n}: U has orthonormal columns`)
			check(diff(M, M.matmul(nil, adjoint(case, V), V), M.identity(k)) <= tol, `{m}x{n}: V has orthonormal columns`)
			local only = R.totable(M.svd_copy(A0, false).values)
			local agree = true
			for i = 1, k do
				agree = agree and math.abs(only[i] - s[i]) <= tol * s[1]
			end
			check(agree, `{m}x{n}: values without vectors agree`)
		end
	end)

	test(`{name} SVD on known matrices`, function()
		local function values(rows: { { number } }): { number }
			return R.totable(M.svd_copy(from_rows(case, rows), false).values)
		end
		local s = values({ { 3, 0, 0 }, { 0, -2, 0 }, { 0, 0, 1 } })
		check(s[1] == 3 and s[2] == 2 and s[3] == 1, "diagonal, with a sign")
		s = values({ { 3, 0 }, { 4, 5 } })
		close(s[1], math.sqrt(45), 4 * eps)
		close(s[2], math.sqrt(5), 4 * eps, "[[3, 0], [4, 5]]")
		-- Rank 1: (1, 2, 3)^T (4, 5) has one singular value |a| |b|.
		s = values({ { 4, 5 }, { 8, 10 }, { 12, 15 } })
		close(s[1], math.sqrt(14) * math.sqrt(41), 8 * eps, "rank 1: largest")
		check(s[2] <= 30 * eps * s[1], "rank 1: second is ~0")
		s = values({ { 0, 0 }, { 0, 0 } })
		check(s[1] == 0 and s[2] == 0, "zero matrix")
		s = values({ { -7 } })
		check(s[1] == 7, "1 by 1")
		s = values({ { 1, 0 }, { 0, 1 } })
		check(s[1] == 1 and s[2] == 1, "identity")
		local E = M.svd(M.matrix(0, 0))
		check(buffer.len(E.values) == 0, "0 by 0")
	end)

	test(`{name} SVD agrees with eigh of A^H A`, function()
		math.randomseed(72)
		local A = random_matrix(case, 12, 7)
		local s = R.totable(M.svd_copy(A, false).values)
		local lambda = R.totable(M.eigh(M.matmul(nil, adjoint(case, A), A), false).values)
		local agree = true
		for i = 1, 7 do
			agree = agree and math.abs(s[i] * s[i] - lambda[8 - i]) <= 1e3 * eps * s[1] * s[1]
		end
		check(agree, "squared singular values are the eigenvalues of A^H A")
	end)

	test(`{name} SVD keeps tiny singular values relatively accurate`, function()
		-- A graded bidiagonal matrix whose singular values span many orders of magnitude.
		-- Their product is |det| = the product of the diagonal; an algorithm that is only
		-- accurate relative to the largest value would get the small ones badly wrong.
		-- Graded both ways: large entries at the top, then at the bottom.
		local g = if name == "f32" or name == "c32" then 1e-4 else 1e-7
		local n = 5
		for _, downward in { true, false } do
			local rows = {}
			for i = 1, n do
				local p = if downward then i - 1 else n - i
				rows[i] = table.create(n, 0)
				rows[i][i] = g ^ p
				if i < n then
					rows[i][i + 1] = g ^ (if downward then p else p - 1)
				end
			end
			local A = from_rows(case, rows)
			local det = 1
			local re = M.mattotable(A)
			for i = 1, n do
				det *= math.abs(re[i][i])
			end
			local s = R.totable(M.svd_copy(A, false).values)
			local product = 1
			for i = 1, n do
				product *= s[i]
			end
			local rel = math.abs(product - det) / det
			local label = if downward then "graded downward" else "graded upward"
			check(rel <= 1e3 * eps, `{label}: product of singular values matches |det| (relative error {rel})`)
			check(s[n] > 0 and s[n] < 10 * g ^ (n - 1), `{label}: smallest value {s[n]} has the right size`)
		end

		-- Not graded at all: every entry is 1 or 2, yet the smallest singular value is about
		-- 2^-n. Its determinant is 1, so the values multiply to 1. Plain shifted QR only
		-- gets s_min to within eps * s_max, which is a large relative error here; the
		-- zero-shift sweeps are what make this accurate.
		local k = 30
		local brows = {}
		for i = 1, k do
			brows[i] = table.create(k, 0)
			brows[i][i] = 1
			if i < k then
				brows[i][i + 1] = 2
			end
		end
		local t = R.totable(M.svd_copy(from_rows(case, brows), false).values)
		local logp = 0
		for i = 1, k do
			logp += math.log(t[i])
		end
		check(t[k] < 1e-8, `ungraded: s_min = {t[k]} is tiny`)
		check(math.abs(logp) <= 1e3 * eps, `ungraded: values multiply to 1 (log of product {logp})`)
	end)
	test(`{name} SVD returns NaN for non-finite input`, function()
		local single = eps == h.eps32
		for _, shape in { { 5, 3 }, { 3, 5 } } do
			for _, bad in { 0 / 0, -math.huge } do
				local re, im = M.mattotable(random_matrix(case, shape[1], shape[2]))
				re[2][3] = bad
				local A = if im then M.matfrom(re, im) else M.matfrom(re)
				local F = M.svd_copy(A)
				local all = h.all_nan(F.values, single)
					and h.all_nan((F.u :: lib.Matrix).data, single)
					and h.all_nan((F.v :: lib.Matrix).data, single)
				check(all, `{shape[1]}x{shape[2]}: {bad} gives NaN`)
				check(h.all_nan(M.svd_copy(A, false).values, single), `{shape[1]}x{shape[2]}: {bad} without vectors gives NaN`)
			end
		end
	end)

	test(`{name} SVD on extreme scales`, function()
		math.randomseed(72)
		local scales = if eps == h.eps32 then { 2 ^ 100, 2 ^ -100 } else { 2 ^ 1000, 2 ^ -1000 }
		for _, shape in { { 7, 4 }, { 4, 7 } } do
			local m, n = shape[1], shape[2]
			local A = random_matrix(case, m, n)
			local base = R.totable(M.svd_copy(A, false).values)
			for _, sc in scales do
				local As = M.matcopy(nil, A)
				As.data = M.scale(nil, sc, As.data)
				local F = M.svd_copy(As)
				local s = R.totable(F.values)
				local tol = 30 * math.max(m, n) * eps
				local agree = true
				for i = 1, #s do
					agree = agree and math.abs(s[i] / sc - base[i]) <= tol * base[1]
				end
				check(agree, `{m}x{n} scale {sc}: values`)
				local recon = M.matmul(nil, M.matmul(nil, F.u :: lib.Matrix, diagonal(case, s)), adjoint(case, F.v :: lib.Matrix))
				local err = diff(M, recon, As) / M.norm(As.data)
				check(err <= tol, `{m}x{n} scale {sc}: |U S V^H - A| / |A| = {err}`)
			end
		end
	end)

	test(`{name} SVD on a larger matrix`, function()
		math.randomseed(73)
		local m, n = 80, 60
		local A = random_matrix(case, m, n)
		local A0 = M.matcopy(nil, A)
		local F = M.svd(A)
		local U, V = F.u :: lib.Matrix, F.v :: lib.Matrix
		local tol = 30 * m * eps
		local recon = M.matmul(nil, M.matmul(nil, U, diagonal(case, R.totable(F.values))), adjoint(case, V))
		local err = diff(M, recon, A0) / M.norm(A0.data)
		check(err <= tol, `{m}x{n}: |U S V^H - A| / |A| = {err}`)
		check(diff(M, M.matmul(nil, adjoint(case, U), U), M.identity(n)) <= tol, `{m}x{n}: U has orthonormal columns`)
		check(diff(M, M.matmul(nil, adjoint(case, V), V), M.identity(n)) <= tol, `{m}x{n}: V has orthonormal columns`)
	end)
	test(`{name} SVD pseudoinverse solve`, function()
		math.randomseed(74)
		local function vec(n: number): buffer
			return if case.complex then M.from(random_values(n), random_values(n)) else M.from(random_values(n))
		end
		local function dist(x: buffer, y: buffer): number
			return M.norm(M.sub(nil, x, y))
		end
		-- |A^H (A x - b)|, zero exactly for least-squares solutions.
		local function normal(A: lib.Matrix, x: buffer, b: buffer): number
			return M.norm(M.matvec(nil, adjoint(case, A), M.sub(nil, M.matvec(nil, A, x), b)))
		end
		local tol = 1e3 * eps

		-- Tall, full rank: the least-squares solution, as QR finds it.
		local A = random_matrix(case, 8, 5)
		local b = vec(8)
		local F = M.svd_copy(A)
		local x, resid, r = M.svd_solve(nil, F, b)
		check(r == 5, "tall: full rank")
		check(dist(x, M.qr_solve(nil, M.qr_copy(A), b)) <= tol * M.norm(x), "tall: agrees with QR")
		close(resid, M.norm(M.sub(nil, M.matvec(nil, A, x), b)), tol, "tall: residual norm")
		check(dist(M.solve(nil, F, b, "svd"), x) == 0, "form \"svd\"")
		local o = M.create(5)
		check(M.solve(o, F, b, "svd") == o, "writes into out")

		-- Rank 3 (A = B C): of all least-squares solutions, the shortest, which has no
		-- component in the null space of A.
		local R3 = M.matmul(nil, random_matrix(case, 7, 3), random_matrix(case, 3, 5))
		local c = vec(7)
		local G = M.svd_copy(R3)
		local y, resid3, r3 = M.svd_solve(nil, G, c)
		check(r3 == 3, `rank 3 detected (got {r3})`)
		check(normal(R3, y, c) <= tol * M.norm(R3.data) ^ 2 * M.norm(y), "rank 3: least squares")
		close(resid3, M.norm(M.sub(nil, M.matvec(nil, R3, y), c)), tol, "rank 3: residual norm")
		local V = G.v :: lib.Matrix
		local Vh = adjoint(case, V)
		local proj = M.matvec(nil, Vh, y)
		local re, im = M.totable(proj)
		local outside = 0
		for i = 4, 5 do
			outside = math.max(outside, math.abs(re[i]), if im then math.abs(im[i]) else 0)
		end
		check(outside <= tol * M.norm(y), `rank 3: no null-space component ({outside})`)
		local basic = M.qrp_solve(nil, M.qrp_copy(R3), c)
		check(M.norm(y) <= M.norm(basic) * (1 + tol), "rank 3: no longer than the basic solution")
		check(normal(R3, basic, c) <= 1e3 * tol * M.norm(R3.data) ^ 2 * M.norm(basic), "rank 3: the basic solution is least squares too")

		-- Wide, full row rank: A x = b exactly, with x = A^H (A A^H)^-1 b.
		local W = random_matrix(case, 3, 6)
		local w = vec(3)
		local z, residw, rw = M.svd_solve(nil, M.svd_copy(W), w)
		check(rw == 3 and residw <= tol * M.norm(w), "wide: A x = b")
		local want = M.matvec(nil, adjoint(case, W), M.solve(nil, M.matmul(nil, W, adjoint(case, W)), w))
		check(dist(z, want) <= tol * M.norm(want), "wide: minimum norm")

		-- A tolerance above a small perturbation drops it.
		local N = M.matmul(nil, random_matrix(case, 6, 1), random_matrix(case, 1, 4))
		N.data = M.add(nil, N.data, M.scale(nil, 1e-3, random_matrix(case, 6, 4).data))
		local NF = M.svd_copy(N)
		local _, _, full = M.svd_solve(nil, NF, vec(6))
		local _, _, one = M.svd_solve(nil, NF, vec(6), 0.1)
		check(full == 4 and one == 1, `tolerance sets the rank ({full}, {one})`)

		-- Degenerate inputs.
		local zb = vec(5)
		local zx, zres, zr = M.svd_solve(nil, M.svd(M.matrix(5, 3)), zb)
		check(zr == 0 and M.norm(zx) == 0, "zero matrix: x = 0")
		close(zres, M.norm(zb), 1e-14, "zero matrix: residual |b|")
		local nre, nim = M.mattotable(random_matrix(case, 4, 3))
		nre[2][2] = 0 / 0
		local nx = M.svd_solve(nil, M.svd(if nim then M.matfrom(nre, nim) else M.matfrom(nre)), vec(4))
		check(h.all_nan(nx, eps == h.eps32), "NaN input gives NaN")
		check(not pcall(function()
			M.svd_solve(nil, M.svd_copy(A, false), b)
		end), "needs the vectors")
		check(not pcall(function()
			M.solve(nil, M.svd_copy(A, false), b, "svd")
		end), "solve: needs the vectors")
		check(not pcall(function()
			M.solve(nil, F, b, "lu")
		end), "solve: SVD passed as \"lu\"")
		check(not pcall(function()
			M.solve(nil, M.lu_copy(random_matrix(case, 3, 3)), vec(3), "svd")
		end), "solve: LU passed as \"svd\"")
		check(not pcall(function()
			M.svd_solve(nil, F, vec(7))
		end), "b of the wrong length")
	end)
	test(`{name} pseudoinverse matrix`, function()
		math.randomseed(75)
		local function mul(A: lib.Matrix, B: lib.Matrix): lib.Matrix
			return M.matmul(nil, A, B)
		end
		-- The four Penrose conditions characterize A^+.
		local function penrose(A: lib.Matrix, X: lib.Matrix): number
			local AX, XA = mul(A, X), mul(X, A)
			return math.max(
				diff(M, mul(AX, A), A) / M.norm(A.data),
				diff(M, mul(XA, X), X) / M.norm(X.data),
				diff(M, adjoint(case, AX), AX),
				diff(M, adjoint(case, XA), XA)
			)
		end
		local tol = 1e3 * eps
		for _, spec in { { 7, 5, 5 }, { 3, 6, 3 }, { 7, 5, 3 }, { 4, 6, 2 } } do
			local m, n, r = spec[1], spec[2], spec[3]
			local A = mul(random_matrix(case, m, r), random_matrix(case, r, n))
			local A0 = M.matcopy(nil, A)
			local X = M.pinv(A)
			check(X.rows == n and X.cols == m, `{m}x{n} rank {r}: shape`)
			check(diff(M, A, A0) == 0, `{m}x{n} rank {r}: A is not modified`)
			local e = penrose(A, X)
			check(e <= 100 * tol, `{m}x{n} rank {r}: Penrose conditions ({e})`)
			local b = random_matrix(case, m, 1).data
			local x = M.svd_solve(nil, M.svd_copy(A), b)
			check(M.norm(M.sub(nil, M.matvec(nil, X, b), x)) <= tol * M.norm(x), `{m}x{n} rank {r}: X b = svd_solve`)
		end
		local S = random_matrix(case, 4, 4)
		local Xs = M.pinv(S)
		check(diff(M, Xs, M.inv_copy(S)) <= tol * M.norm(Xs.data), "square nonsingular: the inverse")

		-- A tolerance above a small perturbation drops it: the result is close to the
		-- pseudoinverse of the rank-1 part.
		local N = mul(random_matrix(case, 5, 1), random_matrix(case, 1, 3))
		local N1 = M.matcopy(nil, N)
		N.data = M.add(nil, N.data, M.scale(nil, 1e-3, random_matrix(case, 5, 3).data))
		local e = penrose(N1, M.pinv(N, 0.1))
		check(e <= 1e-2, `tolerance drops the perturbation ({e})`)

		local Z = M.pinv(M.matrix(3, 2))
		check(Z.rows == 2 and Z.cols == 3 and M.norm(Z.data) == 0, "zero matrix")
		local nre, nim = M.mattotable(random_matrix(case, 3, 2))
		nre[1][2] = 0 / 0
		check(h.all_nan(M.pinv(if nim then M.matfrom(nre, nim) else M.matfrom(nre)).data, eps == h.eps32), "NaN gives NaN")
	end)
end

return nil
