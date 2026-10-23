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
end

return nil
