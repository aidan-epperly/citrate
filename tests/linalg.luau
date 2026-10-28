--!strict

-- The user-facing linear algebra: each function against the library function it wraps,
-- answers in the form asked (tables or arrays), the methods it picks, and its errors.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local linalg, f64 = lib.linalg, lib.f64

local seed = 31
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647 - 0.5
end

local function random_rows(m: number, n: number): { { number } }
	local t = {}
	for i = 1, m do
		t[i] = {}
		for j = 1, n do
			t[i][j] = rnd()
		end
	end
	return t
end

local function spd_rows(n: number): { { number } }
	local A = random_rows(n, n)
	local S = {}
	for i = 1, n do
		S[i] = {}
		for j = 1, n do
			local s = 0
			for k = 1, n do
				s += A[i][k] * A[j][k]
			end
			S[i][j] = s + (if i == j then n else 0)
		end
	end
	return S
end

-- the 1-D Laplacian (symmetric positive definite) as entries
local function laplacian_entries(n: number): { { number } }
	local e = {}
	for i = 1, n do
		table.insert(e, { i, i, 2 })
		if i < n then
			table.insert(e, { i, i + 1, -1 })
			table.insert(e, { i + 1, i, -1 })
		end
	end
	return e
end

local function errors(f: () -> (), pattern: string): boolean
	local ok, message = pcall(f)
	return not ok and string.find(tostring(message), pattern, 1, true) ~= nil
end

test("linalg: solve picks its method and answers in kind", function()
	local n = 12
	local b = {}
	for i = 1, n do
		b[i] = math.sin(i)
	end
	-- symmetric positive definite: Cholesky; general: LU; tables in, tables out
	local x, info = linalg.solve(spd_rows(n), b)
	check(type(x) == "table" and #x == n and info.method == "Cholesky" and (info.residual :: number) < 1e-14, "Cholesky: " .. info.method)
	local G = random_rows(n, n)
	local y, info2 = linalg.solve(G, b)
	check(info2.method == "LU" and (info2.residual :: number) < 1e-13, "LU")
	-- the same answer as the library's own solve
	local direct = f64.solve(nil, linalg.matrix(G), f64.from(b))
	local err = 0
	for i = 1, n do
		err = math.max(err, math.abs(y[i] - f64.get(direct, i)))
	end
	check(err < 1e-12, "agrees with f64.solve")
	-- arrays in, arrays out; the caller's matrix untouched
	local M = linalg.matrix(G)
	local before = f64.copy(nil, M.data)
	local z = linalg.solve(M, f64.from(b))
	check(type(z) == "buffer" and f64.norm(f64.sub(nil, M.data, before)) == 0, "arrays, input untouched")
	-- symmetric but indefinite: LU after Cholesky fails
	local I = { { 1, 2 }, { 2, 1 } }
	local _, info3 = linalg.solve(I, { 1, 1 })
	check(string.find(info3.method, "not positive definite", 1, true) ~= nil, info3.method)
	-- sparse: Cholesky for the Laplacian, LU for a nonsymmetric one
	local S = linalg.sparse(50, 50, laplacian_entries(50))
	local rhs = {}
	for i = 1, 50 do
		rhs[i] = 1
	end
	local xs, info4 = linalg.solve(S, rhs)
	check(info4.method == "sparse Cholesky" and (info4.residual :: number) < 1e-13, "sparse Cholesky")
	close(xs[25], 25 * 26 / 2, 1e-9, "Laplacian solution")
	local entries = laplacian_entries(50)
	table.insert(entries, { 1, 50, 0.5 })
	local _, info5 = linalg.solve(linalg.sparse(50, 50, entries), rhs)
	check(info5.method == "sparse LU" and (info5.residual :: number) < 1e-13, "sparse LU")
end)

test("linalg: solve's checks and warnings", function()
	check(errors(function()
		linalg.solve({ { 1, 2 }, { 3 } }, { 1, 1 })
	end, "row 2 has 1 entries"), "ragged rows")
	check(errors(function()
		linalg.solve({ { 1, 0 }, { 0, 0 / 0 } }, { 1, 1 })
	end, "not a finite number"), "NaN")
	check(errors(function()
		linalg.solve({ { 1, 2, 3 }, { 4, 5, 6 } }, { 1, 1 })
	end, "square"), "not square")
	check(errors(function()
		linalg.solve({ { 1, 2 }, { 2, 4 } }, { 1, 1 })
	end, "singular"), "singular")
	check(errors(function()
		linalg.solve({ { 1, 0 }, { 0, 1 } }, { 1, 1, 1 })
	end, "vector of 2"), "wrong vector length")
	-- nearly singular: an answer, with a warning
	local _, info = linalg.solve({ { 1, 1 }, { 1, 1 + 1e-14 } }, { 2, 2 })
	check(info.warning ~= nil and (info.rcond :: number) < 1e-12, "a warning when nearly singular")
	-- a matrix that is almost but not quite symmetric is treated as general
	local _, info2 = linalg.solve({ { 2, 1 }, { 1.001, 2 } }, { 1, 1 })
	check(info2.method == "LU", "not symmetric within tolerance")
end)

test("linalg: least squares, eigenvalues, svd and the rest", function()
	-- least squares: a line through noisy points
	local A, b = {}, {}
	for i = 1, 20 do
		A[i] = { 1, i }
		b[i] = 2 + 3 * i + 0.01 * math.sin(7 * i)
	end
	local c, info = linalg.lstsq(A, b)
	close(c[1], 2, 0.01, "intercept")
	close(c[2], 3, 1e-3, "slope")
	check(info.rank == 2, "rank")
	local _, info2 = linalg.lstsq({ { 1, 2 }, { 2, 4 }, { 3, 6 } }, { 1, 2, 3 })
	check(info2.rank == 1 and info2.warning ~= nil, "rank deficient")
	-- eigenvalues of a dense symmetric matrix against f64.eigh
	local S = spd_rows(8)
	local values, vectors = linalg.eig(S)
	local E = f64.eigh_copy(linalg.matrix(S), false)
	local err = 0
	for k = 1, 8 do
		err = math.max(err, math.abs(values[k] - f64.get(E.values, k)))
	end
	check(err < 1e-12 and #vectors == 8 and type(vectors[1]) == "table", "dense eig")
	local top = linalg.eig(S, { count = 2, largest = true })
	check(#top == 2 and math.abs(top[2] - values[8]) < 1e-12, "largest two")
	-- the sparse Laplacian's smallest and largest: 2 - 2 cos(k pi / (n + 1))
	local n = 400
	local L = linalg.sparse(n, n, laplacian_entries(n))
	local small, _, info3 = linalg.eig(L, { count = 4 })
	local large = linalg.eig(L, { count = 3, largest = true })
	local exact = function(k: number): number
		return 2 - 2 * math.cos(k * math.pi / (n + 1))
	end
	local e2 = 0
	for k = 1, 4 do
		e2 = math.max(e2, math.abs(small[k] - exact(k)))
	end
	for k = 1, 3 do
		e2 = math.max(e2, math.abs(large[k] - exact(n - 3 + k)))
	end
	check(e2 < 1e-9 and (info3.residual :: number) < 1e-8, string.format("sparse eig (error %.1e)", e2))
	check(errors(function()
		linalg.eig({ { 1, 2 }, { 0, 1 } })
	end, "not symmetric"), "eig of a nonsymmetric matrix")
	-- svd: A = U diag(s) V^T
	local G = random_rows(5, 3)
	local s, U, V = linalg.svd(G)
	local worst = 0
	for i = 1, 5 do
		for j = 1, 3 do
			local a = 0
			for k = 1, 3 do
				a += U[k][i] * s[k] * V[k][j]
			end
			worst = math.max(worst, math.abs(a - G[i][j]))
		end
	end
	check(worst < 1e-12 and s[1] >= s[2] and s[2] >= s[3], "svd reconstructs")
	-- det: dense and sparse agree
	local D = random_rows(6, 6)
	local entries = {}
	for i = 1, 6 do
		for j = 1, 6 do
			table.insert(entries, { i, j, D[i][j] })
		end
	end
	close(linalg.det(linalg.sparse(6, 6, entries)), linalg.det(D), 1e-12, "sparse det")
	close(linalg.det({ { 2, 1 }, { 1, 3 } }), 5, 1e-14, "2 by 2 det")
	-- inv, norm, cond
	local Ai = linalg.inv({ { 4, 7 }, { 2, 6 } })
	close(Ai[1][1], 0.6, 1e-14, "inverse")
	close(linalg.norm({ 3, 4 }), 5, 1e-15, "vector norm")
	close(linalg.norm({ { 3, 0 }, { 0, 4 } }), 4, 1e-14, "matrix norm")
	close(linalg.cond({ { 1, 0 }, { 0, 100 } }), 100, 1e-12, "condition number")
	-- totable round trips
	local T = linalg.totable(linalg.matrix({ { 1, 2 }, { 3, 4 } }))
	check(T[2][1] == 3, "totable of a matrix")
	local Te = linalg.totable(linalg.sparse(2, 2, { { 2, 1, 5 } }))
	check(#Te == 1 and Te[1][1] == 2 and Te[1][3] == 5, "totable of a sparse matrix")
end)

return nil
