--!strict

-- Sparse Cholesky: the elimination tree and column counts against a naive symbolic
-- elimination, AMD's fill against exact minimum degree, and solves on all four types
-- (orderings, relaxed supernodes or not, reuse, indefinite matrices, disconnected graphs).

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check
local graph = lib.graph

local seed = 777
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647
end

-- a random symmetric pattern: edges of a random geometric graph (local, like a mesh)
local function random_graph(n: number, reach: number): ({ number }, { number })
	local xs, ys = {}, {}
	for i = 1, n do
		xs[i], ys[i] = rnd(), rnd()
	end
	local ii, jj = {}, {}
	for i = 1, n do
		for j = i + 1, n do
			if (xs[i] - xs[j]) ^ 2 + (ys[i] - ys[j]) ^ 2 < reach * reach then
				table.insert(ii, i)
				table.insert(jj, j)
			end
		end
	end
	return ii, jj
end

-- the column patterns of L (rows below the diagonal, positions) for vertices ordered by
-- perm, by eliminating with explicit sets
local function naive_symbolic(n: number, ii: { number }, jj: { number }, perm: buffer): { { number } }
	local pos = {}
	for k = 1, n do
		pos[buffer.readi32(perm, 4 * (k - 1))] = k
	end
	local adj: { { [number]: boolean } } = {}
	for k = 1, n do
		adj[k] = {}
	end
	for e = 1, #ii do
		local a, b = pos[ii[e]], pos[jj[e]]
		if a ~= b then
			adj[a][b] = true
			adj[b][a] = true
		end
	end
	local cols = {}
	for k = 1, n do
		local below = {}
		for u in adj[k] do
			if u > k then
				table.insert(below, u)
			end
		end
		table.sort(below)
		cols[k] = below
		for _, a in below do
			for _, b in below do
				if a ~= b then
					adj[a][b] = true
				end
			end
		end
	end
	return cols
end

-- exact minimum degree by explicit elimination (the reference for AMD): nnz(L)
local function exact_md_fill(n: number, ii: { number }, jj: { number }): number
	local adj: { { [number]: boolean } } = {}
	for k = 1, n do
		adj[k] = {}
	end
	for e = 1, #ii do
		if ii[e] ~= jj[e] then
			adj[ii[e]][jj[e]] = true
			adj[jj[e]][ii[e]] = true
		end
	end
	local alive = table.create(n, true)
	local nnz = 0
	for _ = 1, n do
		local best, bd = 0, math.huge
		for v = 1, n do
			if alive[v] then
				local d = 0
				for _ in adj[v] do
					d += 1
				end
				if d < bd then
					best, bd = v, d
				end
			end
		end
		alive[best] = false
		nnz += bd + 1
		local nb = {}
		for u in adj[best] do
			table.insert(nb, u)
			adj[u][best] = nil
		end
		for _, a in nb do
			for _, b in nb do
				if a ~= b then
					adj[a][b] = true
				end
			end
		end
		adj[best] = {}
	end
	return nnz
end

test("sparse Cholesky: elimination tree, postorder and column counts", function()
	for trial = 1, 4 do
		local n = 60
		local ii, jj = random_graph(n, 0.2)
		local G = graph.from_edges(n, ii, jj)
		local perm = if trial % 2 == 0 then graph.amd(G) else graph.rcm(G)
		local cols = naive_symbolic(n, ii, jj, perm)
		local parent, post = graph.etree(G, perm)
		local ok = true
		for k = 1, n do
			local expect = if #cols[k] > 0 then cols[k][1] else 0
			ok = ok and buffer.readi32(parent, 4 * (k - 1)) == expect
		end
		check(ok, "parents, trial " .. trial)
		-- postorder: a permutation, children before parents
		local where = {}
		for k = 1, n do
			where[buffer.readi32(post, 4 * (k - 1))] = k
		end
		local post_ok = #h.ints(post) == n
		for k = 1, n do
			local pk = buffer.readi32(parent, 4 * (k - 1))
			post_ok = post_ok and where[k] ~= nil and (pk == 0 or where[k] < where[pk])
		end
		check(post_ok, "postorder, trial " .. trial)
		-- the analysis (postordered: the same fill) without merging counts exactly nnz(L)
		local sym = graph.chol_analysis(G, perm, { relax = false })
		local nnz = n
		for k = 1, n do
			nnz += #cols[k]
		end
		check(sym.lnz == nnz, string.format("nnz(L) %d against %d, trial %d", sym.lnz, nnz, trial))
	end
end)

test("sparse Cholesky: AMD's fill against exact minimum degree", function()
	for trial = 1, 3 do
		local n = 150
		local ii, jj = random_graph(n, 0.13)
		local G = graph.from_edges(n, ii, jj)
		local perm = graph.amd(G)
		local inv: { [number]: number } = {}
		for k = 1, n do
			local v = buffer.readi32(perm, 4 * (k - 1))
			inv[v] = (inv[v] or 0) + 1
		end
		local valid = true
		for v = 1, n do
			valid = valid and inv[v] == 1
		end
		check(valid, "a permutation, trial " .. trial)
		local amd_fill = graph.chol_analysis(G, perm, { relax = false }).lnz
		local md_fill = exact_md_fill(n, ii, jj)
		local natural = buffer.create(4 * n)
		for k = 1, n do
			buffer.writei32(natural, 4 * (k - 1), k)
		end
		local nat_fill = graph.chol_analysis(G, natural, { relax = false }).lnz
		check(amd_fill <= 1.15 * md_fill, string.format("AMD %d against exact %d, trial %d", amd_fill, md_fill, trial))
		check(amd_fill < nat_fill, string.format("AMD %d against natural %d, trial %d", amd_fill, nat_fill, trial))
	end
	-- a grid: AMD beats the band
	local m = 25
	local ii, jj = {}, {}
	for y = 1, m do
		for x = 1, m do
			local k = (y - 1) * m + x
			if x < m then
				table.insert(ii, k)
				table.insert(jj, k + 1)
			end
			if y < m then
				table.insert(ii, k)
				table.insert(jj, k + m)
			end
		end
	end
	local G = graph.from_edges(m * m, ii, jj)
	local a = graph.chol_analysis(G, graph.amd(G), { relax = false }).lnz
	local r = graph.chol_analysis(G, graph.rcm(G), { relax = false }).lnz
	check(a < 0.8 * r, string.format("grid: AMD %d against RCM %d", a, r))
end)

type Case = { M: any, complex: boolean, eps: number, name: string }
local cases: { Case } = {
	{ M = lib.f64, complex = false, eps = h.eps64, name = "f64" },
	{ M = lib.f32, complex = false, eps = h.eps32, name = "f32" },
	{ M = lib.c64, complex = true, eps = h.eps64, name = "c64" },
	{ M = lib.c32, complex = true, eps = h.eps32, name = "c32" },
}

-- a symmetric (Hermitian) positive definite sparse matrix on the graph: off-diagonal
-- entries in (-1, 1) (complex for complex modules), the diagonal its row's absolute sum
-- plus `shift`
local function spd(case: Case, n: number, ii: { number }, jj: { number }, shift: number): any
	local M = case.M
	local I, J, re, im = {}, {}, {}, {}
	local rowsum = table.create(n, 0)
	for e = 1, #ii do
		local a, b = ii[e], jj[e]
		if a ~= b then
			local xr = 2 * rnd() - 1
			local xi = if case.complex then 2 * rnd() - 1 else 0
			table.insert(I, a)
			table.insert(J, b)
			table.insert(re, xr)
			table.insert(im, xi)
			table.insert(I, b)
			table.insert(J, a)
			table.insert(re, xr)
			table.insert(im, -xi)
			local m = math.sqrt(xr * xr + xi * xi)
			rowsum[a] += m
			rowsum[b] += m
		end
	end
	for i = 1, n do
		table.insert(I, i)
		table.insert(J, i)
		table.insert(re, rowsum[i] + shift)
		table.insert(im, 0)
	end
	local vals = if case.complex then M.from(re, im) else M.from(re)
	local S = M.sparse(n, n, I, J, vals)
	S.symmetric = not case.complex
	S.hermitian = case.complex
	return S
end

local function rhs(case: Case, n: number): buffer
	local re, im = {}, {}
	for i = 1, n do
		re[i] = math.sin(i)
		im[i] = math.cos(3 * i)
	end
	return if case.complex then case.M.from(re, im) else case.M.from(re)
end

local function relres(case: Case, S: any, x: buffer, b: buffer): number
	local M = case.M
	local r = M.sub(nil, M.sparse_matvec(nil, S, x), b)
	return M.norm(r) / M.norm(b)
end

test("sparse Cholesky: solves on all four types", function()
	for _, case in cases do
		local M = case.M
		local n = 120
		local ii, jj = random_graph(n, 0.16)
		local S = spd(case, n, ii, jj, 0.5)
		local b = rhs(case, n)
		local custom = buffer.create(4 * n)
		for k = 1, n do
			buffer.writei32(custom, 4 * (k - 1), n + 1 - k)
		end
		for _, order in { "amd", "rcm", "natural", custom } do
			for _, relax in { true, false } do
				local F = M.sparse_chol(S, { order = order, relax = relax })
				local x = M.sparse_chol_solve(nil, F, b)
				local res = relres(case, S, x, b)
				local label = string.format("%s, order %s, relax %s", case.name, if type(order) == "string" then order else "custom", tostring(relax))
				check(F.definite and res < 100 * case.eps, label .. string.format(": residual %.1e", res))
			end
		end
		-- solving in place
		local F = M.sparse_chol(S)
		local x = M.copy(nil, b)
		M.sparse_chol_solve(x, F, x)
		check(relres(case, S, x, b) < 100 * case.eps, case.name .. ": in place")
	end
end)

test("sparse Cholesky: reuse, indefinite and disconnected matrices", function()
	for _, case in cases do
		local M = case.M
		local n = 90
		local ii, jj = random_graph(n, 0.18)
		local S = spd(case, n, ii, jj, 0.3)
		local F = M.sparse_chol(S)
		-- new values on the same pattern: the same factorization object, refactored
		local S2 = { rows = n, cols = n, colptr = S.colptr, rowind = S.rowind, values = M.scale(nil, 2, S.values), symmetric = S.symmetric, hermitian = S.hermitian }
		local F2 = M.sparse_chol(S2, { reuse = F })
		local b = rhs(case, n)
		check(F2 == F and relres(case, S2, M.sparse_chol_solve(nil, F2, b), b) < 100 * case.eps, case.name .. ": reused")
		-- an indefinite matrix: shift the diagonal far down
		local T = spd(case, n, ii, jj, -50)
		check(not M.sparse_chol(T).definite, case.name .. ": indefinite")
		-- two components and isolated vertices
		local I2, J2 = {}, {}
		for e = 1, #ii do
			if (ii[e] <= 40 and jj[e] <= 40) or (ii[e] > 50 and jj[e] > 50) then
				table.insert(I2, ii[e])
				table.insert(J2, jj[e])
			end
		end
		local U = spd(case, n, I2, J2, 1)
		local Fu = M.sparse_chol(U)
		check(Fu.definite and relres(case, U, M.sparse_chol_solve(nil, Fu, b), b) < 100 * case.eps, case.name .. ": disconnected")
	end
	-- 1 by 1
	local one = lib.f64.sparse(1, 1, { 1 }, { 1 }, lib.f64.from({ 4 }))
	local x = lib.f64.sparse_chol_solve(nil, lib.f64.sparse_chol(one), lib.f64.from({ 2 }))
	check(lib.f64.get(x, 1) == 0.5, "1 by 1")
end)

return nil
