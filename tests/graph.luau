--!strict

-- Graphs: construction, traversal, reverse Cuthill-McKee (a valid permutation that
-- narrows shuffled grids and ropes), shortest paths against Floyd-Warshall, and the
-- symmetric permutations of sparse matrices on all four types.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local graph, f64, c64 = lib.graph, lib.f64, lib.c64

local function neighbors(G: any, v: number): { number }
	local t = {}
	for p = buffer.readu32(G.ptr, 4 * (v - 1)), buffer.readu32(G.ptr, 4 * v) - 1 do
		table.insert(t, buffer.readu32(G.ind, 4 * p) + 1)
	end
	return t
end

local function same(a: { number }, b: { number }): boolean
	if #a ~= #b then
		return false
	end
	for k = 1, #a do
		if a[k] ~= b[k] then
			return false
		end
	end
	return true
end

-- A random permutation of 1..n as a table.
local function shuffled(n: number): { number }
	local t = {}
	for i = 1, n do
		t[i] = i
	end
	for i = n, 2, -1 do
		local j = math.random(i)
		t[i], t[j] = t[j], t[i]
	end
	return t
end

-- The nx by ny grid graph (4 neighbors), vertex (x, y) numbered label[(y - 1) nx + x].
local function grid(nx: number, ny: number, label: { number }?): any
	local ii, jj = {}, {}
	local function id(x: number, y: number): number
		local k = (y - 1) * nx + x
		return if label then label[k] else k
	end
	for y = 1, ny do
		for x = 1, nx do
			if x < nx then
				table.insert(ii, id(x, y))
				table.insert(jj, id(x + 1, y))
			end
			if y < ny then
				table.insert(ii, id(x, y))
				table.insert(jj, id(x, y + 1))
			end
		end
	end
	return (graph.from_edges(nx * ny, ii, jj))
end

local function is_permutation(perm: buffer, n: number): boolean
	local t = h.ints(perm)
	if #t ~= n then
		return false
	end
	local seen = {}
	for _, v in t do
		if v < 1 or v > n or seen[v] then
			return false
		end
		seen[v] = true
	end
	return true
end

test("graph: construction merges, sorts and symmetrizes", function()
	local G, w = graph.from_edges(4, { 1, 3, 1, 2, 2 }, { 3, 1, 3, 2, 4 }, { weights = { 5, 2, 7, 1, 4 } })
	check(same(neighbors(G, 1), { 3 }) and same(neighbors(G, 2), { 4 }) and same(neighbors(G, 3), { 1 }) and same(neighbors(G, 4), { 2 }), "neighbor lists")
	check(w ~= nil and buffer.len(w :: buffer) == 8 * 4, "one weight per stored edge")
	-- the edge 1 - 3 appears three times (both ways); the smallest weight is kept
	check(buffer.readf64(w :: buffer, 0) == 2, "smallest repeated weight")
	local D = graph.from_edges(3, { 1, 2 }, { 2, 3 }, { directed = true })
	check(same(neighbors(D, 1), { 2 }) and #neighbors(D, 3) == 0, "directed edges one way")
	local U = graph.symmetrize(D)
	check(same(neighbors(U, 2), { 1, 3 }) and same(neighbors(U, 3), { 2 }), "symmetrized")
	local T = graph.from_triangles(4, { 1, 2, 3, 2, 3, 4 })
	check(same(neighbors(T, 1), { 2, 3 }) and same(neighbors(T, 2), { 1, 3, 4 }) and same(neighbors(T, 4), { 2, 3 }), "triangle edges")
	check(same(h.ints(graph.degrees(T)), { 2, 3, 3, 2 }), "degrees")
	-- views of matrix patterns share their buffers
	local S = f64.sparse(3, 3, { 1, 2, 3, 1 }, { 1, 1, 2, 3 }, f64.from({ 1, 2, 3, 4 }))
	local V = graph.of(S)
	check(V.n == 3 and V.ptr == S.colptr and same(neighbors(V, 1), { 1, 2 }), "view of a Sparse")
	local B = f64.bsparse(6, 6, 2, { 1, 2, 3 }, { 1, 2, 1 }, f64.create(12))
	local VB = graph.of(B)
	check(VB.n == 3 and same(neighbors(VB, 3), { 1 }), "view of a BlockSparse")
end)

test("graph: breadth-first levels and components", function()
	local G = grid(7, 5)
	local order, level = graph.bfs(G, 1)
	check(buffer.len(order) == 4 * 35 and buffer.readi32(order, 0) == 1, "every vertex visited, the source first")
	local ok = true
	for y = 1, 5 do
		for x = 1, 7 do
			ok = ok and buffer.readi32(level, 4 * ((y - 1) * 7 + x - 1)) == (x - 1) + (y - 1)
		end
	end
	check(ok, "levels are grid distances")
	-- two paths and an isolated vertex
	local P = graph.from_edges(7, { 1, 2, 4, 5 }, { 2, 3, 5, 6 })
	local labels, count = graph.components(P)
	check(count == 3 and same(h.ints(labels), { 1, 1, 1, 2, 2, 2, 3 }), "components")
	local _, l2 = graph.bfs(P, 4)
	check(buffer.readi32(l2, 0) == -1 and buffer.readi32(l2, 4 * 5) == 2, "unreached is -1")
end)

test("graph: reverse Cuthill-McKee narrows shuffled ropes and grids", function()
	-- a rope (path) numbered at random: its band becomes 1
	local n = 200
	local label = shuffled(n)
	local ii, jj = {}, {}
	for k = 1, n - 1 do
		ii[k], jj[k] = label[k], label[k + 1]
	end
	local rope = graph.from_edges(n, ii, jj)
	local perm = graph.rcm(rope)
	check(is_permutation(perm, n), "a permutation")
	check(graph.bandwidth(rope) > 50, "shuffled rope is wide")
	check(graph.bandwidth(rope, perm) == 1, "rope ordered end to end")
	-- a 40 by 25 grid numbered at random: back to about the short side
	local G = grid(40, 25, shuffled(1000))
	local pg = graph.rcm(G)
	check(is_permutation(pg, 1000), "grid permutation")
	local before, after = graph.bandwidth(G), graph.bandwidth(G, pg)
	check(before > 500 and after <= 2 * 25, string.format("grid band %d -> %d", before, after))
	check(graph.profile(G, pg) < graph.profile(G) / 10, "profile shrinks")
	-- the inverse
	local inv = graph.inverse(pg)
	local ok = true
	for k = 1, 1000 do
		ok = ok and buffer.readi32(inv, 4 * (buffer.readi32(pg, 4 * (k - 1)) - 1)) == k
	end
	check(ok, "inverse")
	-- several components, isolated vertices included
	local P = graph.from_edges(9, { 3, 5, 8, 1 }, { 5, 9, 1, 6 })
	local pp = graph.rcm(P)
	check(is_permutation(pp, 9) and graph.bandwidth(P, pp) == 1, "components ordered separately")
	-- a star and a complete graph
	local S = graph.from_edges(6, { 4, 4, 4, 4, 4 }, { 1, 2, 3, 5, 6 })
	check(is_permutation(graph.rcm(S), 6), "star")
	check(graph.bandwidth(graph.from_edges(1, {}, {}), graph.rcm(graph.from_edges(1, {}, {}))) == 0, "single vertex")
end)

-- Floyd-Warshall distances of a weighted graph, as a table of tables.
local function floyd(n: number, ii: { number }, jj: { number }, ww: { number }, directed: boolean): { { number } }
	local d = {}
	for i = 1, n do
		d[i] = table.create(n, math.huge)
		d[i][i] = 0
	end
	for k = 1, #ii do
		local a, b = ii[k], jj[k]
		if a ~= b then
			d[a][b] = math.min(d[a][b], ww[k])
			if not directed then
				d[b][a] = math.min(d[b][a], ww[k])
			end
		end
	end
	for k = 1, n do
		for i = 1, n do
			for j = 1, n do
				if d[i][k] + d[k][j] < d[i][j] then
					d[i][j] = d[i][k] + d[k][j]
				end
			end
		end
	end
	return d
end

test("graph: Dijkstra and A* against Floyd-Warshall", function()
	for trial = 1, 6 do
		local directed = trial % 2 == 0
		local n = 30
		local ii, jj, ww = {}, {}, {}
		for _ = 1, 70 do
			table.insert(ii, math.random(n))
			table.insert(jj, math.random(n))
			table.insert(ww, math.random() * 10)
		end
		local G, w = graph.from_edges(n, ii, jj, { weights = ww, directed = directed })
		local d = floyd(n, ii, jj, ww, directed)
		local ok, paths_ok = true, true
		for s = 1, n, 7 do
			local dist, prev = graph.dijkstra(G, w, s)
			for v = 1, n do
				local x = buffer.readf64(dist, 8 * (v - 1))
				ok = ok and (x == d[s][v] or math.abs(x - d[s][v]) < 1e-12 * d[s][v])
				local p = graph.path(prev, v, dist)
				if d[s][v] == math.huge then
					paths_ok = paths_ok and p == nil
				else
					-- the path starts at s, ends at v, and its edges add up to the distance
					local pt = p :: { number }
					local len = 0
					for k = 1, #pt - 1 do
						local best = math.huge
						for e = 1, #ii do
							if (ii[e] == pt[k] and jj[e] == pt[k + 1]) or (not directed and jj[e] == pt[k] and ii[e] == pt[k + 1]) then
								best = math.min(best, ww[e])
							end
						end
						len += best
					end
					paths_ok = paths_ok and pt[1] == s and pt[#pt] == v and math.abs(len - d[s][v]) < 1e-12 * (1 + d[s][v])
				end
			end
			-- A* with h = 0 is Dijkstra's search
			for t = 1, n, 5 do
				local p, info = graph.astar(G, w, s, t, function(_: number): number
					return 0
				end)
				ok = ok and (info.cost == d[s][t] or math.abs(info.cost - d[s][t]) < 1e-12 * d[s][t]) and ((p == nil) == (d[s][t] == math.huge))
			end
		end
		check(ok, "distances, trial " .. trial)
		check(paths_ok, "paths, trial " .. trial)
	end
	-- several sources, a target and a limit
	local G = grid(10, 10)
	local dist = graph.dijkstra(G, nil, { 1, 100 })
	check(buffer.readf64(dist, 8 * 9) == 9 and buffer.readf64(dist, 8 * 44) == 8, "nearest of two sources")
	local dt = graph.dijkstra(G, nil, 1, { target = 23 })
	check(buffer.readf64(dt, 8 * 22) == 4 and buffer.readf64(dt, 8 * 99) > 4, "stops at the target")
	local dl = graph.dijkstra(G, nil, 1, { limit = 3 })
	check(buffer.readf64(dl, 8 * 3) == 3 and buffer.readf64(dl, 8 * 99) == math.huge, "stops past the limit")
end)

test("graph: A* with a straight-line heuristic expands less", function()
	-- a 30 by 30 grid of unit spacing, with a wall across most of column 15
	local nx = 30
	local ii, jj, ww = {}, {}, {}
	local function blocked(x: number, y: number): boolean
		return x == 15 and y < 25
	end
	for y = 1, nx do
		for x = 1, nx do
			if not blocked(x, y) then
				if x < nx and not blocked(x + 1, y) then
					table.insert(ii, (y - 1) * nx + x)
					table.insert(jj, (y - 1) * nx + x + 1)
					table.insert(ww, 1)
				end
				if y < nx and not blocked(x, y + 1) then
					table.insert(ii, (y - 1) * nx + x)
					table.insert(jj, y * nx + x)
					table.insert(ww, 1)
				end
			end
		end
	end
	local G, w = graph.from_edges(nx * nx, ii, jj, { weights = ww })
	local s, t = 1 + 5 * nx, 30 + 5 * nx
	local function straight(v: number): number
		local x, y = (v - 1) % nx, (v - 1) // nx
		return math.sqrt((x - 29) ^ 2 + (y - 5) ^ 2)
	end
	local p, info = graph.astar(G, w, s, t, straight)
	local p0, info0 = graph.astar(G, w, s, t, function(_: number): number
		return 0
	end)
	local dist = graph.dijkstra(G, w, s)
	close(info.cost, buffer.readf64(dist, 8 * (t - 1)), 1e-12, "A* cost")
	check(p ~= nil and p0 ~= nil and #(p :: { number }) == info.cost + 1, "a path of unit steps")
	check(info.expanded < info0.expanded, string.format("expanded %d against %d", info.expanded, info0.expanded))
end)

test("graph: symmetric permutations of sparse matrices", function()
	for _, M in { lib.f32, lib.f64, lib.c32, lib.c64 } do
		local complex = M == lib.c32 or M == lib.c64
		local n = 9
		local ii, jj, re, im = {}, {}, {}, {}
		for _ = 1, 25 do
			table.insert(ii, math.random(n))
			table.insert(jj, math.random(n))
			table.insert(re, math.random(1, 9))
			table.insert(im, math.random(1, 9))
		end
		local vals = if complex then M.from(re, im) else M.from(re)
		local S = M.sparse(n, n, ii, jj, vals)
		local permt = shuffled(n)
		local perm = buffer.create(4 * n)
		for k = 1, n do
			buffer.writei32(perm, 4 * (k - 1), permt[k])
		end
		local P = M.sparse_permute(S, perm)
		local ok = true
		for k = 1, n do
			for l = 1, n do
				local a, b = M.sparse_get(P, k, l), M.sparse_get(S, permt[k], permt[l])
				ok = ok and a == b
			end
		end
		check(ok and M.sparse_nnz(P) == M.sparse_nnz(S), "sparse_permute")
	end
	-- banded Cholesky of a reordered symmetric matrix solves the original system: the 2-D
	-- Laplacian (plus a shift) on a shuffled grid
	for _, M: any in { f64, c64 } do
		local complex = M == c64
		local nx, ny = 12, 9
		local n = nx * ny
		local label = shuffled(n)
		local G = grid(nx, ny, label)
		local ii, jj, vals = {}, {}, {}
		for v = 1, n do
			table.insert(ii, v)
			table.insert(jj, v)
			table.insert(vals, 4.5)
			for _, u in neighbors(G, v) do
				table.insert(ii, v)
				table.insert(jj, u)
				table.insert(vals, -1)
			end
		end
		local S = M.sparse(n, n, ii, jj, M.from(vals))
		local perm = graph.rcm(graph.of(S))
		local B = M.sparse_symband(S, perm)
		check(B.kd == graph.bandwidth(G, perm) and B.kd <= 2 * ny, "band of the reordered matrix")
		local F = M.band_chol(B)
		check(F.definite, "definite")
		local x = M.create(n)
		for i = 1, n do
			if complex then
				M.set(x, i, math.sin(i), math.cos(i))
			else
				M.set(x, i, math.sin(i))
			end
		end
		local b = M.sparse_matvec(nil, S, x)
		-- solve P S P^T (P x) = P b
		local pb = M.create(n)
		for k = 1, n do
			local j = buffer.readi32(perm, 4 * (k - 1))
			if complex then
				local re, im = M.get(b, j)
				M.set(pb, k, re, im)
			else
				M.set(pb, k, M.get(b, j))
			end
		end
		local px = M.band_chol_solve(nil, F, pb)
		local err = 0
		for k = 1, n do
			local ar, ai = M.get(px, k)
			local er, ei = M.get(x, buffer.readi32(perm, 4 * (k - 1)))
			err = math.max(err, math.abs(ar - er), math.abs((ai or 0) - (ei or 0)))
		end
		check(err < 1e-12, "reordered solve")
		-- reuse
		check(M.sparse_symband(S, perm, B) == B, "storage reused")
	end
end)
return nil
