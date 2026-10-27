--!strict
--!native
--!optimize 2

-- Graphs as sparsity patterns: vertex v's neighbors (0-based) are ind[ptr[v] ..
-- ptr[v + 1] - 1], u32 buffers, the layout of a Sparse matrix's columns (colptr, rowind)
-- and a BlockSparse's block rows (rowptr, colind), so graph.of views either without a
-- copy and a weighted graph is a pattern with values (a Sparse's, or any f64 array
-- aligned with ind). Vertices, orders and permutations are 1-based as elsewhere; a
-- permutation perm (a buffer of 32-bit integers, as pivots are) puts vertex perm[k] k-th.
--
-- Traversal (bfs, components), orderings (rcm: reverse Cuthill-McKee, to narrow a
-- matrix's band before a banded factorization; amd: approximate minimum degree, for the
-- least fill in a sparse Cholesky factorization), the elimination tree and symbolic
-- Cholesky analysis (etree, chol_analysis; used by the array modules' sparse_chol) and
-- shortest paths (dijkstra, astar).

local types = require("./types")

export type Graph = types.Graph
type Sparse = types.Sparse
type BlockSparse = types.BlockSparse

local function u32(b: buffer, i: number): number
	return buffer.readu32(b, 4 * i)
end

local function iget(b: buffer, k: number): number
	return buffer.readi32(b, 4 * (k - 1))
end

local function iset(b: buffer, k: number, v: number)
	buffer.writei32(b, 4 * (k - 1), v)
end

-- The graph of a square Sparse matrix's pattern (vertex j's neighbors are the rows stored
-- in column j) or of a BlockSparse's block pattern, sharing its buffers.
local function of(S: Sparse | BlockSparse): Graph
	local A = S :: any
	assert(A.rows == A.cols, "graph.of needs a square matrix")
	if A.block then
		return { n = A.rows // A.block, ptr = A.rowptr, ind = A.colind }
	end
	return { n = A.rows, ptr = A.colptr, ind = A.rowind }
end

-- The graph with edges src[k] -> dst[k] (0-based, u32 buffers of m entries), weights w
-- (f64, aligned) if given: neighbor lists sorted, repeated edges merged (keeping the
-- smallest weight) and self loops dropped. Two counting sorts, O(n + m).
local function build(n: number, m: number, src: buffer, dst: buffer, w: buffer?): (Graph, buffer?)
	local count = buffer.create(4 * (n + 1))
	local by_dst = buffer.create(4 * m)
	for k = 0, m - 1 do
		local d = u32(dst, k)
		buffer.writeu32(count, 4 * (d + 1), u32(count, d + 1) + 1)
	end
	for v = 1, n do
		buffer.writeu32(count, 4 * v, u32(count, v) + u32(count, v - 1))
	end
	for k = 0, m - 1 do
		local d = u32(dst, k)
		local at = u32(count, d)
		buffer.writeu32(by_dst, 4 * at, k)
		buffer.writeu32(count, 4 * d, at + 1)
	end
	-- stable by source: each source's edges in increasing destination
	buffer.fill(count, 0, 0)
	for k = 0, m - 1 do
		local s = u32(src, k)
		buffer.writeu32(count, 4 * (s + 1), u32(count, s + 1) + 1)
	end
	for v = 1, n do
		buffer.writeu32(count, 4 * v, u32(count, v) + u32(count, v - 1))
	end
	local order = buffer.create(4 * m)
	for t = 0, m - 1 do
		local k = u32(by_dst, t)
		local s = u32(src, k)
		local at = u32(count, s)
		buffer.writeu32(order, 4 * at, k)
		buffer.writeu32(count, 4 * s, at + 1)
	end
	-- compact: drop self loops and repeats
	local ptr = buffer.create(4 * (n + 1))
	local ind = buffer.create(4 * m)
	local wout = if w then buffer.create(8 * m) else nil
	local p, t = 0, 0
	for v = 0, n - 1 do
		buffer.writeu32(ptr, 4 * v, p)
		local start = p
		local stop = u32(count, v)
		while t < stop do
			local k = u32(order, t)
			t += 1
			local d = u32(dst, k)
			if d ~= v then
				if p > start and u32(ind, p - 1) == d then
					if w and wout then
						local x = buffer.readf64(w, 8 * k)
						if x < buffer.readf64(wout, 8 * (p - 1)) then
							buffer.writef64(wout, 8 * (p - 1), x)
						end
					end
				else
					buffer.writeu32(ind, 4 * p, d)
					if w and wout then
						buffer.writef64(wout, 8 * p, buffer.readf64(w, 8 * k))
					end
					p += 1
				end
			end
		end
	end
	buffer.writeu32(ptr, 4 * n, p)
	if p < m then
		local trimmed = buffer.create(4 * p)
		buffer.copy(trimmed, 0, ind, 0, 4 * p)
		ind = trimmed
		if wout then
			local wt = buffer.create(8 * p)
			buffer.copy(wt, 0, wout, 0, 8 * p)
			wout = wt
		end
	end
	return { n = n, ptr = ptr, ind = ind }, wout
end

export type EdgeOptions = { weights: { number }?, directed: boolean? }

-- The graph on vertices 1..n with an edge between ii[k] and jj[k] (both ways unless
-- options.directed), and its weights (an f64 array aligned with the graph's ind) when
-- options.weights gives one per edge. Repeated edges keep the smallest weight; self loops
-- are dropped.
local function from_edges(n: number, ii: { number }, jj: { number }, options: EdgeOptions?): (Graph, buffer?)
	local o: EdgeOptions = options or {}
	local count = #ii
	assert(#jj == count, "need as many ends as starts")
	local ws = o.weights
	assert(ws == nil or #ws == count, "need a weight per edge")
	local both = not o.directed
	local m = if both then 2 * count else count
	local src, dst = buffer.create(4 * m), buffer.create(4 * m)
	local w = if ws then buffer.create(8 * m) else nil
	for k = 1, count do
		local a, b = ii[k], jj[k]
		assert(a % 1 == 0 and b % 1 == 0 and a >= 1 and a <= n and b >= 1 and b <= n, "vertex outside 1..n")
		buffer.writeu32(src, 4 * (k - 1), a - 1)
		buffer.writeu32(dst, 4 * (k - 1), b - 1)
		if both then
			buffer.writeu32(src, 4 * (count + k - 1), b - 1)
			buffer.writeu32(dst, 4 * (count + k - 1), a - 1)
		end
		if w and ws then
			local x = ws[k]
			buffer.writef64(w, 8 * (k - 1), x)
			if both then
				buffer.writef64(w, 8 * (count + k - 1), x)
			end
		end
	end
	return build(n, m, src, dst, w)
end

-- The edges of a triangle mesh on vertices 1..n: tris lists 3 vertices per triangle.
local function from_triangles(n: number, tris: { number }): Graph
	local t = #tris // 3
	assert(#tris == 3 * t, "need 3 vertices per triangle")
	local m = 6 * t
	local src, dst = buffer.create(4 * m), buffer.create(4 * m)
	local e = 0
	for k = 0, t - 1 do
		for c = 1, 3 do
			local a, b = tris[3 * k + c], tris[3 * k + c % 3 + 1]
			assert(a % 1 == 0 and b % 1 == 0 and a >= 1 and a <= n and b >= 1 and b <= n, "vertex outside 1..n")
			buffer.writeu32(src, 4 * e, a - 1)
			buffer.writeu32(dst, 4 * e, b - 1)
			buffer.writeu32(src, 4 * (e + 1), b - 1)
			buffer.writeu32(dst, 4 * (e + 1), a - 1)
			e += 2
		end
	end
	return (build(n, m, src, dst))
end

-- The undirected graph with G's edges both ways (the pattern of A + A^T), self loops
-- dropped.
local function symmetrize(G: Graph): Graph
	local n, ptr, ind = G.n, G.ptr, G.ind
	local nnz = u32(ptr, n)
	local m = 2 * nnz
	local src, dst = buffer.create(4 * m), buffer.create(4 * m)
	local e = 0
	for v = 0, n - 1 do
		for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
			local d = u32(ind, p)
			buffer.writeu32(src, 4 * e, v)
			buffer.writeu32(dst, 4 * e, d)
			buffer.writeu32(src, 4 * (e + 1), d)
			buffer.writeu32(dst, 4 * (e + 1), v)
			e += 2
		end
	end
	return (build(n, m, src, dst))
end

-- Number of neighbors of each vertex: a buffer of 32-bit integers.
local function degrees(G: Graph): buffer
	local n, ptr = G.n, G.ptr
	local d = buffer.create(4 * n)
	for v = 0, n - 1 do
		buffer.writei32(d, 4 * v, u32(ptr, v + 1) - u32(ptr, v))
	end
	return d
end

-- Breadth-first search from s (0-based) through G's edges, over the vertices whose seen
-- entry is not tag (marking them tag): the vertices reached go to queue[0 ..] in the order
-- visited, with their levels (distance from s in edges) in level. Returns how many were
-- reached and the last level. The one search behind bfs, components and rcm.
local function search(G: Graph, s: number, seen: buffer, tag: number, level: buffer, queue: buffer): (number, number)
	local ptr, ind = G.ptr, G.ind
	buffer.writei32(seen, 4 * s, tag)
	buffer.writei32(level, 4 * s, 0)
	buffer.writeu32(queue, 0, s)
	local head, tail = 0, 1
	local depth = 0
	while head < tail do
		local v = u32(queue, head)
		head += 1
		local lv = buffer.readi32(level, 4 * v)
		depth = lv
		for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
			local w = u32(ind, p)
			if buffer.readi32(seen, 4 * w) ~= tag then
				buffer.writei32(seen, 4 * w, tag)
				buffer.writei32(level, 4 * w, lv + 1)
				buffer.writeu32(queue, 4 * tail, w)
				tail += 1
			end
		end
	end
	return tail, depth
end

-- Breadth-first search from source (1-based) through G's edges. Returns the vertices
-- reached in the order visited (a buffer of 32-bit integers, 1-based) and each vertex's
-- level (its distance in edges: 0 at the source, -1 if not reached).
local function bfs(G: Graph, source: number): (buffer, buffer)
	local n = G.n
	assert(source % 1 == 0 and source >= 1 and source <= n, "source outside 1..n")
	local level = buffer.create(4 * n)
	buffer.fill(level, 0, 255) -- -1 everywhere
	local queue = buffer.create(4 * n)
	local count = search(G, source - 1, buffer.create(4 * n), 1, level, queue)
	local order = buffer.create(4 * count)
	for k = 0, count - 1 do
		buffer.writei32(order, 4 * k, u32(queue, k) + 1)
	end
	return order, level
end

-- Connected components of an undirected graph (symmetrize a directed one first): each
-- vertex's component (1-based, numbered in order of their lowest vertex; a buffer of
-- 32-bit integers) and the number of components.
local function components(G: Graph): (buffer, number)
	local n = G.n
	local label = buffer.create(4 * n)
	local level, queue = buffer.create(4 * n), buffer.create(4 * n)
	local count = 0
	for s = 0, n - 1 do
		if buffer.readi32(label, 4 * s) == 0 then
			count += 1
			search(G, s, label, count, level, queue)
		end
	end
	return label, count
end

-- Orderings

-- The positions of a permutation: inv[perm[k]] = k. Errors if perm is not a permutation of
-- 1..n.
local function inverse(perm: buffer): buffer
	local n = buffer.len(perm) // 4
	local inv = buffer.create(4 * n)
	for k = 1, n do
		local v = iget(perm, k)
		assert(v >= 1 and v <= n and iget(inv, v) == 0, "not a permutation")
		iset(inv, v, k)
	end
	return inv
end

-- The half bandwidth of G's adjacency matrix with vertices ordered by perm (default: as
-- numbered): the largest |position(v) - position(w)| over edges v - w.
local function bandwidth(G: Graph, perm: buffer?): number
	local n, ptr, ind = G.n, G.ptr, G.ind
	local inv = if perm then inverse(perm) else nil
	local kd = 0
	for v = 0, n - 1 do
		local pv = if inv then iget(inv, v + 1) else v + 1
		for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
			local w = u32(ind, p)
			local pw = if inv then iget(inv, w + 1) else w + 1
			local d = if pv > pw then pv - pw else pw - pv
			if d > kd then
				kd = d
			end
		end
	end
	return kd
end

-- The profile (envelope size) of a symmetric pattern ordered by perm: the sum over rows i
-- of i minus the first column stored in row i. A skyline or banded factorization's work
-- grows with it.
local function profile(G: Graph, perm: buffer?): number
	local n, ptr, ind = G.n, G.ptr, G.ind
	local inv = if perm then inverse(perm) else nil
	local first = buffer.create(4 * n)
	for k = 1, n do
		iset(first, k, k)
	end
	for v = 0, n - 1 do
		local pv = if inv then iget(inv, v + 1) else v + 1
		for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
			local w = u32(ind, p)
			local pw = if inv then iget(inv, w + 1) else w + 1
			if pw < pv and pw < iget(first, pv) then
				iset(first, pv, pw)
			end
		end
	end
	local total = 0
	for k = 1, n do
		total += k - iget(first, k)
	end
	return total
end

-- Reverse Cuthill-McKee ordering of an undirected graph (symmetrize a directed one first):
-- breadth-first from a pseudo-peripheral vertex of each component (George and Liu's
-- search: repeated searches from a lowest-degree vertex of the last level while the depth
-- grows), visiting each vertex's unvisited neighbors in increasing degree, then reversed.
-- Neighbors in the graph end up near each other in the order, so the matrix's band
-- narrows: on a mesh of n vertices numbered arbitrarily, from up to n to about sqrt(n).
-- Cost O(m log d) for m edges and degree d. Returns the permutation (buffer of 32-bit
-- integers): vertex perm[k] goes k-th.
local function rcm(G: Graph): buffer
	local n, ptr, ind = G.n, G.ptr, G.ind
	local deg = buffer.create(4 * n)
	for v = 0, n - 1 do
		buffer.writeu32(deg, 4 * v, u32(ptr, v + 1) - u32(ptr, v))
	end
	local order = buffer.create(4 * n) -- 0-based vertices, in Cuthill-McKee order
	local placed = buffer.create(n) -- 1 once ordered
	local stamp = buffer.create(4 * n) -- search number that last reached each vertex
	local level = buffer.create(4 * n)
	local queue = buffer.create(4 * n)
	local searches = 0
	-- breadth-first search from s within its component; returns the depth (last level) and
	-- a lowest-degree vertex of the last level
	local function probe(s: number): (number, number)
		searches += 1
		local tail, depth = search(G, s, stamp, searches, level, queue)
		local best, bestdeg = s, math.huge
		for k = tail - 1, 0, -1 do
			local v = u32(queue, k)
			if buffer.readi32(level, 4 * v) ~= depth then
				break
			end
			local d = u32(deg, v)
			if d < bestdeg then
				best, bestdeg = v, d
			end
		end
		return depth, best
	end
	local placed_count = 0
	local scan = 0
	while placed_count < n do
		while buffer.readu8(placed, scan) ~= 0 do
			scan += 1
		end
		-- a pseudo-peripheral start for scan's component
		local start = scan
		local depth, far = probe(start)
		while true do
			local d2, far2 = probe(far)
			if d2 <= depth then
				start = far
				break
			end
			start, depth, far = far, d2, far2
		end
		-- Cuthill-McKee from start
		buffer.writeu8(placed, start, 1)
		buffer.writeu32(order, 4 * placed_count, start)
		local head = placed_count
		placed_count += 1
		while head < placed_count do
			local v = u32(order, head)
			head += 1
			local first = placed_count
			for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
				local w = u32(ind, p)
				if buffer.readu8(placed, w) == 0 then
					buffer.writeu8(placed, w, 1)
					-- insert by degree (ties by vertex number, as the lists are sorted)
					local dw = u32(deg, w)
					local k = placed_count
					while k > first and u32(deg, u32(order, k - 1)) > dw do
						buffer.writeu32(order, 4 * k, u32(order, k - 1))
						k -= 1
					end
					buffer.writeu32(order, 4 * k, w)
					placed_count += 1
				end
			end
		end
	end
	local perm = buffer.create(4 * n)
	for k = 0, n - 1 do
		buffer.writei32(perm, 4 * k, u32(order, n - 1 - k) + 1)
	end
	return perm
end

-- Shortest paths

-- An indexed binary min-heap of vertices keyed by an f64 array, with decrease-key.
type Heap = { size: number, heap: buffer, pos: buffer, key: buffer }

local function heap_new(n: number, key: buffer): Heap
	local pos = buffer.create(4 * n)
	buffer.fill(pos, 0, 255) -- -1: not in the heap
	return { size = 0, heap = buffer.create(4 * n), pos = pos, key = key }
end

local function sift_up(H: Heap, k: number)
	local heap, pos, key = H.heap, H.pos, H.key
	local v = u32(heap, k)
	local kv = buffer.readf64(key, 8 * v)
	while k > 0 do
		local parent = (k - 1) // 2
		local u = u32(heap, parent)
		if buffer.readf64(key, 8 * u) <= kv then
			break
		end
		buffer.writeu32(heap, 4 * k, u)
		buffer.writei32(pos, 4 * u, k)
		k = parent
	end
	buffer.writeu32(heap, 4 * k, v)
	buffer.writei32(pos, 4 * v, k)
end

-- Inserts v, or moves it up after its key decreased.
local function heap_push(H: Heap, v: number)
	local k = buffer.readi32(H.pos, 4 * v)
	if k < 0 then
		k = H.size
		H.size += 1
		buffer.writeu32(H.heap, 4 * k, v)
		buffer.writei32(H.pos, 4 * v, k)
	end
	sift_up(H, k)
end

local function heap_pop(H: Heap): number
	local heap, pos, key = H.heap, H.pos, H.key
	local top = u32(heap, 0)
	buffer.writei32(pos, 4 * top, -1)
	local size = H.size - 1
	H.size = size
	if size > 0 then
		local v = u32(heap, size)
		local kv = buffer.readf64(key, 8 * v)
		local k = 0
		while true do
			local c = 2 * k + 1
			if c >= size then
				break
			end
			local kc = buffer.readf64(key, 8 * u32(heap, c))
			if c + 1 < size then
				local k2 = buffer.readf64(key, 8 * u32(heap, c + 1))
				if k2 < kc then
					c, kc = c + 1, k2
				end
			end
			if kc >= kv then
				break
			end
			local u = u32(heap, c)
			buffer.writeu32(heap, 4 * k, u)
			buffer.writei32(pos, 4 * u, k)
			k = c
		end
		buffer.writeu32(heap, 4 * k, v)
		buffer.writei32(pos, 4 * v, k)
	end
	return top
end

export type PathOptions = { target: number?, limit: number? }

-- Dijkstra's shortest paths from source (a vertex, or a list of them, 1-based) along G's
-- edges, of lengths w (an f64 array aligned with G's ind, nonnegative, such as a Sparse
-- matrix's values; nil for 1 each). options.target stops once that vertex is settled and
-- options.limit once the distance exceeds it. Returns each vertex's distance (an f64
-- array; math.huge if not reached) and predecessor on a shortest path (a buffer of 32-bit
-- integers, 1-based; 0 for sources and unreached vertices): graph.path(prev, v) is the
-- path. O((n + m) log n).
local function dijkstra(G: Graph, w: buffer?, source: number | { number }, options: PathOptions?): (buffer, buffer)
	local n, ptr, ind = G.n, G.ptr, G.ind
	local o: PathOptions = options or {}
	local target = if o.target then o.target - 1 else -1
	local limit = o.limit or math.huge
	assert(w == nil or buffer.len(w) == 8 * u32(ptr, n), "need a weight per edge")
	local dist = buffer.create(8 * n)
	for v = 0, n - 1 do
		buffer.writef64(dist, 8 * v, math.huge)
	end
	local prev = buffer.create(4 * n)
	local H = heap_new(n, dist)
	local sources: { number } = if type(source) == "number" then { source } else source
	for _, s in sources do
		assert(s % 1 == 0 and s >= 1 and s <= n, "source outside 1..n")
		buffer.writef64(dist, 8 * (s - 1), 0)
		heap_push(H, s - 1)
	end
	while H.size > 0 do
		local v = heap_pop(H)
		local dv = buffer.readf64(dist, 8 * v)
		if v == target or dv > limit then
			break
		end
		for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
			local u = u32(ind, p)
			local len = if w then buffer.readf64(w, 8 * p) else 1
			local du = dv + len
			if du < buffer.readf64(dist, 8 * u) then
				buffer.writef64(dist, 8 * u, du)
				buffer.writei32(prev, 4 * u, v + 1)
				heap_push(H, u)
			end
		end
	end
	return dist, prev
end

-- The path to v (1-based) from predecessors prev (as dijkstra's): its vertices from the
-- source to v, or nil if v was not reached (and is not a source).
local function path(prev: buffer, v: number, dist: buffer?): { number }?
	if dist and buffer.readf64(dist, 8 * (v - 1)) == math.huge then
		return nil
	end
	local rev = { v }
	local u = iget(prev, v)
	while u ~= 0 do
		table.insert(rev, u)
		u = iget(prev, u)
		assert(#rev <= buffer.len(prev) // 4, "graph.path: predecessors form a cycle")
	end
	local out = table.create(#rev) :: { number }
	for k = #rev, 1, -1 do
		table.insert(out, rev[k])
	end
	return out
end

export type AstarInfo = { cost: number, expanded: number }

-- A* search for a shortest path from source to target (1-based) with edge lengths w (as
-- dijkstra's) and a heuristic h(v) (1-based) never above v's true distance to target
-- (a straight-line distance, for points joined by straight edges): the closer h is to
-- the true distance, the fewer vertices it expands; h = 0 is Dijkstra's search. Returns
-- the path (vertices from source to target) or nil, and info: the path's cost and the
-- number of vertices expanded.
local function astar(G: Graph, w: buffer?, source: number, target: number, h: (number) -> number): ({ number }?, AstarInfo)
	local n, ptr, ind = G.n, G.ptr, G.ind
	assert(source % 1 == 0 and source >= 1 and source <= n and target % 1 == 0 and target >= 1 and target <= n, "vertex outside 1..n")
	assert(w == nil or buffer.len(w) == 8 * u32(ptr, n), "need a weight per edge")
	local g = buffer.create(8 * n)
	local f = buffer.create(8 * n)
	for v = 0, n - 1 do
		buffer.writef64(g, 8 * v, math.huge)
	end
	local prev = buffer.create(4 * n)
	local H = heap_new(n, f)
	local t = target - 1
	buffer.writef64(g, 8 * (source - 1), 0)
	buffer.writef64(f, 8 * (source - 1), h(source))
	heap_push(H, source - 1)
	local expanded = 0
	while H.size > 0 do
		local v = heap_pop(H)
		if v == t then
			break
		end
		expanded += 1
		local gv = buffer.readf64(g, 8 * v)
		for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
			local u = u32(ind, p)
			local len = if w then buffer.readf64(w, 8 * p) else 1
			local gu = gv + len
			if gu < buffer.readf64(g, 8 * u) then
				buffer.writef64(g, 8 * u, gu)
				-- (re)opened with its new key; an inconsistent heuristic may reopen a vertex
				buffer.writef64(f, 8 * u, gu + h(u + 1))
				buffer.writei32(prev, 4 * u, v + 1)
				heap_push(H, u)
			end
		end
	end
	local cost = buffer.readf64(g, 8 * t)
	return path(prev, target, g), { cost = cost, expanded = expanded }
end

-- Sparse Cholesky: ordering and symbolic analysis

-- Approximate minimum degree ordering of an undirected graph (a symmetric pattern;
-- symmetrize a directed one first), after Amestoy, Davis and Duff (SIAM J. Matrix Anal.
-- Appl., 1996). Eliminating a vertex joins its neighbors into a clique: the ordering
-- eliminates, at each step, a vertex of least degree, which keeps the Cholesky factor L
-- of a matrix with this pattern sparse. The graph of eliminated vertices is kept implicitly
-- (each eliminated vertex an "element": the clique of its remaining neighbors), degrees
-- are bounded from above rather than counted, vertices with the same neighbors are merged
-- and eliminated together, and elements whose cliques lie inside a newer one are absorbed,
-- so the work grows about as the number of edges. All lists live in one integer workspace
-- and are pruned in place (each step removes at least one entry from every list it
-- touches, so the new element fits); the workspace is compacted when a new element does
-- not fit at its end. Returns the permutation (a buffer of 32-bit integers): vertex
-- perm[k] goes k-th.
local function amd(G: Graph): buffer
	local n, ptr, ind = G.n, G.ptr, G.ind
	local function newi(count: number, fill: number?): buffer
		local b = buffer.create(4 * math.max(count, 1))
		if fill == -1 then
			buffer.fill(b, 0, 255)
		end
		return b
	end
	-- per vertex (0-based): its list in the workspace iw (pe, len: elements first, elen of
	-- them, then variables; an element's list is its variables), status (0 variable, 1
	-- element, 2 absorbed element, 3 merged variable), nv (variables it stands for), deg
	-- (bound on its external degree), esize (an element's weighted size)
	local pe, len, elen = newi(n), newi(n), newi(n)
	local nv, deg, esize = newi(n), newi(n), newi(n)
	local status = buffer.create(math.max(n, 1))
	local mark, wv, wtag = newi(n), newi(n), newi(n)
	local merged = newi(n)
	local nnz = u32(ptr, n)
	local iwlen = nnz + (nnz // 2) + 2 * n + 64
	local iw = newi(iwlen)
	local pfree = 0
	for v = 0, n - 1 do
		buffer.writei32(pe, 4 * v, pfree)
		for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
			local w = u32(ind, p)
			if w ~= v then
				buffer.writei32(iw, 4 * pfree, w)
				pfree += 1
			end
		end
		local count = pfree - buffer.readi32(pe, 4 * v)
		buffer.writei32(len, 4 * v, count)
		buffer.writei32(deg, 4 * v, count)
		buffer.writei32(nv, 4 * v, 1)
	end
	-- degree lists: doubly linked (-1 ends a list)
	local dhead, dnext, dprev, dbucket = newi(n + 1, -1), newi(n), newi(n), newi(n)
	local function insert(i: number, d: number)
		local h = buffer.readi32(dhead, 4 * d)
		buffer.writei32(dnext, 4 * i, h)
		buffer.writei32(dprev, 4 * i, -1)
		if h >= 0 then
			buffer.writei32(dprev, 4 * h, i)
		end
		buffer.writei32(dhead, 4 * d, i)
		buffer.writei32(dbucket, 4 * i, d)
	end
	local function remove(i: number)
		local d = buffer.readi32(dbucket, 4 * i)
		local a, b = buffer.readi32(dprev, 4 * i), buffer.readi32(dnext, 4 * i)
		if a >= 0 then
			buffer.writei32(dnext, 4 * a, b)
		else
			buffer.writei32(dhead, 4 * d, b)
		end
		if b >= 0 then
			buffer.writei32(dprev, 4 * b, a)
		end
	end
	for v = 0, n - 1 do
		insert(v, buffer.readi32(deg, 4 * v))
	end
	-- the workspace compacted (into a new buffer with room for `need` more entries): the
	-- lists of variables and live elements, in vertex order
	local function compact(need: number)
		local live = 0
		for v = 0, n - 1 do
			local st = buffer.readu8(status, v)
			if st <= 1 then
				live += buffer.readi32(len, 4 * v)
			end
		end
		local size = math.max(iwlen, (live + need) * 3 // 2 + n)
		local fresh = newi(size)
		local at = 0
		for v = 0, n - 1 do
			local st = buffer.readu8(status, v)
			if st <= 1 then
				local l = buffer.readi32(len, 4 * v)
				buffer.copy(fresh, 4 * at, iw, 4 * buffer.readi32(pe, 4 * v), 4 * l)
				buffer.writei32(pe, 4 * v, at)
				at += l
			end
		end
		iw, iwlen, pfree = fresh, size, at
	end
	local S = newi(n) -- the pivot's variables, Lp
	local hhead, hnext, hkey = newi(n, -1), newi(n), newi(n)
	local order = newi(n) -- pivots in elimination order
	local npiv = 0
	local tag, wt = 0, 0
	local eliminated, mindeg = 0, 0
	while eliminated < n do
		while buffer.readi32(dhead, 4 * mindeg) < 0 do
			mindeg += 1
		end
		local p = buffer.readi32(dhead, 4 * mindeg)
		remove(p)
		tag += 1
		buffer.writei32(mark, 4 * p, tag)
		-- Lp: p's variables, through its elements (absorbed into p) and directly
		local lpn, size = 0, 0
		local pp, pl, pel = buffer.readi32(pe, 4 * p), buffer.readi32(len, 4 * p), buffer.readi32(elen, 4 * p)
		for k = pp, pp + pel - 1 do
			local e = buffer.readi32(iw, 4 * k)
			if buffer.readu8(status, e) == 1 then
				local ep = buffer.readi32(pe, 4 * e)
				for q = ep, ep + buffer.readi32(len, 4 * e) - 1 do
					local i = buffer.readi32(iw, 4 * q)
					if buffer.readu8(status, i) == 0 and buffer.readi32(mark, 4 * i) ~= tag then
						buffer.writei32(mark, 4 * i, tag)
						buffer.writei32(S, 4 * lpn, i)
						lpn += 1
						size += buffer.readi32(nv, 4 * i)
					end
				end
				buffer.writeu8(status, e, 2)
			end
		end
		for k = pp + pel, pp + pl - 1 do
			local i = buffer.readi32(iw, 4 * k)
			if buffer.readu8(status, i) == 0 and buffer.readi32(mark, 4 * i) ~= tag then
				buffer.writei32(mark, 4 * i, tag)
				buffer.writei32(S, 4 * lpn, i)
				lpn += 1
				size += buffer.readi32(nv, 4 * i)
			end
		end
		-- p becomes an element, its list Lp
		buffer.writeu8(status, p, 1)
		if pfree + lpn > iwlen then
			compact(lpn)
		end
		buffer.copy(iw, 4 * pfree, S, 0, 4 * lpn)
		buffer.writei32(pe, 4 * p, pfree)
		buffer.writei32(len, 4 * p, lpn)
		buffer.writei32(elen, 4 * p, 0)
		pfree += lpn
		buffer.writei32(esize, 4 * p, size)
		buffer.writei32(order, 4 * npiv, p)
		npiv += 1
		eliminated += buffer.readi32(nv, 4 * p)
		for k = 0, lpn - 1 do
			remove(buffer.readi32(S, 4 * k))
		end
		-- |Le \ Lp| for the other elements of Lp's variables
		wt += 1
		for k = 0, lpn - 1 do
			local i = buffer.readi32(S, 4 * k)
			local nvi = buffer.readi32(nv, 4 * i)
			local ip = buffer.readi32(pe, 4 * i)
			for q = ip, ip + buffer.readi32(elen, 4 * i) - 1 do
				local e = buffer.readi32(iw, 4 * q)
				if buffer.readu8(status, e) == 1 then
					if buffer.readi32(wtag, 4 * e) ~= wt then
						buffer.writei32(wtag, 4 * e, wt)
						buffer.writei32(wv, 4 * e, buffer.readi32(esize, 4 * e))
					end
					buffer.writei32(wv, 4 * e, buffer.readi32(wv, 4 * e) - nvi)
				end
			end
		end
		-- each variable of Lp: its list pruned in place (p added as an element), its degree
		-- bounded, its list hashed
		local remaining = n - eliminated
		for k = 0, lpn - 1 do
			local i = buffer.readi32(S, 4 * k)
			local ip, il, ie = buffer.readi32(pe, 4 * i), buffer.readi32(len, 4 * i), buffer.readi32(elen, 4 * i)
			local wpos = ip
			local esum = 0
			for q = ip, ip + ie - 1 do
				local e = buffer.readi32(iw, 4 * q)
				if buffer.readu8(status, e) == 1 then
					local we = buffer.readi32(wv, 4 * e)
					if we <= 0 then
						buffer.writeu8(status, e, 2) -- its variables all in Lp: absorbed into p
					else
						buffer.writei32(iw, 4 * wpos, e)
						wpos += 1
						esum += we
					end
				end
			end
			local vstart = ip + ie
			local vend = ip + il - 1
			if wpos < vstart then
				buffer.writei32(iw, 4 * wpos, p)
				wpos += 1
			else
				-- no element removed: then p is one of i's variables; it moves to the front of
				-- the variables and becomes the last element
				local found = false
				for q = vstart, vend do
					if buffer.readi32(iw, 4 * q) == p then
						buffer.writei32(iw, 4 * q, buffer.readi32(iw, 4 * vstart))
						buffer.writei32(iw, 4 * vstart, p)
						found = true
						break
					end
				end
				assert(found, "graph.amd: the pattern is not symmetric")
				wpos += 1
				vstart += 1
			end
			local newelen = wpos - ip
			local asum = 0
			for q = vstart, vend do
				local j = buffer.readi32(iw, 4 * q)
				if buffer.readu8(status, j) == 0 and buffer.readi32(mark, 4 * j) ~= tag then
					buffer.writei32(iw, 4 * wpos, j)
					wpos += 1
					asum += buffer.readi32(nv, 4 * j)
				end
			end
			buffer.writei32(len, 4 * i, wpos - ip)
			buffer.writei32(elen, 4 * i, newelen)
			local nvi = buffer.readi32(nv, 4 * i)
			local ext = size - nvi
			local d = math.min(remaining - nvi, buffer.readi32(deg, 4 * i) + ext, asum + ext + esum)
			buffer.writei32(deg, 4 * i, if d > 0 then d else 0)
			local h = 0
			for q = ip, wpos - 1 do
				h += buffer.readi32(iw, 4 * q)
			end
			h %= n
			buffer.writei32(hkey, 4 * i, h)
			buffer.writei32(hnext, 4 * i, buffer.readi32(hhead, 4 * h))
			buffer.writei32(hhead, 4 * h, i)
		end
		-- variables of Lp with the same lists (equal hashes, then compared) are merged
		for k = 0, lpn - 1 do
			local h = buffer.readi32(hkey, 4 * buffer.readi32(S, 4 * k))
			local a = buffer.readi32(hhead, 4 * h)
			if a >= 0 then
				buffer.writei32(hhead, 4 * h, -1)
				while a >= 0 do
					local anext = buffer.readi32(hnext, 4 * a)
					if buffer.readu8(status, a) == 0 and anext >= 0 then
						local ap, al, ae = buffer.readi32(pe, 4 * a), buffer.readi32(len, 4 * a), buffer.readi32(elen, 4 * a)
						local marked = false
						local b = anext
						while b >= 0 do
							if buffer.readu8(status, b) == 0 and buffer.readi32(len, 4 * b) == al and buffer.readi32(elen, 4 * b) == ae then
								if not marked then
									tag += 1
									for q = ap, ap + al - 1 do
										buffer.writei32(mark, 4 * buffer.readi32(iw, 4 * q), tag)
									end
									marked = true
								end
								local same = true
								local bp = buffer.readi32(pe, 4 * b)
								for q = bp, bp + al - 1 do
									if buffer.readi32(mark, 4 * buffer.readi32(iw, 4 * q)) ~= tag then
										same = false
										break
									end
								end
								if same then
									local nvb = buffer.readi32(nv, 4 * b)
									local da = buffer.readi32(deg, 4 * a) - nvb
									buffer.writei32(deg, 4 * a, if da > 0 then da else 0)
									buffer.writei32(nv, 4 * a, buffer.readi32(nv, 4 * a) + nvb)
									buffer.writei32(nv, 4 * b, 0)
									buffer.writeu8(status, b, 3)
									buffer.writei32(merged, 4 * b, a)
								end
							end
							b = buffer.readi32(hnext, 4 * b)
						end
					end
					a = anext
				end
			end
		end
		-- the remaining variables of Lp back in the degree lists; p's list pruned to them
		local wpos = buffer.readi32(pe, 4 * p)
		for k = 0, lpn - 1 do
			local i = buffer.readi32(S, 4 * k)
			if buffer.readu8(status, i) == 0 then
				local d = buffer.readi32(deg, 4 * i)
				insert(i, d)
				if d < mindeg then
					mindeg = d
				end
				buffer.writei32(iw, 4 * wpos, i)
				wpos += 1
			end
		end
		buffer.writei32(len, 4 * p, wpos - buffer.readi32(pe, 4 * p))
	end
	-- the order: each pivot, then the variables merged into it
	local rank = newi(n)
	for k = 0, npiv - 1 do
		buffer.writei32(rank, 4 * buffer.readi32(order, 4 * k), k)
	end
	local start = newi(npiv + 1)
	local key = newi(n)
	for v = 0, n - 1 do
		local r = v
		while buffer.readu8(status, r) == 3 do
			r = buffer.readi32(merged, 4 * r)
		end
		local kv = buffer.readi32(rank, 4 * r)
		buffer.writei32(key, 4 * v, kv)
		buffer.writei32(start, 4 * (kv + 1), buffer.readi32(start, 4 * (kv + 1)) + 1)
	end
	for k = 1, npiv do
		buffer.writei32(start, 4 * k, buffer.readi32(start, 4 * k) + buffer.readi32(start, 4 * (k - 1)))
	end
	local perm = buffer.create(4 * n)
	for pass = 1, 2 do
		for v = 0, n - 1 do
			if (buffer.readu8(status, v) ~= 3) == (pass == 1) then
				local kv = buffer.readi32(key, 4 * v)
				local at = buffer.readi32(start, 4 * kv)
				buffer.writei32(perm, 4 * at, v + 1)
				buffer.writei32(start, 4 * kv, at + 1)
			end
		end
	end
	return perm
end

-- The elimination tree of a symmetric pattern with its vertices ordered by perm (default as
-- numbered), by Liu's algorithm with path compression: column j's parent is the first row
-- below the diagonal in column j of the Cholesky factor. Returns the parents (a buffer of
-- 32-bit integers, positions in the order, 1-based; 0 at a root) and a postorder (a
-- permutation of the positions, children before parents).
local function etree(G: Graph, perm: buffer?): (buffer, buffer)
	local n, ptr, ind = G.n, G.ptr, G.ind
	local inv = if perm then inverse(perm) else nil
	local parent = buffer.create(4 * n)
	local ancestor = buffer.create(4 * n)
	for k = 1, n do
		local old = if perm then iget(perm, k) else k
		for p = u32(ptr, old - 1), u32(ptr, old) - 1 do
			local i = u32(ind, p) + 1
			if inv then
				i = iget(inv, i)
			end
			while i ~= 0 and i < k do
				local nx = iget(ancestor, i)
				iset(ancestor, i, k)
				if nx == 0 then
					iset(parent, i, k)
				end
				i = nx
			end
		end
	end
	-- postorder: depth first, children in increasing order (first child and next sibling
	-- lists; ancestor reused as the stack)
	local first, sibling = buffer.create(4 * n), buffer.create(4 * n)
	for j = n, 1, -1 do
		local pj = iget(parent, j)
		if pj ~= 0 then
			iset(sibling, j, iget(first, pj))
			iset(first, pj, j)
		end
	end
	local post = buffer.create(4 * n)
	local stack = ancestor
	local count = 0
	for r = 1, n do
		if iget(parent, r) == 0 then
			local top = 1
			iset(stack, 1, r)
			while top > 0 do
				local v = iget(stack, top)
				local c = iget(first, v)
				if c ~= 0 then
					iset(first, v, iget(sibling, c))
					top += 1
					iset(stack, top, c)
				else
					top -= 1
					count += 1
					iset(post, count, v)
				end
			end
		end
	end
	return parent, post
end

export type CholOptions = { relax: boolean? }
export type CholSymbolic = {
	n: number,
	perm: buffer, -- i32, 1-based: position k holds vertex perm[k]
	nsuper: number,
	super: buffer, -- u32: supernode s (0-based) is columns super[s] .. super[s + 1] - 1 (0-based)
	rowptr: buffer, -- u32: its rows are rows[rowptr[s] .. rowptr[s + 1] - 1], its columns first
	rows: buffer, -- u32, 0-based positions
	valptr: buffer, -- u32: its values (rows by columns, row by row) start at entry valptr[s]
	col2super: buffer, -- u32
	amap: buffer, -- u32 per stored entry of the matrix: 1 + its entry in L's values, or 0
	nnz: number, -- entries stored for L (whole blocks: explicit zeros included)
	lnz: number, -- entries of L on and below the diagonal (with amalgamation's zeros)
	madds: number, -- multiply-adds of the numeric factorization
	ptr: buffer, -- the pattern analyzed (to recognize it when reused)
	ind: buffer,
}

-- The symbolic Cholesky factorization of a symmetric pattern with vertices ordered by perm:
-- the order is postordered (its elimination tree's children before parents, which keeps L
-- the same and makes its supernodes contiguous), then the columns of L are grouped into
-- supernodes, runs of columns with the same rows below their diagonal block, stored as
-- dense blocks. With options.relax (default true), a supernode is merged into its parent
-- when that adds few explicit zeros (up to 4 columns while under 50% of the entries are
-- zeros, up to 16 under 30%, up to 48 under 10%, any width under 5%): wider blocks for the
-- dense kernels, at the cost of multiply-adds on zeros. These limits measured best on the
-- drum and cloth matrices (CHOLMOD's looser 80% for up to 16 columns doubled the drum's
-- multiply-adds and cost 25% in time). Also maps each stored entry of the matrix to its
-- place in L.
local function chol_analysis(G: Graph, perm: buffer, options: CholOptions?): CholSymbolic
	local o: CholOptions = options or {}
	local n, ptr, ind = G.n, G.ptr, G.ind
	-- postorder the elimination tree: the order P, its inverse, and the parents relabeled
	local parent0, post = etree(G, perm)
	local P = buffer.create(4 * n)
	for k = 1, n do
		iset(P, k, iget(perm, iget(post, k)))
	end
	local inv = inverse(P)
	local postinv = inverse(post)
	local parent = buffer.create(4 * n)
	for k = 1, n do
		local pk = iget(parent0, iget(post, k))
		iset(parent, k, if pk == 0 then 0 else iget(postinv, pk))
	end
	-- column counts of L (diagonal included), after Gilbert, Ng and Peyton (SIAM J. Matrix
	-- Anal. Appl., 1994), in time about proportional to the matrix's entries: row i of L is
	-- the subtree spanned by the columns j < i of row i of the matrix, whose leaves are those
	-- j whose first descendant comes after the previous leaf's; each leaf adds one to its
	-- column and the least common ancestor of consecutive leaves takes one back, so summing
	-- up the tree counts each row's columns once. (The order is postordered, so a node's
	-- descendants come just before it.)
	local counts = buffer.create(4 * n) -- delta, then summed up the tree
	local fdesc = buffer.create(4 * n) -- first descendant
	for k = 1, n do
		if iget(fdesc, k) == 0 then
			iset(counts, k, 1) -- a leaf
			local j = k
			while j ~= 0 and iget(fdesc, j) == 0 do
				iset(fdesc, j, k)
				j = iget(parent, j)
			end
		end
	end
	local ancestor = buffer.create(4 * n)
	local maxfirst = buffer.create(4 * n)
	local prevleaf = buffer.create(4 * n)
	for k = 1, n do
		iset(ancestor, k, k)
	end
	for j = 1, n do
		local pj = iget(parent, j)
		if pj ~= 0 then
			iset(counts, pj, iget(counts, pj) - 1)
		end
		local fj = iget(fdesc, j)
		local old = iget(P, j)
		for p = u32(ptr, old - 1), u32(ptr, old) - 1 do
			local i = iget(inv, u32(ind, p) + 1)
			if i > j and fj > iget(maxfirst, i) then
				-- j is a leaf of row i's subtree
				iset(maxfirst, i, fj)
				local jprev = iget(prevleaf, i)
				iset(prevleaf, i, j)
				iset(counts, j, iget(counts, j) + 1)
				if jprev ~= 0 then
					-- the least common ancestor of jprev and j, compressing the path
					local q = jprev
					while iget(ancestor, q) ~= q do
						q = iget(ancestor, q)
					end
					local t = jprev
					while t ~= q do
						local nx = iget(ancestor, t)
						iset(ancestor, t, q)
						t = nx
					end
					iset(counts, q, iget(counts, q) - 1)
				end
			end
		end
		if pj ~= 0 then
			iset(ancestor, j, pj)
		end
	end
	for j = 1, n do
		local pj = iget(parent, j)
		if pj ~= 0 then
			iset(counts, pj, iget(counts, pj) + iget(counts, j))
		end
	end
	local mark = ancestor -- reused below
	-- fundamental supernodes (1-based numbers; fsuper[j] the one holding column j): column
	-- j joins column j - 1's when j is its parent and its count is one less
	local fsuper = buffer.create(4 * n)
	local nf = 0
	for j = 1, n do
		if j == 1 or not (iget(parent, j - 1) == j and iget(counts, j) == iget(counts, j - 1) - 1) then
			nf += 1
		end
		iset(fsuper, j, nf)
	end
	-- per fundamental supernode: first column, columns, rows, explicit zeros, and the one it
	-- was merged into (0: none)
	local first, ncols, nrows, zeros, into = buffer.create(4 * nf), buffer.create(4 * nf), buffer.create(4 * nf), buffer.create(4 * nf), buffer.create(4 * nf)
	for j = n, 1, -1 do
		local s = iget(fsuper, j)
		iset(first, s, j)
		iset(ncols, s, iget(ncols, s) + 1)
	end
	for s = 1, nf do
		iset(nrows, s, iget(counts, iget(first, s)))
	end
	local function find(s: number): number
		while iget(into, s) ~= 0 do
			s = iget(into, s)
		end
		return s
	end
	local function stored(K: number, R: number): number
		return K * R - K * (K - 1) / 2
	end
	-- relaxed amalgamation, from the last supernode down: s merges into the supernode
	-- holding s + 1 when that holds its parent column
	if o.relax ~= false then
		for s = nf - 1, 1, -1 do
			local last = iget(first, s + 1) - 1
			local pc = iget(parent, last)
			if pc ~= 0 then
				local T = find(iget(fsuper, pc))
				if T == find(s + 1) then
					local Ks, KT = iget(ncols, s), iget(ncols, T)
					local Rs, RT = iget(nrows, s), iget(nrows, T)
					local K = Ks + KT
					local R = Ks + RT
					local total = stored(K, R)
					local z = iget(zeros, s) + iget(zeros, T) + total - stored(Ks, Rs) - stored(KT, RT)
					local frac = z / total
					if (K <= 4 and frac < 0.5) or (K <= 16 and frac < 0.3) or (K <= 48 and frac < 0.1) or frac < 0.05 then
						iset(into, s, T)
						iset(first, T, iget(first, s))
						iset(ncols, T, K)
						iset(nrows, T, R)
						iset(zeros, T, z)
					end
				end
			end
		end
	end
	-- the final supernodes (0-based numbers), in column order: a supernode starts at each
	-- unmerged fundamental supernode's first column
	local isstart = buffer.create(n + 1)
	for s = 1, nf do
		if iget(into, s) == 0 then
			buffer.writeu8(isstart, iget(first, s) - 1, 1)
		end
	end
	local ns = 0
	for j = 0, n - 1 do
		ns += buffer.readu8(isstart, j)
	end
	local super = buffer.create(4 * (ns + 1))
	local col2super = buffer.create(4 * n)
	local s = -1
	for j = 0, n - 1 do
		if buffer.readu8(isstart, j) == 1 then
			s += 1
			buffer.writeu32(super, 4 * s, j)
		end
		buffer.writeu32(col2super, 4 * j, s)
	end
	buffer.writeu32(super, 4 * ns, n)
	-- each supernode's parent: the supernode of its last column's parent (every column's
	-- rows below the block lie in the last column's)
	local sparent = buffer.create(4 * ns) -- -1 at a root
	for t = 0, ns - 1 do
		local last = buffer.readu32(super, 4 * (t + 1)) -- 1-based last column
		local pc = iget(parent, last)
		buffer.writei32(sparent, 4 * t, if pc == 0 then -1 else u32(col2super, pc - 1))
	end
	-- the rows below each supernode's block, by row subtrees in the supernodal tree: row k
	-- (in increasing order, so each list comes out sorted) belongs to every supernode met
	-- climbing from the supernode of each i < k in column k of the matrix, up to k's own
	local below = buffer.create(4 * (ns + 1))
	local smark = buffer.create(4 * ns)
	buffer.fill(smark, 0, 255)
	local function climb(k: number, fill: buffer, at: buffer)
		local own = u32(col2super, k - 1)
		local old = iget(P, k)
		for p = u32(ptr, old - 1), u32(ptr, old) - 1 do
			local i = iget(inv, u32(ind, p) + 1)
			if i < k then
				local t = u32(col2super, i - 1)
				while t ~= own and t >= 0 and buffer.readi32(smark, 4 * t) ~= k do
					buffer.writei32(smark, 4 * t, k)
					local q = buffer.readu32(at, 4 * t)
					buffer.writeu32(fill, 4 * q, k - 1)
					buffer.writeu32(at, 4 * t, q + 1)
					t = buffer.readi32(sparent, 4 * t)
				end
			end
		end
	end
	-- (every column's rows below the block lie in the last column's, so a supernode has as
	-- many as its last column has below the diagonal)
	for t = 0, ns - 1 do
		local last = buffer.readu32(super, 4 * (t + 1))
		buffer.writeu32(below, 4 * t, iget(counts, last) - 1)
	end
	local rowptr = buffer.create(4 * (ns + 1))
	local valptr = buffer.create(4 * (ns + 1))
	local at, vat, madds, lnz = 0, 0, 0, 0
	for t = 0, ns - 1 do
		local K = buffer.readu32(super, 4 * (t + 1)) - buffer.readu32(super, 4 * t)
		local R = K + buffer.readu32(below, 4 * t)
		buffer.writeu32(rowptr, 4 * t, at)
		buffer.writeu32(valptr, 4 * t, vat)
		at += R
		vat += R * K
		lnz += stored(K, R)
		for c = R - K + 1, R do
			madds += c * (c - 1) / 2
		end
	end
	buffer.writeu32(rowptr, 4 * ns, at)
	buffer.writeu32(valptr, 4 * ns, vat)
	local rows = buffer.create(4 * at)
	-- each supernode's own columns first, then (by the second climb) the rows below
	local fillat = buffer.create(4 * ns)
	for t = 0, ns - 1 do
		local c0 = buffer.readu32(super, 4 * t)
		local K = buffer.readu32(super, 4 * (t + 1)) - c0
		local r0 = buffer.readu32(rowptr, 4 * t)
		for q = 0, K - 1 do
			buffer.writeu32(rows, 4 * (r0 + q), c0 + q)
		end
		buffer.writeu32(fillat, 4 * t, r0 + K)
	end
	for k = 1, n do
		climb(k, rows, fillat)
	end
	-- where each stored entry of the matrix goes in L (its lower triangle, after ordering)
	local nnzA = u32(ptr, n)
	local amap = buffer.create(4 * nnzA)
	local rel = mark -- reused: a row's position within the current supernode
	for t = 0, ns - 1 do
		local c0 = buffer.readu32(super, 4 * t)
		local K = buffer.readu32(super, 4 * (t + 1)) - c0
		local base = buffer.readu32(valptr, 4 * t)
		local r0 = buffer.readu32(rowptr, 4 * t)
		local r1 = buffer.readu32(rowptr, 4 * (t + 1))
		for r = r0, r1 - 1 do
			buffer.writeu32(rel, 4 * buffer.readu32(rows, 4 * r), r - r0)
		end
		for j = c0 + 1, c0 + K do
			local old = iget(P, j)
			for p = u32(ptr, old - 1), u32(ptr, old) - 1 do
				local i = iget(inv, u32(ind, p) + 1)
				if i >= j then
					buffer.writeu32(amap, 4 * p, 1 + base + buffer.readu32(rel, 4 * (i - 1)) * K + (j - 1 - c0))
				end
			end
		end
	end
	return {
		n = n,
		perm = P,
		nsuper = ns,
		super = super,
		rowptr = rowptr,
		rows = rows,
		valptr = valptr,
		col2super = col2super,
		amap = amap,
		nnz = vat,
		lnz = lnz,
		madds = madds,
		ptr = ptr,
		ind = ind,
	}
end

return {
	of = of,
	from_edges = from_edges,
	from_triangles = from_triangles,
	symmetrize = symmetrize,
	degrees = degrees,
	bfs = bfs,
	components = components,
	inverse = inverse,
	bandwidth = bandwidth,
	profile = profile,
	rcm = rcm,
	dijkstra = dijkstra,
	path = path,
	astar = astar,
	amd = amd,
	etree = etree,
	chol_analysis = chol_analysis,
}
