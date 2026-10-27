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
-- matrix's band before a banded factorization) and shortest paths (dijkstra, astar).

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

-- Breadth-first search from source (1-based) through G's edges. Returns the vertices
-- reached in the order visited (a buffer of 32-bit integers, 1-based) and each vertex's
-- level (its distance in edges: 0 at the source, -1 if not reached).
local function bfs(G: Graph, source: number): (buffer, buffer)
	local n, ptr, ind = G.n, G.ptr, G.ind
	assert(source % 1 == 0 and source >= 1 and source <= n, "source outside 1..n")
	local level = buffer.create(4 * n)
	buffer.fill(level, 0, 255) -- -1 everywhere
	local queue = buffer.create(4 * n)
	iset(level, source, 0)
	buffer.writeu32(queue, 0, source - 1)
	local head, tail = 0, 1
	while head < tail do
		local v = u32(queue, head)
		head += 1
		local lv = buffer.readi32(level, 4 * v) + 1
		for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
			local w = u32(ind, p)
			if buffer.readi32(level, 4 * w) < 0 then
				buffer.writei32(level, 4 * w, lv)
				buffer.writeu32(queue, 4 * tail, w)
				tail += 1
			end
		end
	end
	local order = buffer.create(4 * tail)
	for k = 0, tail - 1 do
		buffer.writei32(order, 4 * k, u32(queue, k) + 1)
	end
	return order, level
end

-- Connected components of an undirected graph (symmetrize a directed one first): each
-- vertex's component (1-based, numbered in order of their lowest vertex; a buffer of
-- 32-bit integers) and the number of components.
local function components(G: Graph): (buffer, number)
	local n, ptr, ind = G.n, G.ptr, G.ind
	local label = buffer.create(4 * n)
	local queue = buffer.create(4 * n)
	local count = 0
	for s = 0, n - 1 do
		if buffer.readi32(label, 4 * s) == 0 then
			count += 1
			buffer.writei32(label, 4 * s, count)
			buffer.writeu32(queue, 0, s)
			local head, tail = 0, 1
			while head < tail do
				local v = u32(queue, head)
				head += 1
				for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
					local w = u32(ind, p)
					if buffer.readi32(label, 4 * w) == 0 then
						buffer.writei32(label, 4 * w, count)
						buffer.writeu32(queue, 4 * tail, w)
						tail += 1
					end
				end
			end
		end
	end
	return label, count
end

-- Orderings

-- The positions of a permutation: inv[perm[k]] = k.
local function inverse(perm: buffer): buffer
	local n = buffer.len(perm) // 4
	local inv = buffer.create(4 * n)
	for k = 1, n do
		iset(inv, iget(perm, k), k)
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
	-- breadth-first search from s within its component, unmarked by placed; returns the
	-- depth (last level) and a lowest-degree vertex of the last level
	local function probe(s: number): (number, number)
		searches += 1
		buffer.writeu32(stamp, 4 * s, searches)
		buffer.writeu32(level, 4 * s, 0)
		buffer.writeu32(queue, 0, s)
		local head, tail = 0, 1
		local depth = 0
		while head < tail do
			local v = u32(queue, head)
			head += 1
			local lv = u32(level, v)
			depth = lv
			for p = u32(ptr, v), u32(ptr, v + 1) - 1 do
				local w = u32(ind, p)
				if u32(stamp, w) ~= searches then
					buffer.writeu32(stamp, 4 * w, searches)
					buffer.writeu32(level, 4 * w, lv + 1)
					buffer.writeu32(queue, 4 * tail, w)
					tail += 1
				end
			end
		end
		local best, bestdeg = s, math.huge
		for k = tail - 1, 0, -1 do
			local v = u32(queue, k)
			if u32(level, v) ~= depth then
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
}
