--!strict
--!native
--!optimize 2

-- Spatial queries on point sets ({ vector }, 3-D; leave z = 0 for 2-D): which points lie
-- near a place, and which pairs lie near each other.
--
-- Two structures, for different uses:
--   spatial.grid    a uniform grid (cells hashed, so a sparse world costs nothing),
--                   rebuilt in O(n) by counting sort: the fastest for many queries of one
--                   radius on points that move every frame (flocks, particles). With
--                   about 12 neighbors each, all pairs among 2000 points took 1.2 ms
--                   natively, against 5.8 by testing every pair.
--   spatial.kdtree  a k-d tree (median splits, O(n log n) to build): nearest and k nearest
--                   points, radii that vary per query, clustered points, and fixed sets
--                   queried many times (waypoints, spawn points); refit keeps its
--                   structure for points that moved a little.
-- Indices are 1-based positions in the points array. Queries that return lists take an
-- optional table to fill (cleared first), so a per-frame query need not allocate.

local types = require("./types")
local graph = require("./graph")

type Graph = types.Graph

-- Uniform grid

export type Grid = {
	cell: number,
	n: number,
	sorted: { vector }, -- the points by bucket: sorted[k] is point order[k]
	order: { number },
	lo: vector,
	buckets: number,
	start: buffer, -- u32: bucket b holds sorted positions start[b] .. start[b + 1] - 1
	key: buffer, -- u32 scratch: each point's bucket
	stamp: buffer, -- u32: the query that last scanned each bucket
	tick: number,
	rebuild: (self: Grid, points: { vector }) -> Grid,
	radius: (self: Grid, p: vector, r: number, out: { number }?) -> ({ number }, number),
	pairs: (self: Grid, r: number, ii: { number }?, jj: { number }?) -> ({ number }, { number }, number),
	graph: (self: Grid, r: number) -> (Graph, buffer),
}

local Grid = {}
Grid.__index = Grid

-- The bucket of cell (ix, iy, iz) (nonnegative integers): a hash into `buckets` slots.
-- Distinct cells may share a bucket; queries test distances, and scan each bucket once.
local function bucket(ix: number, iy: number, iz: number, buckets: number): number
	return (ix * 73856093 + iy * 19349663 + iz * 83492791) % buckets
end

local function grid_rebuild(self: Grid, P: { vector }): Grid
	local n = #P
	local inv = 1 / self.cell
	local lo = if n > 0 then P[1] else vector.zero
	for i = 2, n do
		lo = vector.min(lo, P[i])
	end
	-- about two slots a point (at least 64), a power of 2
	local buckets = 64
	while buckets < 2 * n do
		buckets *= 2
	end
	local start = self.start
	if buffer.len(start) < 4 * (buckets + 1) then
		start = buffer.create(4 * (buckets + 1))
		self.start = start
		self.stamp = buffer.create(4 * buckets)
		self.tick = 0
	else
		buffer.fill(start, 0, 0, 4 * (buckets + 1))
	end
	local key = self.key
	if buffer.len(key) < 4 * n then
		key = buffer.create(4 * n)
		self.key = key
	end
	for i = 0, n - 1 do
		local q = (P[i + 1] - lo) * inv
		local b = bucket(math.floor(q.x), math.floor(q.y), math.floor(q.z), buckets)
		buffer.writeu32(key, 4 * i, b)
		buffer.writeu32(start, 4 * (b + 1), buffer.readu32(start, 4 * (b + 1)) + 1)
	end
	for b = 1, buckets do
		buffer.writeu32(start, 4 * b, buffer.readu32(start, 4 * b) + buffer.readu32(start, 4 * (b - 1)))
	end
	-- each point goes to its bucket's next free position (start[b] counting up, so that
	-- afterwards it holds bucket b + 1's start: shifted back by one bucket at the end)
	local sorted, order = self.sorted, self.order
	for i = 0, n - 1 do
		local b = buffer.readu32(key, 4 * i)
		local at = buffer.readu32(start, 4 * b)
		sorted[at + 1] = P[i + 1]
		order[at + 1] = i + 1
		buffer.writeu32(start, 4 * b, at + 1)
	end
	for k = #order, n + 1, -1 do
		order[k] = nil
		sorted[k] = nil
	end
	for b = buckets, 1, -1 do
		buffer.writeu32(start, 4 * b, buffer.readu32(start, 4 * (b - 1)))
	end
	buffer.writeu32(start, 0, 0)
	self.n, self.lo, self.buckets = n, lo, buckets
	return self
end

-- A new query number for the bucket stamps (cleared when it would overflow).
local function next_tick(self: Grid): number
	local t = self.tick + 1
	if t >= 2 ^ 32 then
		buffer.fill(self.stamp, 0, 0)
		t = 1
	end
	self.tick = t
	return t
end

-- The points within r of p (indices into the points array, in no particular order) and
-- their number, into out if given. Scans the cells within r: 27 when r <= cell.
local function grid_radius(self: Grid, p: vector, r: number, out: { number }?): ({ number }, number)
	local res = out or {}
	table.clear(res)
	local count = 0
	local buckets, start, stamp, xs, order = self.buckets, self.start, self.stamp, self.sorted, self.order
	local inv = 1 / self.cell
	local q = (p - self.lo) * inv
	local reach = math.ceil(r * inv)
	local cx, cy, cz = math.floor(q.x), math.floor(q.y), math.floor(q.z)
	local r2 = r * r
	local t = next_tick(self)
	for iz = math.max(cz - reach, 0), cz + reach do
		for iy = math.max(cy - reach, 0), cy + reach do
			for ix = math.max(cx - reach, 0), cx + reach do
				local b = bucket(ix, iy, iz, buckets)
				if buffer.readu32(stamp, 4 * b) ~= t then
					buffer.writeu32(stamp, 4 * b, t)
					for m = buffer.readu32(start, 4 * b), buffer.readu32(start, 4 * b + 4) - 1 do
						local d = xs[m + 1] - p
						if vector.dot(d, d) <= r2 then
							count += 1
							res[count] = order[m + 1]
						end
					end
				end
			end
		end
	end
	return res, count
end

-- Every pair of points within r of each other, i < j, as two lists (into ii and jj if
-- given) and their number.
local function grid_pairs(self: Grid, r: number, ii: { number }?, jj: { number }?): ({ number }, { number }, number)
	local I, J = ii or {}, jj or {}
	table.clear(I)
	table.clear(J)
	local count = 0
	local buckets, start, stamp, xs, order, lo = self.buckets, self.start, self.stamp, self.sorted, self.order, self.lo
	local inv = 1 / self.cell
	local reach = math.ceil(r * inv)
	local r2 = r * r
	for k = 1, self.n do
		local a = xs[k]
		local q = (a - lo) * inv
		local cx, cy, cz = math.floor(q.x), math.floor(q.y), math.floor(q.z)
		local oa = order[k]
		local t = next_tick(self)
		for iz = math.max(cz - reach, 0), cz + reach do
			for iy = math.max(cy - reach, 0), cy + reach do
				for ix = math.max(cx - reach, 0), cx + reach do
					local b = bucket(ix, iy, iz, buckets)
					if buffer.readu32(stamp, 4 * b) ~= t then
						buffer.writeu32(stamp, 4 * b, t)
						-- each pair once: from the point earlier in sorted order
						for m = math.max(buffer.readu32(start, 4 * b), k), buffer.readu32(start, 4 * b + 4) - 1 do
							local d = xs[m + 1] - a
							if vector.dot(d, d) <= r2 then
								local ob = order[m + 1]
								count += 1
								if oa < ob then
									I[count], J[count] = oa, ob
								else
									I[count], J[count] = ob, oa
								end
							end
						end
					end
				end
			end
		end
	end
	return I, J, count
end

-- The graph joining points within r of each other, both ways (see graph.luau), and its
-- edge lengths (an f64 array aligned with the graph's ind).
local function grid_graph(self: Grid, r: number): (Graph, buffer)
	local I, J, count = grid_pairs(self, r)
	local P = table.create(self.n) :: { vector }
	for k = 1, self.n do
		P[self.order[k]] = self.sorted[k]
	end
	local W = table.create(count) :: { number }
	for e = 1, count do
		local d = P[J[e]] - P[I[e]]
		W[e] = math.sqrt(vector.dot(d, d))
	end
	local G, w = graph.from_edges(self.n, I, J, { weights = W })
	return G, w or buffer.create(0)
end

Grid.rebuild = grid_rebuild
Grid.radius = grid_radius
Grid.pairs = grid_pairs
Grid.graph = grid_graph

-- A uniform grid of the points with cells `cell` wide (a query radius, typically: queries
-- of radius up to cell scan 27 cells). grid:rebuild(points) refills it for moved points
-- (or another array), reusing its storage.
local function new_grid(points: { vector }, cell: number): Grid
	assert(cell > 0, "the cell size must be positive")
	local self = setmetatable({
		cell = cell,
		n = 0,
		sorted = {},
		order = {},
		lo = vector.zero,
		buckets = 0,
		start = buffer.create(0),
		key = buffer.create(0),
		stamp = buffer.create(0),
		tick = 0,
	}, Grid) :: any
	return grid_rebuild(self, points)
end

-- k-d tree

export type KdTree = {
	n: number,
	leaf: number,
	sorted: { vector }, -- the points in tree order: sorted[k] is point order[k]
	order: { number },
	-- nodes (1 is the root; children after their parent): the sorted positions first ..
	-- last, the children (0 at a leaf), the bounding box
	first: { number },
	last: { number },
	left: { number },
	right: { number },
	bmin: { vector },
	bmax: { vector },
	stack: { number },
	heapd: { number },
	heapi: { number },
	rebuild: (self: KdTree, points: { vector }) -> KdTree,
	refit: (self: KdTree, points: { vector }) -> KdTree,
	nearest: (self: KdTree, p: vector) -> (number, number),
	knn: (self: KdTree, p: vector, k: number, out: { number }?, dist: { number }?) -> ({ number }, { number }),
	radius: (self: KdTree, p: vector, r: number, out: { number }?) -> ({ number }, number),
	graph: (self: KdTree, r: number) -> (Graph, buffer),
}

local KdTree = {}
KdTree.__index = KdTree

local function axis(v: vector, a: number): number
	return if a == 1 then v.x elseif a == 2 then v.y else v.z
end

-- The squared distance from p to the box [lo, hi] (0 inside).
local function box_dist2(p: vector, lo: vector, hi: vector): number
	local d = vector.max(vector.max(lo - p, p - hi), vector.zero)
	return vector.dot(d, d)
end

local function kd_rebuild(self: KdTree, P: { vector }): KdTree
	local n = #P
	local leaf = self.leaf
	local sorted, order = self.sorted, self.order
	for i = 1, n do
		sorted[i] = P[i]
		order[i] = i
	end
	for k = #order, n + 1, -1 do
		sorted[k] = nil
		order[k] = nil
	end
	local first, last, left, right, bmin, bmax = self.first, self.last, self.left, self.right, self.bmin, self.bmax
	local count = 0
	-- the node for sorted positions lo .. hi: split at the median along the box's longest
	-- side (by quickselect)
	local function build(lo: number, hi: number): number
		count += 1
		local id = count
		first[id] = lo
		last[id] = hi
		local mn, mx = sorted[lo], sorted[lo]
		for k = lo + 1, hi do
			local v = sorted[k]
			mn = vector.min(mn, v)
			mx = vector.max(mx, v)
		end
		bmin[id] = mn
		bmax[id] = mx
		if hi - lo < leaf then
			left[id] = 0
			right[id] = 0
			return id
		end
		local e = mx - mn
		local a = if e.x >= e.y and e.x >= e.z then 1 elseif e.y >= e.z then 2 else 3
		local mid = (lo + hi) // 2
		local l, h = lo, hi
		while l < h do
			local pv = axis(sorted[(l + h) // 2], a)
			local i, j = l, h
			while i <= j do
				while axis(sorted[i], a) < pv do
					i += 1
				end
				while axis(sorted[j], a) > pv do
					j -= 1
				end
				if i <= j then
					sorted[i], sorted[j] = sorted[j], sorted[i]
					order[i], order[j] = order[j], order[i]
					i += 1
					j -= 1
				end
			end
			if mid <= j then
				h = j
			elseif mid >= i then
				l = i
			else
				break
			end
		end
		local lc = build(lo, mid)
		local rc = build(mid + 1, hi)
		left[id] = lc
		right[id] = rc
		return id
	end
	if n > 0 then
		build(1, n)
	end
	for k = #first, count + 1, -1 do
		first[k] = nil
		last[k] = nil
		left[k] = nil
		right[k] = nil
		bmin[k] = nil
		bmax[k] = nil
	end
	self.n = n
	return self
end

-- Updates the tree for points that moved (the same array, or one in the same order): the
-- points and bounding boxes are refreshed and the splits kept, in O(n). Queries stay
-- correct and slow down as the points drift from where the tree was built; rebuild then.
local function kd_refit(self: KdTree, P: { vector }): KdTree
	assert(#P == self.n, "kdtree:refit needs as many points as the tree was built with")
	local sorted, order = self.sorted, self.order
	for k = 1, self.n do
		sorted[k] = P[order[k]]
	end
	local first, last, left, right, bmin, bmax = self.first, self.last, self.left, self.right, self.bmin, self.bmax
	-- children come after their parents, so going backwards visits children first
	for id = #first, 1, -1 do
		local lc = left[id]
		if lc == 0 then
			local mn, mx = sorted[first[id]], sorted[first[id]]
			for k = first[id] + 1, last[id] do
				local v = sorted[k]
				mn = vector.min(mn, v)
				mx = vector.max(mx, v)
			end
			bmin[id] = mn
			bmax[id] = mx
		else
			local rc = right[id]
			bmin[id] = vector.min(bmin[lc], bmin[rc])
			bmax[id] = vector.max(bmax[lc], bmax[rc])
		end
	end
	return self
end

-- Pushes node id's children on the stack, the nearer one to p on top.
local function push_children(stack: { number }, top: number, p: vector, lc: number, rc: number, bmin: { vector }, bmax: { vector }): number
	if box_dist2(p, bmin[lc], bmax[lc]) <= box_dist2(p, bmin[rc], bmax[rc]) then
		stack[top + 1] = rc
		stack[top + 2] = lc
	else
		stack[top + 1] = lc
		stack[top + 2] = rc
	end
	return top + 2
end

-- The nearest point to p: its index and distance (0 and math.huge for an empty tree).
local function kd_nearest(self: KdTree, p: vector): (number, number)
	if self.n == 0 then
		return 0, math.huge
	end
	local sorted, first, last, left, right, bmin, bmax = self.sorted, self.first, self.last, self.left, self.right, self.bmin, self.bmax
	local stack = self.stack
	local best, bestk = math.huge, 0
	local top = 1
	stack[1] = 1
	while top > 0 do
		local id = stack[top]
		top -= 1
		if box_dist2(p, bmin[id], bmax[id]) < best then
			local lc = left[id]
			if lc == 0 then
				for k = first[id], last[id] do
					local d = sorted[k] - p
					local d2 = vector.dot(d, d)
					if d2 < best then
						best = d2
						bestk = k
					end
				end
			else
				top = push_children(stack, top, p, lc, right[id], bmin, bmax)
			end
		end
	end
	return self.order[bestk], math.sqrt(best)
end

-- Removes the top of the max-heap (hd, hi) of size `size` and returns the new size.
local function heap_pop(hd: { number }, hi: { number }, size: number): number
	local d2, m = hd[size], hi[size]
	size -= 1
	local c = 1
	while true do
		local ch = 2 * c
		if ch > size then
			break
		end
		if ch < size and hd[ch + 1] > hd[ch] then
			ch += 1
		end
		if hd[ch] <= d2 then
			break
		end
		hd[c] = hd[ch]
		hi[c] = hi[ch]
		c = ch
	end
	if size > 0 then
		hd[c] = d2
		hi[c] = m
	end
	return size
end

-- The k nearest points to p, nearest first: their indices and distances (into out and
-- dist if given); all of them if the tree has fewer than k points.
local function kd_knn(self: KdTree, p: vector, k: number, out: { number }?, dist: { number }?): ({ number }, { number })
	local res, rd = out or {}, dist or {}
	table.clear(res)
	table.clear(rd)
	k = math.min(k, self.n)
	if k <= 0 then
		return res, rd
	end
	-- a max-heap of the k nearest so far, by squared distance
	local hd, hi = self.heapd, self.heapi
	local size = 0
	local sorted, first, last, left, right, bmin, bmax = self.sorted, self.first, self.last, self.left, self.right, self.bmin, self.bmax
	local stack = self.stack
	local top = 1
	stack[1] = 1
	while top > 0 do
		local id = stack[top]
		top -= 1
		local bound = if size < k then math.huge else hd[1]
		if box_dist2(p, bmin[id], bmax[id]) < bound then
			local lc = left[id]
			if lc == 0 then
				for m = first[id], last[id] do
					local d = sorted[m] - p
					local d2 = vector.dot(d, d)
					if size < k then
						-- add, sifting up
						size += 1
						local c = size
						while c > 1 do
							local parent = c // 2
							if hd[parent] >= d2 then
								break
							end
							hd[c] = hd[parent]
							hi[c] = hi[parent]
							c = parent
						end
						hd[c] = d2
						hi[c] = m
					elseif d2 < hd[1] then
						-- replace the farthest, sifting down
						local c = 1
						while true do
							local ch = 2 * c
							if ch > size then
								break
							end
							if ch < size and hd[ch + 1] > hd[ch] then
								ch += 1
							end
							if hd[ch] <= d2 then
								break
							end
							hd[c] = hd[ch]
							hi[c] = hi[ch]
							c = ch
						end
						hd[c] = d2
						hi[c] = m
					end
				end
			else
				top = push_children(stack, top, p, lc, right[id], bmin, bmax)
			end
		end
	end
	-- farthest first off the heap, into the back of the result
	local order = self.order
	for s = size, 1, -1 do
		res[s] = order[hi[1]]
		rd[s] = math.sqrt(hd[1])
		size = heap_pop(hd, hi, size)
	end
	return res, rd
end

-- The points within r of p (indices, in no particular order) and their number, into out
-- if given.
local function kd_radius(self: KdTree, p: vector, r: number, out: { number }?): ({ number }, number)
	local res = out or {}
	table.clear(res)
	local count = 0
	if self.n == 0 then
		return res, 0
	end
	local r2 = r * r
	local sorted, order, first, last, left, right, bmin, bmax = self.sorted, self.order, self.first, self.last, self.left, self.right, self.bmin, self.bmax
	local stack = self.stack
	local top = 1
	stack[1] = 1
	while top > 0 do
		local id = stack[top]
		top -= 1
		if box_dist2(p, bmin[id], bmax[id]) <= r2 then
			local lc = left[id]
			if lc == 0 then
				for m = first[id], last[id] do
					local d = sorted[m] - p
					if vector.dot(d, d) <= r2 then
						count += 1
						res[count] = order[m]
					end
				end
			else
				stack[top + 1] = right[id]
				stack[top + 2] = lc
				top += 2
			end
		end
	end
	return res, count
end

-- The graph joining points within r of each other (as grid:graph).
local function kd_graph(self: KdTree, r: number): (Graph, buffer)
	local I, J, W = {}, {}, {}
	local r2 = r * r
	local sorted, order, first, last, left, right, bmin, bmax = self.sorted, self.order, self.first, self.last, self.left, self.right, self.bmin, self.bmax
	local stack = self.stack
	for k = 1, self.n do
		local p = sorted[k]
		local top = 1
		stack[1] = 1
		while top > 0 do
			local id = stack[top]
			top -= 1
			-- only nodes holding positions after k, so each pair is found once
			if last[id] > k and box_dist2(p, bmin[id], bmax[id]) <= r2 then
				local lc = left[id]
				if lc == 0 then
					for m = math.max(first[id], k + 1), last[id] do
						local d = sorted[m] - p
						local d2 = vector.dot(d, d)
						if d2 <= r2 then
							table.insert(I, order[k])
							table.insert(J, order[m])
							table.insert(W, math.sqrt(d2))
						end
					end
				else
					stack[top + 1] = right[id]
					stack[top + 2] = lc
					top += 2
				end
			end
		end
	end
	local G, w = graph.from_edges(self.n, I, J, { weights = W })
	return G, w or buffer.create(0)
end

KdTree.rebuild = kd_rebuild
KdTree.refit = kd_refit
KdTree.nearest = kd_nearest
KdTree.knn = kd_knn
KdTree.radius = kd_radius
KdTree.graph = kd_graph

export type KdOptions = { leaf: number? }

-- A k-d tree of the points (options.leaf: points per leaf, default 8). tree:rebuild(points)
-- builds it again, reusing its storage; tree:refit(points) keeps its splits.
local function new_kdtree(points: { vector }, options: KdOptions?): KdTree
	local o: KdOptions = options or {}
	local self = setmetatable({
		n = 0,
		leaf = o.leaf or 8,
		sorted = {},
		order = {},
		first = {},
		last = {},
		left = {},
		right = {},
		bmin = {},
		bmax = {},
		stack = {},
		heapd = {},
		heapi = {},
	}, KdTree) :: any
	return kd_rebuild(self, points)
end

return {
	grid = new_grid,
	kdtree = new_kdtree,
}
