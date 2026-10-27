--!strict
--!native
-- Spatial queries in two scenes, usable outside Roblox:
--
-- A flock (boids): each boid steers by its neighbors within a radius (keep apart, match
-- their heading, move toward their center) and turns back from the walls of a box. The
-- neighbors are found each frame by citrate's uniform grid (spatial.grid), its k-d tree
-- (spatial.kdtree) or by testing every pair; flock.step takes the method and reports the
-- time it spent finding neighbors.
--
-- Navigation: waypoints scattered among box obstacles on a floor, joined to the waypoints
-- within reach (kdtree:graph) when the straight segment between them clears every
-- obstacle; routes by A* (graph.astar) with the straight-line distance as heuristic.

local citrate = require("../../src")
local spatial, graph = citrate.spatial, citrate.graph

local flock = {}

local seed = 1
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647
end

-- Flock

export type Flock = {
	n: number,
	pos: { vector },
	vel: { vector },
	half: vector, -- the box is [-half, half]
	radius: number,
	grid: any,
	tree: any,
	-- this frame's neighbor lists, flat: boid i's are nbr[first[i] .. first[i + 1] - 1]
	nbr: { number },
	first: { number },
	found: { number },
	acc: { vector },
}

flock.SPEED = 12 -- studs per second (between 0.5 and 1.5 times)
flock.WEIGHTS = { separate = 1.6, align = 1.0, cohere = 0.8, wall = 2.5 }

-- n boids in a box of half-size `half`, seeing neighbors within `radius`.
function flock.new(n: number, half: vector, radius: number, s: number?): Flock
	seed = s or 1
	local pos, vel = table.create(n) :: { vector }, table.create(n) :: { vector }
	for i = 1, n do
		pos[i] = vector.create((2 * rnd() - 1) * half.x, (2 * rnd() - 1) * half.y, (2 * rnd() - 1) * half.z) * 0.9
		local d = vector.normalize(vector.create(2 * rnd() - 1, 0.3 * (2 * rnd() - 1), 2 * rnd() - 1))
		vel[i] = d * flock.SPEED
	end
	return {
		n = n,
		pos = pos,
		vel = vel,
		half = half,
		radius = radius,
		grid = spatial.grid(pos, radius),
		tree = spatial.kdtree(pos),
		nbr = {},
		first = {},
		found = {},
		acc = table.create(n, vector.zero) :: { vector },
	}
end

-- Fills the neighbor lists by method "grid", "kdtree" or "brute", or by a function
-- query(i, out) -> count (Roblox's own queries, in the client).
local function find_neighbors(F: Flock, method: string | (number, { number }) -> number)
	local n, pos, r = F.n, F.pos, F.radius
	local nbr, first, found = F.nbr, F.first, F.found
	local count = 0
	if method == "brute" then
		local r2 = r * r
		for i = 1, n do
			first[i] = count + 1
			local a = pos[i]
			for j = 1, n do
				if j ~= i then
					local d = pos[j] - a
					if vector.dot(d, d) <= r2 then
						count += 1
						nbr[count] = j
					end
				end
			end
		end
	else
		local query: (number, { number }) -> number
		if method == "grid" then
			F.grid:rebuild(pos)
			query = function(i: number, out: { number }): number
				local _, c = F.grid:radius(pos[i], r, out)
				return c
			end
		elseif method == "kdtree" then
			F.tree:rebuild(pos)
			query = function(i: number, out: { number }): number
				local _, c = F.tree:radius(pos[i], r, out)
				return c
			end
		else
			query = method :: (number, { number }) -> number
		end
		for i = 1, n do
			first[i] = count + 1
			local c = query(i, found)
			for k = 1, c do
				local j = found[k]
				if j ~= i then
					count += 1
					nbr[count] = j
				end
			end
		end
	end
	first[n + 1] = count + 1
end

-- Advances the flock by dt, finding neighbors by `method` (see find_neighbors). Returns the
-- milliseconds spent finding neighbors and the mean number of neighbors.
function flock.step(F: Flock, dt: number, method: string | (number, { number }) -> number): (number, number)
	local t0 = os.clock()
	find_neighbors(F, method)
	local ms = (os.clock() - t0) * 1000
	local n, pos, vel, acc, nbr, first = F.n, F.pos, F.vel, F.acc, F.nbr, F.first
	local W = flock.WEIGHTS
	local r = F.radius
	local half = F.half
	local speed = flock.SPEED
	for i = 1, n do
		local p, v = pos[i], vel[i]
		local sep, ali, coh = vector.zero, vector.zero, vector.zero
		local k0, k1 = first[i], first[i + 1] - 1
		for k = k0, k1 do
			local j = nbr[k]
			local d = p - pos[j]
			local dd = vector.dot(d, d)
			if dd < 0.25 * r * r and dd > 1e-8 then
				sep += d / dd
			end
			ali += vel[j]
			coh += pos[j]
		end
		local a = vector.zero
		local m = k1 - k0 + 1
		if m > 0 then
			a += W.separate * sep * r * speed * 0.5
			a += W.align * (ali / m - v)
			a += W.cohere * (coh / m - p)
		end
		-- turn back from the walls (within a radius of them)
		local over = vector.max(p - (half - vector.one * r), vector.zero) - vector.max(-(half - vector.one * r) - p, vector.zero)
		a -= W.wall * over * speed
		acc[i] = a
	end
	for i = 1, n do
		local v = vel[i] + acc[i] * dt
		local s = vector.magnitude(v)
		if s > 1.5 * speed then
			v *= 1.5 * speed / s
		elseif s < 0.5 * speed then
			v *= 0.5 * speed / math.max(s, 1e-6)
		end
		vel[i] = v
		pos[i] = vector.clamp(pos[i] + v * dt, -F.half, F.half)
	end
	return ms, (first[n + 1] - 1) / n
end

-- Navigation

export type Box = { center: vector, half: vector } -- on the floor (y = 0), half.y its height
export type Nav = {
	size: number,
	boxes: { Box },
	points: { vector },
	tree: any,
	graph: any,
	weights: buffer,
	edges: number,
	candidates: number,
}

-- Whether the segment a-b (in the floor plane) passes through box B grown by `margin`.
local function blocked(a: vector, b: vector, B: Box, margin: number): boolean
	local lo = B.center - B.half - vector.create(margin, 0, margin)
	local hi = B.center + B.half + vector.create(margin, 0, margin)
	local d = b - a
	local t0, t1 = 0, 1
	for _, axis in { "x", "z" } do
		local da = (d :: any)[axis] :: number
		local pa = (a :: any)[axis] :: number
		local l, h = (lo :: any)[axis] :: number, (hi :: any)[axis] :: number
		if math.abs(da) < 1e-9 then
			if pa < l or pa > h then
				return false
			end
		else
			local u, w = (l - pa) / da, (h - pa) / da
			if u > w then
				u, w = w, u
			end
			t0, t1 = math.max(t0, u), math.min(t1, w)
			if t0 > t1 then
				return false
			end
		end
	end
	return true
end

local function inside(p: vector, boxes: { Box }, margin: number): boolean
	for _, B in boxes do
		local d = p - B.center
		if math.abs(d.x) <= B.half.x + margin and math.abs(d.z) <= B.half.z + margin then
			return true
		end
	end
	return false
end

flock.MARGIN = 1.5 -- clearance kept from obstacles, studs

-- A floor `size` studs square with `nboxes` obstacles and about `npoints` waypoints (plus
-- four outside each obstacle's corners, so routes can hug them), joined within `reach`.
function flock.nav(size: number, nboxes: number, npoints: number, reach: number, s: number?): Nav
	seed = s or 7
	local margin = flock.MARGIN
	local boxes: { Box } = {}
	while #boxes < nboxes do
		local half = vector.create(2 + rnd() * 10, 3 + rnd() * 6, 2 + rnd() * 10)
		local c = vector.create((rnd() - 0.5) * (size - 2 * half.x), half.y, (rnd() - 0.5) * (size - 2 * half.z))
		-- keep a gap between obstacles
		if not inside(c, boxes, math.max(half.x, half.z) + 2 * margin + 2) then
			table.insert(boxes, { center = c, half = half })
		end
	end
	local points: { vector } = {}
	for _, B in boxes do
		for _, sx in { -1, 1 } do
			for _, sz in { -1, 1 } do
				local p = vector.create(B.center.x + sx * (B.half.x + 1.2 * margin), 0, B.center.z + sz * (B.half.z + 1.2 * margin))
				if math.abs(p.x) < size / 2 and math.abs(p.z) < size / 2 and not inside(p, boxes, margin) then
					table.insert(points, p)
				end
			end
		end
	end
	local tries = 0
	while #points < npoints and tries < 20 * npoints do
		tries += 1
		local p = vector.create((rnd() - 0.5) * size, 0, (rnd() - 0.5) * size)
		if not inside(p, boxes, margin) then
			table.insert(points, p)
		end
	end
	local tree = spatial.kdtree(points)
	local G0, w0 = tree:graph(reach)
	-- keep the edges that clear every obstacle (tested against the obstacles near them)
	local ii, jj, ww = {}, {}, {}
	for v = 0, G0.n - 1 do
		for p = buffer.readu32(G0.ptr, 4 * v), buffer.readu32(G0.ptr, 4 * v + 4) - 1 do
			local u = buffer.readu32(G0.ind, 4 * p)
			if u > v then
				local a, b = points[v + 1], points[u + 1]
				local clear = true
				for _, B in boxes do
					if blocked(a, b, B, margin * 0.9) then
						clear = false
						break
					end
				end
				if clear then
					table.insert(ii, v + 1)
					table.insert(jj, u + 1)
					table.insert(ww, buffer.readf64(w0, 8 * p))
				end
			end
		end
	end
	local G, w = graph.from_edges(#points, ii, jj, { weights = ww })
	return {
		size = size,
		boxes = boxes,
		points = points,
		tree = tree,
		graph = G,
		weights = w :: buffer,
		edges = #ii,
		candidates = buffer.len(G0.ind) // 8,
	}
end

export type RouteInfo = { ms: number, expanded: number, dijkstra_ms: number, length: number }

-- A route from a to b: a, the waypoints of a shortest path between the waypoints nearest
-- them (by A*), b; or nil if they are not connected. info compares A* with Dijkstra's
-- search.
function flock.route(N: Nav, a: vector, b: vector): ({ vector }?, RouteInfo)
	local s = N.tree:nearest(a)
	local t = N.tree:nearest(b)
	local target = N.points[t]
	local t0 = os.clock()
	local path, info = graph.astar(N.graph, N.weights, s, t, function(v: number): number
		return vector.magnitude(N.points[v] - target)
	end)
	local ms = (os.clock() - t0) * 1000
	t0 = os.clock()
	graph.dijkstra(N.graph, N.weights, s, { target = t })
	local dms = (os.clock() - t0) * 1000
	local result: RouteInfo = { ms = ms, expanded = info.expanded, dijkstra_ms = dms, length = info.cost }
	if path == nil then
		return nil, result
	end
	local route = { a }
	for _, v in path do
		table.insert(route, N.points[v])
	end
	table.insert(route, b)
	return route, result
end

return flock
