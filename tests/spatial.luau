--!strict

-- Spatial queries against brute force: uniform, clustered, coincident, flat (2-D) and
-- widely spread point sets; the grid's radius, pairs and graph, the k-d tree's nearest,
-- k nearest, radius and graph queries, refit after motion, and rebuilds of other sizes.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check
local spatial, graph = lib.spatial, lib.graph

local seed = 4242
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647
end
local function gauss(): number
	return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())
end

type Set = { name: string, points: { vector }, r: number }

local function sets(): { Set }
	local out: { Set } = {}
	local uniform = {}
	for i = 1, 600 do
		uniform[i] = vector.create(rnd() * 10, rnd() * 10, rnd() * 10)
	end
	table.insert(out, { name = "uniform", points = uniform, r = 1.2 })
	local clustered = {}
	for i = 1, 600 do
		local c = (i % 4) * 7
		clustered[i] = vector.create(c + 0.3 * gauss(), 0.3 * gauss(), c + 0.3 * gauss())
	end
	table.insert(out, { name = "clustered", points = clustered, r = 0.25 })
	local coincident = {}
	for i = 1, 300 do
		coincident[i] = if i % 3 == 0 then vector.create(1, 2, 3) else vector.create(math.floor(rnd() * 5), math.floor(rnd() * 5), 0)
	end
	table.insert(out, { name = "coincident", points = coincident, r = 1 })
	local flat = {}
	for i = 1, 500 do
		flat[i] = vector.create(rnd() * 20 - 10, rnd() * 20 - 10, 0)
	end
	table.insert(out, { name = "flat", points = flat, r = 1.5 })
	local spread = {}
	for i = 1, 400 do
		spread[i] = vector.create(rnd() * 1e5, rnd() * 3, rnd() * 1e5)
	end
	-- pairs of near neighbors far apart from the rest
	for i = 2, 400, 2 do
		spread[i] = spread[i - 1] + vector.create(rnd() * 2, 0, 0)
	end
	table.insert(out, { name = "spread", points = spread, r = 1.5 })
	return out
end

local function brute_radius(P: { vector }, p: vector, r: number): { number }
	local t = {}
	for i, q in P do
		local d = q - p
		if vector.dot(d, d) <= r * r then
			table.insert(t, i)
		end
	end
	return t
end

local function same_set(a: { number }, b: { number }, count: number?): boolean
	local n = count or #a
	if n ~= #b then
		return false
	end
	local seen: { [number]: number } = {}
	for k = 1, n do
		seen[a[k]] = (seen[a[k]] or 0) + 1
	end
	for _, v in b do
		if not seen[v] or seen[v] == 0 then
			return false
		end
		seen[v] -= 1
	end
	return true
end

-- the probes: every point, and some points nearby
local function probes(P: { vector }): { vector }
	local t = {}
	for i = 1, #P, 7 do
		table.insert(t, P[i])
		table.insert(t, P[i] + vector.create(0.31, -0.17, 0.05))
	end
	return t
end

test("spatial: grid radius queries and pairs against brute force", function()
	for _, set in sets() do
		local P, r = set.points, set.r
		for _, cell in { r, r / 2.5 } do
			local G = spatial.grid(P, cell)
			local ok = true
			local out = {}
			for _, p in probes(P) do
				local res, count = G:radius(p, r, out)
				ok = ok and same_set(res, brute_radius(P, p, r), count)
			end
			check(ok, set.name .. ": radius queries, cell " .. cell)
			-- pairs: each once, i < j
			local I, J, count = G:pairs(r)
			local expect = 0
			local pairs_ok = true
			for i = 1, #P do
				for j = i + 1, #P do
					local d = P[j] - P[i]
					if vector.dot(d, d) <= r * r then
						expect += 1
					end
				end
			end
			local seen = {}
			for e = 1, count do
				local i, j = I[e], J[e]
				local d = P[j] - P[i]
				local key = i * 100000 + j
				pairs_ok = pairs_ok and i < j and vector.dot(d, d) <= r * r and not seen[key]
				seen[key] = true
			end
			check(pairs_ok and count == expect, string.format("%s: pairs, cell %g (%d of %d)", set.name, cell, count, expect))
		end
	end
end)

test("spatial: k-d tree queries against brute force", function()
	for _, set in sets() do
		local P, r = set.points, set.r
		for _, leaf in { 1, 8 } do
			local T = spatial.kdtree(P, { leaf = leaf })
			local near_ok, knn_ok, radius_ok = true, true, true
			for _, p in probes(P) do
				-- (single-precision coordinates: distances good to about 1e-7 of |p|)
				local tol = 1e-6 * math.max(1, vector.magnitude(p))
				-- brute distances, sorted
				local ds: { number } = {}
				for i, q in P do
					ds[i] = vector.magnitude(q - p)
				end
				local sorted = table.clone(ds)
				table.sort(sorted)
				local i: number, d: number = T:nearest(p)
				near_ok = near_ok and math.abs(d - sorted[1]) <= tol and math.abs(ds[i] - d) <= tol
				local idx, dist = T:knn(p, 10)
				for k = 1, 10 do
					knn_ok = knn_ok and math.abs(dist[k] - sorted[k]) <= tol and math.abs(ds[idx[k]] - dist[k]) <= tol
				end
				local seen = {}
				for k = 1, 10 do
					knn_ok = knn_ok and not seen[idx[k]]
					seen[idx[k]] = true
				end
				local res, count = T:radius(p, r)
				radius_ok = radius_ok and same_set(res, brute_radius(P, p, r), count)
			end
			check(near_ok, set.name .. ": nearest, leaf " .. leaf)
			check(knn_ok, set.name .. ": k nearest, leaf " .. leaf)
			check(radius_ok, set.name .. ": radius, leaf " .. leaf)
		end
	end
end)

test("spatial: radius graphs agree", function()
	for _, set in sets() do
		local P, r = set.points, set.r
		local G1, w1 = spatial.grid(P, r):graph(r)
		local G2, w2 = spatial.kdtree(P):graph(r)
		local ok = G1.n == #P and buffer.len(G1.ind) == buffer.len(G2.ind) and buffer.len(w1) == buffer.len(w2)
		if ok then
			for k = 0, buffer.len(G1.ind) // 4 - 1 do
				ok = ok and buffer.readu32(G1.ind, 4 * k) == buffer.readu32(G2.ind, 4 * k)
				ok = ok and math.abs(buffer.readf64(w1, 8 * k) - buffer.readf64(w2, 8 * k)) < 1e-9
			end
			for v = 0, G1.n do -- ptr has n + 1 entries
				ok = ok and buffer.readu32(G1.ptr, 4 * v) == buffer.readu32(G2.ptr, 4 * v)
			end
		end
		-- the edges are the pairs (both ways), coincident points included
		local expect = 0
		for i = 1, #P do
			for j = i + 1, #P do
				local d = P[j] - P[i]
				if vector.dot(d, d) <= r * r then
					expect += 2
				end
			end
		end
		check(ok and buffer.len(G1.ind) == 4 * expect, set.name .. ": grid and tree graphs")
		-- usable by the graph module
		local _, count = graph.components(G1)
		check(count >= 1, set.name .. ": components")
	end
end)

test("spatial: moving points, rebuilds and small sets", function()
	local P = {}
	for i = 1, 400 do
		P[i] = vector.create(rnd() * 8, rnd() * 8, rnd() * 8)
	end
	local T = spatial.kdtree(P)
	local G = spatial.grid(P, 1)
	-- move every point, then refit the tree and rebuild the grid in place
	for i = 1, 400 do
		P[i] += vector.create(gauss(), gauss(), gauss()) * 0.5
	end
	T:refit(P)
	G:rebuild(P)
	local ok = true
	for _, p in probes(P) do
		local a, ca = T:radius(p, 1)
		local b, cb = G:radius(p, 1)
		local e = brute_radius(P, p, 1)
		ok = ok and same_set(a, e, ca) and same_set(b, e, cb)
		local _, d = T:nearest(p)
		local best = math.huge
		for _, q in P do
			best = math.min(best, vector.magnitude(q - p))
		end
		ok = ok and math.abs(d - best) < 1e-6
	end
	check(ok, "refit tree and rebuilt grid")
	-- rebuilds with fewer and more points
	for _, n in { 50, 900 } do
		local Q = {}
		for i = 1, n do
			Q[i] = vector.create(rnd() * 6, rnd() * 6, rnd() * 6)
		end
		T:rebuild(Q)
		G:rebuild(Q)
		local good = #T.sorted == n and G.n == n
		for _, p in probes(Q) do
			local a, ca = T:radius(p, 1)
			local b, cb = G:radius(p, 1)
			local e = brute_radius(Q, p, 1)
			good = good and same_set(a, e, ca) and same_set(b, e, cb)
		end
		check(good, "rebuilt with " .. n .. " points")
	end
	-- empty and single-point sets
	local E = spatial.kdtree({})
	local i, d = E:nearest(vector.zero)
	check(i == 0 and d == math.huge and #E:knn(vector.zero, 3) == 0, "empty tree")
	local _, c0 = spatial.grid({}, 1):radius(vector.zero, 1)
	check(c0 == 0, "empty grid")
	local S = spatial.kdtree({ vector.create(1, 1, 1) })
	local idx, dist = S:knn(vector.zero, 5)
	check(#idx == 1 and idx[1] == 1 and math.abs(dist[1] - math.sqrt(3)) < 1e-6, "one point, k = 5")
end)

return nil
