--!strict
-- Spatial queries (luau -O2 [--codegen] bench/spatial.luau): every point's neighbors
-- within r (a flock's query each frame) by brute force, the grid and the k-d tree, on
-- uniform and clustered points with about 12 neighbors each; k nearest queries; and how a
-- refit tree's queries slow as its points drift, against rebuilding.
local lib = require("../src")
local spatial = lib.spatial

local seed = 12345
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647
end
local function gauss(): number
	return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())
end

-- n points with about 12 neighbors within 1 each: uniform in a cube, or in 20 Gaussian
-- clusters (denser inside, so r is chosen for the same mean neighbor count)
local function uniform(n: number): { vector }
	local L = (n * (4 / 3) * math.pi / 12) ^ (1 / 3)
	local P = table.create(n) :: { vector }
	for i = 1, n do
		P[i] = vector.create(rnd() * L, rnd() * L, rnd() * L)
	end
	return P
end
local function clustered(n: number): { vector }
	local L = (n * (4 / 3) * math.pi / 12) ^ (1 / 3)
	local centers = {}
	for c = 1, 20 do
		centers[c] = vector.create(rnd() * L, rnd() * L, rnd() * L)
	end
	local s = L / 12
	local P = table.create(n) :: { vector }
	for i = 1, n do
		P[i] = centers[i % 20 + 1] + vector.create(gauss(), gauss(), gauss()) * s
	end
	return P
end

-- the radius giving about k neighbors per point on average (by sampling with the tree)
local function radius_for(P: { vector }, k: number): number
	local T = spatial.kdtree(P)
	local sum, count = 0, 0
	for i = 1, #P, math.max(1, #P // 200) do
		local _, d = T:knn(P[i], k + 1)
		sum += d[k + 1]
		count += 1
	end
	return sum / count
end

local function best_ms(f: () -> ()): number
	local best = math.huge
	for _ = 1, 3 do
		local t0 = os.clock()
		f()
		best = math.min(best, os.clock() - t0)
	end
	return best * 1000
end

local function brute(P: { vector }, r: number)
	local r2 = r * r
	local n = #P
	for i = 1, n do
		local a = P[i]
		for j = i + 1, n do
			local d = P[j] - a
			if vector.dot(d, d) <= r2 then
				continue
			end
		end
	end
end

print("## Every point's neighbors within r (about 12 each)\n")
print("| points | layout | brute force | grid: build + a radius query per point | grid: pairs | k-d tree: build + a radius query per point | k-d tree: 8 nearest of every point |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for _, n in { 500, 2000, 10000 } do
	for _, layout in { "uniform", "clustered" } do
		local P = if layout == "uniform" then uniform(n) else clustered(n)
		local r = radius_for(P, 12)
		local out = {}
		local tb = if n <= 2000 then string.format("%.2f ms", best_ms(function() brute(P, r) end)) else "-"
		local G = spatial.grid(P, r)
		local tg = best_ms(function()
			G:rebuild(P)
			for i = 1, n do
				G:radius(P[i], r, out)
			end
		end)
		local I, J = {}, {}
		local tp = best_ms(function()
			G:rebuild(P)
			G:pairs(r, I, J)
		end)
		local T = spatial.kdtree(P)
		local tk = best_ms(function()
			T:rebuild(P)
			for i = 1, n do
				T:radius(P[i], r, out)
			end
		end)
		local idx, dist = {}, {}
		local tn = best_ms(function()
			for i = 1, n do
				T:knn(P[i], 8, idx, dist)
			end
		end)
		print(string.format("| %d | %s | %s | %.2f ms | %.2f ms | %.2f ms | %.2f ms |", n, layout, tb, tg, tp, tk, tn))
	end
end

print("\n## A refit tree as its points drift (2000 uniform points, r for about 12 neighbors)\n")
print("| drift (in r) | refit | radius queries after refit | rebuild | radius queries after rebuild |")
print("| --- | --- | --- | --- | --- |")
do
	local P0 = uniform(2000)
	local r = radius_for(P0, 12)
	local dirs = {}
	for i = 1, #P0 do
		dirs[i] = vector.normalize(vector.create(gauss(), gauss(), gauss()))
	end
	local out = {}
	for _, drift in { 0, 0.25, 1, 4 } do
		local T = spatial.kdtree(P0)
		local P = table.create(#P0) :: { vector }
		for i = 1, #P0 do
			P[i] = P0[i] + dirs[i] * (drift * r)
		end
		local tr = best_ms(function() T:refit(P) end)
		local tq = best_ms(function()
			for i = 1, #P do
				T:radius(P[i], r, out)
			end
		end)
		local U = spatial.kdtree(P0)
		local tb = best_ms(function() U:rebuild(P) end)
		local tq2 = best_ms(function()
			for i = 1, #P do
				U:radius(P[i], r, out)
			end
		end)
		print(string.format("| %g | %.2f ms | %.2f ms | %.2f ms | %.2f ms |", drift, tr, tq, tb, tq2))
	end
end
