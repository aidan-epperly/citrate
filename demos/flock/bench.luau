--!strict
-- The flock and navigation scenes outside Roblox (luau -O2 [--codegen] demos/flock/bench.luau):
-- the time per frame spent finding neighbors by each method, and routes by A* against
-- Dijkstra's search.
local flock = require("./flock")

print("| boids | grid | k-d tree | every pair | neighbors each |")
print("| --- | --- | --- | --- | --- |")
for _, n in { 500, 1000, 2000 } do
	local row = {}
	local mean = 0
	for _, method in { "grid", "kdtree", "brute" } do
		local F = flock.new(n, vector.create(40, 15, 40), 4)
		local total, frames = 0, 0
		for frame = 1, 120 do
			local ms, m = flock.step(F, 1 / 60, method)
			if frame > 60 then
				total += ms
				frames += 1
				mean = m
			end
		end
		table.insert(row, string.format("%.2f ms", total / frames))
	end
	print(string.format("| %d | %s | %s | %s | %.1f |", n, row[1], row[2], row[3], mean))
end

print("\n| waypoints | candidate edges (within reach) | clear edges | A* | expanded | Dijkstra to the target |")
print("| --- | --- | --- | --- | --- | --- |")
for _, npoints in { 300, 1000 } do
	local N = flock.nav(160, 18, npoints, 18)
	local astar, dij, expanded, routes = 0, 0, 0, 0
	for k = 1, 20 do
		local a = N.points[(k * 37) % #N.points + 1]
		local b = N.points[(k * 101 + 13) % #N.points + 1]
		local route, info = flock.route(N, a, b)
		if route then
			astar += info.ms
			dij += info.dijkstra_ms
			expanded += info.expanded
			routes += 1
		end
	end
	print(string.format("| %d | %d | %d | %.3f ms | %.0f | %.3f ms |", #N.points, N.candidates, N.edges, astar / routes, expanded / routes, dij / routes))
end
