--!strict
-- Terrain benchmark (standalone: luau -O2 [--codegen] demos/terrain/bench.luau): energy the
-- marbles lose with no friction (ideally none) and the time per marble step, on the
-- bilinear and cubic surfaces.
local terrain = require("./terrain")

local h = terrain.heights()
local linear, cubic = terrain.surfaces(h)
local M = 200
local start = terrain.marbles(M)

for _, substeps in { 1, 2, 4, 8 } do
	local dt = 1 / 60 / substeps
	for _, side: any in { { "bilinear", linear }, { "cubic", cubic } } do
		local S = side[2]
		local m = terrain.copy(start)
		local e0 = terrain.energy(S, m)
		-- the energy available to lose: down to the lowest grid point
		local low = math.huge
		for i = 1, terrain.N do
			for j = 1, terrain.N do
				low = math.min(low, h[i][j])
			end
		end
		local avail = e0 - M * terrain.GRAVITY * low
		local lost10 = 0
		for frame = 1, 60 * 30 do
			for _ = 1, substeps do
				terrain.step(S, m, dt, 0)
			end
			if frame == 600 then
				lost10 = (e0 - terrain.energy(S, m)) / avail
			end
		end
		local lost30 = (e0 - terrain.energy(S, m)) / avail
		print(string.format("substeps %d, %-8s energy lost: %5.1f%% after 10 s, %5.1f%% after 30 s", substeps, side[1], 100 * lost10, 100 * lost30))
	end
end

for _, side: any in { { "bilinear", linear }, { "cubic", cubic } } do
	local m = terrain.copy(start)
	local t0 = os.clock()
	for _ = 1, 300 do
		terrain.step(side[2], m, 1 / 60, 0)
	end
	print(string.format("%-8s %.2f us per marble step (two surface samples)", side[1], (os.clock() - t0) / (300 * M) * 1e6))
end
