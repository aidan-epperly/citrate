--!strict

-- Curve costs on a 50-segment Catmull-Rom path, with vector points and with f64 buffers.
-- Run with codegen for Roblox-like numbers: luau -O2 --codegen bench/curve.luau

local lib = require("../src")
local curve, f64 = lib.curve, lib.f64

local function time(label: string, reps: number, body: () -> ())
	local t0 = os.clock()
	body()
	print(string.format("%-28s %8.3f us", label, (os.clock() - t0) / reps * 1e6))
end

math.randomseed(1)
local pts: { vector } = {}
for i = 1, 51 do
	pts[i] = vector.create(4 * i, 10 * math.random(), 10 * math.random())
end
local bufs: { buffer } = {}
for i, p in pts do
	bufs[i] = f64.from({ p.x, p.y, p.z })
end

for _, case in { { "vector", curve.catmull_rom(pts) :: any }, { "f64 buffer", curve.catmull_rom(bufs, { space = f64.space }) :: any } } do
	local name, C = case[1], case[2]
	print(name)
	local a, b = curve.domain(C)
	local L = 0
	time("arclength build (once)", 1, function()
		L = curve.length(C)
	end)
	time("eval", 1e5, function()
		for i = 1, 1e5 do
			curve.eval(C, a + (b - a) * (i % 997) / 997)
		end
	end)
	time("at_length", 1e5, function()
		for i = 1, 1e5 do
			curve.at_length(C, L * (i % 997) / 997)
		end
	end)
	time("tangent", 1e5, function()
		for i = 1, 1e5 do
			curve.tangent(C, a + (b - a) * (i % 997) / 997)
		end
	end)
	local q = if name == "vector" then vector.create(100, 5, 5) :: any else f64.from({ 100, 5, 5 })
	time("closest", 1e3, function()
		for _ = 1, 1e3 do
			curve.closest(C, q)
		end
	end)
	-- Tracking: q moves along the curve (0.2 units per step), starting from the last answer.
	local t = a
	time("closest_near (tracking)", 1e4, function()
		for i = 1, 1e4 do
			local s = (L * i / 1e4) % L
			local target = curve.at_length(C, s)
			local qi = if name == "vector" then (target :: any) + vector.create(0, 0.3, 0) else f64.add(nil, target, f64.from({ 0, 0.3, 0 }))
			t = curve.closest_near(C, qi, t)
		end
	end)
end
