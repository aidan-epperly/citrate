--!strict
-- The tails outside Roblox (luau -O2 [--codegen] demos/tails/bench.luau): for each method
-- and rate, the tip's droop at rest, how long the tail keeps moving after one shake, and
-- the time per step.
local tails = require("./tails")

local SHAKE = 4 / 3
local motion = tails.motion(0.8, SHAKE, 1e9) -- one shake, then rest
local T = 8

local function run(make: (number) -> any, rate: number): (number, number, number)
	local tail = make(rate)
	local tips: { number } = {}
	local t0 = os.clock()
	local steps = 0
	while tail.t < T - 1e-9 do
		tail:step(motion)
		steps += 1
		local k = math.round(tail.t * 15)
		if math.abs(tail.t * 15 - k) < 1e-6 then
			local nodes = tail:nodes()
			tips[k] = nodes[#nodes].y
		end
	end
	local us = (os.clock() - t0) / steps * 1e6
	-- the last time after the shake that the tip moves more than 0.01 studs in 1/15 s
	local last = SHAKE
	for k = math.ceil(SHAKE * 15) + 1, T * 15 do
		if math.abs(tips[k] - tips[k - 1]) > 0.01 then
			last = k / 15
		end
	end
	return tips[T * 15], last - SHAKE, us
end

print("method          rate    tip at rest   moving after the shake   time per update")
local methods: { { name: string, make: (number) -> any } } = {
	{ name = "SmartBone rule", make = tails.smartbone },
	{ name = "implicit", make = tails.physical },
	{ name = "XPBD 240 Hz", make = function(rate)
		return tails.xpbd(rate)
	end },
	{ name = "XPBD look", make = function(rate)
		return tails.xpbd(rate, tails.SMARTBONE_LOOK)
	end },
	{ name = "XPBD 2/update", make = function(rate)
		return tails.xpbd(rate, { substeps = 2 })
	end },
}
for _, m in methods do
	for _, rate in { 60, 30, 15 } do
		local rest, moving, us = run(m.make, rate)
		print(string.format("%-14s %3d Hz   %7.2f studs   %5.2f s                  %7.1f us", m.name, rate, rest, moving, us))
	end
end
