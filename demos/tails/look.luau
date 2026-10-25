--!strict
-- How closely a tail follows SmartBone's 60 Hz tail: the RMS distance between corresponding
-- bone tips (studs, on a 9-stud tail), frame by frame at 60 Hz, under three root motions
-- (the fitting set) and a fourth (held out).
--
--   luau -O2 --codegen demos/tails/look.luau      the table in README.md
--   luau ... look.luau -a <rest_compliance> <rest_damping> <drag> <rod_compliance>
--       <rest_growth> <substep_rate> <rate> <mass_taper>
--                                                 one XPBD tail: "fit <error> check <error>"
--
-- tails.SMARTBONE_LOOK was fitted by minimizing the fitting set's error by Nelder-Mead in log
-- space with the second form as the objective.
local tails = require("./tails")

local function envelope(t: number, len: number): number
	return if t < len then math.sin(math.pi * t / len) else 0
end
local MOTIONS: { (number) -> vector } = {
	function(t) -- a vertical shake
		return vector.create(0, 0.8 * math.sin(2 * math.pi * 1.5 * t) * envelope(t, 4 / 3), 0)
	end,
	function(t) -- a sideways shake
		return vector.create(0, 0, 1.0 * math.sin(2 * math.pi * 1.0 * t) * envelope(t, 2))
	end,
	function(t) -- a swing along an arc that stops
		local u = math.min(t, 2.5)
		return vector.create(3 * math.sin(1.6 * u), 0.5 * math.sin(3 * u), 3 * (1 - math.cos(1.6 * u)))
	end,
	function(t) -- held out: a wobble in all three axes
		return vector.create(0.5 * math.sin(5.1 * t), 0.6 * math.sin(7.3 * t + 1), 0.7 * math.sin(3.7 * t + 2)) * envelope(t, 3)
	end,
}
local T = 5

local function frames(tail: any, motion: (number) -> vector): { { vector } }
	local out = {}
	for frame = 1, T * 60 do
		while tail.t < frame / 60 - 1e-9 do
			tail:step(motion)
		end
		out[frame] = tail:nodes()
	end
	return out
end

-- (RMS over the fitting motions, RMS on the held-out one)
local function measure(make: () -> any): (number, number)
	local s, c = 0, 0
	local held = 0
	for k, m in MOTIONS do
		local a, b = frames(tails.smartbone(60), m), frames(make(), m)
		local sk, ck = 0, 0
		for f = 1, #a do
			for i = 2, #a[f] do
				local d = a[f][i] - b[f][i]
				sk += d.x * d.x + d.y * d.y + d.z * d.z
				ck += 1
			end
		end
		if k < #MOTIONS then
			s += sk
			c += ck
		else
			held = math.sqrt(sk / ck)
		end
	end
	return math.sqrt(s / c), held
end

local args = { ... }
if #args > 0 then
	local opts = {
		rest_compliance = tonumber(args[1]),
		rest_damping = tonumber(args[2]),
		drag = tonumber(args[3]),
		rod_compliance = tonumber(args[4]),
		rest_growth = tonumber(args[5]),
		substep_rate = tonumber(args[6]),
		mass_taper = tonumber(args[8]),
	}
	local rate = tonumber(args[7]) or 60
	local fit, held = measure(function()
		return tails.xpbd(rate, opts :: any)
	end)
	print(string.format("fit %.6f check %.6f", fit, held))
	return
end

print("distance from SmartBone's 60 Hz tail (RMS over the bone tips, studs): fitting motions, held-out motion")
for _, rate in { 30, 15 } do
	local fit, held = measure(function()
		return tails.smartbone(rate)
	end)
	print(string.format("SmartBone rule, %d Hz          %.2f   %.2f", rate, fit, held))
end
for _, rate in { 60, 30, 15 } do
	local fit, held = measure(function()
		return tails.xpbd(rate, tails.SMARTBONE_LOOK)
	end)
	print(string.format("XPBD, SMARTBONE_LOOK, %d Hz    %.2f   %.2f", rate, fit, held))
end
for _, rate in { 60, 15 } do
	local fit, held = measure(function()
		return tails.xpbd(rate)
	end)
	print(string.format("XPBD, physical options, %d Hz  %.2f   %.2f", rate, fit, held))
end
