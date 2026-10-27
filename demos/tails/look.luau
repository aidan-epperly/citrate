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
-- space with the second form as the objective (in Python, with SciPy);
--   luau -O2 --codegen demos/tails/look.luau -a fit [lm|nm|lbfgs]
-- fits it with citrate.optimize instead, by several methods, for comparison.
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

-- Fitting with citrate.optimize: the parameters in log space (rest compliance, rest
-- damping, drag, rest growth, mass taper; rods rigid, 120 Hz substeps), from the starting
-- point the Python fit used, by each method: Levenberg-Marquardt on the per-frame tip
-- differences (its damping by Moré's iteration or AAA), Nelder-Mead on the RMS distance (as
-- the Python fit), and L-BFGS on the squared distance with difference gradients (Wolfe or
-- Chebyshev line search). The simulation is in single precision, so the differences take
-- relative steps of 1e-4.
if args[1] == "fit" then
	local lib = require("../../src")
	local f64, opt = lib.f64, lib.optimize
	local FIT = { MOTIONS[1], MOTIONS[2], MOTIONS[3] }
	local reference: { { { vector } } } = {}
	for k, m in FIT do
		reference[k] = frames(tails.smartbone(60), m)
	end
	local count = 0
	for _, a in reference do
		for f = 1, #a do
			count += #a[f] - 1
		end
	end
	local function options(z: buffer): any
		local e = function(i: number): number
			return math.exp(buffer.readf64(z, 8 * (i - 1)))
		end
		return { substep_rate = 120, rod_compliance = 0, rest_compliance = e(1), rest_damping = e(2), drag = e(3), rest_growth = e(4), mass_taper = e(5) }
	end
	-- residuals: tip differences / sqrt(count), so |r|^2 is the mean squared distance
	local scale = 1 / math.sqrt(count)
	local function residuals(z: buffer, out: buffer?): buffer
		local o = out or buffer.create(8 * 3 * count)
		local opts = options(z)
		local at = 0
		for k, m in FIT do
			local b = frames(tails.xpbd(60, opts), m)
			local a = reference[k]
			for f = 1, #a do
				for i = 2, #a[f] do
					local d = (a[f][i] - b[f][i]) * scale
					buffer.writef64(o, at, d.x)
					buffer.writef64(o, at + 8, d.y)
					buffer.writef64(o, at + 16, d.z)
					at += 24
				end
			end
		end
		return o
	end
	local function mean_square(z: buffer): number
		local r = residuals(z)
		return f64.dot(r, r)
	end
	local z0 = f64.from({ math.log(3.5e-3), math.log(1.0), math.log(2.37), math.log(1.0), math.log(0.44) })
	local function report(name: string, z: buffer, evaluations: number, seconds: number)
		local fit, held = measure(function()
			return tails.xpbd(60, options(z))
		end)
		local o = options(z)
		print(string.format("%-34s %5d evaluations %6.1f s   fit %.4f check %.4f   rest_compliance %.4g rest_damping %.4g drag %.4g rest_growth %.4g mass_taper %.4g",
			name, evaluations, seconds, fit, held, o.rest_compliance, o.rest_damping, o.drag, o.rest_growth, o.mass_taper))
	end
	local which = args[2]
	for _, lam in { "newton", "aaa" } do
		if which == nil or which == "lm" then
			local t0 = os.clock()
			local z, info = opt.least_squares(f64.space, residuals, f64.copy(nil, z0), { lambda = lam, diff_step = 1e-4, ftol = 1e-6, xtol = 1e-6 })
			report(`Levenberg-Marquardt ({lam})`, z, info.evaluations, os.clock() - t0)
		end
	end
	if which == nil or which == "nm" then
		local t0 = os.clock()
		local z, _, info = opt.minimize(f64.space, function(z: buffer): number
			return math.sqrt(mean_square(z))
		end, f64.copy(nil, z0), { method = "nelder-mead", xatol = 1e-3, fatol = 1e-4 })
		report("Nelder-Mead (as the Python fit)", z, info.evaluations, os.clock() - t0)
	end
	for _, ls in { "wolfe", "chebyshev" } do
		if which == nil or which == "lbfgs" then
			local t0 = os.clock()
			local z, _, info = opt.minimize(f64.space, mean_square, f64.copy(nil, z0), { line_search = ls, diff_step = 1e-4, gtol = 1e-7, ftol = 1e-8 })
			report(`L-BFGS, differences ({ls})`, z, info.evaluations, os.clock() - t0)
		end
	end
	local fit, held = measure(function()
		return tails.xpbd(60, tails.SMARTBONE_LOOK)
	end)
	print(string.format("%-34s                         fit %.4f check %.4f", "SMARTBONE_LOOK (the Python fit)", fit, held))
	return
end

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
