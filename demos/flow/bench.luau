--!strict
-- Flow past a cylinder outside Roblox (luau -O2 --codegen demos/flow/bench.luau): the time
-- per step, the divergence left after the projection, and the vortex shedding, measured
-- by the transverse velocity two diameters behind the cylinder: its Strouhal number
-- St = f D / U and its amplitude relative to U.
local flow = require("./flow")
local NX, NY, U, D, DT = 192, 96, 67.5, 12, 1 / 60
local function run(advection: string, re: number, seconds: number)
	local f = flow.new({ nx = NX, ny = NY, speed = U, reynolds = re, advection = advection })
	f:set_obstacle("cylinder", NX / 4, NY / 2, D)
	-- a small transverse kick behind the cylinder to break the symmetry
	for j = NX / 4 + D, NX / 4 + 2 * D do
		for i = NY / 2, NY / 2 + 8 do
			buffer.writef64(f.v, 8 * (j * NY + i), 0.1 * U)
		end
	end
	local probe = 8 * ((NX / 4 + 2 * D) * NY + NY / 2)
	local samples = {}
	local t0 = os.clock()
	local steps = math.floor(seconds / DT)
	for s = 1, steps do
		f:step(DT)
		samples[s] = buffer.readf64(f.v, probe)
	end
	local ms = (os.clock() - t0) * 1000 / steps
	-- shedding over the second half: upward zero crossings and the amplitude
	local first, last, crossings, amp = 0, 0, 0, 0
	for s = steps // 2 + 1, steps do
		amp = math.max(amp, math.abs(samples[s]))
		if samples[s - 1] < 0 and samples[s] >= 0 then
			crossings += 1
			if first == 0 then first = s end
			last = s
		end
	end
	local freq = if crossings > 1 then (crossings - 1) / ((last - first) * DT) else 0
	local shedding = if amp < 0.01 * U then "no shedding" else string.format("St %.3f, amplitude %.2f U", freq * D / U, amp / U)
	print(string.format("%-15s Re %4d: %5.1f ms/step, max |div| %.0e, %s", advection, re, ms, f:divergence(), shedding))
end
for _, re in { 30, 100, 200 } do
	run("maccormack", re, 24)
end
run("semilagrangian", 30, 24)
run("semilagrangian", 100, 24)
