--!strict
-- The reaching arm outside Roblox (luau -O2 [--codegen] demos/reach/bench.luau): 10
-- seconds of the target's path at 60 Hz, each frame solved by each method from the last
-- frame's angles. Reported: time per frame, the hand's distance from the target (mean, and
-- the largest while the target is within reach), the largest change of any angle in a
-- frame after the first (jumps show as twitching), and the distance from the rest pose.
local reach = require("./reach")

local FRAMES = 600
local REACH = 0
for _, l in reach.LENGTHS do
	REACH += l
end

local function run(name: string, step: (number, vector) -> { number })
	local t0 = os.clock()
	local sum, worst, jump, rest = 0, 0, 0, 0
	local prev: { number }? = nil
	local reachable = 0
	for f = 1, FRAMES do
		local target = reach.target(f / 60)
		local theta = step(f, target)
		local pts = reach.forward(theta)
		local d = vector.magnitude(pts[#pts] - target)
		sum += d
		if vector.magnitude(target) < 0.95 * REACH then
			worst = math.max(worst, d)
			reachable += 1
		end
		local r = 0
		for i = 1, reach.N do
			r += (theta[i] - reach.REST[i]) ^ 2
			if prev and f > 2 then
				jump = math.max(jump, math.abs(theta[i] - (prev :: { number })[i]))
			end
		end
		rest += math.sqrt(r)
		prev = table.clone(theta)
	end
	local ms = (os.clock() - t0) / FRAMES * 1000
	print(string.format("%-34s %7.3f ms/frame   distance mean %.4f, largest in reach %.4f   largest angle jump %.3f rad   rest distance %.2f",
		name, ms, sum / FRAMES, worst, jump, rest / FRAMES))
	local _ = reachable
end

for _, v in { { "lm", "newton" }, { "lm", "aaa" }, { "lbfgs", "wolfe" }, { "lbfgs", "chebyshev" } } do
	local solve = reach.citrate_solver(v[1], v[2])
	local phi = reach.phi_of(reach.REST)
	run(`citrate {v[1]} ({v[2]})`, function(_f: number, target: vector): { number }
		phi = solve(target, phi)
		return reach.theta_of(phi)
	end)
end
for _, sweeps in { 1, 4, 16 } do
	local theta = table.clone(reach.REST)
	run(`CCD, {sweeps} sweeps a frame`, function(_f: number, target: vector): { number }
		reach.ccd(theta, target, sweeps)
		return theta
	end)
end
