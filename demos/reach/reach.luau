--!strict
-- A reaching arm: 5 segments from a base, each joint turning about two axes (yaw about the
-- segment's up axis, then pitch about its right axis), within limits. Two ways to point its
-- hand at a target each frame:
--   citrate: least squares (Levenberg-Marquardt) on the hand's distance from the target
--     plus a weak pull toward a rest pose, warm-started from last frame's angles; the
--     limits are kept smoothly by solving for phi with angle = limit * tanh(phi).
--   CCD (cyclic coordinate descent), the usual approach: each joint axis in turn, from the
--     hand back to the base, turned to swing the hand toward the target, clamped to its
--     limits, for a number of sweeps.
-- Usable outside Roblox (bench.luau).
local lib = require("../../src")
local f64, opt = lib.f64, lib.optimize

local reach = {}

reach.LENGTHS = { 2.4, 2.0, 1.6, 1.2, 0.8 }
local J = #reach.LENGTHS
reach.N = 2 * J -- unknowns: yaw and pitch per joint
-- limits (radians): the base turns freely in yaw; the others less
reach.LIMITS = {}
for j = 1, J do
	reach.LIMITS[2 * j - 1] = if j == 1 then math.pi else math.rad(80)
	reach.LIMITS[2 * j] = math.rad(110)
end
-- the rest pose: a gentle curl forward
reach.REST = {}
for j = 1, J do
	reach.REST[2 * j - 1] = 0
	reach.REST[2 * j] = if j == 1 then 0.3 else 0.35
end
reach.REST_WEIGHT = 0.2

-- rotate v about the unit axis a by angle t (Rodrigues)
local function rotate(v: vector, a: vector, t: number): vector
	local c, s = math.cos(t), math.sin(t)
	return v * c + vector.cross(a, v) * s + a * (vector.dot(a, v) * (1 - c))
end

-- The joint positions (base first, hand last) for angles theta (a table of N), from the
-- base at origin pointing up; also each joint's axes (up, right) after its rotation, for CCD.
function reach.forward(theta: { number }, origin: vector?): ({ vector }, { vector }, { vector })
	local p = origin or vector.zero
	local up, right = vector.create(0, 1, 0), vector.create(1, 0, 0)
	local points, ups, rights = { p }, {}, {}
	for j = 1, J do
		-- yaw about up, then pitch about the (yawed) right
		right = rotate(right, up, theta[2 * j - 1])
		local fwd = vector.cross(right, up)
		up = rotate(up, right, theta[2 * j])
		fwd = rotate(fwd, right, theta[2 * j])
		ups[j], rights[j] = up, right
		p += up * reach.LENGTHS[j]
		points[j + 1] = p
		local _ = fwd
	end
	return points, ups, rights
end

-- theta from phi (smooth limits) and back
function reach.theta_of(phi: buffer): { number }
	local t = {}
	for i = 1, reach.N do
		t[i] = reach.LIMITS[i] * math.tanh(buffer.readf64(phi, 8 * (i - 1)))
	end
	return t
end
function reach.phi_of(theta: { number }): buffer
	local b = buffer.create(8 * reach.N)
	for i = 1, reach.N do
		local u = math.clamp(theta[i] / reach.LIMITS[i], -0.999999, 0.999999)
		buffer.writef64(b, 8 * (i - 1), 0.5 * math.log((1 + u) / (1 - u)))
	end
	return b
end

-- The forward kinematics is in single precision (vectors), so derivatives by differences
-- take relative steps of 1e-4 (the default, sqrt of double precision's roundoff, would
-- difference rounding noise).
local DIFF = 1e-4

-- A solver for the citrate arm: solve(target, phi) improves phi (an f64 array, in place
-- returned) toward reaching target. method: "lm" (Levenberg-Marquardt; lambda by Moré's
-- iteration or "aaa") or "lbfgs" (with a line search: "wolfe" or "chebyshev").
function reach.citrate_solver(method: string, variant: string?): (vector, buffer) -> (buffer, any)
	local target = vector.zero
	local m = 3 + reach.N
	local function residuals(phi: buffer, out: buffer?): buffer
		local o = out or buffer.create(8 * m)
		local th = reach.theta_of(phi)
		local pts = reach.forward(th)
		local d = pts[#pts] - target
		buffer.writef64(o, 0, d.x)
		buffer.writef64(o, 8, d.y)
		buffer.writef64(o, 16, d.z)
		for i = 1, reach.N do
			buffer.writef64(o, 8 * (2 + i), reach.REST_WEIGHT * (th[i] - reach.REST[i]))
		end
		return o
	end
	local r = buffer.create(8 * m)
	local function cost(phi: buffer): number
		r = residuals(phi, r)
		return 0.5 * f64.dot(r, r)
	end
	return function(t: vector, phi: buffer): (buffer, any)
		target = t
		if method == "lm" then
			return opt.least_squares(f64.space, residuals, phi, { lambda = variant or "newton", diff_step = DIFF, ftol = 1e-6, xtol = 1e-6, gtol = 1e-6, maxiter = 20 })
		end
		local x, _, info = opt.minimize(f64.space, cost, phi, { line_search = variant or "wolfe", diff_step = DIFF, gtol = 1e-6, ftol = 1e-9, maxiter = 40 })
		return x, info
	end
end

-- CCD: `sweeps` passes over the joint axes from the hand back to the base, each turned
-- (and clamped) to swing the hand toward the target. theta is changed in place.
function reach.ccd(theta: { number }, target: vector, sweeps: number)
	for _ = 1, sweeps do
		for j = J, 1, -1 do
			for k = 2, 1, -1 do -- pitch, then yaw
				local i = 2 * j - 2 + k
				local pts, ups, rights = reach.forward(theta)
				local pivot = pts[j]
				-- the axis this angle turns about: pitch about joint j's right axis; yaw about
				-- the up axis of the joint before (the base's: world up)
				local axis = if k == 2 then rights[j] else (if j > 1 then ups[j - 1] else vector.create(0, 1, 0))
				local e = pts[#pts] - pivot
				local w = target - pivot
				e -= axis * vector.dot(axis, e)
				w -= axis * vector.dot(axis, w)
				if vector.magnitude(e) > 1e-9 and vector.magnitude(w) > 1e-9 then
					local turn = math.atan2(vector.dot(vector.cross(e, w), axis), vector.dot(e, w))
					-- (pitch turns the opposite way to the right-handed rotation about right)
					theta[i] = math.clamp(theta[i] + turn, -reach.LIMITS[i], reach.LIMITS[i])
				end
			end
		end
	end
end

-- The target's path: a figure moving through and beyond the arm's reach.
function reach.target(t: number): vector
	return vector.create(4.5 * math.sin(0.7 * t), 4 + 2.5 * math.sin(1.1 * t + 0.5), 3.5 + 2.5 * math.cos(0.9 * t))
end

return reach
