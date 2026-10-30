--!strict

-- Rotations: constructors and conversions against each other and against matrices, Euler
-- angles in every order, slerp, the exponential map, integration (exact, and fourth order
-- with the Magnus step), means, point-set fits, splines (through keys, continuity, angle
-- reparameterization) and frames in tables and buffers.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local rot, random = lib.rotation, lib.random

local rng = random.new(5)

local function rand_quat(): any
	return rot.quat(rng:normal(), rng:normal(), rng:normal(), rng:normal())
end

-- the distance between two rotations as unit quaternions, q ~ -q
local function qdist(a: any, b: any): number
	local d = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z
	local s = if d < 0 then -1 else 1
	return math.max(math.abs(a.w - s * b.w), math.abs(a.x - s * b.x), math.abs(a.y - s * b.y), math.abs(a.z - s * b.z))
end

local function matmul(A: { { number } }, B: { { number } }): { { number } }
	local C = {}
	for i = 1, 3 do
		C[i] = {}
		for j = 1, 3 do
			C[i][j] = A[i][1] * B[1][j] + A[i][2] * B[2][j] + A[i][3] * B[3][j]
		end
	end
	return C
end
local function matdist(A: { { number } }, B: { { number } }): number
	local d = 0
	for i = 1, 3 do
		for j = 1, 3 do
			d = math.max(d, math.abs(A[i][j] - B[i][j]))
		end
	end
	return d
end

test("rotation: construction, matrices, composition and Euler angles", function()
	local q = rot.axis_angle(vector.create(0, 1, 0), math.pi / 2)
	local v = rot.apply(q, vector.create(1, 0, 0))
	check(vector.magnitude(v - vector.create(0, 0, -1)) < 1e-7, `quarter turn about y: {v}`)
	-- matrices both ways, including half turns (every branch of Shepperd's method)
	local worst = 0
	for k = 1, 200 do
		local r = if k <= 4 then rot.axis_angle(({ vector.create(1, 0, 0), vector.create(0, 1, 0), vector.create(0, 0, 1), vector.create(1, 1, 1) })[k], math.pi) else rand_quat()
		worst = math.max(worst, qdist(rot.from_matrix(rot.matrix(r)), r))
	end
	check(worst < 1e-15, string.format("matrix round trips: %.1e", worst))
	-- composition is the matrix product, b first; inverse
	local a, b = rand_quat(), rand_quat()
	close(matdist(rot.matrix(rot.compose(a, b)), matmul(rot.matrix(a), rot.matrix(b))), 0, 1e-15, "mul is the matrix product")
	close(qdist(rot.compose(a, rot.inverse(a)), rot.identity()), 0, 1e-15, "inverse")
	-- Euler angles: every order, against the product of axis rotations, and back
	local ex, ey, ez = vector.create(1, 0, 0), vector.create(0, 1, 0), vector.create(0, 0, 1)
	local axis = { X = ex, Y = ey, Z = ez }
	worst = 0
	local back = 0
	for _, order in { "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX" } do
		for _ = 1, 20 do
			local angles = { rng:uniform(-3, 3), rng:uniform(-1.4, 1.4), rng:uniform(-3, 3) }
			-- the middle axis's angle in [-pi/2, pi/2] so the angles come back unchanged
			local byaxis = { X = 0, Y = 0, Z = 0 }
			for p = 1, 3 do
				byaxis[order:sub(p, p)] = angles[p]
			end
			local r = rot.from_euler(byaxis.X, byaxis.Y, byaxis.Z, order)
			local M = matmul(matmul(rot.matrix(rot.axis_angle(axis[order:sub(1, 1)], angles[1])), rot.matrix(rot.axis_angle(axis[order:sub(2, 2)], angles[2]))), rot.matrix(rot.axis_angle(axis[order:sub(3, 3)], angles[3])))
			worst = math.max(worst, matdist(rot.matrix(r), M))
			local rx, ry, rz = rot.to_euler(r, order)
			back = math.max(back, math.abs(rx - byaxis.X), math.abs(ry - byaxis.Y), math.abs(rz - byaxis.Z))
		end
		-- gimbal lock: the rotation comes back, if not the angles
		for _, mid in { math.pi / 2, -math.pi / 2 } do
			local byaxis = { X = 0.3, Y = -0.7, Z = 1.1 }
			byaxis[order:sub(2, 2)] = mid
			local r = rot.from_euler(byaxis.X, byaxis.Y, byaxis.Z, order)
			local rx, ry, rz = rot.to_euler(r, order)
			back = math.max(back, qdist(rot.from_euler(rx, ry, rz, order), r))
		end
	end
	check(worst < 1e-14, string.format("Euler angles: products of axis rotations (%.1e)", worst))
	check(back < 1e-9, string.format("Euler angles back, and the rotation at gimbal lock (%.1e)", back))
	-- between: directions, and opposite ones
	worst = 0
	for _ = 1, 50 do
		local u = vector.create(rng:normal(), rng:normal(), rng:normal())
		local w = vector.create(rng:normal(), rng:normal(), rng:normal())
		for _, target in { w, -u } do
			local r = rot.between(u, target)
			worst = math.max(worst, vector.magnitude(rot.apply(r, vector.normalize(u)) - vector.normalize(target)))
		end
	end
	check(worst < 1e-6, string.format("between (%.1e, single-precision vectors)", worst))
end)

test("rotation: exponential map, slerp, integration", function()
	-- exp and log, at tiny, ordinary and nearly half-turn angles
	local worst = 0
	for _, t in { 1e-12, 1e-6, 0.5, 2, math.pi - 1e-6 } do
		local q = rot.axis_angle(vector.create(1, 2, 2), t)
		local v = rot.log(q)
		close(vector.magnitude(v), t, 1e-7 * t + 1e-15, string.format("log angle at %g", t))
		worst = math.max(worst, qdist(rot.exp(v), q))
	end
	check(worst < 1e-7, string.format("exp(log q) = q (%.1e; vectors are single precision)", worst))
	-- slerp: ends, constant angular speed, the shorter arc
	local a, b = rand_quat(), rand_quat()
	close(qdist(rot.slerp(a, b, 0), a), 0, 1e-15, "slerp at 0")
	close(qdist(rot.slerp(a, b, 1), b), 0, 1e-14, "slerp at 1")
	local total = rot.angle(a, b)
	for _, t in { 0.25, 0.6 } do
		close(rot.angle(a, rot.slerp(a, b, t)), t * total, 1e-13, `constant speed at {t}`)
	end
	local nb = { w = -b.w, x = -b.x, y = -b.y, z = -b.z }
	close(qdist(rot.slerp(a, nb, 0.4), rot.slerp(a, b, 0.4)), 0, 1e-14, "-q is the same rotation")
	-- integration with a constant angular velocity is exact: exp(omega t) q
	local omega = vector.create(0.3, -1.2, 0.8)
	local q = a
	for _ = 1, 100 do
		q = rot.integrate(q, omega, 0.01)
	end
	close(qdist(q, rot.compose(rot.exp(omega * 1), a)), 0, 1e-7, "constant omega: exact")
	-- a linearly changing omega: the Magnus step is fourth order (error / 16 per halving)
	local w0, w1 = vector.create(1, 0, 0.5), vector.create(-0.5, 2, 0)
	local function run(steps: number): any
		local r = a
		for k = 0, steps - 1 do
			local t0, t1 = k / steps, (k + 1) / steps
			r = rot.integrate(r, w0 + (w1 - w0) * t0, 1 / steps, w0 + (w1 - w0) * t1)
		end
		return r
	end
	local ref = run(2048)
	local e1, e2 = qdist(run(8), ref), qdist(run(16), ref)
	check(e1 / e2 > 12 and e1 / e2 < 20, string.format("Magnus: errors %.1e, %.1e (ratio %.1f)", e1, e2, e1 / e2))
	-- angular velocity between two rotations
	local b2 = rot.integrate(a, omega, 0.5)
	check(vector.magnitude(rot.angular_velocity(a, b2, 0.5) - omega) < 1e-6, "angular_velocity")
end)

test("rotation: means and fits", function()
	-- rotations spread symmetrically about a center average to it, whatever their signs
	local c = rand_quat()
	local list = {}
	for _, d in { vector.create(0.3, 0, 0), vector.create(-0.3, 0, 0), vector.create(0, 0.2, 0.1), vector.create(0, -0.2, -0.1) } do
		local r = rot.compose(c, rot.exp(d))
		if #list % 2 == 1 then
			r = { w = -r.w, x = -r.x, y = -r.y, z = -r.z }
		end
		table.insert(list, r)
	end
	check(qdist(rot.mean(list), c) < 1e-3, "mean of a symmetric spread (to second order)")
	close(qdist(rot.mean({ c, c, c }), c), 0, 1e-14, "mean of equal rotations")
	close(qdist(rot.mean({ c, rand_quat() }, { 1, 0 }), c), 0, 1e-14, "weights")
	-- fit: a random rigid motion recovered from 20 points
	local R = rand_quat()
	local p = vector.create(3, -1, 2)
	local A, B = {}, {}
	for i = 1, 20 do
		local x = { rng:normal(), rng:normal(), rng:normal() }
		local w = rot.apply(R, x) + p
		A[i], B[i] = x, { w.x, w.y, w.z }
	end
	local F, rms = rot.fit(A, B)
	check(qdist(F, R) < 1e-6 and math.abs(F.px - 3) < 1e-6 and rms < 1e-6, string.format("fit: rotation %.1e, rms %.1e", qdist(F, R), rms))
end)

test("rotation: splines", function()
	local keys = { rand_quat(), rand_quat(), rand_quat(), rand_quat(), rand_quat() }
	local times = { 0, 1, 1.5, 3, 4 }
	for _, kind in { "smooth", "slerp" } do
		local S = rot.spline(keys, { kind = kind, times = times })
		local worst = 0
		for k = 1, 5 do
			worst = math.max(worst, qdist(S:at(times[k]), keys[k]))
		end
		check(worst < 1e-12, string.format("%s: through the keys (%.1e)", kind, worst))
		-- angular velocity across interior knots: continuous for smooth, not for slerp
		local jump = 0
		for k = 2, 4 do
			jump = math.max(jump, vector.magnitude(S:angular_velocity(times[k] - 1e-7) - S:angular_velocity(times[k] + 1e-7)))
		end
		if kind == "smooth" then
			check(jump < 1e-3, string.format("smooth: angular velocity continuous (%.1e)", jump))
		else
			check(jump > 0.1, "slerp: angular velocity jumps at keys")
		end
		-- by angle: constant angular speed, and the total
		local da = 1e-3
		local spread = 0
		for _, s in { 0.1, 0.4, 0.8 } do
			local a = s * S.angle
			spread = math.max(spread, math.abs(rot.angle(S:at_angle(a), S:at_angle(a + da)) - da))
		end
		check(spread < 1e-7, string.format("%s: constant angular speed by angle (%.1e)", kind, spread))
		close(qdist(S:at_angle(S.angle), keys[5]), 0, 1e-9, `{kind}: at_angle(total) is the last key`)
	end
	-- slerp's total angle is the sum of the steps
	local S = rot.spline(keys, { kind = "slerp" })
	local sum = 0
	for k = 1, 4 do
		sum += rot.angle(keys[k], keys[k + 1])
	end
	close(S.angle, sum, 1e-9, "slerp: total angle")
	-- B-spline: angular velocity continuous, and smooth enough that its change is too
	local Bs = rot.spline(keys, { kind = "bspline" })
	local jump = 0
	for k = 2, 4 do
		local t = k - 1
		jump = math.max(jump, vector.magnitude(Bs:angular_velocity(t - 1e-6) - Bs:angular_velocity(t + 1e-6)))
	end
	check(jump < 1e-3, string.format("bspline: angular velocity continuous (%.1e)", jump))
	-- closed: back to the first key, smoothly
	local C = rot.spline(keys, { closed = true })
	local a, b = C:domain()
	close(qdist(C:at(b), keys[1]), 0, 1e-12, "closed: returns to the first key")
	check(vector.magnitude(C:angular_velocity(a + 1e-7) - C:angular_velocity(b - 1e-7)) < 1e-3, "closed: smooth at the seam")
	check(not pcall(function()
		rot.spline(keys, { kind = "bspline", times = times })
	end), "bspline needs even times")
end)

test("rotation: frames, buffers, and errors", function()
	local f = rot.place(rot.axis_angle(vector.create(0, 0, 1), math.pi / 2), vector.create(1, 2, 3))
	local p = rot.apply(f, vector.create(1, 0, 0))
	check(vector.magnitude(p - vector.create(1, 3, 3)) < 1e-6, `frame applied: {p}`)
	local g = rot.place(rand_quat(), vector.create(-1, 0, 2))
	local fg = rot.compose(f, g)
	local q = vector.create(0.3, -0.2, 0.5)
	check(vector.magnitude(rot.apply(fg, q) - rot.apply(f, rot.apply(g, q))) < 1e-5, "frames compose")
	local back = rot.apply(rot.inverse(f), rot.apply(f, q))
	check(vector.magnitude(back - q) < 1e-6, "frame inverse")
	-- buffers of 7 numbers a frame
	local B = rot.frames(3)
	rot.set(B, 1, f)
	rot.set(B, 2, g)
	rot.compose_into(B, 3, B, 1, B, 2)
	local h3 = rot.get(B, 3)
	close(qdist(h3, fg) + math.abs(h3.px - fg.px) + math.abs(h3.pz - fg.pz), 0, 1e-14, "compose_into matches mul")
	close(qdist(rot.get(rot.frames(1), 1), rot.identity()), 0, 0, "new frames are the identity")
	check(not pcall(rot.axis_angle, vector.zero, 1), "zero axis")
	check(not pcall(rot.identity, "nope"), "unknown kind")
	check(not pcall(rot.compose, { 1, 2 }, rot.identity()), "not a rotation")
	-- a space of one's own: rotation matrices as rows, from the primitives only
	local function mm(A: any, B: any): any
		return matmul(A, B)
	end
	local matrices: any = {
		name = "matrices",
		eps = 2 ^ -53,
		is = function(v: any): boolean
			return type(v) == "table" and type(v[1]) == "table"
		end,
		identity = function(): any
			return { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
		end,
		compose = mm,
		inverse = function(A: any): any
			return { { A[1][1], A[2][1], A[3][1] }, { A[1][2], A[2][2], A[3][2] }, { A[1][3], A[2][3], A[3][3] } }
		end,
		apply = function(A: any, x: number, y: number, z: number): (number, number, number)
			return A[1][1] * x + A[1][2] * y + A[1][3] * z, A[2][1] * x + A[2][2] * y + A[2][3] * z, A[3][1] * x + A[3][2] * y + A[3][3] * z
		end,
		exp = function(x: number, y: number, z: number): any
			return rot.matrix(rot.exp(vector.create(x, y, z)))
		end,
		log = function(A: any): (number, number, number)
			local v = rot.log(rot.from_matrix(A))
			return v.x, v.y, v.z
		end,
	}
	local qs = { rand_quat(), rand_quat(), rand_quat() }
	local ms = { rot.matrix(qs[1]), rot.matrix(qs[2]), rot.matrix(qs[3]) }
	local Sq = rot.spline(qs)
	local Sm = rot.spline(ms, { space = matrices })
	check(matdist(Sm:at(1.3), rot.matrix(Sq:at(1.3))) < 1e-6, "a spline in a space of one's own")
	-- CFrames: made in Roblox, refused elsewhere with a clear message
	local ok, cf = pcall(rot.identity, "cframe")
	local kind: string = typeof(cf)
	check(if ok then kind == "CFrame" else string.find(tostring(cf), "Roblox") ~= nil, "CFrames only in Roblox")
end)

-- In Roblox: the same operations on CFrames, by CFrame methods, agree with quaternions to
-- the CFrames' single precision.
if rot.cframe then
	test("rotation: CFrames", function()
		local C = rot.cframe
		local qs = { rand_quat(), rand_quat(), rand_quat(), rand_quat() }
		local cs = {}
		for k, q in qs do
			cs[k] = rot.convert(q, C)
		end
		local function cdist(c: any, q: any): number
			return qdist(rot.convert(c, rot.quaternion), q)
		end
		check(cdist(rot.compose(cs[1], cs[2]), rot.compose(qs[1], qs[2])) < 1e-6, "compose")
		check(cdist(rot.slerp(cs[1], cs[2], 0.3), rot.slerp(qs[1], qs[2], 0.3)) < 1e-6, "slerp")
		check(cdist(rot.from_euler(0.3, -0.7, 1.1, "YXZ", C), rot.from_euler(0.3, -0.7, 1.1, "YXZ")) < 1e-6, "from_euler")
		for _, kind in { "smooth", "slerp", "bspline" } do
			local Sq, Sc = rot.spline(qs, { kind = kind }), rot.spline(cs, { kind = kind })
			local worst = 0
			for _, t in { 0.3, 1.5, 2.9 } do
				worst = math.max(worst, cdist(Sc:at(t), Sq:at(t)))
			end
			check(worst < 1e-5 and math.abs(Sc.angle - Sq.angle) < 1e-4, string.format("%s spline of CFrames (%.1e, angle %.1e)", kind, worst, Sc.angle - Sq.angle))
		end
		check(cdist(rot.mean(cs), rot.mean(qs)) < 1e-5, "mean of CFrames")
		local w = vector.create(0.3, -1, 0.5)
		check(cdist(rot.integrate(cs[1], w, 0.1), rot.integrate(qs[1], w, 0.1)) < 1e-6, "integrate a CFrame")
	end)
end

return nil
