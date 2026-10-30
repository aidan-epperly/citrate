--!strict
--!native
--!optimize 2

-- Rotations and rigid frames: unit quaternions, with every function also accepting Roblox
-- CFrames and frames (a position plus a rotation).
--
--   local q = rotation.axis_angle(vector.create(0, 1, 0), math.pi / 2)   -- { w, x, y, z }
--   local v = rotation.apply(q, vector.create(1, 0, 0))                   -- (0, 0, -1)
--   local r = rotation.slerp(q, rotation.identity(), 0.25)
--   local S = rotation.spline({ q1, q2, q3, q4 }, { times = { 0, 1, 3, 4 } })
--   S:at(2.5); S:angular_velocity(2.5); S:at_angle(1.2)
--   local mean = rotation.mean({ q1, q2, q3 })
--   q = rotation.integrate(q, omega, dt)                                  -- dq/dt = omega q / 2
--
-- Values: a quaternion is a table { w, x, y, z } (unit length; q and -q are the same
-- rotation); a frame is { px, py, pz, w, x, y, z }, the position and the rotation; in Roblox a
-- CFrame is both. Functions take any of them and return the kind they were given (the first
-- argument's); constructors take an optional kind ("quat", the default, "frame" or
-- "cframe"). For many frames, a buffer of 7 numbers a frame (px, py, pz, w, x, y, z) avoids
-- a table each: rotation.frames(n), get, set and compose_into. Arithmetic is in double
-- precision on numbers, never on `vector`s (single precision); vectors in and out
-- (rotation vectors, angular velocities, points) are rounded once.
--
-- Conventions: q rotates v as q v q*, composition mul(a, b) applies b first (as a * b for
-- CFrames), angular velocities are in world coordinates (dq/dt = omega q / 2, as Roblox's
-- AssemblyAngularVelocity), and log(q) is the rotation vector (angle times axis, angle in
-- [0, pi]). Euler angles follow CFrame.fromEulerAngles: order "XYZ" (the default) is
-- R = Rx(rx) Ry(ry) Rz(rz).

local types = require("./types")
local f64 = require("./f64")
local cheb = require("./cheb")
local host = require("./host")

type Matrix = types.Matrix
export type Quat = { w: number, x: number, y: number, z: number }
export type Frame = { px: number, py: number, pz: number, w: number, x: number, y: number, z: number }

local function fail(who: string, message: string): never
	error("rotation." .. who .. ": " .. message, 3)
end

-- Quaternion arithmetic on numbers (w, x, y, z)

local function mul(aw: number, ax: number, ay: number, az: number, bw: number, bx: number, by: number, bz: number): (number, number, number, number)
	return aw * bw - ax * bx - ay * by - az * bz,
		aw * bx + ax * bw + ay * bz - az * by,
		aw * by - ax * bz + ay * bw + az * bx,
		aw * bz + ax * by - ay * bx + az * bw
end

-- q p q* for a point p: p + w t + u x t with t = 2 u x p (u = (x, y, z))
local function rotate(w: number, x: number, y: number, z: number, px: number, py: number, pz: number): (number, number, number)
	local tx, ty, tz = 2 * (y * pz - z * py), 2 * (z * px - x * pz), 2 * (x * py - y * px)
	return px + w * tx + (y * tz - z * ty), py + w * ty + (z * tx - x * tz), pz + w * tz + (x * ty - y * tx)
end

local function normalize(w: number, x: number, y: number, z: number): (number, number, number, number)
	local n = math.sqrt(w * w + x * x + y * y + z * z)
	if not (n > 0 and n < math.huge) then
		error("rotation: a quaternion of length zero or not finite", 3)
	end
	return w / n, x / n, y / n, z / n
end

-- The rotation by angle |v| about v.
local function qexp(vx: number, vy: number, vz: number): (number, number, number, number)
	local t = math.sqrt(vx * vx + vy * vy + vz * vz)
	-- sin(t / 2) / t, by its series near 0
	local s = if t < 1e-4 then 0.5 - t * t / 48 else math.sin(t / 2) / t
	return math.cos(t / 2), s * vx, s * vy, s * vz
end

-- The rotation vector of a unit quaternion (angle in [0, pi]).
local function qlog(w: number, x: number, y: number, z: number): (number, number, number)
	if w < 0 then
		w, x, y, z = -w, -x, -y, -z
	end
	local s = math.sqrt(x * x + y * y + z * z)
	-- angle / s = 2 atan2(s, w) / s, near s = 0 by its series 2 / w (1 - s^2 / (3 w^2))
	local f = if s < 1e-5 then 2 / w * (1 - s * s / (3 * w * w)) else 2 * math.atan2(s, w) / s
	return f * x, f * y, f * z
end

-- The point along the shorter arc from a to b at fraction t, at constant angular speed.
local function slerp(aw: number, ax: number, ay: number, az: number, bw: number, bx: number, by: number, bz: number, t: number): (number, number, number, number)
	-- d = a* b
	local dw, dx, dy, dz = mul(aw, -ax, -ay, -az, bw, bx, by, bz)
	local vx, vy, vz = qlog(dw, dx, dy, dz)
	local ew, ex, ey, ez = qexp(t * vx, t * vy, t * vz)
	return mul(aw, ax, ay, az, ew, ex, ey, ez)
end

-- Rows of the rotation matrix.
local function to_matrix(w: number, x: number, y: number, z: number): (number, number, number, number, number, number, number, number, number)
	return 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
		2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
		2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
end

-- Shepperd's method: from the largest of 1 + trace and the diagonal, for accuracy at every
-- angle.
local function from_matrix(r00: number, r01: number, r02: number, r10: number, r11: number, r12: number, r20: number, r21: number, r22: number): (number, number, number, number)
	local tr = r00 + r11 + r22
	local w, x, y, z
	if tr >= r00 and tr >= r11 and tr >= r22 then
		local s = 2 * math.sqrt(math.max(1 + tr, 0))
		w, x, y, z = s / 4, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s
	elseif r00 >= r11 and r00 >= r22 then
		local s = 2 * math.sqrt(math.max(1 + r00 - r11 - r22, 0))
		w, x, y, z = (r21 - r12) / s, s / 4, (r01 + r10) / s, (r02 + r20) / s
	elseif r11 >= r22 then
		local s = 2 * math.sqrt(math.max(1 + r11 - r00 - r22, 0))
		w, x, y, z = (r02 - r20) / s, (r01 + r10) / s, s / 4, (r12 + r21) / s
	else
		local s = 2 * math.sqrt(math.max(1 + r22 - r00 - r11, 0))
		w, x, y, z = (r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, s / 4
	end
	return normalize(w, x, y, z)
end

-- Values: quaternions, frames and CFrames

-- The rotation and position of a value, and its kind.
local function load(r: any, who: string): (number, number, number, number, number, number, number, string)
	if type(r) == "table" then
		local w, x, y, z = r.w, r.x, r.y, r.z
		if type(w) ~= "number" or type(x) ~= "number" or type(y) ~= "number" or type(z) ~= "number" then
			fail(who, "expected a quaternion { w, x, y, z }, a frame or a CFrame")
		end
		if r.px ~= nil then
			return w, x, y, z, r.px, r.py, r.pz, "frame"
		end
		return w, x, y, z, 0, 0, 0, "quat"
	end
	local tr: string = typeof(r)
	if tr == "CFrame" then
		local px, py, pz, r00, r01, r02, r10, r11, r12, r20, r21, r22 = (r :: any):GetComponents()
		local w, x, y, z = from_matrix(r00, r01, r02, r10, r11, r12, r20, r21, r22)
		return w, x, y, z, px, py, pz, "cframe"
	end
	error(`rotation.{who}: expected a quaternion \{ w, x, y, z }, a frame or a CFrame`, 3)
end

local function store(kind: string, w: number, x: number, y: number, z: number, px: number, py: number, pz: number): any
	if kind == "quat" then
		return { w = w, x = x, y = y, z = z }
	elseif kind == "frame" then
		return { px = px, py = py, pz = pz, w = w, x = x, y = y, z = z }
	elseif kind == "cframe" then
		local CF = host.CFrame
		if CF == nil then
			error("rotation: CFrames exist only in Roblox", 3)
		end
		return CF.new(px, py, pz, x, y, z, w)
	end
	error(`rotation: kind must be "quat", "frame" or "cframe" (got {kind})`, 3)
end

local function vec3(v: any, who: string): (number, number, number)
	if type(v) == "vector" then
		return v.x, v.y, v.z
	end
	local tv: string = typeof(v)
	if tv == "Vector3" then
		return v.X, v.Y, v.Z
	elseif type(v) == "table" and type(v[1]) == "number" then
		return v[1], v[2] or 0, v[3] or 0
	end
	error(`rotation.{who}: expected a vector (or \{ x, y, z })`, 3)
end

-- Constructors

-- A unit quaternion from components (normalized).
local function quat(w: number, x: number, y: number, z: number): Quat
	local a, b, c, d = normalize(w, x, y, z)
	return { w = a, x = b, y = c, z = d }
end

local function identity(kind: string?): any
	return store(kind or "quat", 1, 0, 0, 0, 0, 0, 0)
end

-- A frame from a position and a rotation (any kind; its own position is replaced).
local function frame(position: any, r: any, kind: string?): any
	local px, py, pz = vec3(position, "frame")
	local w, x, y, z = load(r, "frame")
	return store(kind or "frame", w, x, y, z, px, py, pz)
end

-- The rotation by angle about axis (any nonzero length).
local function axis_angle(axis: any, angle: number, kind: string?): any
	local ax, ay, az = vec3(axis, "axis_angle")
	local n = math.sqrt(ax * ax + ay * ay + az * az)
	if not (n > 0) then
		fail("axis_angle", "the axis is zero")
	end
	local w, x, y, z = qexp(ax / n * angle, ay / n * angle, az / n * angle)
	return store(kind or "quat", w, x, y, z, 0, 0, 0)
end

-- The rotation by the rotation vector v (angle |v| about v): the inverse of log.
local function exp(v: any, kind: string?): any
	local vx, vy, vz = vec3(v, "exp")
	local w, x, y, z = qexp(vx, vy, vz)
	return store(kind or "quat", w, x, y, z, 0, 0, 0)
end

-- The shortest rotation taking direction a to direction b (about a x b; for opposite
-- directions, a half turn about some axis perpendicular to a).
local function between(a: any, b: any, kind: string?): any
	local ax, ay, az = vec3(a, "between")
	local bx, by, bz = vec3(b, "between")
	local na, nb = math.sqrt(ax * ax + ay * ay + az * az), math.sqrt(bx * bx + by * by + bz * bz)
	if not (na > 0 and nb > 0) then
		fail("between", "a direction is zero")
	end
	ax, ay, az, bx, by, bz = ax / na, ay / na, az / na, bx / nb, by / nb, bz / nb
	-- (1 + a.b, a x b), normalized: the half-way construction, accurate except near a = -b
	local d = ax * bx + ay * by + az * bz
	local w, x, y, z
	if d < -1 + 1e-12 then
		-- perpendicular axis: a x e for the coordinate axis e least aligned with a
		local ex, ey, ez = 0, 0, 0
		if math.abs(ax) <= math.abs(ay) and math.abs(ax) <= math.abs(az) then
			ex = 1
		elseif math.abs(ay) <= math.abs(az) then
			ey = 1
		else
			ez = 1
		end
		w, x, y, z = 0, ay * ez - az * ey, az * ex - ax * ez, ax * ey - ay * ex
	else
		w, x, y, z = 1 + d, ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
	end
	w, x, y, z = normalize(w, x, y, z)
	return store(kind or "quat", w, x, y, z, 0, 0, 0)
end

-- From a rotation matrix: rows { { r00, r01, r02 }, ... } or a 3 by 3 f64 Matrix.
local function from_rotation_matrix(R: any, kind: string?): any
	local function at(i: number, j: number): number
		if type(R) == "table" and R.data ~= nil then
			return f64.matget(R, i, j)
		end
		return R[i][j]
	end
	local w, x, y, z = from_matrix(at(1, 1), at(1, 2), at(1, 3), at(2, 1), at(2, 2), at(2, 3), at(3, 1), at(3, 2), at(3, 3))
	return store(kind or "quat", w, x, y, z, 0, 0, 0)
end

-- The rotation matrix, as rows.
local function matrix(r: any): { { number } }
	local w, x, y, z = load(r, "matrix")
	local a, b, c, d, e, f, g, h, i = to_matrix(w, x, y, z)
	return { { a, b, c }, { d, e, f }, { g, h, i } }
end

local AXES: { [string]: number } = { X = 1, Y = 2, Z = 3 }

-- The order's three axes (a permutation of X, Y, Z).
local function axes_of(order: string, who: string): (number, number, number)
	local i, j, k = AXES[order:sub(1, 1)], AXES[order:sub(2, 2)], AXES[order:sub(3, 3)]
	if #order ~= 3 or not i or not j or not k or i == j or j == k or i == k then
		fail(who, "order must be three different axes, such as \"XYZ\" or \"YXZ\"")
	end
	return i, j, k
end

-- From Euler angles about X, Y and Z (radians), applied as the order says: "XYZ" (the
-- default) is Rx(rx) Ry(ry) Rz(rz), as CFrame.fromEulerAngles(rx, ry, rz, order).
local function from_euler(rx: number, ry: number, rz: number, order: string?, kind: string?): any
	local i, j, k = axes_of(order or "XYZ", "from_euler")
	local angles = { rx, ry, rz }
	local function about(a: number): (number, number, number, number)
		local h = angles[a] / 2
		local c, s = math.cos(h), math.sin(h)
		return c, if a == 1 then s else 0, if a == 2 then s else 0, if a == 3 then s else 0
	end
	local aw, ax, ay, az = about(i)
	local bw, bx, by, bz = about(j)
	local cw, cx, cy, cz = about(k)
	local w, x, y, z = mul(aw, ax, ay, az, bw, bx, by, bz)
	w, x, y, z = mul(w, x, y, z, cw, cx, cy, cz)
	return store(kind or "quat", w, x, y, z, 0, 0, 0)
end

-- The Euler angles (rx, ry, rz) of a rotation for the order (see from_euler); the middle
-- angle is in [-pi/2, pi/2], and at its ends (gimbal lock) the last is taken as 0.
local function to_euler(r: any, order: string?): (number, number, number)
	local i, j, k = axes_of(order or "XYZ", "to_euler")
	local w, x, y, z = load(r, "to_euler")
	local m = { to_matrix(w, x, y, z) }
	local function R(a: number, b: number): number
		return m[3 * (a - 1) + b]
	end
	-- +1 for the cyclic orders (XYZ, YZX, ZXY), -1 for the others
	local s = if (j - i) % 3 == 1 then 1 else -1
	local sb = math.clamp(s * R(i, k), -1, 1)
	local cb = math.sqrt(R(i, i) ^ 2 + R(i, j) ^ 2)
	local a, c
	-- the middle angle by atan2, accurate near +-pi/2 where asin is not
	local b = math.atan2(sb, cb)
	if cb > 1e-12 then
		a = math.atan2(-s * R(j, k), R(k, k))
		c = math.atan2(-s * R(i, j), R(i, i))
	else
		-- gimbal lock: R = R_i(a) R_j(+-pi/2), so a from the j-th row
		a = math.atan2(sb * R(j, i), R(j, j))
		c = 0
	end
	local out = { 0, 0, 0 }
	out[i], out[j], out[k] = a, b, c
	return out[1], out[2], out[3]
end

-- Operations

-- a b: b, then a (as a * b for CFrames; for frames, positions compose too).
local function compose(a: any, b: any): any
	local aw, ax, ay, az, apx, apy, apz, kind = load(a, "mul")
	local bw, bx, by, bz, bpx, bpy, bpz = load(b, "mul")
	local w, x, y, z = mul(aw, ax, ay, az, bw, bx, by, bz)
	local qx, qy, qz = rotate(aw, ax, ay, az, bpx, bpy, bpz)
	return store(kind, w, x, y, z, apx + qx, apy + qy, apz + qz)
end

local function inverse(r: any): any
	local w, x, y, z, px, py, pz, kind = load(r, "inverse")
	local qx, qy, qz = rotate(w, -x, -y, -z, px, py, pz)
	return store(kind, w, -x, -y, -z, -qx, -qy, -qz)
end

-- The point p rotated (and, for a frame or CFrame, moved): a vector.
local function apply(r: any, p: any): vector
	local w, x, y, z, px, py, pz = load(r, "apply")
	local vx, vy, vz = vec3(p, "apply")
	local qx, qy, qz = rotate(w, x, y, z, vx, vy, vz)
	return vector.create(px + qx, py + qy, pz + qz)
end

-- The angle of the rotation from a to b, in [0, pi].
local function angle(a: any, b: any): number
	local aw, ax, ay, az = load(a, "angle")
	local bw, bx, by, bz = load(b, "angle")
	-- 2 acos(|a . b|), through atan2 for accuracy near 0
	local d = math.abs(aw * bw + ax * bx + ay * by + az * bz)
	local _, dx, dy, dz = mul(aw, -ax, -ay, -az, bw, bx, by, bz)
	return 2 * math.atan2(math.sqrt(dx * dx + dy * dy + dz * dz), d)
end

-- The rotation vector (angle times axis; angle in [0, pi]).
local function log(r: any): vector
	local w, x, y, z = load(r, "log")
	local vx, vy, vz = qlog(w, x, y, z)
	return vector.create(vx, vy, vz)
end

-- From a toward b at fraction t (0: a, 1: b), along the shorter arc at constant angular
-- speed; frames' positions move linearly.
local function interpolate(a: any, b: any, t: number): any
	local aw, ax, ay, az, apx, apy, apz, kind = load(a, "slerp")
	local bw, bx, by, bz, bpx, bpy, bpz = load(b, "slerp")
	local w, x, y, z = slerp(aw, ax, ay, az, bw, bx, by, bz, t)
	return store(kind, w, x, y, z, apx + t * (bpx - apx), apy + t * (bpy - apy), apz + t * (bpz - apz))
end

-- The angular velocity (world coordinates) that turns a into b in time dt at a constant
-- rate: log(b a*) / dt.
local function angular_velocity(a: any, b: any, dt: number): vector
	local aw, ax, ay, az = load(a, "angular_velocity")
	local bw, bx, by, bz = load(b, "angular_velocity")
	local vx, vy, vz = qlog(mul(bw, bx, by, bz, aw, -ax, -ay, -az))
	return vector.create(vx / dt, vy / dt, vz / dt)
end

-- r advanced by dt under angular velocity omega (world coordinates; dq/dt = omega q / 2):
-- exactly for a constant omega. With omega_end (omega at the end of the step, omega
-- changing linearly between), by the fourth-order Magnus step: the rotation vector
-- dt (omega + omega_end) / 2 + dt^2 / 12 (omega_end x omega). Frames keep their position.
local function integrate(r: any, omega: any, dt: number, omega_end: any?): any
	local w, x, y, z, px, py, pz, kind = load(r, "integrate")
	local ox, oy, oz = vec3(omega, "integrate")
	local vx, vy, vz = ox * dt, oy * dt, oz * dt
	if omega_end ~= nil then
		local ex, ey, ez = vec3(omega_end, "integrate")
		local c = dt * dt / 12
		vx = (ox + ex) * dt / 2 + c * (ey * oz - ez * oy)
		vy = (oy + ey) * dt / 2 + c * (ez * ox - ex * oz)
		vz = (oz + ez) * dt / 2 + c * (ex * oy - ey * ox)
	end
	local ew, ex2, ey2, ez2 = qexp(vx, vy, vz)
	w, x, y, z = normalize(mul(ew, ex2, ey2, ez2, w, x, y, z))
	return store(kind, w, x, y, z, px, py, pz)
end

-- Several rotations

-- The weighted mean rotation (Markley et al., 2007): the unit quaternion maximizing
-- sum w_i (q . q_i)^2, the top eigenvector of sum w_i q_i q_i^T, which does not depend on
-- the signs of the q_i. Of the first rotation's kind.
local function mean(rotations: { any }, weights: { number }?): any
	local n = #rotations
	if n == 0 then
		fail("mean", "no rotations")
	end
	local M = f64.matrix(4, 4)
	local kind = "quat"
	for k = 1, n do
		local w, x, y, z, _, _, _, kk = load(rotations[k], "mean")
		if k == 1 then
			kind = kk
		end
		local c = if weights then weights[k] else 1
		local q = { w, x, y, z }
		for a = 1, 4 do
			for b = 1, 4 do
				f64.matset(M, a, b, f64.matget(M, a, b) + c * q[a] * q[b])
			end
		end
	end
	local E = f64.eigh_copy(M, true)
	local V = E.vectors :: Matrix
	local w, x, y, z = normalize(f64.matget(V, 1, 4), f64.matget(V, 2, 4), f64.matget(V, 3, 4), f64.matget(V, 4, 4))
	if w < 0 then
		w, x, y, z = -w, -x, -y, -z
	end
	local _, _, _, _, px, py, pz = load(rotations[1], "mean")
	return store(kind, w, x, y, z, px, py, pz)
end

-- The rigid motion best taking points a_i to b_i (least squares, weighted): Horn's method,
-- the rotation from the top eigenvector of a 4 by 4 matrix of the centered points' cross
-- covariances, and the translation matching the centroids. Points: vectors or { x, y, z }.
-- Returns a frame (or kind), and the root-mean-square distance remaining.
local function fit(a: { any }, b: { any }, weights: { number }?, kind: string?): (any, number)
	local n = #a
	if n == 0 or #b ~= n then
		fail("fit", "need the same number of points in each set, at least one")
	end
	local A, B = table.create(3 * n, 0), table.create(3 * n, 0)
	local ca, cb, total = { 0, 0, 0 }, { 0, 0, 0 }, 0
	for i = 1, n do
		local c = if weights then weights[i] else 1
		local x1, y1, z1 = vec3(a[i], "fit")
		local x2, y2, z2 = vec3(b[i], "fit")
		A[3 * i - 2], A[3 * i - 1], A[3 * i] = x1, y1, z1
		B[3 * i - 2], B[3 * i - 1], B[3 * i] = x2, y2, z2
		for d = 1, 3 do
			ca[d] += c * A[3 * i - 3 + d]
			cb[d] += c * B[3 * i - 3 + d]
		end
		total += c
	end
	if not (total > 0) then
		fail("fit", "the weights sum to zero")
	end
	for d = 1, 3 do
		ca[d] /= total
		cb[d] /= total
	end
	local S = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
	for i = 1, n do
		local c = if weights then weights[i] else 1
		for r = 1, 3 do
			for s = 1, 3 do
				S[r][s] += c * (A[3 * i - 3 + r] - ca[r]) * (B[3 * i - 3 + s] - cb[s])
			end
		end
	end
	local sxx, sxy, sxz = S[1][1], S[1][2], S[1][3]
	local syx, syy, syz = S[2][1], S[2][2], S[2][3]
	local szx, szy, szz = S[3][1], S[3][2], S[3][3]
	local N = f64.matfrom({
		{ sxx + syy + szz, syz - szy, szx - sxz, sxy - syx },
		{ syz - szy, sxx - syy - szz, sxy + syx, szx + sxz },
		{ szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy },
		{ sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz },
	})
	local E = f64.eigh_copy(N, true)
	local V = E.vectors :: Matrix
	local w, x, y, z = normalize(f64.matget(V, 1, 4), f64.matget(V, 2, 4), f64.matget(V, 3, 4), f64.matget(V, 4, 4))
	if w < 0 then
		w, x, y, z = -w, -x, -y, -z
	end
	local rx, ry, rz = rotate(w, x, y, z, ca[1], ca[2], ca[3])
	local px, py, pz = cb[1] - rx, cb[2] - ry, cb[3] - rz
	local sq = 0
	for i = 1, n do
		local c = if weights then weights[i] else 1
		local qx, qy, qz = rotate(w, x, y, z, A[3 * i - 2], A[3 * i - 1], A[3 * i])
		sq += c * ((px + qx - B[3 * i - 2]) ^ 2 + (py + qy - B[3 * i - 1]) ^ 2 + (pz + qz - B[3 * i]) ^ 2)
	end
	return store(kind or "frame", w, x, y, z, px, py, pz), math.sqrt(sq / total)
end

-- Many frames in a buffer: 7 numbers a frame (px, py, pz, w, x, y, z), frame i (1-based) at
-- byte 56 (i - 1).

-- A buffer of n frames, each at the origin with no rotation.
local function frames(n: number): buffer
	local b = buffer.create(56 * n)
	for i = 0, n - 1 do
		buffer.writef64(b, 56 * i + 24, 1)
	end
	return b
end

-- Frame i of a buffer, as a frame (or kind).
local function get(b: buffer, i: number, kind: string?): any
	local o = 56 * (i - 1)
	return store(kind or "frame", buffer.readf64(b, o + 24), buffer.readf64(b, o + 32), buffer.readf64(b, o + 40), buffer.readf64(b, o + 48),
		buffer.readf64(b, o), buffer.readf64(b, o + 8), buffer.readf64(b, o + 16))
end

-- Store a rotation, frame or CFrame as frame i.
local function set(b: buffer, i: number, r: any)
	local w, x, y, z, px, py, pz = load(r, "set")
	local o = 56 * (i - 1)
	buffer.writef64(b, o, px)
	buffer.writef64(b, o + 8, py)
	buffer.writef64(b, o + 16, pz)
	buffer.writef64(b, o + 24, w)
	buffer.writef64(b, o + 32, x)
	buffer.writef64(b, o + 40, y)
	buffer.writef64(b, o + 48, z)
end

-- Frame i of out := (frame j of A)(frame k of B), without tables (out may be A or B).
local function compose_into(out: buffer, i: number, A: buffer, j: number, B: buffer, k: number)
	local a, b, o = 56 * (j - 1), 56 * (k - 1), 56 * (i - 1)
	local aw, ax, ay, az = buffer.readf64(A, a + 24), buffer.readf64(A, a + 32), buffer.readf64(A, a + 40), buffer.readf64(A, a + 48)
	local bw, bx, by, bz = buffer.readf64(B, b + 24), buffer.readf64(B, b + 32), buffer.readf64(B, b + 40), buffer.readf64(B, b + 48)
	local qx, qy, qz = rotate(aw, ax, ay, az, buffer.readf64(B, b), buffer.readf64(B, b + 8), buffer.readf64(B, b + 16))
	local px, py, pz = buffer.readf64(A, a) + qx, buffer.readf64(A, a + 8) + qy, buffer.readf64(A, a + 16) + qz
	local w, x, y, z = mul(aw, ax, ay, az, bw, bx, by, bz)
	buffer.writef64(out, o, px)
	buffer.writef64(out, o + 8, py)
	buffer.writef64(out, o + 16, pz)
	buffer.writef64(out, o + 24, w)
	buffer.writef64(out, o + 32, x)
	buffer.writef64(out, o + 40, y)
	buffer.writef64(out, o + 48, z)
end

-- Splines through a sequence of rotations

export type SplineOptions = { kind: string?, times: { number }?, closed: boolean? }
export type RotationSpline = {
	at: (self: RotationSpline, t: number) -> any,
	angular_velocity: (self: RotationSpline, t: number) -> vector,
	at_angle: (self: RotationSpline, s: number) -> any,
	time_at_angle: (self: RotationSpline, s: number) -> number,
	domain: (self: RotationSpline) -> (number, number),
	angle: number,
	[string]: any,
}

local Spline = {}
Spline.__index = Spline

-- A spline through (or near) rotations at times (default 0, 1, 2, ...); the keys may be of
-- any kind, and the spline returns the first one's kind (frames and CFrames at the
-- first key's position: it interpolates orientation only). options.kind:
--   "smooth" (the default)  through every key, with continuous angular velocity: on each
--                           interval a Bezier curve on the sphere (Shoemake's, evaluated by
--                           slerps), its end angular velocities those of the neighbors
--                           (Catmull-Rom, weighted for uneven times);
--   "slerp"                 through every key at constant angular speed between keys (the
--                           angular velocity jumps at each);
--   "bspline"               a cumulative cubic B-spline (Kim, Kim and Shin, 1995): angular
--                           acceleration continuous too, but it passes near the keys, not
--                           through them; even times only.
-- closed = true joins the last key to the first. S:at(t), S:angular_velocity(t) (world
-- coordinates), S.angle (the total angle turned) and S:at_angle(s) (the rotation after
-- turning through angle s: constant angular speed along any spline).
local function spline(keys: { any }, options: SplineOptions?): RotationSpline
	local o: SplineOptions = options or {}
	local kind = o.kind or "smooth"
	if kind ~= "smooth" and kind ~= "slerp" and kind ~= "bspline" then
		fail("spline", "kind must be \"smooth\", \"slerp\" or \"bspline\"")
	end
	local n = #keys
	if n < 2 then
		fail("spline", "need at least two keys")
	end
	local closed = o.closed == true
	-- the keys as numbers, each on the same side as the one before (q and -q are one rotation)
	local W, X, Y, Z = table.create(n, 0), table.create(n, 0), table.create(n, 0), table.create(n, 0)
	local out_kind, px, py, pz = "quat", 0, 0, 0
	for k = 1, n do
		local w, x, y, z, a, b, c, kk = load(keys[k], "spline")
		w, x, y, z = normalize(w, x, y, z)
		if k == 1 then
			out_kind, px, py, pz = kk, a, b, c
		elseif w * W[k - 1] + x * X[k - 1] + y * Y[k - 1] + z * Z[k - 1] < 0 then
			w, x, y, z = -w, -x, -y, -z
		end
		W[k], X[k], Y[k], Z[k] = w, x, y, z
	end
	local m = if closed then n + 1 else n -- knots
	if closed then
		local w, x, y, z = W[1], X[1], Y[1], Z[1]
		if w * W[n] + x * X[n] + y * Y[n] + z * Z[n] < 0 then
			w, x, y, z = -w, -x, -y, -z
		end
		W[n + 1], X[n + 1], Y[n + 1], Z[n + 1] = w, x, y, z
	end
	local T: { number } = table.create(m, 0)
	if o.times then
		if #o.times ~= m then
			fail("spline", string.format("need %d times (one per key%s)", m, if closed then ", and one to return to the first" else ""))
		end
		for k = 1, m do
			T[k] = o.times[k]
			if k > 1 and not (T[k] > T[k - 1]) then
				fail("spline", "times must increase")
			end
		end
	else
		for k = 1, m do
			T[k] = k - 1
		end
	end
	if kind == "bspline" then
		local h = T[2] - T[1]
		for k = 3, m do
			if math.abs(T[k] - T[k - 1] - h) > 1e-9 * math.max(1, math.abs(h)) then
				fail("spline", "a \"bspline\" needs evenly spaced times")
			end
		end
	end
	local function key(k: number): (number, number, number, number)
		-- with the ends held (open) or wrapped (closed)
		if closed then
			local i = (k - 1) % n + 1
			local w, x, y, z = W[i], X[i], Y[i], Z[i]
			return w, x, y, z
		end
		local i = math.clamp(k, 1, n)
		return W[i], X[i], Y[i], Z[i]
	end
	-- the rotation vector from key k to key k + 1 (in either's coordinates: the same)
	local function step(k: number): (number, number, number)
		local aw, ax, ay, az = key(k)
		local bw, bx, by, bz = key(k + 1)
		return qlog(mul(aw, -ax, -ay, -az, bw, bx, by, bz))
	end
	local S: any = setmetatable({ kind = kind, out = out_kind, px = px, py = py, pz = pz, T = T, m = m, n = n, closed = closed }, Spline)
	if kind == "smooth" then
		-- the angular velocity (body coordinates) at each knot, and the Bezier control points
		local VX, VY, VZ = table.create(m, 0), table.create(m, 0), table.create(m, 0)
		for k = 1, m do
			local has_prev = k > 1 or closed
			local has_next = k < m or closed
			local dp = if k > 1 then T[k] - T[k - 1] elseif closed then T[m] - T[m - 1] else 0
			local dn = if k < m then T[k + 1] - T[k] elseif closed then T[2] - T[1] else 0
			local ux, uy, uz = 0, 0, 0
			local vx, vy, vz = 0, 0, 0
			if has_prev then
				ux, uy, uz = step(if k > 1 then k - 1 else n)
				ux, uy, uz = ux / dp, uy / dp, uz / dp
			end
			if has_next then
				vx, vy, vz = step(if k < m then k else 1)
				vx, vy, vz = vx / dn, vy / dn, vz / dn
			end
			if has_prev and has_next then
				local c = dp + dn
				VX[k], VY[k], VZ[k] = (dn * ux + dp * vx) / c, (dn * uy + dp * vy) / c, (dn * uz + dp * vz) / c
			elseif has_next then
				VX[k], VY[k], VZ[k] = vx, vy, vz
			else
				VX[k], VY[k], VZ[k] = ux, uy, uz
			end
		end
		local B1, B2 = {}, {}
		for k = 1, m - 1 do
			local d = (T[k + 1] - T[k]) / 3
			local aw, ax, ay, az = key(k)
			local bw, bx, by, bz = key(k + 1)
			local ew, ex, ey, ez = qexp(VX[k] * d, VY[k] * d, VZ[k] * d)
			B1[k] = { mul(aw, ax, ay, az, ew, ex, ey, ez) }
			local fw, fx, fy, fz = qexp(-VX[k + 1] * d, -VY[k + 1] * d, -VZ[k + 1] * d)
			B2[k] = { mul(bw, bx, by, bz, fw, fx, fy, fz) }
		end
		S.B1, S.B2 = B1, B2
	end
	S.key = key
	S.step = step
	-- the total angle, by Chebyshev interpolation of the angular speed on each interval
	local cum = table.create(m, 0)
	local funs = {}
	for k = 1, m - 1 do
		local a, b = T[k], T[k + 1]
		local F = cheb.cumsum(cheb.fun(function(t: number): number
			return S:_speed(t, k)
		end, { a, b }, { n = 33 }))
		funs[k] = F
		cum[k + 1] = cum[k] + cheb.eval(F, b)
	end
	S.cum, S.funs = cum, funs
	S.angle = cum[m]
	return S :: any
end

-- The interval holding t (1 .. m - 1), with t clamped to the domain.
local function interval(S: any, t: number): (number, number)
	local T, m = S.T, S.m
	t = math.clamp(t, T[1], T[m])
	local lo, hi = 1, m - 1
	while lo < hi do
		local mid = (lo + hi + 1) // 2
		if T[mid] <= t then
			lo = mid
		else
			hi = mid - 1
		end
	end
	return lo, t
end

-- The spline's rotation at t in interval k, as numbers.
function Spline._eval(self: RotationSpline, t: number, k: number): (number, number, number, number)
	local s = self :: any
	local T = s.T
	local u = (t - T[k]) / (T[k + 1] - T[k])
	local key = s.key
	if s.kind == "slerp" then
		local aw, ax, ay, az = key(k)
		local vx, vy, vz = s.step(k)
		local ew, ex, ey, ez = qexp(u * vx, u * vy, u * vz)
		return mul(aw, ax, ay, az, ew, ex, ey, ez)
	elseif s.kind == "smooth" then
		-- de Casteljau by slerps
		local p0w, p0x, p0y, p0z = key(k)
		local p3w, p3x, p3y, p3z = key(k + 1)
		local b1, b2 = s.B1[k], s.B2[k]
		local p1w, p1x, p1y, p1z = b1[1], b1[2], b1[3], b1[4]
		local p2w, p2x, p2y, p2z = b2[1], b2[2], b2[3], b2[4]
		local aw, ax, ay, az = slerp(p0w, p0x, p0y, p0z, p1w, p1x, p1y, p1z, u)
		local bw, bx, by, bz = slerp(p1w, p1x, p1y, p1z, p2w, p2x, p2y, p2z, u)
		local cw, cx, cy, cz = slerp(p2w, p2x, p2y, p2z, p3w, p3x, p3y, p3z, u)
		local dw, dx, dy, dz = slerp(aw, ax, ay, az, bw, bx, by, bz, u)
		local ew, ex, ey, ez = slerp(bw, bx, by, bz, cw, cx, cy, cz, u)
		return slerp(dw, dx, dy, dz, ew, ex, ey, ez, u)
	end
	-- cumulative B-spline: q_{k-1} exp(B1 w1) exp(B2 w2) exp(B3 w3), w_j the steps from key
	-- k - 2 + j to k - 1 + j
	local u2, u3 = u * u, u * u * u
	local c1, c2, c3 = (5 + 3 * u - 3 * u2 + u3) / 6, (1 + 3 * u + 3 * u2 - 2 * u3) / 6, u3 / 6
	local w, x, y, z = key(k - 1)
	local a1, a2, a3 = s.step(k - 1)
	local ew, ex, ey, ez = qexp(c1 * a1, c1 * a2, c1 * a3)
	w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
	a1, a2, a3 = s.step(k)
	ew, ex, ey, ez = qexp(c2 * a1, c2 * a2, c2 * a3)
	w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
	a1, a2, a3 = s.step(k + 1)
	ew, ex, ey, ez = qexp(c3 * a1, c3 * a2, c3 * a3)
	return mul(w, x, y, z, ew, ex, ey, ez)
end

-- The angular speed at t in interval k (exact for slerp; else a central difference of
-- step 1e-5 of the interval, accurate to about 1e-9).
function Spline._speed(self: RotationSpline, t: number, k: number): number
	local s = self :: any
	local T = s.T
	if s.kind == "slerp" then
		local vx, vy, vz = s.step(k)
		return math.sqrt(vx * vx + vy * vy + vz * vz) / (T[k + 1] - T[k])
	end
	local h = 1e-5 * (T[k + 1] - T[k])
	local aw, ax, ay, az = self:_eval(t - h, k)
	local bw, bx, by, bz = self:_eval(t + h, k)
	local vx, vy, vz = qlog(mul(aw, -ax, -ay, -az, bw, bx, by, bz))
	return math.sqrt(vx * vx + vy * vy + vz * vz) / (2 * h)
end

function Spline.domain(self: RotationSpline): (number, number)
	local s = self :: any
	return s.T[1], s.T[s.m]
end

function Spline.at(self: RotationSpline, t: number): any
	local s = self :: any
	local k, tc = interval(s, t)
	local w, x, y, z = self:_eval(tc, k)
	return store(s.out, w, x, y, z, s.px, s.py, s.pz)
end

-- The angular velocity at t, in world coordinates (a central difference).
function Spline.angular_velocity(self: RotationSpline, t: number): vector
	local s = self :: any
	local k, tc = interval(s, t)
	local T = s.T
	local h = 1e-5 * (T[k + 1] - T[k])
	local lo, hi = math.max(tc - h, T[k]), math.min(tc + h, T[k + 1])
	local aw, ax, ay, az = self:_eval(lo, k)
	local bw, bx, by, bz = self:_eval(hi, k)
	local vx, vy, vz = qlog(mul(bw, bx, by, bz, aw, -ax, -ay, -az))
	return vector.create(vx / (hi - lo), vy / (hi - lo), vz / (hi - lo))
end

-- The time at which the spline has turned through angle s (clamped to [0, S.angle]).
function Spline.time_at_angle(self: RotationSpline, a: number): number
	local s = self :: any
	local cum, T, m = s.cum, s.T, s.m
	a = math.clamp(a, 0, cum[m])
	local k = 1
	while k < m - 1 and cum[k + 1] < a do
		k += 1
	end
	local want = a - cum[k]
	local F = s.funs[k]
	local span = cum[k + 1] - cum[k]
	-- Newton on the interval's cumulative angle (its derivative is the speed), safeguarded
	local lo, hi = T[k], T[k + 1]
	local t = if span > 0 then lo + (hi - lo) * want / span else lo
	for _ = 1, 30 do
		local f = cheb.eval(F, t) - want
		if f > 0 then
			hi = t
		else
			lo = t
		end
		local d = self:_speed(t, k)
		local nt = if d > 0 then t - f / d else (lo + hi) / 2
		if not (nt > lo and nt < hi) then
			nt = (lo + hi) / 2
		end
		if math.abs(nt - t) <= 1e-14 * math.max(1, math.abs(t)) then
			t = nt
			break
		end
		t = nt
	end
	return t
end

-- The rotation after turning through angle s along the spline.
function Spline.at_angle(self: RotationSpline, a: number): any
	return self:at(self:time_at_angle(a))
end

-- A value's quaternion and position as numbers (w, x, y, z, px, py, pz), for code that
-- composes many rotations without tables.
local function components(r: any): (number, number, number, number, number, number, number)
	local w, x, y, z, px, py, pz = load(r, "components")
	return w, x, y, z, px, py, pz
end

return {
	quat = quat,
	identity = identity,
	frame = frame,
	axis_angle = axis_angle,
	exp = exp,
	between = between,
	from_matrix = from_rotation_matrix,
	matrix = matrix,
	from_euler = from_euler,
	to_euler = to_euler,
	mul = compose,
	inverse = inverse,
	apply = apply,
	angle = angle,
	log = log,
	slerp = interpolate,
	angular_velocity = angular_velocity,
	integrate = integrate,
	mean = mean,
	fit = fit,
	spline = spline,
	frames = frames,
	get = get,
	set = set,
	compose_into = compose_into,
	components = components,
}
