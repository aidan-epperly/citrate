--!strict
--!native
--!optimize 2

-- Rotations and rigid frames of three-dimensional space in any representation: unit
-- quaternions, frames (a position and a rotation), Roblox CFrames, or your own. (Three
-- dimensions by nature: unit quaternions and CFrames are rotations of 3-D space.)
--
--   local q = rotation.axis_angle(vector.create(0, 1, 0), math.pi / 2)   -- a quaternion
--   local c = rotation.axis_angle(vector.create(0, 1, 0), math.pi / 2, rotation.cframe)
--   rotation.slerp(c, CFrame.identity, 0.25)                             -- CFrames in, CFrames out
--   local S = rotation.spline({ c1, c2, c3, c4 }, { times = { 0, 1, 3, 4 } })
--   S:at(2.5); S:angular_velocity(2.5); S:at_angle(1.2)
--   rotation.mean({ q1, q2, q3 }); rotation.integrate(q, omega, dt)
--
-- A rotation space supplies a representation's own primitives (see RotationSpace):
-- composing, inverting, applying to a point, and the exponential and logarithm maps between
-- rotations and rotation vectors (angle times axis). Everything else (slerp, Euler angles,
-- splines, means, fits, integration) is written in terms of those, so a value is never
-- converted: CFrames are worked on by CFrame methods, quaternions by quaternion arithmetic.
-- The spaces provided:
--   rotation.quaternion  tables { w, x, y, z } (unit; q and -q are the same rotation),
--                        arithmetic in double precision on numbers;
--   rotation.frame       tables { px, py, pz, w, x, y, z }: a position and a rotation;
--   rotation.cframe      Roblox CFrames (in Roblox; nil elsewhere): Roblox's own operations,
--                        in single precision as CFrames are.
-- Functions find the space from their arguments (a table with px is a frame, one with w a
-- quaternion, a CFrame a CFrame); constructors take one (default rotation.quaternion), and
-- options.space or a last argument gives a space of your own. rotation.convert changes
-- representation when you want it to.
--
-- Conventions: compose(a, b) applies b first (a * b for CFrames), angular velocities are in
-- world coordinates (dq/dt = omega q / 2, as Roblox's AssemblyAngularVelocity), log gives
-- the angle in [0, pi], and Euler angles follow CFrame.fromEulerAngles: order "XYZ" (the
-- default) is Rx(rx) Ry(ry) Rz(rz).
--
-- For many frames as numbers, a buffer of 7 numbers a frame (px, py, pz, w, x, y, z):
-- rotation.frames(n), get, set and compose_into.

local types = require("./types")
local f64 = require("./f64")
local cheb = require("./cheb")
local host = require("./host")

type Matrix = types.Matrix
export type Quat = { w: number, x: number, y: number, z: number }
export type Frame = { px: number, py: number, pz: number, w: number, x: number, y: number, z: number }

-- A representation of rotations (or frames). compose(a, b) applies b first; apply rotates
-- (and, for frames, moves) a point; exp turns a rotation vector (angle |v| about v) into a
-- rotation at the origin and log is its inverse on the rotation part (angle in [0, pi]).
-- Optional: slerp (else a exp(t log(a^-1 b))), compose_exp3 (a exp(v1) exp(v2) exp(v3) in
-- one call, which is a whole spline evaluation), position and place (for values with a
-- position: read it, and set it keeping the rotation), and components and from_components
-- (the rotation as a unit quaternion and the position as numbers, for rotation.convert and
-- the frame buffers). eps is the representation's precision (unit roundoff).
export type RotationSpace<R> = {
	name: string,
	eps: number,
	is: (v: any) -> boolean,
	identity: () -> R,
	compose: (a: R, b: R) -> R,
	inverse: (a: R) -> R,
	apply: (a: R, x: number, y: number, z: number) -> (number, number, number),
	exp: (x: number, y: number, z: number) -> R,
	log: (a: R) -> (number, number, number),
	slerp: ((a: R, b: R, t: number) -> R)?,
	compose_exp3: ((a: R, x1: number, y1: number, z1: number, x2: number, y2: number, z2: number, x3: number, y3: number, z3: number) -> R)?,
	position: ((a: R) -> (number, number, number))?,
	place: ((a: R, x: number, y: number, z: number) -> R)?,
	components: ((a: R) -> (number, number, number, number, number, number, number))?,
	from_components: ((w: number, x: number, y: number, z: number, px: number, py: number, pz: number) -> R)?,
}

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

local function qslerp(aw: number, ax: number, ay: number, az: number, bw: number, bx: number, by: number, bz: number, t: number): (number, number, number, number)
	local dw, dx, dy, dz = mul(aw, -ax, -ay, -az, bw, bx, by, bz)
	local vx, vy, vz = qlog(dw, dx, dy, dz)
	local ew, ex, ey, ez = qexp(t * vx, t * vy, t * vz)
	return mul(aw, ax, ay, az, ew, ex, ey, ez)
end

-- Shepperd's method: a unit quaternion from a rotation matrix, from the largest of
-- 1 + trace and the diagonal, for accuracy at every angle.
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

-- The spaces

local quaternion: RotationSpace<Quat> = {
	name = "quaternion",
	eps = 2 ^ -53,
	is = function(v: any): boolean
		return type(v) == "table" and type(v.w) == "number" and v.px == nil
	end,
	identity = function(): Quat
		return { w = 1, x = 0, y = 0, z = 0 }
	end,
	compose = function(a: Quat, b: Quat): Quat
		local w, x, y, z = mul(a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z)
		return { w = w, x = x, y = y, z = z }
	end,
	inverse = function(a: Quat): Quat
		return { w = a.w, x = -a.x, y = -a.y, z = -a.z }
	end,
	apply = function(a: Quat, x: number, y: number, z: number): (number, number, number)
		return rotate(a.w, a.x, a.y, a.z, x, y, z)
	end,
	exp = function(x: number, y: number, z: number): Quat
		local w, a, b, c = qexp(x, y, z)
		return { w = w, x = a, y = b, z = c }
	end,
	log = function(a: Quat): (number, number, number)
		return qlog(a.w, a.x, a.y, a.z)
	end,
	slerp = function(a: Quat, b: Quat, t: number): Quat
		local w, x, y, z = qslerp(a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z, t)
		return { w = w, x = x, y = y, z = z }
	end,
	compose_exp3 = function(a: Quat, x1: number, y1: number, z1: number, x2: number, y2: number, z2: number, x3: number, y3: number, z3: number): Quat
		local w, x, y, z = a.w, a.x, a.y, a.z
		local ew, ex, ey, ez = qexp(x1, y1, z1)
		w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
		ew, ex, ey, ez = qexp(x2, y2, z2)
		w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
		ew, ex, ey, ez = qexp(x3, y3, z3)
		w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
		return { w = w, x = x, y = y, z = z }
	end,
	components = function(a: Quat): (number, number, number, number, number, number, number)
		return a.w, a.x, a.y, a.z, 0, 0, 0
	end,
	from_components = function(w: number, x: number, y: number, z: number): Quat
		return { w = w, x = x, y = y, z = z }
	end,
}

local function new_frame(px: number, py: number, pz: number, w: number, x: number, y: number, z: number): Frame
	return { px = px, py = py, pz = pz, w = w, x = x, y = y, z = z }
end

local frame: RotationSpace<Frame> = {
	name = "frame",
	eps = 2 ^ -53,
	is = function(v: any): boolean
		return type(v) == "table" and type(v.w) == "number" and type(v.px) == "number"
	end,
	identity = function(): Frame
		return new_frame(0, 0, 0, 1, 0, 0, 0)
	end,
	compose = function(a: Frame, b: Frame): Frame
		local w, x, y, z = mul(a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z)
		local qx, qy, qz = rotate(a.w, a.x, a.y, a.z, b.px, b.py, b.pz)
		return new_frame(a.px + qx, a.py + qy, a.pz + qz, w, x, y, z)
	end,
	inverse = function(a: Frame): Frame
		local qx, qy, qz = rotate(a.w, -a.x, -a.y, -a.z, a.px, a.py, a.pz)
		return new_frame(-qx, -qy, -qz, a.w, -a.x, -a.y, -a.z)
	end,
	apply = function(a: Frame, x: number, y: number, z: number): (number, number, number)
		local qx, qy, qz = rotate(a.w, a.x, a.y, a.z, x, y, z)
		return a.px + qx, a.py + qy, a.pz + qz
	end,
	exp = function(x: number, y: number, z: number): Frame
		local w, a, b, c = qexp(x, y, z)
		return new_frame(0, 0, 0, w, a, b, c)
	end,
	log = function(a: Frame): (number, number, number)
		return qlog(a.w, a.x, a.y, a.z)
	end,
	-- the rotation by slerp, the position linearly (as CFrame:Lerp)
	slerp = function(a: Frame, b: Frame, t: number): Frame
		local w, x, y, z = qslerp(a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z, t)
		return new_frame(a.px + t * (b.px - a.px), a.py + t * (b.py - a.py), a.pz + t * (b.pz - a.pz), w, x, y, z)
	end,
	compose_exp3 = function(a: Frame, x1: number, y1: number, z1: number, x2: number, y2: number, z2: number, x3: number, y3: number, z3: number): Frame
		local w, x, y, z = a.w, a.x, a.y, a.z
		local ew, ex, ey, ez = qexp(x1, y1, z1)
		w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
		ew, ex, ey, ez = qexp(x2, y2, z2)
		w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
		ew, ex, ey, ez = qexp(x3, y3, z3)
		w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
		return new_frame(a.px, a.py, a.pz, w, x, y, z)
	end,
	position = function(a: Frame): (number, number, number)
		return a.px, a.py, a.pz
	end,
	place = function(a: Frame, x: number, y: number, z: number): Frame
		return new_frame(x, y, z, a.w, a.x, a.y, a.z)
	end,
	components = function(a: Frame): (number, number, number, number, number, number, number)
		return a.w, a.x, a.y, a.z, a.px, a.py, a.pz
	end,
	from_components = function(w: number, x: number, y: number, z: number, px: number, py: number, pz: number): Frame
		return new_frame(px, py, pz, w, x, y, z)
	end,
}

-- CFrames, by their own methods (only in Roblox).
local cframe: RotationSpace<any>? = nil
local CF, V3 = host.CFrame, host.Vector3
if CF ~= nil and V3 ~= nil then
	local identity_cf = CF.identity
	cframe = ({
		name = "cframe",
		eps = 2 ^ -24,
		is = function(v: any): boolean
			local t: string = typeof(v)
			return t == "CFrame"
		end,
		identity = function(): any
			return identity_cf
		end,
		compose = function(a: any, b: any): any
			return a * b
		end,
		inverse = function(a: any): any
			return a:Inverse()
		end,
		apply = function(a: any, x: number, y: number, z: number): (number, number, number)
			local v = a * V3.new(x, y, z)
			return v.X, v.Y, v.Z
		end,
		exp = function(x: number, y: number, z: number): any
			local t = math.sqrt(x * x + y * y + z * z)
			if t == 0 then
				return identity_cf
			end
			return CF.fromAxisAngle(V3.new(x / t, y / t, z / t), t)
		end,
		-- from the matrix by Shepperd's method, not ToAxisAngle: that loses the angle near 0
		-- (measured: splines' total angles off by 3e-3 relative), where this keeps it
		log = function(a: any): (number, number, number)
			local _, _, _, r00, r01, r02, r10, r11, r12, r20, r21, r22 = a:GetComponents()
			return qlog(from_matrix(r00, r01, r02, r10, r11, r12, r20, r21, r22))
		end,
		slerp = function(a: any, b: any, t: number): any
			return a:Lerp(b, t)
		end,
		-- exp(v1) exp(v2) exp(v3) as one rotation (a quaternion product in numbers), then one
		-- CFrame and one CFrame product
		compose_exp3 = function(a: any, x1: number, y1: number, z1: number, x2: number, y2: number, z2: number, x3: number, y3: number, z3: number): any
			local w, x, y, z = qexp(x1, y1, z1)
			local ew, ex, ey, ez = qexp(x2, y2, z2)
			w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
			ew, ex, ey, ez = qexp(x3, y3, z3)
			w, x, y, z = mul(w, x, y, z, ew, ex, ey, ez)
			return a * CF.new(0, 0, 0, x, y, z, w)
		end,
		position = function(a: any): (number, number, number)
			local p = a.Position
			return p.X, p.Y, p.Z
		end,
		place = function(a: any, x: number, y: number, z: number): any
			return CF.new(x, y, z) * a.Rotation
		end,
		components = function(a: any): (number, number, number, number, number, number, number)
			local px, py, pz, r00, r01, r02, r10, r11, r12, r20, r21, r22 = a:GetComponents()
			local w, x, y, z = from_matrix(r00, r01, r02, r10, r11, r12, r20, r21, r22)
			return w, x, y, z, px, py, pz
		end,
		from_components = function(w: number, x: number, y: number, z: number, px: number, py: number, pz: number): any
			return CF.new(px, py, pz, x, y, z, w)
		end,
	} :: any)
end

-- The space of a value (one type test for the common cases; spaces of one's own are
-- passed explicitly).
local function space_of(v: any, who: string): RotationSpace<any>
	if type(v) == "table" then
		if v.px ~= nil then
			return frame :: any
		elseif v.w ~= nil then
			return quaternion :: any
		end
	elseif cframe then
		local t: string = typeof(v)
		if t == "CFrame" then
			return cframe
		end
	end
	error(`rotation.{who}: expected a quaternion \{ w, x, y, z }, a frame \{ px, py, pz, w, x, y, z } or a CFrame (or pass a space)`, 3)
end

-- A constructor's space: given, by name, or the quaternions.
local function space_arg(space: any, who: string): RotationSpace<any>
	if space == nil then
		return quaternion :: any
	elseif type(space) == "table" and type(space.compose) == "function" then
		return space
	elseif space == "quaternion" or space == "quat" then
		return quaternion :: any
	elseif space == "frame" then
		return frame :: any
	elseif space == "cframe" then
		if not cframe then
			fail(who, "CFrames exist only in Roblox")
		end
		return cframe :: any
	end
	error(`rotation.{who}: space must be rotation.quaternion, rotation.frame, rotation.cframe or a RotationSpace`, 3)
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

-- Derived operations, from a space's primitives

local function slerp_in(S: RotationSpace<any>, a: any, b: any, t: number): any
	local sl = S.slerp
	if sl then
		return sl(a, b, t)
	end
	local x, y, z = S.log(S.compose(S.inverse(a), b))
	return S.compose(a, S.exp(t * x, t * y, t * z))
end

-- |log(a^-1 b)|: the angle between.
local function angle_in(S: RotationSpace<any>, a: any, b: any): number
	local x, y, z = S.log(S.compose(S.inverse(a), b))
	return math.sqrt(x * x + y * y + z * z)
end

-- The rotation matrix's rows (from the images of the axes, less that of the origin).
local function rows_in(S: RotationSpace<any>, r: any): { number }
	local ox, oy, oz = S.apply(r, 0, 0, 0)
	local ax, ay, az = S.apply(r, 1, 0, 0)
	local bx, by, bz = S.apply(r, 0, 1, 0)
	local cx, cy, cz = S.apply(r, 0, 0, 1)
	return { ax - ox, bx - ox, cx - ox, ay - oy, by - oy, cy - oy, az - oz, bz - oz, cz - oz }
end

-- Constructors

-- A unit quaternion from components (normalized).
local function quat(w: number, x: number, y: number, z: number): Quat
	local a, b, c, d = normalize(w, x, y, z)
	return { w = a, x = b, y = c, z = d }
end

local function identity(space: any?): any
	return space_arg(space, "identity").identity()
end

-- The rotation by the rotation vector v (angle |v| about v): the inverse of log.
local function exp(v: any, space: any?): any
	local x, y, z = vec3(v, "exp")
	return space_arg(space, "exp").exp(x, y, z)
end

-- The rotation by angle about axis (any nonzero length).
local function axis_angle(axis: any, angle: number, space: any?): any
	local x, y, z = vec3(axis, "axis_angle")
	local n = math.sqrt(x * x + y * y + z * z)
	if not (n > 0) then
		fail("axis_angle", "the axis is zero")
	end
	return space_arg(space, "axis_angle").exp(x / n * angle, y / n * angle, z / n * angle)
end

-- The shortest rotation taking direction a to direction b (about a x b; for opposite
-- directions, a half turn about an axis perpendicular to a).
local function between(a: any, b: any, space: any?): any
	local S = space_arg(space, "between")
	local ax, ay, az = vec3(a, "between")
	local bx, by, bz = vec3(b, "between")
	local na, nb = math.sqrt(ax * ax + ay * ay + az * az), math.sqrt(bx * bx + by * by + bz * bz)
	if not (na > 0 and nb > 0) then
		fail("between", "a direction is zero")
	end
	ax, ay, az, bx, by, bz = ax / na, ay / na, az / na, bx / nb, by / nb, bz / nb
	local d = ax * bx + ay * by + az * bz
	local cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
	local s = math.sqrt(cx * cx + cy * cy + cz * cz)
	if s < 1e-12 and d < 0 then
		-- a half turn about a x e, e the coordinate axis least aligned with a
		local ex, ey, ez = 0, 0, 0
		if math.abs(ax) <= math.abs(ay) and math.abs(ax) <= math.abs(az) then
			ex = 1
		elseif math.abs(ay) <= math.abs(az) then
			ey = 1
		else
			ez = 1
		end
		cx, cy, cz = ay * ez - az * ey, az * ex - ax * ez, ax * ey - ay * ex
		s = math.sqrt(cx * cx + cy * cy + cz * cz)
		return S.exp(math.pi * cx / s, math.pi * cy / s, math.pi * cz / s)
	end
	if s == 0 then
		return S.identity()
	end
	local t = math.atan2(s, d)
	return S.exp(t * cx / s, t * cy / s, t * cz / s)
end

-- From a rotation matrix: rows { { r00, r01, r02 }, ... } or a 3 by 3 f64 Matrix.
local function from_rotation_matrix(R: any, space: any?): any
	local function at(i: number, j: number): number
		if type(R) == "table" and R.data ~= nil then
			return f64.matget(R, i, j)
		end
		return R[i][j]
	end
	local x, y, z = qlog(from_matrix(at(1, 1), at(1, 2), at(1, 3), at(2, 1), at(2, 2), at(2, 3), at(3, 1), at(3, 2), at(3, 3)))
	return space_arg(space, "from_matrix").exp(x, y, z)
end

-- The rotation matrix, as rows.
local function matrix(r: any, space: any?): { { number } }
	local S: RotationSpace<any> = space or space_of(r, "matrix")
	local m = rows_in(S, r)
	return { { m[1], m[2], m[3] }, { m[4], m[5], m[6] }, { m[7], m[8], m[9] } }
end

local AXES: { [string]: number } = { X = 1, Y = 2, Z = 3 }

local function axes_of(order: string, who: string): (number, number, number)
	local i, j, k = AXES[order:sub(1, 1)], AXES[order:sub(2, 2)], AXES[order:sub(3, 3)]
	if #order ~= 3 or not i or not j or not k or i == j or j == k or i == k then
		fail(who, "order must be three different axes, such as \"XYZ\" or \"YXZ\"")
	end
	return i, j, k
end

-- From Euler angles about X, Y and Z (radians), composed as the order says: "XYZ" (the
-- default) is Rx(rx) Ry(ry) Rz(rz), as CFrame.fromEulerAngles(rx, ry, rz, order).
local function from_euler(rx: number, ry: number, rz: number, order: string?, space: any?): any
	local S = space_arg(space, "from_euler")
	local i, j, k = axes_of(order or "XYZ", "from_euler")
	local angles = { rx, ry, rz }
	local function about(a: number): any
		local t = angles[a]
		return S.exp(if a == 1 then t else 0, if a == 2 then t else 0, if a == 3 then t else 0)
	end
	return S.compose(S.compose(about(i), about(j)), about(k))
end

-- The Euler angles (rx, ry, rz) for the order (see from_euler); the middle angle is in
-- [-pi/2, pi/2], and at its ends (gimbal lock) the last is taken as 0.
local function to_euler(r: any, order: string?, space: any?): (number, number, number)
	local S: RotationSpace<any> = space or space_of(r, "to_euler")
	local i, j, k = axes_of(order or "XYZ", "to_euler")
	local m = rows_in(S, r)
	local function R(a: number, b: number): number
		return m[3 * (a - 1) + b]
	end
	-- +1 for the cyclic orders (XYZ, YZX, ZXY), -1 for the others
	local s = if (j - i) % 3 == 1 then 1 else -1
	local sb = math.clamp(s * R(i, k), -1, 1)
	local cb = math.sqrt(R(i, i) ^ 2 + R(i, j) ^ 2)
	-- the middle angle by atan2, accurate near +-pi/2 where asin is not
	local b = math.atan2(sb, cb)
	local a, c
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

-- Operations (on values of one space, found from the first)

-- a b: b, then a (a * b for CFrames; for frames, positions compose too).
local function compose(a: any, b: any, space: any?): any
	local S: RotationSpace<any> = space or space_of(a, "compose")
	return S.compose(a, b)
end

local function inverse(r: any, space: any?): any
	local S: RotationSpace<any> = space or space_of(r, "inverse")
	return S.inverse(r)
end

-- The point p rotated (and, for a frame or CFrame, moved): a vector.
local function apply(r: any, p: any, space: any?): vector
	local S: RotationSpace<any> = space or space_of(r, "apply")
	local x, y, z = vec3(p, "apply")
	local u, v, w = S.apply(r, x, y, z)
	return vector.create(u, v, w)
end

-- The angle of the rotation from a to b, in [0, pi].
local function angle(a: any, b: any, space: any?): number
	local S: RotationSpace<any> = space or space_of(a, "angle")
	return angle_in(S, a, b)
end

-- The rotation vector (angle times axis; angle in [0, pi]).
local function log(r: any, space: any?): vector
	local S: RotationSpace<any> = space or space_of(r, "log")
	local x, y, z = S.log(r)
	return vector.create(x, y, z)
end

-- From a toward b at fraction t (0: a, 1: b), along the shorter arc at constant angular
-- speed (frames' and CFrames' positions move linearly, as CFrame:Lerp).
local function slerp(a: any, b: any, t: number, space: any?): any
	local S: RotationSpace<any> = space or space_of(a, "slerp")
	return slerp_in(S, a, b, t)
end

-- The angular velocity (world coordinates) that turns a into b in time dt at a constant
-- rate: log(b a^-1) / dt.
local function angular_velocity(a: any, b: any, dt: number, space: any?): vector
	local S: RotationSpace<any> = space or space_of(a, "angular_velocity")
	local x, y, z = S.log(S.compose(b, S.inverse(a)))
	return vector.create(x / dt, y / dt, z / dt)
end

-- r advanced by dt under angular velocity omega (world coordinates; dq/dt = omega q / 2):
-- exp(omega dt) r, exact for a constant omega. With omega_end (omega at the step's end,
-- changing linearly between), the fourth-order Magnus step: the rotation vector
-- dt (omega + omega_end) / 2 + dt^2 / 12 (omega_end x omega). Positions stay.
local function integrate(r: any, omega: any, dt: number, omega_end: any?, space: any?): any
	local S: RotationSpace<any> = space or space_of(r, "integrate")
	local ox, oy, oz = vec3(omega, "integrate")
	local vx, vy, vz = ox * dt, oy * dt, oz * dt
	if omega_end ~= nil then
		local ex, ey, ez = vec3(omega_end, "integrate")
		local c = dt * dt / 12
		vx = (ox + ex) * dt / 2 + c * (ey * oz - ez * oy)
		vy = (oy + ey) * dt / 2 + c * (ez * ox - ex * oz)
		vz = (oz + ez) * dt / 2 + c * (ex * oy - ey * ox)
	end
	local turned = S.compose(S.exp(vx, vy, vz), r)
	local position, place = S.position, S.place
	if position and place then
		local px, py, pz = position(r)
		return place(turned, px, py, pz)
	end
	return turned
end

-- Several rotations

-- The weighted geodesic (Karcher) mean: the rotation m with sum w_i log(m^-1 r_i) = 0,
-- found by m <- m exp(the weighted mean of the log(m^-1 r_i)) until that step stops
-- shrinking (a few iterations; it needs the rotations within a half turn of each other).
-- Frames' positions are averaged too.
local function mean(rotations: { any }, weights: { number }?, space: any?): any
	local n = #rotations
	if n == 0 then
		fail("mean", "no rotations")
	end
	local S: RotationSpace<any> = space or space_of(rotations[1], "mean")
	local total = 0
	for k = 1, n do
		total += if weights then weights[k] else 1
	end
	if not (total > 0) then
		fail("mean", "the weights sum to zero")
	end
	local m = rotations[1]
	local last = math.huge
	for _ = 1, 50 do
		local sx, sy, sz = 0, 0, 0
		local mi = S.inverse(m)
		for k = 1, n do
			local c = (if weights then weights[k] else 1) / total
			local x, y, z = S.log(S.compose(mi, rotations[k]))
			sx, sy, sz = sx + c * x, sy + c * y, sz + c * z
		end
		local size = math.sqrt(sx * sx + sy * sy + sz * sz)
		if size >= last then
			break
		end
		m = S.compose(m, S.exp(sx, sy, sz))
		last = size
		if size <= 4 * S.eps then
			break
		end
	end
	local position, place = S.position, S.place
	if position and place then
		local px, py, pz = 0, 0, 0
		for k = 1, n do
			local c = (if weights then weights[k] else 1) / total
			local x, y, z = position(rotations[k])
			px, py, pz = px + c * x, py + c * y, pz + c * z
		end
		m = place(m, px, py, pz)
	end
	return m
end

-- The rigid motion best taking points a_i to b_i (least squares, weighted): Horn's method,
-- the rotation from the top eigenvector of a 4 by 4 matrix of the centered points' cross
-- covariances, built in the space by exp. Returns the motion (with its translation, in a
-- space with positions: rotation.frame by default) and the root-mean-square distance
-- remaining; in a space without positions, the rotation, the translation and the distance.
local function fit(a: { any }, b: { any }, weights: { number }?, space: any?): (any, any, number?)
	local S = space_arg(space or frame, "fit")
	local n = #a
	if n == 0 or #b ~= n then
		fail("fit", "need the same number of points in each set, at least one")
	end
	local A, B = table.create(3 * n, 0), table.create(3 * n, 0)
	local ca, cb, total = { 0, 0, 0 }, { 0, 0, 0 }, 0
	for i = 1, n do
		local c = if weights then weights[i] else 1
		A[3 * i - 2], A[3 * i - 1], A[3 * i] = vec3(a[i], "fit")
		B[3 * i - 2], B[3 * i - 1], B[3 * i] = vec3(b[i], "fit")
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
	local M = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
	for i = 1, n do
		local c = if weights then weights[i] else 1
		for r = 1, 3 do
			for s = 1, 3 do
				M[r][s] += c * (A[3 * i - 3 + r] - ca[r]) * (B[3 * i - 3 + s] - cb[s])
			end
		end
	end
	local sxx, sxy, sxz = M[1][1], M[1][2], M[1][3]
	local syx, syy, syz = M[2][1], M[2][2], M[2][3]
	local szx, szy, szz = M[3][1], M[3][2], M[3][3]
	local N = f64.matfrom({
		{ sxx + syy + szz, syz - szy, szx - sxz, sxy - syx },
		{ syz - szy, sxx - syy - szz, sxy + syx, szx + sxz },
		{ szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy },
		{ sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz },
	})
	local V = f64.eigh_copy(N, true).vectors :: Matrix
	local w, x, y, z = normalize(f64.matget(V, 1, 4), f64.matget(V, 2, 4), f64.matget(V, 3, 4), f64.matget(V, 4, 4))
	local R = S.exp(qlog(w, x, y, z))
	local rx, ry, rz = rotate(w, x, y, z, ca[1], ca[2], ca[3])
	local px, py, pz = cb[1] - rx, cb[2] - ry, cb[3] - rz
	local sq = 0
	for i = 1, n do
		local c = if weights then weights[i] else 1
		local qx, qy, qz = rotate(w, x, y, z, A[3 * i - 2], A[3 * i - 1], A[3 * i])
		sq += c * ((px + qx - B[3 * i - 2]) ^ 2 + (py + qy - B[3 * i - 1]) ^ 2 + (pz + qz - B[3 * i]) ^ 2)
	end
	local rms = math.sqrt(sq / total)
	local place = S.place
	if place then
		return place(R, px, py, pz), rms
	end
	return R, vector.create(px, py, pz), rms
end

-- r placed at a position (keeping its rotation): in r's space if that has positions, else
-- (for a quaternion) a frame.
local function place(r: any, position: any, space: any?): any
	local S: RotationSpace<any> = space or space_of(r, "place")
	local x, y, z = vec3(position, "place")
	local put = S.place
	if put then
		return put(r, x, y, z)
	end
	local get_c = S.components
	if get_c == nil then
		error("rotation.place: the space has neither positions nor components", 2)
	end
	local w, a, b, c = get_c(r)
	return new_frame(x, y, z, w, a, b, c)
end

-- A value in another space (through its rotation as a quaternion and its position; both
-- spaces need components and from_components).
local function convert(r: any, space: any, from: any?): any
	local S = if from then space_arg(from, "convert") else space_of(r, "convert")
	local T = space_arg(space, "convert")
	local get_c, make = S.components, T.from_components
	if get_c == nil or make == nil then
		error("rotation.convert: both spaces need components and from_components", 2)
	end
	return make(get_c(r))
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

-- Frame i of a buffer, in a space (default rotation.frame).
local function get(b: buffer, i: number, space: any?): any
	local T = space_arg(space or frame, "get")
	local make = T.from_components
	if make == nil then
		error("rotation.get: the space needs from_components", 2)
	end
	local o = 56 * (i - 1)
	return make(buffer.readf64(b, o + 24), buffer.readf64(b, o + 32), buffer.readf64(b, o + 40), buffer.readf64(b, o + 48),
		buffer.readf64(b, o), buffer.readf64(b, o + 8), buffer.readf64(b, o + 16))
end

-- Store a value as frame i.
local function set(b: buffer, i: number, r: any, space: any?)
	local S: RotationSpace<any> = space or space_of(r, "set")
	local get_c = S.components
	if get_c == nil then
		error("rotation.set: the space needs components", 2)
	end
	local w, x, y, z, px, py, pz = get_c(r)
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

export type SplineOptions = { kind: string?, times: { number }?, closed: boolean?, space: any? }
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

-- A spline through (or near) rotations at times (default 0, 1, 2, ...), in the keys' space
-- (options.space for your own). options.kind:
--   "smooth" (the default)  through every key, with continuous angular velocity: on each
--                           interval a cubic Bezier curve of rotations in the cumulative
--                           form of Kim, Kim and Shin (1995), its end angular velocities
--                           those of the neighbors (Catmull-Rom, weighted for uneven times);
--   "slerp"                 through every key at constant angular speed between keys (the
--                           angular velocity jumps at each);
--   "bspline"               a cumulative cubic B-spline (Kim, Kim and Shin, 1995): angular
--                           acceleration continuous too, but it passes near the keys, not
--                           through them; even times only.
-- closed = true joins the last key to the first. S:at(t), S:angular_velocity(t) (world
-- coordinates), S.angle (the total angle turned) and S:at_angle(s) (the rotation after
-- turning through angle s: constant angular speed along any spline). Positions of frames
-- and CFrames are carried along by the space's slerp (between keys' positions); for a path,
-- interpolate positions with a curve.
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
	local S = if o.space then space_arg(o.space, "spline") else space_of(keys[1], "spline")
	for k = 2, n do
		if not S.is(keys[k]) then
			fail("spline", `key {k} is not in the first key's space ({S.name})`)
		end
	end
	local closed = o.closed == true
	local m = if closed then n + 1 else n -- knots
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
	-- key k, with the ends held (open) or wrapped (closed)
	local function key(k: number): any
		return keys[if closed then (k - 1) % n + 1 else math.clamp(k, 1, n)]
	end
	-- the rotation vector from key k to key k + 1 (the same in either's coordinates)
	local steps = {}
	local function step(k: number): (number, number, number)
		local s = steps[k]
		if s == nil then
			s = { S.log(S.compose(S.inverse(key(k)), key(k + 1))) }
			steps[k] = s
		end
		return s[1], s[2], s[3]
	end
	local self: any = setmetatable({ kind = kind, space = S, T = T, m = m, key = key, step = step, chain = S.compose_exp3 }, Spline)
	if kind == "smooth" then
		-- the angular velocity (body coordinates) at each knot, and the Bezier control points
		local VX, VY, VZ = table.create(m, 0), table.create(m, 0), table.create(m, 0)
		for k = 1, m do
			local has_prev = k > 1 or closed
			local has_next = k < m or closed
			local dp = if k > 1 then T[k] - T[k - 1] elseif closed then T[m] - T[m - 1] else 0
			local dn = if k < m then T[k + 1] - T[k] elseif closed then T[2] - T[1] else 0
			local ux, uy, uz, vx, vy, vz = 0, 0, 0, 0, 0, 0
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
		-- each interval's Bezier control points q, b1, b2, q' and the rotation vectors between
		-- them, for the cumulative form
		local W = {}
		for k = 1, m - 1 do
			local d = (T[k + 1] - T[k]) / 3
			local q0, q3 = key(k), key(k + 1)
			local b1 = S.compose(q0, S.exp(VX[k] * d, VY[k] * d, VZ[k] * d))
			local b2 = S.compose(q3, S.exp(-VX[k + 1] * d, -VY[k + 1] * d, -VZ[k + 1] * d))
			local x1, y1, z1 = S.log(S.compose(S.inverse(q0), b1))
			local x2, y2, z2 = S.log(S.compose(S.inverse(b1), b2))
			local x3, y3, z3 = S.log(S.compose(S.inverse(b2), q3))
			W[k] = { x1, y1, z1, x2, y2, z2, x3, y3, z3 }
		end
		self.W = W
	end
	-- the total angle, by Chebyshev interpolation of the angular speed on each interval
	local cum = table.create(m, 0)
	local funs = {}
	for k = 1, m - 1 do
		local F = cheb.cumsum(cheb.fun(function(t: number): number
			return self:_speed(t, k)
		end, { T[k], T[k + 1] }, { n = 33 }))
		funs[k] = F
		cum[k + 1] = cum[k] + cheb.eval(F, T[k + 1])
	end
	self.cum, self.funs = cum, funs
	self.angle = cum[m]
	return self :: any
end

-- The interval holding t (1 .. m - 1), with t clamped to the domain.
local function interval(s: any, t: number): (number, number)
	local T, m = s.T, s.m
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

-- The spline's value at t in interval k.
function Spline._eval(self: RotationSpline, t: number, k: number): any
	local s = self :: any
	local S, T, key = s.space, s.T, s.key
	local u = (t - T[k]) / (T[k + 1] - T[k])
	if s.kind == "slerp" then
		return slerp_in(S, key(k), key(k + 1), u)
	end
	-- the cumulative forms: r exp(c1 w1) exp(c2 w2) exp(c3 w3)
	local r, c1, c2, c3
	local x1, y1, z1, x2, y2, z2, x3, y3, z3
	local u2, u3 = u * u, u * u * u
	if s.kind == "smooth" then
		-- the Bezier curve through control points q, b1, b2, q' (Kim, Kim and Shin): w_j the
		-- rotation vectors between consecutive control points, cumulative Bernstein weights
		local w = s.W[k]
		x1, y1, z1, x2, y2, z2, x3, y3, z3 = w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9]
		local v = 1 - u
		r, c1, c2, c3 = key(k), 1 - v * v * v, 3 * u2 - 2 * u3, u3
	else
		-- the B-spline: from key k - 1, w_j the steps from key k - 2 + j to k - 1 + j
		x1, y1, z1 = s.step(k - 1)
		x2, y2, z2 = s.step(k)
		x3, y3, z3 = s.step(k + 1)
		r, c1, c2, c3 = key(k - 1), (5 + 3 * u - 3 * u2 + u3) / 6, (1 + 3 * u + 3 * u2 - 2 * u3) / 6, u3 / 6
	end
	local chain = s.chain
	if chain then
		return chain(r, c1 * x1, c1 * y1, c1 * z1, c2 * x2, c2 * y2, c2 * z2, c3 * x3, c3 * y3, c3 * z3)
	end
	r = S.compose(r, S.exp(c1 * x1, c1 * y1, c1 * z1))
	r = S.compose(r, S.exp(c2 * x2, c2 * y2, c2 * z2))
	return S.compose(r, S.exp(c3 * x3, c3 * y3, c3 * z3))
end

-- A central difference's step for the space's precision (about eps^(1/3) of the interval).
local function diff_step(s: any, k: number): number
	return s.space.eps ^ (1 / 3) * (s.T[k + 1] - s.T[k])
end

-- The angular speed at t in interval k (exact for slerp; else a central difference).
function Spline._speed(self: RotationSpline, t: number, k: number): number
	local s = self :: any
	local T = s.T
	if s.kind == "slerp" then
		local vx, vy, vz = s.step(k)
		return math.sqrt(vx * vx + vy * vy + vz * vz) / (T[k + 1] - T[k])
	end
	local h = diff_step(s, k)
	return angle_in(s.space, self:_eval(t - h, k), self:_eval(t + h, k)) / (2 * h)
end

function Spline.domain(self: RotationSpline): (number, number)
	local s = self :: any
	return s.T[1], s.T[s.m]
end

function Spline.at(self: RotationSpline, t: number): any
	local k, tc = interval(self, t)
	return self:_eval(tc, k)
end

-- The angular velocity at t, in world coordinates (a central difference).
function Spline.angular_velocity(self: RotationSpline, t: number): vector
	local s = self :: any
	local S = s.space
	local k, tc = interval(s, t)
	local h = diff_step(s, k)
	local lo, hi = math.max(tc - h, s.T[k]), math.min(tc + h, s.T[k + 1])
	local x, y, z = S.log(S.compose(self:_eval(hi, k), S.inverse(self:_eval(lo, k))))
	return vector.create(x / (hi - lo), y / (hi - lo), z / (hi - lo))
end

-- The time at which the spline has turned through angle a (clamped to [0, S.angle]).
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

-- The rotation after turning through angle a along the spline.
function Spline.at_angle(self: RotationSpline, a: number): any
	return self:at(self:time_at_angle(a))
end

-- The operations with the space fixed (no space to find on each call), for code that calls
-- them often: local R = rotation.bind(rotation.cframe); R.slerp(a, b, t).
local function bind(S: RotationSpace<any>): { [string]: any }
	return {
		space = S,
		identity = S.identity,
		compose = S.compose,
		inverse = S.inverse,
		exp = function(v: any): any
			local x, y, z = vec3(v, "exp")
			return S.exp(x, y, z)
		end,
		axis_angle = function(axis: any, a: number): any
			return axis_angle(axis, a, S)
		end,
		between = function(a: any, b: any): any
			return between(a, b, S)
		end,
		from_euler = function(rx: number, ry: number, rz: number, order: string?): any
			return from_euler(rx, ry, rz, order, S)
		end,
		to_euler = function(r: any, order: string?): (number, number, number)
			return to_euler(r, order, S)
		end,
		from_matrix = function(M: any): any
			return from_rotation_matrix(M, S)
		end,
		matrix = function(r: any): { { number } }
			return matrix(r, S)
		end,
		apply = function(r: any, p: any): vector
			return apply(r, p, S)
		end,
		angle = function(a: any, b: any): number
			return angle_in(S, a, b)
		end,
		log = function(r: any): vector
			local x, y, z = S.log(r)
			return vector.create(x, y, z)
		end,
		slerp = S.slerp or function(a: any, b: any, t: number): any
			return slerp_in(S, a, b, t)
		end,
		angular_velocity = function(a: any, b: any, dt: number): vector
			return angular_velocity(a, b, dt, S)
		end,
		integrate = function(r: any, omega: any, dt: number, omega_end: any?): any
			return integrate(r, omega, dt, omega_end, S)
		end,
		mean = function(list: { any }, weights: { number }?): any
			return mean(list, weights, S)
		end,
		fit = function(a: { any }, b: { any }, weights: { number }?): (any, any, number?)
			return fit(a, b, weights, S)
		end,
		spline = function(keys: { any }, options: SplineOptions?): RotationSpline
			local o: any = table.clone(options or {})
			o.space = S
			return spline(keys, o)
		end,
	}
end

return {
	bind = bind,
	quaternion = quaternion,
	frame = frame,
	cframe = cframe,
	quat = quat,
	identity = identity,
	exp = exp,
	axis_angle = axis_angle,
	between = between,
	from_matrix = from_rotation_matrix,
	matrix = matrix,
	from_euler = from_euler,
	to_euler = to_euler,
	compose = compose,
	inverse = inverse,
	apply = apply,
	angle = angle,
	log = log,
	slerp = slerp,
	angular_velocity = angular_velocity,
	integrate = integrate,
	mean = mean,
	fit = fit,
	convert = convert,
	place = place,
	spline = spline,
	frames = frames,
	get = get,
	set = set,
	compose_into = compose_into,
}
