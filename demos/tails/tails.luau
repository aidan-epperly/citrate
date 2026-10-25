--!strict
--!native
-- Tails of bones hanging off a moving root, stepped at a fixed rate, in two ways:
--
-- * SmartBone 2's rule (github.com/smartbone-org/SmartBone-2; Bone:StepPhysics and its
--   "Spring" constraint with the default settings, without wind and colliders): Verlet
--   steps with damping applied per step, then each bone pulled a fraction of the way back
--   to its rest position and set at its length from its parent. SmartBone lowers the rate
--   with camera distance (60 Hz up to 15 studs, 0 at 45).
--   SmartBone 2 is Copyright (c) 2023 SmartBone contributors, MIT license.
-- * The same tail as a physical system: stiff rods with dashpots, springs toward the rest
--   shape, drag and gravity, advanced by citrate's implicit mechanics integrator with the
--   rational step (rho_inf = 0.5), one solve per step.
--
-- Usable outside Roblox.
local citrate = require("../../src")
local f64, ode = citrate.f64, citrate.ode

local tails = {}

tails.BONES = 6
tails.LENGTH = 1.5
local N, L = tails.BONES, tails.LENGTH
local REST = vector.create(L, 0, 0) -- each bone's rest offset from its parent
local GRAVITY = vector.create(0, -25 + 0.2, 0) -- SmartBone's Gravity + Force defaults

export type Tail = {
	rate: number, -- steps per second
	step: (self: Tail, root: (number) -> vector) -> (),
	nodes: (self: Tail) -> { vector }, -- the bone tips, root first
	t: number,
}

-- SmartBone: Damping 0.1, Elasticity 3, Stiffness 0.2, MAX_SPEED 50.
function tails.smartbone(rate: number): Tail
	local D, E, STIFF = 0.1, 3, 0.2
	local dt = 1 / rate
	local pos, last = {}, {}
	for i = 1, N do
		pos[i] = REST * i
		last[i] = pos[i]
	end
	local root = vector.zero
	local self = { rate = rate, t = 0 }
	function self.step(me: Tail, motion: (number) -> vector)
		me.t += dt
		root = motion(me.t)
		for i = 1, N do -- StepPhysics
			local vel = (pos[i] - last[i]) / dt
			if vector.magnitude(vel) > 50 then
				vel = vector.normalize(vel) * 50
			end
			last[i] = pos[i]
			pos[i] += vel * (1 - D) * dt + GRAVITY * dt * dt
		end
		for i = 1, N do -- Constrain: the spring constraint
			local parent = if i == 1 then root else pos[i - 1]
			local p = pos[i]
			local rest = parent + REST
			p += (rest - p) * math.min(E * dt, 1)
			local sd = rest - p
			local len = vector.magnitude(sd)
			local maxlen = L * (1 - STIFF) * 2
			if len > maxlen then
				p += sd * ((len - maxlen) / len)
			end
			local diff = parent - p
			local dl = vector.magnitude(diff)
			if dl > 0 then
				p += diff * ((dl - L) / dl)
			end
			pos[i] = p
		end
	end
	function self.nodes(_: Tail): { vector }
		local out = { root }
		for i = 1, N do
			out[i + 1] = pos[i]
		end
		return out
	end
	return self :: any
end

-- The physical tail: masses 1 at the bone tips. KREST and DRAG were chosen so that at 60
-- Hz it droops as far as SmartBone's (0.78 studs at the tip) and settles about as fast.
local KROD = 2e4
local KREST, DRAG = 660, 2.5
local CROD = 2 * math.sqrt(KROD) -- critical damping along each rod
function tails.physical(rate: number): Tail
	local dt = 1 / rate
	local motion: (number) -> vector = function(_)
		return vector.zero
	end
	local function root_at(t: number): (number, number, number, number, number, number)
		local r = motion(t)
		local h = 1e-4 -- the root's velocity, by a central difference
		local v = (motion(t + h) - motion(t - h)) / (2 * h)
		return r.x, r.y, r.z, v.x, v.y, v.z
	end
	-- F(t, q, v), in doubles (float32 vectors put noise into the stiff rod forces).
	local function force(t: number, q: buffer, v: buffer, out: buffer): buffer
		buffer.fill(out, 0, 0)
		local function get(i: number): (number, number, number, number, number, number)
			if i == 0 then
				return root_at(t)
			end
			local o = 3 * (i - 1)
			return f64.get(q, o + 1), f64.get(q, o + 2), f64.get(q, o + 3), f64.get(v, o + 1), f64.get(v, o + 2), f64.get(v, o + 3)
		end
		local function add(i: number, x: number, y: number, z: number)
			local o = 3 * (i - 1)
			f64.set(out, o + 1, f64.get(out, o + 1) + x)
			f64.set(out, o + 2, f64.get(out, o + 2) + y)
			f64.set(out, o + 3, f64.get(out, o + 3) + z)
		end
		for i = 1, N do
			local px, py, pz, pvx, pvy, pvz = get(i - 1)
			local qx, qy, qz, vx, vy, vz = get(i)
			local dx, dy, dz = qx - px, qy - py, qz - pz
			local len = math.sqrt(dx * dx + dy * dy + dz * dz)
			local ux, uy, uz = dx / len, dy / len, dz / len
			-- the rod (spring and dashpot along it) and the spring toward the rest shape act
			-- on the bone and, oppositely, on its parent
			local f = -KROD * (len - L) - CROD * ((vx - pvx) * ux + (vy - pvy) * uy + (vz - pvz) * uz)
			local ex, ey, ez = KREST * (px + REST.x - qx), KREST * (py + REST.y - qy), KREST * (pz + REST.z - qz)
			add(i, f * ux + ex + GRAVITY.x - DRAG * vx, f * uy + ey + GRAVITY.y - DRAG * vy, f * uz + ez + GRAVITY.z - DRAG * vz)
			if i > 1 then
				add(i - 1, -f * ux - ex, -f * uy - ey, -f * uz - ez)
			end
		end
		return out
	end
	-- K = dF/dq and D = dF/dv: 3 by 3 blocks, block tridiagonal, refilled each step.
	local bi, bj = {}, {}
	for i = 1, N do
		table.insert(bi, i)
		table.insert(bj, i)
		if i > 1 then
			table.insert(bi, i)
			table.insert(bj, i - 1)
			table.insert(bi, i - 1)
			table.insert(bj, i)
		end
	end
	local pattern = f64.bsparse(3 * N, 3 * N, 3, bi, bj, f64.create(9 * #bi))
	local KV, DV = f64.create(f64.bsparse_nnz(pattern)), f64.create(f64.bsparse_nnz(pattern))
	local BK = { rows = 3 * N, cols = 3 * N, block = 3, rowptr = pattern.rowptr, colind = pattern.colind, values = KV, symmetric = true }
	local BD = { rows = 3 * N, cols = 3 * N, block = 3, rowptr = pattern.rowptr, colind = pattern.colind, values = DV, symmetric = true }
	local slot = {}
	for i = 1, N do
		slot[i] = { f64.bsparse_block(pattern, i, i), if i > 1 then f64.bsparse_block(pattern, i - 1, i - 1) else nil, if i > 1 then f64.bsparse_block(pattern, i, i - 1) else nil, if i > 1 then f64.bsparse_block(pattern, i - 1, i) else nil }
	end
	local function add_block(V: buffer, p: number?, sg: number, m: { number })
		local o = 9 * ((p :: number) - 1)
		for k = 1, 9 do
			f64.set(V, o + k, f64.get(V, o + k) + sg * m[k])
		end
	end
	local DRAGB = { DRAG, 0, 0, 0, DRAG, 0, 0, 0, DRAG }
	local function jacobian(t: number, q: buffer, _v: buffer): (any, any)
		buffer.fill(KV, 0, 0)
		buffer.fill(DV, 0, 0)
		local rx, ry, rz = root_at(t)
		for i = 1, N do
			local px = if i > 1 then f64.get(q, 3 * i - 5) else rx
			local py = if i > 1 then f64.get(q, 3 * i - 4) else ry
			local pz = if i > 1 then f64.get(q, 3 * i - 3) else rz
			local dx, dy, dz = f64.get(q, 3 * i - 2) - px, f64.get(q, 3 * i - 1) - py, f64.get(q, 3 * i) - pz
			local len = math.sqrt(dx * dx + dy * dy + dz * dz)
			local u = { dx / len, dy / len, dz / len }
			local tt = math.max(1 - L / len, 0)
			local Kb, Db = table.create(9, 0), table.create(9, 0)
			for a = 1, 3 do
				for b = 1, 3 do
					local uu = u[a] * u[b]
					Kb[3 * (b - 1) + a] = KROD * ((1 - tt) * uu + (if a == b then tt else 0)) + (if a == b then KREST else 0)
					Db[3 * (b - 1) + a] = CROD * uu
				end
			end
			local s = slot[i]
			add_block(KV, s[1], -1, Kb)
			add_block(DV, s[1], -1, Db)
			add_block(DV, s[1], -1, DRAGB)
			if i > 1 then
				add_block(KV, s[2], -1, Kb)
				add_block(KV, s[3], 1, Kb)
				add_block(KV, s[4], 1, Kb)
				add_block(DV, s[2], -1, Db)
				add_block(DV, s[3], 1, Db)
				add_block(DV, s[4], 1, Db)
			end
		end
		return BK, BD
	end
	local q0, v0 = f64.create(3 * N), f64.create(3 * N)
	for i = 1, N do
		f64.set(q0, 3 * (i - 1) + 1, L * i)
	end
	local it = ode.implicit_mechanics(f64.space, force, 0, q0, v0, { jacobian = jacobian, rho_inf = 0.5 })
	local self = { rate = rate, t = 0 }
	function self.step(me: Tail, m: (number) -> vector)
		motion = m
		it:step(dt)
		me.t = it.t
	end
	function self.nodes(me: Tail): { vector }
		local out = { motion(me.t) }
		for i = 1, N do
			out[i + 1] = vector.create(f64.get(it.q, 3 * i - 2), f64.get(it.q, 3 * i - 1), f64.get(it.q, 3 * i))
		end
		return out
	end
	return self :: any
end

-- The root's motion: a shake of `amplitude` studs for `shake` seconds, then rest, repeating
-- every `period` seconds.
function tails.motion(amplitude: number, shake: number, period: number): (number) -> vector
	return function(t: number): vector
		local u = t % period
		if u >= shake then
			return vector.zero
		end
		local w = 2 * math.pi * 1.5
		return vector.create(0, amplitude * math.sin(w * u) * math.sin(math.pi * u / shake), 0)
	end
end

return tails
