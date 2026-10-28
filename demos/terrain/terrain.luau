--!strict
--!native
-- Marbles rolling over a heightfield known only at grid points (every SPACING studs), the
-- surface between them interpolated in two ways: bilinearly (the usual approach; its slope
-- jumps at every grid line, as a triangle mesh's does at every edge) and by citrate's
-- interp.grid (Keys' cubic convolution: its slope is continuous). Both sides step the
-- marbles the same way; only the surface differs. Usable outside Roblox.
local citrate = require("../../src")
local interp, random = citrate.interp, citrate.random

local terrain = {}

terrain.N = 25 -- grid points each way
terrain.SPACING = 5 -- studs between them
terrain.SIZE = (terrain.N - 1) * terrain.SPACING
terrain.GRAVITY = 196.2 -- studs/s^2, Roblox's default
terrain.RADIUS = 1
local ROLL = 5 / 7 -- a rolling solid ball accelerates at 5/7 of a sliding one's rate

-- The heights: seeded hills and pits on a shallow bowl (so marbles stay on the terrain).
function terrain.heights(seed: number?): { { number } }
	local rng = random.new(seed or 7)
	local bumps = {}
	for _ = 1, 14 do
		table.insert(bumps, {
			x = rng:uniform(0.1, 0.9) * terrain.SIZE,
			z = rng:uniform(0.1, 0.9) * terrain.SIZE,
			w = rng:uniform(7, 16),
			a = rng:uniform(-9, 14),
		})
	end
	local c = terrain.SIZE / 2
	local h = {}
	for i = 1, terrain.N do
		h[i] = {}
		for j = 1, terrain.N do
			local x, z = (i - 1) * terrain.SPACING, (j - 1) * terrain.SPACING
			local y = 0.004 * ((x - c) ^ 2 + (z - c) ^ 2)
			for _, b in bumps do
				y += b.a * math.exp(-((x - b.x) ^ 2 + (z - b.z) ^ 2) / (b.w * b.w))
			end
			h[i][j] = y
		end
	end
	return h
end

-- The two surfaces: x and z in studs from the grid's corner.
function terrain.surfaces(h: { { number } }): (any, any)
	local o: any = { dx = terrain.SPACING, dy = terrain.SPACING }
	return interp.grid(h, { dx = o.dx, dy = o.dy, method = "linear" }), interp.grid(h, o)
end

-- Marbles: positions x, z and velocities (3D, tangent to the surface) in flat arrays.
export type Marbles = { n: number, x: { number }, z: { number }, vx: { number }, vy: { number }, vz: { number } }

function terrain.marbles(n: number, seed: number?): Marbles
	local rng = random.new(seed or 11)
	local m: Marbles = { n = n, x = {}, z = {}, vx = table.create(n, 0), vy = table.create(n, 0), vz = table.create(n, 0) }
	for k = 1, n do
		m.x[k] = rng:uniform(0.08, 0.92) * terrain.SIZE
		m.z[k] = rng:uniform(0.08, 0.92) * terrain.SIZE
	end
	return m
end

function terrain.copy(m: Marbles): Marbles
	return { n = m.n, x = table.clone(m.x), z = table.clone(m.z), vx = table.clone(m.vx), vy = table.clone(m.vy), vz = table.clone(m.vz) }
end

-- One step of dt for every marble on surface S, the way a game rolls a ball on terrain:
-- gravity's component along the surface (the normal from the slope) speeds it up, it moves,
-- and its velocity is made tangent to the surface where it lands (losing the part along the
-- new normal, which at a crease is a bump). friction: a drag rate (1/s). Marbles bounce off
-- the terrain's edges.
function terrain.step(S: any, m: Marbles, dt: number, friction: number)
	local g = terrain.GRAVITY * ROLL
	local size = terrain.SIZE
	local damp = 1 - friction * dt
	local x, z, vx, vy, vz = m.x, m.z, m.vx, m.vy, m.vz
	for k = 1, m.n do
		local _, hx, hz = S:sample(x[k], z[k])
		-- unit normal (-hx, 1, -hz) / l; gravity (0, -g, 0) minus its normal part
		local l2 = 1 + hx * hx + hz * hz
		local ax, ay, az = -g * hx / l2, -g * (1 - 1 / l2), -g * hz / l2
		local ux, uy, uz = (vx[k] + ax * dt) * damp, (vy[k] + ay * dt) * damp, (vz[k] + az * dt) * damp
		local px, pz = x[k] + ux * dt, z[k] + uz * dt
		if px < 0 or px > size then
			ux, px = -ux, math.clamp(px, 0, size)
		end
		if pz < 0 or pz > size then
			uz, pz = -uz, math.clamp(pz, 0, size)
		end
		local _, gx, gz = S:sample(px, pz)
		local d = (ux * -gx + uy + uz * -gz) / (1 + gx * gx + gz * gz)
		x[k], z[k] = px, pz
		vx[k], vy[k], vz[k] = ux + d * gx, uy - d, uz + d * gz
	end
end

-- The marbles' total energy per unit mass (a rolling ball's: 7/10 v^2 + g h).
function terrain.energy(S: any, m: Marbles): number
	local e = 0
	for k = 1, m.n do
		e += 0.7 * (m.vx[k] ^ 2 + m.vy[k] ^ 2 + m.vz[k] ^ 2) + terrain.GRAVITY * S:at(m.x[k], m.z[k])
	end
	return e
end

-- A marble's position in studs (its center, a radius above the surface along the normal).
function terrain.position(S: any, m: Marbles, k: number): vector
	local h, hx, hz = S:sample(m.x[k], m.z[k])
	local l = math.sqrt(1 + hx * hx + hz * hz)
	local r = terrain.RADIUS / l
	return vector.create(m.x[k] - r * hx, h + r, m.z[k] - r * hz)
end

return terrain
