--!strict
--!native
--!optimize 2

-- Extended position-based dynamics (XPBD; Macklin, Müller and Chentanez 2016) for
-- particles joined by constraints, with substeps (Macklin et al. 2019, "Small steps in
-- physics simulation"): each substep predicts the positions from the velocities, then
-- moves the particles to satisfy each constraint C(x) = 0 in turn (one Gauss-Seidel sweep),
-- and takes the velocities from the change in position.
--
-- A constraint with compliance alpha (the inverse of its stiffness) and damping beta is the
-- force -grad C^T (C / alpha + beta dC/dt). Its update is
--   dlambda = (-C - gamma grad C (x - x_prev)) / ((1 + gamma) sum w |grad C|^2 + alpha / h^2)
--   x += w grad C^T dlambda,   gamma = alpha beta / h,
-- with w the inverse masses and h the substep: one Gauss-Seidel sweep on the Schur
-- complement of the backward Euler step's saddle-point system, so stiffness and damping
-- mean the same at every substep size (the h^2 in alpha / h^2), unlike plain
-- position-based dynamics. alpha = 0 makes a constraint rigid. Each substep costs a few
-- dozen operations per constraint.
--
-- Positions are 3D, kept in double precision. Drag (per second) multiplies the velocities
-- by exp(-drag h) each substep, which is exact for drag alone and so the same at any
-- substep size; gravity is then added whole, so a constraint holding a weight at rest
-- balances exactly m g at every substep size.

type Constraint = { kind: number, a: number, b: number, length: number, dx: number, dy: number, dz: number, alpha: number, beta: number }

export type World = {
	n: number,
	x: buffer, -- positions (x, y, z per particle, f64)
	v: buffer, -- velocities
	prev: buffer, -- positions at the start of the substep
	w: buffer, -- inverse masses (0: kinematic, moved by the caller)
	gravity: vector,
	drag: number,
	constraints: { Constraint },
	start: buffer, -- kinematic particles: positions at the start of the step
	target: buffer, -- and where the caller moves them to by its end
	kinematic: { number },
	t: number,
	stats: { steps: number, substeps: number },
	distance: (self: World, a: number, b: number, compliance: number, damping: number?, length: number?) -> (),
	offset: (self: World, a: number, b: number, d: vector, compliance: number, damping: number?) -> (),
	pin: (self: World, i: number) -> (),
	move: (self: World, i: number, p: vector) -> (),
	position: (self: World, i: number) -> vector,
	velocity: (self: World, i: number) -> vector,
	step: (self: World, dt: number, substeps: number?) -> (),
}

local DISTANCE, OFFSET = 1, 2

local function get(b: buffer, i: number): (number, number, number)
	local o = 24 * (i - 1)
	return buffer.readf64(b, o), buffer.readf64(b, o + 8), buffer.readf64(b, o + 16)
end
local function set(b: buffer, i: number, x: number, y: number, z: number)
	local o = 24 * (i - 1)
	buffer.writef64(b, o, x)
	buffer.writef64(b, o + 8, y)
	buffer.writef64(b, o + 16, z)
end

-- A world of particles at the given positions (index i is particle i). options: mass (a
-- number, or one per particle; default 1), gravity (default 0), drag (per second; default
-- 0). Its methods:
--   world:distance(a, b, compliance, damping?, length?)  keeps particles a and b `length`
--       apart (default: as far apart as they are now); compliance 0 makes it rigid
--   world:offset(a, b, d, compliance, damping?)  keeps x_b - x_a = d (a vector): a spring
--       toward a rest shape, along each axis
--   world:pin(i)  makes particle i kinematic: constraints do not move it, world:move does
--   world:move(i, p)  where kinematic particle i is to be at the end of the next step (it
--       moves there in a straight line over the substeps)
--   world:step(dt, substeps?)  advances dt in `substeps` substeps (default 1), each with
--       one sweep over the constraints in the order they were added
--   world:position(i), world:velocity(i)  as vectors
-- and fields x, v (positions and velocities, 3 per particle, f64), t and stats (steps,
-- substeps).
local function new(positions: { vector }, options: { mass: (number | { number })?, gravity: vector?, drag: number? }?): World
	local o: { mass: (number | { number })?, gravity: vector?, drag: number? } = options or {}
	local n = #positions
	local x, w = buffer.create(24 * n), buffer.create(8 * n)
	for i, p in positions do
		set(x, i, p.x, p.y, p.z)
		local m = if type(o.mass) == "table" then (o.mass :: { number })[i] else (o.mass :: number?) or 1
		assert(m > 0, "xpbd.new: masses must be positive")
		buffer.writef64(w, 8 * (i - 1), 1 / m)
	end
	local prev = buffer.create(24 * n)
	buffer.copy(prev, 0, x)
	local world: any = {
		n = n,
		x = x,
		v = buffer.create(24 * n),
		prev = prev,
		w = w,
		gravity = o.gravity or vector.zero,
		drag = o.drag or 0,
		constraints = {},
		start = buffer.create(24 * n),
		target = buffer.create(24 * n),
		kinematic = {},
		t = 0,
		stats = { steps = 0, substeps = 0 },
	}

	-- Keeps particles a and b `length` apart (default: as far apart as they are now).
	function world.distance(self: World, a: number, b: number, compliance: number, damping: number?, length: number?)
		local ax, ay, az = get(self.x, a)
		local bx, by, bz = get(self.x, b)
		local len = length or math.sqrt((bx - ax) ^ 2 + (by - ay) ^ 2 + (bz - az) ^ 2)
		table.insert(self.constraints, { kind = DISTANCE, a = a, b = b, length = len, dx = 0, dy = 0, dz = 0, alpha = compliance, beta = damping or 0 })
	end

	-- Keeps x_b - x_a = d (three constraints, one per axis): a spring toward a rest shape.
	function world.offset(self: World, a: number, b: number, d: vector, compliance: number, damping: number?)
		table.insert(self.constraints, { kind = OFFSET, a = a, b = b, length = 0, dx = d.x, dy = d.y, dz = d.z, alpha = compliance, beta = damping or 0 })
	end

	-- Makes particle i kinematic: constraints do not move it; move() does.
	function world.pin(self: World, i: number)
		buffer.writef64(self.w, 8 * (i - 1), 0)
		table.insert(self.kinematic, i)
		local px, py, pz = get(self.x, i)
		set(self.target, i, px, py, pz)
	end

	-- Where kinematic particle i is to be at the end of the next step (it moves there in a
	-- straight line over the substeps).
	function world.move(self: World, i: number, p: vector)
		set(self.target, i, p.x, p.y, p.z)
	end

	function world.position(self: World, i: number): vector
		local px, py, pz = get(self.x, i)
		return vector.create(px, py, pz)
	end

	function world.velocity(self: World, i: number): vector
		local px, py, pz = get(self.v, i)
		return vector.create(px, py, pz)
	end

	-- Advances dt by `substeps` substeps (default 1), each with one sweep over the
	-- constraints.
	function world.step(self: World, dt: number, substeps: number?)
		local count = substeps or 1
		local h = dt / count
		local X, V, P, W = self.x, self.v, self.prev, self.w
		local gx, gy, gz = self.gravity.x * h, self.gravity.y * h, self.gravity.z * h
		local keep = math.exp(-self.drag * h)
		local cs = self.constraints
		for _, i in self.kinematic do
			buffer.copy(self.start, 24 * (i - 1), X, 24 * (i - 1), 24)
		end
		for s = 1, count do
			-- predict
			buffer.copy(P, 0, X)
			for i = 0, self.n - 1 do
				local o = 24 * i
				if buffer.readf64(W, 8 * i) > 0 then
					-- drag on the velocity, gravity added whole: at rest a constraint then
					-- balances exactly m g (a spring stretches m g alpha at any substep)
					local vx = buffer.readf64(V, o) * keep + gx
					local vy = buffer.readf64(V, o + 8) * keep + gy
					local vz = buffer.readf64(V, o + 16) * keep + gz
					buffer.writef64(X, o, buffer.readf64(X, o) + h * vx)
					buffer.writef64(X, o + 8, buffer.readf64(X, o + 8) + h * vy)
					buffer.writef64(X, o + 16, buffer.readf64(X, o + 16) + h * vz)
				end
			end
			for _, i in self.kinematic do
				local u = s / count
				local ax, ay, az = get(self.start, i)
				local bx, by, bz = get(self.target, i)
				set(X, i, ax + u * (bx - ax), ay + u * (by - ay), az + u * (bz - az))
			end
			-- one Gauss-Seidel sweep over the constraints
			for _, c in cs do
				local oa, ob = 24 * (c.a - 1), 24 * (c.b - 1)
				local wa, wb = buffer.readf64(W, 8 * (c.a - 1)), buffer.readf64(W, 8 * (c.b - 1))
				local wsum = wa + wb
				if wsum == 0 then
					continue
				end
				local at = c.alpha / (h * h)
				local gamma = c.alpha * c.beta / h
				local ax, ay, az = buffer.readf64(X, oa), buffer.readf64(X, oa + 8), buffer.readf64(X, oa + 16)
				local bx, by, bz = buffer.readf64(X, ob), buffer.readf64(X, ob + 8), buffer.readf64(X, ob + 16)
				-- displacement since the substep began (for the damping term)
				local dax, day, daz = ax - buffer.readf64(P, oa), ay - buffer.readf64(P, oa + 8), az - buffer.readf64(P, oa + 16)
				local dbx, dby, dbz = bx - buffer.readf64(P, ob), by - buffer.readf64(P, ob + 8), bz - buffer.readf64(P, ob + 16)
				if c.kind == DISTANCE then
					local ex, ey, ez = bx - ax, by - ay, bz - az
					local len = math.sqrt(ex * ex + ey * ey + ez * ez)
					if len > 0 then
						-- grad C = (-u, u) with u the unit vector from a to b
						local ux, uy, uz = ex / len, ey / len, ez / len
						local C = len - c.length
						local rate = ux * (dbx - dax) + uy * (dby - day) + uz * (dbz - daz)
						local dl = (-C - gamma * rate) / ((1 + gamma) * wsum + at)
						ax, ay, az = ax - wa * dl * ux, ay - wa * dl * uy, az - wa * dl * uz
						bx, by, bz = bx + wb * dl * ux, by + wb * dl * uy, bz + wb * dl * uz
					end
				else
					-- C_k = (x_b - x_a - d)_k, grad C_k = (-e_k, e_k)
					local den = (1 + gamma) * wsum + at
					local lx = (-(bx - ax - c.dx) - gamma * (dbx - dax)) / den
					local ly = (-(by - ay - c.dy) - gamma * (dby - day)) / den
					local lz = (-(bz - az - c.dz) - gamma * (dbz - daz)) / den
					ax, ay, az = ax - wa * lx, ay - wa * ly, az - wa * lz
					bx, by, bz = bx + wb * lx, by + wb * ly, bz + wb * lz
				end
				buffer.writef64(X, oa, ax)
				buffer.writef64(X, oa + 8, ay)
				buffer.writef64(X, oa + 16, az)
				buffer.writef64(X, ob, bx)
				buffer.writef64(X, ob + 8, by)
				buffer.writef64(X, ob + 16, bz)
			end
			-- velocities from the change in position
			for i = 0, self.n - 1 do
				local o = 24 * i
				for k = 0, 16, 8 do
					buffer.writef64(V, o + k, (buffer.readf64(X, o + k) - buffer.readf64(P, o + k)) / h)
				end
			end
		end
		self.t += dt
		self.stats.steps += 1
		self.stats.substeps += count
	end

	return world :: World
end

return {
	new = new,
}
