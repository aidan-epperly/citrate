--!strict
--!native
--!optimize 2

-- Extended position-based dynamics (XPBD; Macklin, Müller and Chentanez 2016) for
-- particles joined by constraints, with substeps (Macklin et al. 2019, "Small steps in
-- physics simulation"): each substep predicts the positions from the velocities, then
-- moves the particles to satisfy each constraint C(x) = 0 in turn (one Gauss-Seidel sweep),
-- pushes them out of colliders, and takes the velocities from the change in position.
--
-- A constraint with compliance alpha (the inverse of its stiffness) and damping beta is the
-- force -grad C^T (C / alpha + beta dC/dt). Its update is
--   dlambda = (-C - gamma grad C (x - x_prev)) / ((1 + gamma) sum w |grad C|^2 + alpha / h^2)
--   x += w grad C^T dlambda,   gamma = alpha beta / h,
-- with w the inverse masses and h the substep: one Gauss-Seidel sweep on the Schur
-- complement of the backward Euler step's saddle-point system, so stiffness and damping
-- mean the same at every substep size (the h^2 in alpha / h^2), unlike plain
-- position-based dynamics. alpha = 0 makes a constraint rigid.
--
-- Drag (per second) pulls the velocities toward the wind's, by exp(-drag h) each substep
-- (exact for drag alone, so the same at any substep size); gravity is then added whole, so
-- a constraint holding a weight at rest balances exactly m g at every substep size.
--
-- The computation keeps the caller's representation: positions given as vectors are
-- stored and updated as vectors (single precision); with options.space = f64.space or
-- f32.space (or positions given as a buffer, x y z per particle) they are kept in a buffer
-- of that precision. The core is generated for each storage (xpbd_vec, xpbd_f64,
-- xpbd_f32) from one template.

local f64 = require("./f64")
local f32 = require("./f32")
local cores = {
	vec = require("./xpbd_vec"),
	f64 = require("./xpbd_f64"),
	f32 = require("./xpbd_f32"),
}

-- A world of particles at the given positions (index i is particle i): a list of vectors,
-- or a buffer (x, y, z per particle) of options.space's precision. options: space
-- (f64.space or f32.space: keep the positions in a buffer of that precision; default: as
-- given), mass (a number, or one per particle; default 1), radius (for collisions; a number
-- or one per particle; default 0), gravity, drag (per second), wind (the air's velocity, the
-- drag's reference; default 0), friction (the fraction of a particle's motion along a
-- collider's surface removed while touching it; default 0). Its methods:
--   world:distance(a, b, compliance, damping?, length?)  keeps particles a and b `length`
--       apart (default: as far apart as they are now); compliance 0 makes it rigid
--   world:offset(a, b, d, compliance, damping?)  keeps x_b - x_a = d (a vector): a spring
--       toward a rest shape, along each axis
--       (both return the constraint's number k, for world:set_offset(k, d),
--       world:set_length(k, length) and world:set_compliance(k, compliance, damping?),
--       which change it between steps: a rest shape that turns with an animated parent)
--   world:pin(i)  makes particle i kinematic: constraints do not move it, world:move does
--   world:move(i, p)  where kinematic particle i is to be at the end of the next step (it
--       moves there in a straight line over the substeps)
--   world:sphere(center, r), world:capsule(a, b, r), world:plane(point, normal)  add a
--       collider and return it; change its fields (a, b, radius) to move it
--   world:step(dt, substeps?)  advances dt in `substeps` substeps (default 1), each with
--       one sweep over the constraints in the order they were added
--   world:position(i), world:velocity(i)  as vectors
-- and fields x, v (positions and velocities in the world's storage), gravity, drag, wind,
-- friction (settable between steps), colliders, t and stats (steps, substeps).
local function new(positions: { vector } | buffer, options: { space: any?, mass: (number | { number })?, radius: (number | { number })?, gravity: vector?, drag: number?, wind: vector?, friction: number? }?): any
	local o: any = options or {}
	local space = o.space
	if type(positions) == "buffer" then
		space = space or f64.space
	end
	if space == nil then
		local list = positions :: { vector }
		return cores.vec.new(list, #list, o)
	end
	local M, tag = f64, "f64"
	if space == f32.space then
		M, tag = f32, "f32"
	else
		assert(space == f64.space, "xpbd.new: options.space must be f64.space or f32.space")
	end
	local data: buffer
	if type(positions) == "buffer" then
		data = positions :: buffer
	else
		local list = positions :: { vector }
		data = M.create(3 * #list)
		for i, p in list do
			M.set(data, 3 * i - 2, p.x)
			M.set(data, 3 * i - 1, p.y)
			M.set(data, 3 * i, p.z)
		end
	end
	return cores[tag].new(data, M.length(data) // 3, o)
end

return {
	new = new,
}
