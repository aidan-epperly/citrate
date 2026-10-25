--!strict

-- XPBD: rigid and compliant constraints, their rest states against Hooke's law at every
-- substep size, damping, and kinematic particles.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local xpbd = lib.xpbd
local V = vector.create

test("xpbd: a rigid pendulum keeps its length", function()
	local w = xpbd.new({ V(0, 0, 0), V(2, 0, 0) }, { gravity = V(0, -9.8, 0) })
	w:pin(1)
	w:distance(1, 2, 0)
	local worst, lowest = 0, 0
	for _ = 1, 120 do
		w:step(1 / 60, 4)
		worst = math.max(worst, math.abs(vector.magnitude(w:position(2) - w:position(1)) - 2))
		lowest = math.min(lowest, w:position(2).y)
	end
	check(worst < 1e-5, `length kept (largest error {worst})`)
	check(lowest < -1.9, `it swings down through the bottom ({lowest})`)
	check(w.stats.steps == 120 and w.stats.substeps == 480, "statistics")
end)

test("xpbd: compliant constraints rest where Hooke's law says, at any substep", function()
	-- a mass m on a spring of compliance alpha under gravity g stretches by m g alpha
	local m, g, alpha = 2, 9.8, 1e-3
	for _, substeps in { 1, 4, 16 } do
		local w = xpbd.new({ V(0, 0, 0), V(0, -1, 0) }, { mass = { 1, m }, gravity = V(0, -g, 0), drag = 4 })
		w:pin(1)
		w:distance(1, 2, alpha)
		for _ = 1, 600 do
			w:step(1 / 60, substeps)
		end
		local stretch = vector.magnitude(w:position(2)) - 1
		close(stretch, m * g * alpha, 1e-4, `distance, {substeps} substeps`)
		-- the same as a spring toward an offset (three axes)
		local w2 = xpbd.new({ V(0, 0, 0), V(1, 0, 0) }, { mass = { 1, m }, gravity = V(0, -g, 0), drag = 4 })
		w2:pin(1)
		w2:offset(1, 2, V(1, 0, 0), alpha)
		for _ = 1, 600 do
			w2:step(1 / 60, substeps)
		end
		close(-w2:position(2).y, m * g * alpha, 1e-4, `offset, {substeps} substeps`)
	end
end)

test("xpbd: damping removes energy", function()
	local function swing(beta: number): number
		local w = xpbd.new({ V(0, 0, 0), V(0, -1.2, 0) })
		w:pin(1)
		w:distance(1, 2, 1e-2, beta, 1)
		for _ = 1, 60 do
			w:step(1 / 60, 4)
		end
		return math.abs(vector.magnitude(w:position(2)) - 1) + vector.magnitude(w:velocity(2)) * 0.1
	end
	check(swing(5) < 0.5 * swing(0), `damped {swing(5)} against undamped {swing(0)}`)
end)

test("xpbd: kinematic particles follow move()", function()
	local w = xpbd.new({ V(0, 0, 0), V(1, 0, 0) })
	w:pin(1)
	w:distance(1, 2, 0)
	w:move(1, V(0, 3, 0))
	w:step(1 / 60, 5)
	check(vector.magnitude(w:position(1) - V(0, 3, 0)) < 1e-6, "pinned particle reaches its target")
	close(vector.magnitude(w:position(2) - w:position(1)), 1, 1e-5, "and drags the other along")
end)

test("xpbd: every storage gives the same motion", function()
	-- a 4-link chain swinging from a pin, in vectors, f64 and f32 buffers
	local pts = { V(0, 0, 0), V(1, 0, 0), V(2, 0, 0), V(3, 0, 0), V(4, 0, 0) }
	local function run(space: any): vector
		local w = xpbd.new(pts, { space = space, gravity = V(0, -9.8, 0), drag = 0.5 })
		w:pin(1)
		for i = 1, 4 do
			w:distance(i, i + 1, 1e-4, 1)
			w:offset(i, i + 1, V(1, 0, 0), 2e-2)
		end
		for _ = 1, 60 do
			w:step(1 / 60, 4)
		end
		return w:position(5)
	end
	local a, b, c = run(nil), run(lib.f64.space), run(lib.f32.space)
	check(vector.magnitude(a - b) < 1e-4 and vector.magnitude(c - b) < 1e-4, `vectors {a}, f64 {b}, f32 {c}`)
	local buf = lib.f64.from({ 0, 0, 0, 1, 0, 0 })
	local w = xpbd.new(buf)
	check(type(w.x) == "buffer" and w.n == 2, "a buffer of positions is kept in a buffer")
end)

test("xpbd: colliders keep particles out", function()
	-- particles of radius 0.2 dropped onto a sphere, a capsule and a floor
	local w = xpbd.new({ V(0, 3, 0.1), V(5, 3, 0), V(10, 3, 0) }, { gravity = V(0, -9.8, 0), radius = 0.2 })
	w:sphere(V(0, 0, 0), 1)
	w:capsule(V(4, 0, 0), V(6, 0, 0), 0.5)
	w:plane(V(0, -5, 0), V(0, 1, 0))
	for _ = 1, 240 do
		w:step(1 / 60, 2)
	end
	check(vector.magnitude(w:position(1)) >= 1.2 - 1e-4, `outside the sphere: {vector.magnitude(w:position(1))}`)
	local p2 = w:position(2)
	check(math.sqrt(p2.y * p2.y + p2.z * p2.z) >= 0.7 - 1e-4 or p2.x < 4 or p2.x > 6, `outside the capsule: {p2}`)
	close(w:position(3).y, -4.8, 1e-3, "resting on the floor")
end)

test("xpbd: drag carries particles to the wind's velocity", function()
	local w = xpbd.new({ V(0, 0, 0) }, { drag = 3, wind = V(4, 0, 0) })
	for _ = 1, 300 do
		w:step(1 / 60, 2)
	end
	close(w:velocity(1).x, 4, 1e-3, "terminal velocity equals the wind")
end)

test("xpbd: friction holds a particle on a slope", function()
	-- a 30 degree slope; tan 30 < 1, so full friction holds it, none lets it slide
	local function slide(friction: number): number
		local n = V(-math.sin(math.pi / 6), math.cos(math.pi / 6), 0)
		local w = xpbd.new({ V(0, 0, 0) }, { gravity = V(0, -9.8, 0), friction = friction })
		w:plane(V(0, 0, 0), n)
		for _ = 1, 120 do
			w:step(1 / 60, 2)
		end
		return vector.magnitude(w:position(1))
	end
	check(slide(1) < 0.05 and slide(0) > 1, `with friction {slide(1)}, without {slide(0)}`)
end)

return nil
