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

return nil
