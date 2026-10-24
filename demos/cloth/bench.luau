--!strict
-- Cloth benchmark (standalone: luau demos/cloth/bench.luau): the cost of one 60 Hz frame
-- of the implicit step at several sizes, and how stiff the springs can be before
-- explicit substepping blows up.

local root = require("../../src")
local cloth = require("./cloth")
local f64, ode = root.f64, root.ode

local H = 1 / 60

local function sag(model: any, q: buffer): number
	-- The lowest point's height (the cloth hangs below y = 0).
	local low = 0
	for p = 1, model.particles do
		low = math.min(low, f64.get(q, 3 * p - 1))
	end
	return low
end

local function finite(q: buffer): boolean
	return f64.norm(q) < math.huge
end

print("implicit (linearized backward Euler + CG), one step per 60 Hz frame")
for _, size in { 10, 20, 30 } do
	for _, k in { 500, 5000, 50000 } do
		local model = cloth.new({ nx = size, ny = size, spacing = 1 / (size - 1), k_struct = k, k_shear = k / 5, k_bend = k / 25 })
		local it = ode.implicit_mechanics(f64.space, model.force, 0, model.q0, model.v0, {
			jacobian = model.jacobian,
			mass = model.mass,
			fixed = model.fixed,
			tol = 1e-4,
		})
		local frames = 120
		local start = os.clock()
		for _ = 1, frames do
			it:step(H)
		end
		local ms = (os.clock() - start) / frames * 1000
		print(string.format("  %2dx%-2d k=%-6d %7.2f ms/frame  %5.1f CG its/frame  lowest point %.3f  %s", size, size, k, ms, it.stats.iterations / frames, sag(model, it.q), if finite(it.q) then "ok" else "BLEW UP"))
	end
end

print("explicit (symplectic Euler substeps): stiffest structural k that survives 2 s")
for _, size in { 10, 20 } do
	for _, substeps in { 1, 10, 50 } do
		local best = 0
		for _, k in { 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000 } do
			local model = cloth.new({ nx = size, ny = size, spacing = 1 / (size - 1), k_struct = k, k_shear = k / 5, k_bend = k / 25 })
			local q, v = f64.copy(nil, model.q0), f64.copy(nil, model.v0)
			for _ = 1, 120 do
				q, v = model.explicit_step(q, v, H, substeps)
			end
			if finite(q) and math.abs(sag(model, q)) < 10 then
				best = k
			else
				break
			end
		end
		local model = cloth.new({ nx = size, ny = size, spacing = 1 / (size - 1) })
		local q, v = f64.copy(nil, model.q0), f64.copy(nil, model.v0)
		local start = os.clock()
		for _ = 1, 60 do
			q, v = model.explicit_step(q, v, H, substeps)
		end
		local ms = (os.clock() - start) / 60 * 1000
		print(string.format("  %2dx%-2d %2d substeps: k up to %-6d  %6.2f ms/frame", size, size, substeps, best, ms))
	end
end
