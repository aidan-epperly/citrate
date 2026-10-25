--!strict
-- Numerical energy loss of the implicit step (luau -O2 --codegen bench/rational_step.luau):
-- a 30 by 30 cloth with no physical damping, its top edge pinned, released horizontal, so
-- its mechanical energy should stay constant; the energy lost (as a fraction of the drop's
-- potential energy) after 1, 2 and 4 seconds, for backward Euler and the rational step.
local root = require("../src")
local cloth = require("../demos/cloth/cloth")
local f64, ode = root.f64, root.ode
local N = 30
for _, k in { 5000, 1e6 } do
	print(string.format("stiffness %g: energy lost (fraction of the drop's potential energy) after 1, 2, 4 s", k))
	for _, s in { false, 0, 0.5, 1 } :: { any } do
		local model = cloth.new({ nx = N, ny = N, spacing = 1 / (N - 1), k_struct = k, k_shear = k / 5, k_bend = k / 25, pin_top = true, hanging = false, damping = 0, air = 0 })
		local it = ode.implicit_mechanics(f64.space, model.force, 0, model.q0, model.v0, { jacobian = model.block_jacobian, mass = model.mass, fixed = model.fixed, tol = 1e-8, amg = { smooth = false }, rho_inf = if s == false then nil else s } :: any)
		local E0 = model.energy(it.q, it.v)
		-- the drop: the potential energy the cloth would release hanging straight down (about m g L / 2)
		local scale = 9 * 9.81 * 0.5 * (0.01 * 900) / 9
		local out = {}
		for f = 1, 240 do
			it:step(1 / 60)
			if f == 60 or f == 120 or f == 240 then
				table.insert(out, string.format("%6.3f", (E0 - model.energy(it.q, it.v)) / scale))
			end
		end
		print(string.format("  %-22s %s", if s == false then "backward Euler" else string.format("rational rho_inf=%.1f", s), table.concat(out, "  ")))
	end
end
