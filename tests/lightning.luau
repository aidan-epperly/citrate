--!strict

-- The lightning Laplace solver against exact solutions: a smooth one with mixed data, the
-- r^(1/2) singularity where a Dirichlet side meets a Neumann side in a straight line, and
-- Poisson's equation with a solution that depends on one coordinate.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check
local lightning = lib.lightning

local function zero(): number
	return 0
end

test("lightning: smooth solution with mixed data", function()
	local function u(x: number, y: number): number
		return math.exp(x) * math.cos(y)
	end
	local S = lightning.laplace({ { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } }, {
		{ kind = "dirichlet", value = u },
		{ kind = "neumann", value = u }, -- x = 1: du/dn = du/dx
		{ kind = "dirichlet", value = u },
		{ kind = "neumann", value = function(x: number, y: number): number
			return -u(x, y)
		end },
	}, { poles = 8 })
	local worst = 0
	for _, p in { { 0.3, 0.4 }, { 0.9, 0.1 }, { 0.5, 0.95 } } do
		worst = math.max(worst, math.abs(S.eval(p[1], p[2]) - u(p[1], p[2])))
	end
	check(worst < 1e-10, `interior error {worst}`)
	check(S.error < 1e-9, `residual {S.error}`)
	local gx, gy = S.gradient(0.4, 0.6)
	check(math.abs(gx - u(0.4, 0.6)) < 1e-8 and math.abs(gy + math.exp(0.4) * math.sin(0.6)) < 1e-8, "gradient")
end)

test("lightning: r^(1/2) singularity, root-exponential convergence", function()
	local function u(x: number, y: number): number
		local r, t = math.sqrt(x * x + y * y), math.atan2(y, x)
		return math.sqrt(r) * math.sin(t / 2)
	end
	local poly = { { -1, 0 }, { 0, 0 }, { 1, 0 }, { 1, 2 }, { -1, 2 } }
	local conds: { lib.LightningCondition } = {
		{ kind = "neumann", value = zero },
		{ kind = "dirichlet", value = zero },
		{ kind = "dirichlet", value = u },
		{ kind = "dirichlet", value = u },
		{ kind = "dirichlet", value = u },
	}
	local errs = {}
	for i, N in { 8, 16, 32 } do
		local S = lightning.laplace(poly, conds, { poles = N })
		local worst = 0
		for _, p in { { 0.01, 0.01 }, { -0.3, 0.2 }, { 0.5, 1 }, { 1e-4, 1e-4 } } do
			worst = math.max(worst, math.abs(S.eval(p[1], p[2]) - u(p[1], p[2])))
		end
		errs[i] = worst
	end
	-- measured: 2e-3, 9e-7, 1.3e-9
	check(errs[2] < 1e-5 and errs[3] < 1e-7, `errors {errs[1]}, {errs[2]}, {errs[3]}`)
	check(errs[3] < errs[2] / 100, "faster than algebraic")
end)

test("lightning: Poisson's equation and validation", function()
	-- The whole top edge pinned, the rest free: u = (1 - y^2) / 2 solves Laplacian(u) = -1.
	local S = lightning.laplace({ { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } }, {
		{ kind = "neumann", value = zero },
		{ kind = "neumann", value = zero },
		{ kind = "dirichlet", value = zero },
		{ kind = "neumann", value = zero },
	}, { poles = 12, source = -1 })
	check(math.abs(S.eval(0.5, 0) - 0.5) < 1e-10 and math.abs(S.eval(0.2, 0.7) - 0.255) < 1e-10, "one-dimensional solution")
	-- Pinned along the middle half only: converges (the sag is 0.6097242064 to 10 digits).
	local rod = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0.75, 1 }, { 0.25, 1 }, { 0, 1 } }
	local conds = {}
	for i = 1, 6 do
		conds[i] = { kind = if i == 4 then "dirichlet" else "neumann", value = zero }
	end
	local R = lightning.laplace(rod, conds, { poles = 20, source = -1 })
	check(math.abs(R.eval(0.5, 0) - 0.6097242064) < 1e-6, `hanging membrane sag {R.eval(0.5, 0)}`)
	check(not pcall(function()
		lightning.laplace({ { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 } }, conds, nil)
	end), "clockwise vertices rejected")
end)

return nil
