--!strict

-- The user-facing PDE functions against exact answers: grid Laplacians (their stencils,
-- boundaries and eigenvalues), meshes, Poisson's equation (exact for linear solutions,
-- second-order otherwise), heat flow's decay and steady state, the unit square's vibration
-- modes, and the errors for bad input.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local pde, linalg, f64 = lib.pde, lib.linalg, lib.f64

local function errors(f: () -> (), pattern: string): boolean
	local ok, message = pcall(f)
	return not ok and string.find(tostring(message), pattern, 1, true) ~= nil
end

test("pde: grid Laplacians", function()
	-- 1-D Dirichlet: tridiag(-1, 2, -1) / h^2
	local L = pde.grid_laplacian(5, nil, nil, { h = 0.5 })
	check(f64.sparse_get(L, 3, 3) == 8 and f64.sparse_get(L, 3, 4) == -4 and f64.sparse_get(L, 1, 1) == 8 and L.symmetric == true, "1-D stencil")
	-- 2-D Dirichlet: the smallest eigenvalue is the sum of the 1-D ones
	local nx, ny = 12, 9
	local L2 = pde.grid_laplacian(nx, ny)
	local values = linalg.eig(L2, { count = 1 })
	local exact = 4 * math.sin(math.pi / (2 * (nx + 1))) ^ 2 + 4 * math.sin(math.pi / (2 * (ny + 1))) ^ 2
	close(values[1], exact, 1e-10, "2-D smallest eigenvalue")
	-- Neumann and periodic: every row sums to zero (constants are in the null space)
	for _, kind in { "neumann", "periodic" } do
		local S = pde.grid_laplacian(4, 3, 2, { boundary = kind })
		local ones = f64.from(table.create(24, 1))
		local r = f64.sparse_matvec(nil, S, ones)
		check(f64.amax(r) < 1e-14 and S.rows == 24, kind .. ": rows sum to zero")
	end
	-- periodic: the corner point's neighbors wrap around
	local P = pde.grid_laplacian(4, 4, nil, { boundary = "periodic" })
	check(f64.sparse_get(P, 1, 4) == -1 and f64.sparse_get(P, 1, 13) == -1 and f64.sparse_get(P, 1, 1) == 4, "periodic wrap")
	-- 3-D Dirichlet: the 7-point stencil
	local L3 = pde.grid_laplacian(3, 3, 3)
	check(f64.sparse_get(L3, 14, 14) == 6 and f64.sparse_get(L3, 14, 5) == -1, "3-D stencil")
	check(errors(function()
		pde.grid_laplacian(0)
	end, "positive whole numbers"), "bad sizes")
	check(errors(function()
		pde.grid_laplacian(3, 3, 1, { boundary = "robin" })
	end, "boundary must be"), "bad boundary")
end)

test("pde: meshes and their assembly", function()
	local points, triangles = pde.rect_mesh(4, 3, 2, 1.5)
	check(#points == 20 and #triangles == 24, "rect_mesh sizes")
	local D = pde.fem(points, triangles)
	close(D.area, 3, 1e-14, "area")
	local total = 0
	for _, m in D.mass do
		total += m
	end
	close(total, 3, 1e-14, "the mass adds up to the area")
	check(#D.boundary == 2 * (4 + 3) and #D.interior == 3 * 2, "boundary and interior points")
	local r = f64.sparse_matvec(nil, D.stiffness, f64.from(table.create(D.n, 1)))
	check(f64.amax(r) < 1e-13, "constants have zero energy")
	-- vectors as points
	local vp = {}
	for k, p in points do
		vp[k] = vector.create(p[1], p[2], 0)
	end
	check(pde.fem(vp, triangles).n == 20, "points as vectors")
	check(errors(function()
		pde.fem({ { 0, 0 }, { 1, 0 }, { 2, 0 } }, { { 1, 2, 3 } })
	end, "zero area"), "a flat triangle")
	check(errors(function()
		pde.fem({ { 0, 0 }, { 1, 0 }, { 0, 1 }, { 5, 5 } }, { { 1, 2, 3 } })
	end, "belongs to no triangle"), "an unused point")
	check(errors(function()
		pde.fem({ { 0, 0 }, { 1, 0 }, { 0, 1 } }, { { 1, 2, 4 } })
	end, "not one of the 3 points"), "a bad index")
end)

test("pde: Poisson's equation", function()
	-- a linear solution is exact
	local D = pde.fem(pde.rect_mesh(6, 5))
	local exact = function(x: number, y: number): number
		return 1 + 2 * x - 3 * y
	end
	local u, info = pde.poisson(D, 0, exact)
	local err = 0
	for k = 1, D.n do
		err = math.max(err, math.abs(u[k] - exact(D.x[k], D.y[k])))
	end
	check(err < 1e-12 and info.method == "sparse Cholesky", string.format("linear solution (error %.1e, %s)", err, info.method))
	-- sin(pi x) sin(pi y): second order
	local function error_at(n: number): number
		local Dn = pde.fem(pde.rect_mesh(n, n))
		local un = pde.poisson(Dn, function(x: number, y: number): number
			return 2 * math.pi ^ 2 * math.sin(math.pi * x) * math.sin(math.pi * y)
		end)
		local e = 0
		for k = 1, Dn.n do
			e = math.max(e, math.abs(un[k] - math.sin(math.pi * Dn.x[k]) * math.sin(math.pi * Dn.y[k])))
		end
		return e
	end
	local e1, e2 = error_at(8), error_at(16)
	check(e2 < 0.01 and e1 / e2 > 3.5, string.format("second order: %.2e then %.2e", e1, e2))
	-- tables of values
	local f = table.create(D.n, 1)
	local _, info2 = pde.poisson(D, f, 0)
	check((info2.residual :: number) < 1e-12, "f as a table")
	check(errors(function()
		pde.poisson(D, { 1, 2 }, 0)
	end, "has 2 values where the mesh has"), "a short table")
	check(errors(function()
		pde.poisson(pde.fem({ { 0, 0 }, { 1, 0 }, { 0, 1 } }, { { 1, 2, 3 } }), 1, 0)
	end, "nothing to solve for"), "a mesh with no interior")
end)

test("pde: heat flow", function()
	local D = pde.fem(pde.rect_mesh(16, 16))
	-- sin(pi x) sin(pi y) decays as exp(-2 pi^2 t)
	local H = pde.heat(D, function(x: number, y: number): number
		return math.sin(math.pi * x) * math.sin(math.pi * y)
	end, { dt = 1e-3 })
	H:step(50)
	close(H.t, 0.05, 1e-12, "time")
	local center = 0
	for k = 1, D.n do
		if math.abs(D.x[k] - 0.5) < 1e-9 and math.abs(D.y[k] - 0.5) < 1e-9 then
			center = H.u[k]
		end
	end
	local expect = math.exp(-2 * math.pi ^ 2 * 0.05)
	check(math.abs(center - expect) < 0.03 * expect, string.format("decay: %.4f against %.4f", center, expect))
	-- held at 1 on the boundary, it fills up to 1
	local H2 = pde.heat(D, 0, { dt = 0.05, boundary = 1, kappa = 2 })
	H2:step(100)
	local lowest = math.huge
	for k = 1, D.n do
		lowest = math.min(lowest, H2.u[k])
	end
	check(math.abs(lowest - 1) < 1e-6, "steady state")
	check(errors(function()
		pde.heat(D, 0, {} :: any)
	end, "dt"), "no time step")
end)

test("pde: vibration modes", function()
	-- the unit square: pi^2 (a^2 + b^2): 2, 5, 5, 8 times pi^2
	for _, n in { 20, 40 } do
		local D = pde.fem(pde.rect_mesh(n, n))
		local values, modes, info = pde.modes(D, 4)
		local exact = { 2, 5, 5, 8 }
		local worst = 0
		for k = 1, 4 do
			worst = math.max(worst, math.abs(values[k] / (exact[k] * math.pi ^ 2) - 1))
		end
		check(worst < 0.03, string.format("n = %d: eigenvalues within %.1f%% (%s)", n, 100 * worst, info.method))
		-- normalized: the integral of u^2 is 1, zero on the boundary, positive at its peak
		local norm, edge = 0, 0
		for k = 1, D.n do
			norm += D.mass[k] * modes[1][k] ^ 2
		end
		for _, k in D.boundary do
			edge = math.max(edge, math.abs(modes[1][k]))
		end
		close(norm, 1, 1e-10, "normalized")
		check(edge == 0, "zero on the boundary")
	end
end)

return nil
