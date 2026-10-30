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
	close(D.volume, 3, 1e-14, "area")
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

test("pde: grids and meshes in any dimension", function()
	-- the table form is the positional one
	local A, B = pde.grid_laplacian(4, 3, 2, { boundary = "neumann" }), pde.grid_laplacian({ 4, 3, 2 }, { boundary = "neumann" })
	local x = f64.create(24)
	for i = 1, 24 do
		f64.set(x, i, math.sin(i))
	end
	local d = f64.sub(nil, f64.sparse_matvec(nil, A, x), f64.sparse_matvec(nil, B, x))
	check(f64.amax(d) == 0, "grid_laplacian({ ... }) is grid_laplacian(nx, ny, nz)")
	-- 4-D: interior rows are 8 / h^2 on the diagonal; Neumann and periodic rows sum to zero
	local L4 = pde.grid_laplacian({ 3, 3, 3, 3 }, { h = 0.5 })
	close(f64.sparse_get(L4, 41, 41), 32, 0, "4-D interior diagonal")
	for _, kind in { "neumann", "periodic" } do
		local S = pde.grid_laplacian({ 3, 2, 4, 3 }, { boundary = kind })
		local r = f64.sparse_matvec(nil, S, f64.from(table.create(72, 1)))
		check(f64.amax(r) == 0, kind .. " rows sum to zero in 4-D")
	end
	-- the Kuhn mesh: d! simplices a cell, filling the box
	for dims = 1, 4 do
		local counts, lengths = {}, {}
		local cells, box = 1, 1
		for a = 1, dims do
			counts[a], lengths[a] = a + 1, 0.5 * a
			cells *= a + 1
			box *= 0.5 * a
		end
		local points, simplices = pde.grid_mesh(counts, lengths)
		local fact = 1
		for a = 2, dims do
			fact *= a
		end
		check(#simplices == fact * cells, string.format("%d-D: %d simplices", dims, #simplices))
		local D = pde.fem(points, simplices)
		close(D.volume, box, 1e-13, string.format("%d-D: volume", dims))
		local r = f64.sparse_matvec(nil, D.stiffness, f64.from(table.create(D.n, 1)))
		check(f64.amax(r) < 1e-12, string.format("%d-D: constants have zero energy", dims))
		local inner = 1
		for a = 1, dims do
			inner *= counts[a] - 1
		end
		check(#D.interior == inner, string.format("%d-D: interior points", dims))
	end
	-- rect_mesh is grid_mesh in 2-D
	local p1, t1 = pde.rect_mesh(3, 2, 2, 1)
	local p2, t2 = pde.grid_mesh({ 3, 2 }, { 2, 1 })
	check(#p1 == #p2 and #t1 == #t2 and p1[12][1] == p2[12][1] and t1[5][3] == t2[5][3], "rect_mesh")
	check(errors(function()
		pde.fem({ { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } }, { { 1, 2, 3, 4 } })
	end, "tetrahedron 1 is flat"), "a flat tetrahedron")
	check(errors(function()
		pde.fem({ { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } }, { { 1, 2, 3, 4 } })
	end, "a simplex needs 2 to 3"), "a simplex too big for its points")
	check(errors(function()
		pde.fem({ { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } }, { { 1, 2, 3 }, { 2, 4 } })
	end, "simplex 2 does not have 3 points"), "mixed simplices")
	check(errors(function()
		pde.fem({ { 0, 0 }, { 1, 0, 0 }, { 0, 1 } }, { { 1, 2, 3 } })
	end, "point 2 is not like point 1"), "mixed points")
end)

test("pde: Poisson's equation on tetrahedra and in 4-D", function()
	-- linear solutions are exact in any dimension
	for dims = 1, 4 do
		local counts = {}
		for a = 1, dims do
			counts[a] = if dims == 4 then 3 else 4
		end
		local D = pde.fem(pde.grid_mesh(counts))
		local function exact(...: number): number
			local v, c = 1, { ... }
			for a = 1, #c do
				v += a * c[a]
			end
			return v
		end
		local u, info = pde.poisson(D, 0, exact)
		local err = 0
		local pt = {}
		for p = 1, D.n do
			for a = 1, dims do
				pt[a] = D.coords[a][p]
			end
			err = math.max(err, math.abs(u[p] - exact(table.unpack(pt))))
		end
		check(err < 1e-12, string.format("%d-D: a linear solution is exact (error %.1e, %s)", dims, err, info.method))
	end
	-- sin(pi x) sin(pi y) sin(pi z) on the unit cube: second order
	local function error_at(n: number): number
		local D = pde.fem(pde.grid_mesh({ n, n, n }))
		local u = pde.poisson(D, function(x: number, y: number, z: number): number
			return 3 * math.pi ^ 2 * math.sin(math.pi * x) * math.sin(math.pi * y) * math.sin(math.pi * z)
		end)
		local Y, Z = D.y, D.z
		local e = 0
		for p = 1, D.n do
			e = math.max(e, math.abs(u[p] - math.sin(math.pi * D.x[p]) * math.sin(math.pi * Y[p]) * math.sin(math.pi * Z[p])))
		end
		return e
	end
	local e1, e2 = error_at(6), error_at(12)
	check(e2 < 0.03 and e1 / e2 > 3.5, string.format("tetrahedra, second order: %.2e then %.2e", e1, e2))
end)

-- An icosahedron's triangles subdivided levels times, on the unit sphere.
local function sphere(levels: number): ({ { number } }, { { number } })
	local t = (1 + math.sqrt(5)) / 2
	local P = {
		{ -1, t, 0 }, { 1, t, 0 }, { -1, -t, 0 }, { 1, -t, 0 },
		{ 0, -1, t }, { 0, 1, t }, { 0, -1, -t }, { 0, 1, -t },
		{ t, 0, -1 }, { t, 0, 1 }, { -t, 0, -1 }, { -t, 0, 1 },
	}
	local T = {
		{ 1, 12, 6 }, { 1, 6, 2 }, { 1, 2, 8 }, { 1, 8, 11 }, { 1, 11, 12 },
		{ 2, 6, 10 }, { 6, 12, 5 }, { 12, 11, 3 }, { 11, 8, 7 }, { 8, 2, 9 },
		{ 4, 10, 5 }, { 4, 5, 3 }, { 4, 3, 7 }, { 4, 7, 9 }, { 4, 9, 10 },
		{ 5, 10, 6 }, { 3, 5, 12 }, { 7, 3, 11 }, { 9, 7, 8 }, { 10, 9, 2 },
	}
	for _ = 1, levels do
		local mids: { [number]: number } = {}
		local function mid(a: number, b: number): number
			local key = math.min(a, b) * 100000 + math.max(a, b)
			if mids[key] == nil then
				table.insert(P, { (P[a][1] + P[b][1]) / 2, (P[a][2] + P[b][2]) / 2, (P[a][3] + P[b][3]) / 2 })
				mids[key] = #P
			end
			return mids[key]
		end
		local T2 = {}
		for _, tri in T do
			local a, b, c = tri[1], tri[2], tri[3]
			local ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
			table.insert(T2, { a, ab, ca })
			table.insert(T2, { b, bc, ab })
			table.insert(T2, { c, ca, bc })
			table.insert(T2, { ab, bc, ca })
		end
		T = T2
	end
	for _, p in P do
		local r = math.sqrt(p[1] ^ 2 + p[2] ^ 2 + p[3] ^ 2)
		p[1], p[2], p[3] = p[1] / r, p[2] / r, p[3] / r
	end
	return P, T
end

test("pde: curves and surfaces", function()
	-- a plane tilted into 3-D gives the same answers as the plane
	local points, triangles = pde.rect_mesh(8, 8)
	local c, s = math.cos(0.7), math.sin(0.7)
	local tilted = {}
	for k, p in points do
		tilted[k] = { p[1], c * p[2], s * p[2] }
	end
	local D2, D3 = pde.fem(points, triangles), pde.fem(tilted, triangles)
	local u2 = pde.poisson(D2, 1, 0)
	local u3 = pde.poisson(D3, 1, 0)
	local diff = 0
	for k = 1, D2.n do
		diff = math.max(diff, math.abs(u2[k] - u3[k]))
	end
	check(diff < 1e-13 and math.abs(D3.volume - 1) < 1e-13, string.format("a tilted plane (difference %.1e)", diff))
	-- a string of length 1 held at its ends: (j pi)^2; as a helix of the same length, the same
	local n = 64
	local line, segments, helix = {}, {}, {}
	for i = 0, n do
		local t = i / n
		line[i + 1] = { t }
		helix[i + 1] = { math.cos(4 * t) / 5, math.sin(4 * t) / 5, t * math.sqrt(1 - (4 / 5) ^ 2) }
		if i > 0 then
			table.insert(segments, { i, i + 1 })
		end
	end
	local l1 = pde.modes(pde.fem(line, segments), 3)
	local Dh = pde.fem(helix, segments)
	local lh = pde.modes(Dh, 3)
	-- the helix's chords are all the same length: the straight string scaled by its length
	local worst, gap = 0, 0
	for j = 1, 3 do
		worst = math.max(worst, math.abs(l1[j] / (j * math.pi) ^ 2 - 1))
		gap = math.max(gap, math.abs(lh[j] * Dh.volume ^ 2 / l1[j] - 1))
	end
	check(worst < 2e-3 and gap < 1e-10 and #Dh.boundary == 2, string.format("a string, straight (%.1e) and as a helix (%.1e)", worst, gap))
	-- the unit sphere: no boundary; eigenvalues l (l + 1), 2 l + 1 of each, second order
	local errs = {}
	for i, levels in { 3, 4 } do
		local D = pde.fem(sphere(levels))
		check(#D.boundary == 0, "a closed surface has no boundary")
		local values = pde.modes(D, 9)
		local e = math.abs(values[1])
		for j = 2, 4 do
			e = math.max(e, math.abs(values[j] - 2))
		end
		for j = 5, 9 do
			e = math.max(e, math.abs(values[j] - 6))
		end
		errs[i] = e
	end
	check(errs[2] < 0.02 and errs[1] / errs[2] > 3.5, string.format("the sphere's eigenvalues, second order: %.2e then %.2e", errs[1], errs[2]))
	check(errors(function()
		pde.poisson(pde.fem(sphere(1)), 1)
	end, "no boundary"), "Poisson on a closed surface")
end)

return nil
