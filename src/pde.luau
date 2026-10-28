--!strict

-- Partial differential equations made simple: the Laplacian on grids, and Poisson's
-- equation, heat flow and vibration modes on triangle meshes. Real numbers only.
--
-- Two ways in:
--   pde.grid_laplacian(nx, ny, nz)  the Laplacian on a grid as a sparse matrix, to use with
--                                   linalg (or the library's sparse solvers) directly;
--   pde.fem(points, triangles)      a domain from a triangle mesh (pde.rect_mesh makes a
--                                   simple one), for pde.poisson, pde.heat and pde.modes.
-- On a mesh, values live at the points: a field is a table with one number per point, and
-- the boundary (the points on edges that belong to only one triangle) is held at given
-- values. Data can be given as a number (the same everywhere), a function of (x, y), or a
-- table of one number per point.
--
-- Nothing here is a new algorithm: the meshes are assembled with linear finite elements
-- and the equations solved by linalg and the sparse solvers. These functions favor safety
-- over speed: they check their input and report what they did.

local types = require("./types")
local f64 = require("./f64")
local linalg = require("./linalg")

type Sparse = types.Sparse

local function fail(who: string, message: string): never
	error("pde." .. who .. ": " .. message, 3)
end

local function whole(x: any): boolean
	return type(x) == "number" and x >= 1 and x % 1 == 0
end

-- The Laplacian on a grid

export type GridOptions = { h: number?, boundary: string? }

-- The matrix of -(the Laplacian) on an nx by ny by nz grid of points spaced h apart
-- (ny and nz default to 1: a line or a plane): row k is (2 d u_k - the sum of its 2 d
-- neighbors) / h^2 in d dimensions, the 3-, 5- or 7-point stencil. Points are numbered with
-- x fastest: point (i, j, k) is i + nx (j - 1) + nx ny (k - 1). options.boundary says what
-- lies past the edges:
--   "dirichlet" (the default)  zero: the matrix is symmetric positive definite;
--   "neumann"                  no flow out: each row counts only the neighbors it has (the
--                              matrix is singular: adding a constant changes nothing);
--   "periodic"                 the grid wraps around (singular too).
-- Returns a sparse matrix marked symmetric.
local function grid_laplacian(nx: number, ny: number?, nz: number?, options: GridOptions?): Sparse
	local o: GridOptions = options or {}
	local my, mz = ny or 1, nz or 1
	if not (whole(nx) and whole(my) and whole(mz)) then
		fail("grid_laplacian", "the grid sizes must be positive whole numbers")
	end
	local h = o.h or 1
	if not (h > 0) then
		fail("grid_laplacian", "the spacing h must be positive")
	end
	local kind = o.boundary or "dirichlet"
	if kind ~= "dirichlet" and kind ~= "neumann" and kind ~= "periodic" then
		fail("grid_laplacian", "boundary must be \"dirichlet\", \"neumann\" or \"periodic\"")
	end
	local sizes = { nx, my, mz }
	local strides = { 1, nx, nx * my }
	local N = nx * my * mz
	local w = 1 / (h * h)
	local ii, jj, vv = {}, {}, {}
	for p = 0, N - 1 do
		local coords = { p % nx, (p // nx) % my, p // (nx * my) }
		local diag = 0
		for axis = 1, 3 do
			local m = sizes[axis]
			if m > 1 or kind == "dirichlet" and (nx > 1 and axis == 1 or my > 1 and axis == 2 or mz > 1 and axis == 3) then
				for _, step in { -1, 1 } do
					local c = coords[axis] + step
					local inside = c >= 0 and c < m
					if kind == "periodic" and not inside then
						c = (c + m) % m
						inside = m > 1
					end
					if inside then
						local q = p + (c - coords[axis]) * strides[axis]
						if q ~= p then
							table.insert(ii, q + 1)
							table.insert(jj, p + 1)
							table.insert(vv, -w)
							diag += w
						end
					elseif kind == "dirichlet" then
						diag += w
					end
				end
			end
		end
		table.insert(ii, p + 1)
		table.insert(jj, p + 1)
		table.insert(vv, diag)
	end
	local S = f64.sparse(N, N, ii, jj, f64.from(vv))
	S.symmetric = true
	return S
end

-- Meshes

-- A rectangle width by height (default 1 by 1) cut into nx by ny cells, each cut into two
-- triangles: the points ({ x, y }, row by row from the corner (0, 0)) and the triangles
-- ({ a, b, c }, indices of points), for pde.fem.
local function rect_mesh(nx: number, ny: number, width: number?, height: number?): ({ { number } }, { { number } })
	if not (whole(nx) and whole(ny)) then
		fail("rect_mesh", "the numbers of cells must be positive whole numbers")
	end
	local wd, ht = width or 1, height or 1
	if not (wd > 0 and ht > 0) then
		fail("rect_mesh", "the width and height must be positive")
	end
	local points, triangles = {}, {}
	for j = 0, ny do
		for i = 0, nx do
			table.insert(points, { wd * i / nx, ht * j / ny })
		end
	end
	local function id(i: number, j: number): number
		return j * (nx + 1) + i + 1
	end
	for j = 0, ny - 1 do
		for i = 0, nx - 1 do
			table.insert(triangles, { id(i, j), id(i + 1, j), id(i + 1, j + 1) })
			table.insert(triangles, { id(i, j), id(i + 1, j + 1), id(i, j + 1) })
		end
	end
	return points, triangles
end

export type Domain = {
	n: number, -- the number of points
	x: { number }, -- their coordinates
	y: { number },
	triangles: { { number } },
	area: number,
	boundary: { number }, -- the points on the boundary
	interior: { number }, -- the others (the unknowns)
	stiffness: Sparse, -- K: u^T K u is the integral of |grad u|^2 (all points)
	mass: { number }, -- the lumped mass: each point's share of the area
	[string]: any,
}

-- A domain for pde.poisson, pde.heat and pde.modes from a triangle mesh: points (each
-- { x, y } or a vector) and triangles (each { a, b, c }, indices of points). Assembled with
-- linear finite elements: on a triangle of area A with corners p1, p2, p3, the stiffness is
-- (b_s b_t + c_s c_t) / (4 A) with b_s = y_u - y_w and c_s = x_w - x_u ((s, u, w) cyclic),
-- and each corner gets A / 3 of the (lumped) mass. Errors on a triangle of zero area, an
-- index outside the points, or a point no triangle uses.
local function fem(points: { any }, triangles: { { number } }): Domain
	local n = #points
	if n < 3 or #triangles < 1 then
		fail("fem", "need at least 3 points and 1 triangle")
	end
	local x, y = table.create(n, 0) :: { number }, table.create(n, 0) :: { number }
	for k, p in points do
		local px: any, py: any
		if type(p) == "vector" then
			px, py = (p :: any).x, (p :: any).y
		elseif type(p) == "table" then
			px, py = p[1], p[2]
		end
		if type(px) ~= "number" or type(py) ~= "number" or px ~= px or py ~= py then
			fail("fem", string.format("point %d is not { x, y } with numbers", k))
		end
		x[k], y[k] = px, py
	end
	local used = table.create(n, false)
	local edges: { [number]: number } = {} -- edge (a, b), a < b, as a * (n + 1) + b: its triangles
	local ii, jj, vv = {}, {}, {}
	local mass = table.create(n, 0) :: { number }
	local area = 0
	for t, tri in triangles do
		local p = { tri[1], tri[2], tri[3] }
		for s = 1, 3 do
			if not whole(p[s]) or p[s] > n then
				fail("fem", string.format("triangle %d refers to point %s, not one of the %d points", t, tostring(p[s]), n))
			end
			used[p[s]] = true
		end
		local A = ((x[p[2]] - x[p[1]]) * (y[p[3]] - y[p[1]]) - (x[p[3]] - x[p[1]]) * (y[p[2]] - y[p[1]])) / 2
		A = math.abs(A)
		if not (A > 0) then
			fail("fem", string.format("triangle %d has zero area", t))
		end
		area += A
		local b, c = {}, {}
		for s = 1, 3 do
			local u, w = p[s % 3 + 1], p[(s + 1) % 3 + 1]
			b[s], c[s] = y[u] - y[w], x[w] - x[u]
		end
		for s = 1, 3 do
			mass[p[s]] += A / 3
			for t2 = 1, 3 do
				table.insert(ii, p[s])
				table.insert(jj, p[t2])
				table.insert(vv, (b[s] * b[t2] + c[s] * c[t2]) / (4 * A))
			end
			local a1, a2 = p[s], p[s % 3 + 1]
			local key = math.min(a1, a2) * (n + 1) + math.max(a1, a2)
			edges[key] = (edges[key] or 0) + 1
		end
	end
	for k = 1, n do
		if not used[k] then
			fail("fem", string.format("point %d belongs to no triangle", k))
		end
	end
	-- the boundary: the points on edges of only one triangle
	local on_boundary = table.create(n, false)
	for key, count in edges do
		if count == 1 then
			on_boundary[key // (n + 1)] = true
			on_boundary[key % (n + 1)] = true
		end
	end
	local boundary, interior = {}, {}
	local index = table.create(n, 0) -- a point's place among the unknowns (0 on the boundary)
	for k = 1, n do
		if on_boundary[k] then
			table.insert(boundary, k)
		else
			table.insert(interior, k)
			index[k] = #interior
		end
	end
	local K = f64.sparse(n, n, ii, jj, f64.from(vv))
	K.symmetric = true
	-- the stiffness between unknowns only
	local ki, kj, kv = {}, {}, {}
	for e = 1, #ii do
		local a, c = index[ii[e]], index[jj[e]]
		if a > 0 and c > 0 then
			table.insert(ki, a)
			table.insert(kj, c)
			table.insert(kv, vv[e])
		end
	end
	local m = #interior
	local Kii: Sparse? = nil
	if m > 0 then
		Kii = f64.sparse(m, m, ki, kj, f64.from(kv))
		;(Kii :: Sparse).symmetric = true
	end
	return {
		n = n,
		x = x,
		y = y,
		triangles = triangles,
		area = area,
		boundary = boundary,
		interior = interior,
		stiffness = K,
		mass = mass,
		_index = index,
		_Kii = Kii,
	}
end

-- Data given as a number, a function of (x, y) or a table of one number per point, as a
-- table of values at the points.
local function field(D: Domain, data: any, who: string, what: string): { number }
	local out = table.create(D.n, 0) :: { number }
	if data == nil then
		return out
	elseif type(data) == "number" then
		for k = 1, D.n do
			out[k] = data
		end
	elseif type(data) == "function" then
		for k = 1, D.n do
			local v = (data :: (number, number) -> any)(D.x[k], D.y[k])
			if type(v) ~= "number" or v ~= v then
				fail(who, string.format("%s returned %s at point %d, not a number", what, tostring(v), k))
			end
			out[k] = v
		end
	elseif type(data) == "table" then
		if #data ~= D.n then
			fail(who, string.format("%s has %d values where the mesh has %d points", what, #data, D.n))
		end
		for k = 1, D.n do
			out[k] = data[k]
		end
	else
		fail(who, what .. " must be a number, a function of (x, y) or a table of one number per point")
	end
	return out
end

local function unknowns(D: Domain, who: string): Sparse
	local K = D._Kii
	if K == nil then
		fail(who, "every point of the mesh is on its boundary: there is nothing to solve for")
	end
	return K :: Sparse
end

-- The right-hand side for the unknowns: mass-weighted f, less the boundary values' pull
-- (K's columns at boundary points times their values).
local function load_vector(D: Domain, f: { number }, g: { number }): { number }
	local gb = f64.create(D.n)
	for _, k in D.boundary do
		f64.set(gb, k, g[k])
	end
	local Kg = f64.sparse_matvec(nil, D.stiffness, gb)
	local r = table.create(#D.interior, 0) :: { number }
	for a, k in D.interior do
		r[a] = D.mass[k] * f[k] - f64.get(Kg, k)
	end
	return r
end

-- Poisson's equation -(the Laplacian of u) = f on a mesh domain, with u = g on the boundary
-- (f and g: numbers, functions of (x, y) or tables per point; both default to 0). Returns u
-- (a table, one value per point) and the solver's info (linalg.solve).
local function poisson(D: Domain, f: any, g: any): ({ number }, linalg.Info)
	local K = unknowns(D, "poisson")
	local fv = field(D, f, "poisson", "f")
	local gv = field(D, g, "poisson", "the boundary values")
	local uI, info = linalg.solve(K, load_vector(D, fv, gv))
	local u = table.clone(gv)
	for a, k in D.interior do
		u[k] = uI[a]
	end
	return u, info
end

export type HeatOptions = { dt: number, kappa: number?, boundary: any }
export type Heat = { t: number, u: { number }, dt: number, step: (self: Heat, count: number?) -> Heat, [string]: any }

-- Heat flow du/dt = kappa (the Laplacian of u) on a mesh domain from u0 (a number, a
-- function of (x, y) or a table per point), with u held at options.boundary on the
-- boundary (default: u0's boundary values) and diffusivity options.kappa (default 1).
-- Returns a stepper: heat:step(count) advances count steps of options.dt (default 1 step);
-- heat.u is the field and heat.t the time. Backward Euler, stable for any dt (large steps
-- smooth more than the true flow would); its matrix M + dt kappa K is factored once
-- (sparse Cholesky) and reused every step.
local function heat(D: Domain, u0: any, options: HeatOptions): Heat
	local o = options
	if type(o) ~= "table" or type(o.dt) ~= "number" or not (o.dt > 0) then
		fail("heat", "options.dt (the time step) must be a positive number")
	end
	local kappa = o.kappa or 1
	if not (kappa > 0) then
		fail("heat", "kappa must be positive")
	end
	local K = unknowns(D, "heat")
	local u = field(D, u0, "heat", "u0")
	local g = if o.boundary ~= nil then field(D, o.boundary, "heat", "the boundary values") else table.clone(u)
	for _, k in D.boundary do
		u[k] = g[k]
	end
	-- A = M + dt kappa K on the unknowns (K's pattern holds its diagonal)
	local m = K.rows
	local values = f64.scale(nil, o.dt * kappa, K.values)
	for a = 1, m do
		local k = D.interior[a]
		for p = buffer.readu32(K.colptr, 4 * (a - 1)), buffer.readu32(K.colptr, 4 * a) - 1 do
			if buffer.readu32(K.rowind, 4 * p) == a - 1 then
				f64.set(values, p + 1, f64.get(values, p + 1) + D.mass[k])
			end
		end
	end
	local A: Sparse = { rows = m, cols = m, colptr = K.colptr, rowind = K.rowind, values = values, symmetric = true }
	local F = f64.sparse_chol(A)
	if not F.definite then
		fail("heat", "the step's matrix is not positive definite (is the mesh valid?)")
	end
	-- the boundary's pull on each step: dt kappa K_IB g
	local zero = table.create(D.n, 0) :: { number }
	local pull = load_vector(D, zero, g)
	local rhs = f64.create(m)
	local sol = f64.create(m)
	local heat: any = { t = 0, u = u, dt = o.dt, method = "backward Euler, sparse Cholesky" }
	function heat.step(self: any, count: number?): any
		for _ = 1, count or 1 do
			for a = 1, m do
				local k = D.interior[a]
				f64.set(rhs, a, D.mass[k] * self.u[k] + o.dt * kappa * pull[a])
			end
			f64.sparse_chol_solve(sol, F, rhs)
			for a = 1, m do
				self.u[D.interior[a]] = f64.get(sol, a)
			end
			self.t += o.dt
		end
		return self
	end
	return heat
end

-- The count (default 6) lowest vibration modes of a mesh domain held at zero on its
-- boundary: -(the Laplacian of u) = lambda u, the drum's equation. Returns the eigenvalues
-- lambda (ascending; a mode's frequency is proportional to sqrt(lambda)), the modes (tables
-- of one value per point, zero on the boundary, scaled so the integral of u^2 is 1) and the
-- solver's info (linalg.eig on M^-1/2 K M^-1/2).
local function modes(D: Domain, count: number?): ({ number }, { { number } }, linalg.Info)
	local K = unknowns(D, "modes")
	local m = K.rows
	local k = math.min(count or 6, m)
	if not whole(k) then
		fail("modes", "count must be a positive whole number")
	end
	-- A = M^-1/2 K M^-1/2 (the lumped mass is diagonal)
	local dinv = table.create(m, 0) :: { number }
	for a = 1, m do
		dinv[a] = 1 / math.sqrt(D.mass[D.interior[a]])
	end
	local values = f64.create(f64.length(K.values))
	for j = 0, m - 1 do
		for p = buffer.readu32(K.colptr, 4 * j), buffer.readu32(K.colptr, 4 * j + 4) - 1 do
			local i = buffer.readu32(K.rowind, 4 * p)
			f64.set(values, p + 1, f64.get(K.values, p + 1) * dinv[i + 1] * dinv[j + 1])
		end
	end
	local A: Sparse = { rows = m, cols = m, colptr = K.colptr, rowind = K.rowind, values = values, symmetric = true }
	local lambdas, vectors, info = linalg.eig(A, { count = k })
	local modes = {}
	for r, v in vectors do
		local u = table.create(D.n, 0) :: { number }
		local big, at = 0, 0
		for a = 1, m do
			local val = (if type(v) == "buffer" then f64.get(v, a) else v[a]) * dinv[a]
			u[D.interior[a]] = val
			if math.abs(val) > big then
				big, at = math.abs(val), D.interior[a]
			end
		end
		-- the sign: positive at its largest value
		if at > 0 and u[at] < 0 then
			for i = 1, D.n do
				u[i] = -u[i]
			end
		end
		modes[r] = u
	end
	return lambdas, modes, info
end

return {
	grid_laplacian = grid_laplacian,
	rect_mesh = rect_mesh,
	fem = fem,
	poisson = poisson,
	heat = heat,
	modes = modes,
}
