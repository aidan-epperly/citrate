--!strict
--!native
--!optimize 2

-- Partial differential equations made simple: the Laplacian on grids, and Poisson's
-- equation, heat flow and vibration modes on meshes of simplices of any dimension
-- (segments, triangles, tetrahedra, ...), including curves and surfaces in a larger space.
-- Real numbers only.
--
-- Two ways in:
--   pde.grid_laplacian(sizes)  the Laplacian on a grid of any dimension as a sparse matrix,
--                              to use with linalg (or the library's sparse solvers) directly;
--   pde.fem(points, simplices) a domain from a mesh (pde.grid_mesh makes a box's), for
--                              pde.poisson, pde.heat and pde.modes.
-- On a mesh, values live at the points: a field is a table with one number per point, and
-- the boundary (the points on faces that belong to only one simplex) is held at given
-- values. Data can be given as a number (the same everywhere), a function of the
-- coordinates (x, y, ...), or a table of one number per point.
--
-- Nothing here is a new algorithm: the meshes are assembled with linear finite elements
-- and the equations solved by linalg and the sparse solvers. These functions favor safety
-- over speed: they check their input and report what they did.

local types = require("./types")
local f64 = require("./f64")
local linalg = require("./linalg")

type Sparse = types.Sparse

local EPS = 2 ^ -52

local function fail(who: string, message: string): never
	error("pde." .. who .. ": " .. message, 3)
end

local function whole(x: any): boolean
	return type(x) == "number" and x >= 1 and x % 1 == 0
end

-- The Laplacian on a grid

export type GridOptions = { h: number?, boundary: string? }

-- The matrix of -(the Laplacian) on a grid of points spaced h apart, in any number of
-- dimensions: grid_laplacian({ n1, n2, ..., nd }, options), or grid_laplacian(nx, ny, nz,
-- options) for up to three (ny and nz default to 1). Row k is (2 d u_k - the sum of its 2 d
-- neighbors) / h^2, counting only the axes with more than one point: the 3-, 5-, 7-, ...
-- point stencil. Points are numbered with the first axis fastest: point (i1, ..., id) is
-- 1 + (i1 - 1) + n1 (i2 - 1) + n1 n2 (i3 - 1) + .... options.boundary says what lies past
-- the edges:
--   "dirichlet" (the default)  zero: the matrix is symmetric positive definite;
--   "neumann"                  no flow out: each row counts only the neighbors it has (the
--                              matrix is singular: adding a constant changes nothing);
--   "periodic"                 the grid wraps around (singular too).
-- Returns a sparse matrix marked symmetric.
local function grid_laplacian(n1: number | { number }, ny: (number | GridOptions)?, nz: number?, options: GridOptions?): Sparse
	local sizes: { number }
	local o: GridOptions
	if type(n1) == "table" then
		sizes = n1
		o = (ny :: any) or {}
	else
		sizes = { n1, (ny :: number?) or 1, nz or 1 }
		o = options or ({} :: any)
	end
	local dims = #sizes
	if dims < 1 then
		fail("grid_laplacian", "the grid needs at least one size")
	end
	for a = 1, dims do
		if not whole(sizes[a]) then
			fail("grid_laplacian", "the grid sizes must be positive whole numbers")
		end
	end
	local h = o.h or 1
	if not (h > 0) then
		fail("grid_laplacian", "the spacing h must be positive")
	end
	local kind = o.boundary or "dirichlet"
	if kind ~= "dirichlet" and kind ~= "neumann" and kind ~= "periodic" then
		fail("grid_laplacian", "boundary must be \"dirichlet\", \"neumann\" or \"periodic\"")
	end
	local strides = table.create(dims, 1) :: { number }
	for a = 2, dims do
		strides[a] = strides[a - 1] * sizes[a - 1]
	end
	local N = strides[dims] * sizes[dims]
	local w = 1 / (h * h)
	local ii, jj, vv = {}, {}, {}
	local coords = table.create(dims, 0) :: { number } -- point p's position, counted from 0
	for p = 0, N - 1 do
		local diag = 0
		for axis = 1, dims do
			local m = sizes[axis]
			if m > 1 then
				local ca = coords[axis]
				for step = -1, 1, 2 do
					local c = ca + step
					local inside = c >= 0 and c < m
					if kind == "periodic" and not inside then
						c = (c + m) % m
						inside = true
					end
					if inside then
						local q = p + (c - ca) * strides[axis]
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
		-- the next point's position
		for a = 1, dims do
			coords[a] += 1
			if coords[a] < sizes[a] then
				break
			end
			coords[a] = 0
		end
	end
	local S = f64.sparse(N, N, ii, jj, f64.from(vv))
	S.symmetric = true
	return S
end

-- Meshes

-- A box lengths[1] by lengths[2] by ... (default 1 on each axis) in any number of
-- dimensions, cut into counts[1] by counts[2] by ... cells, each cut into d! simplices (the
-- Kuhn triangulation: one per order of the axes, from the cell's lowest corner to its
-- highest by unit steps along the axes in that order; in 2-D, two triangles, in 3-D six
-- tetrahedra). Returns the points (tables of d numbers, the first axis fastest) and the
-- simplices (tables of d + 1 indices of points), for pde.fem.
local function grid_mesh(counts: { number }, lengths: { number }?): ({ { number } }, { { number } })
	local d = #counts
	if d < 1 then
		fail("grid_mesh", "need at least one count")
	end
	for a = 1, d do
		if not whole(counts[a]) then
			fail("grid_mesh", "the numbers of cells must be positive whole numbers")
		end
		local L = if lengths then lengths[a] else 1
		if type(L) ~= "number" or not (L > 0) then
			fail("grid_mesh", "the lengths must be positive numbers, one per axis")
		end
	end
	local strides = table.create(d, 1) :: { number }
	for a = 2, d do
		strides[a] = strides[a - 1] * (counts[a - 1] + 1)
	end
	local npoints = strides[d] * (counts[d] + 1)
	local points: { { number } } = table.create(npoints) :: any
	local at = table.create(d, 0) :: { number }
	for p = 1, npoints do
		local pt = table.create(d, 0) :: { number }
		for a = 1, d do
			pt[a] = (if lengths then lengths[a] else 1) * at[a] / counts[a]
		end
		points[p] = pt
		for a = 1, d do
			at[a] += 1
			if at[a] <= counts[a] then
				break
			end
			at[a] = 0
		end
	end
	-- the orders of the axes (Heap's algorithm), as offsets of the successive corners
	local orders: { { number } } = {}
	local perm = table.create(d, 0) :: { number }
	for a = 1, d do
		perm[a] = a
	end
	local function record()
		local offs = table.create(d, 0) :: { number }
		local off = 0
		for s = 1, d do
			off += strides[perm[s]]
			offs[s] = off
		end
		table.insert(orders, offs)
	end
	local c = table.create(d, 1) :: { number }
	record()
	local i = 1
	while i <= d do
		if c[i] < i then
			local j = if i % 2 == 1 then 1 else c[i]
			perm[j], perm[i] = perm[i], perm[j]
			record()
			c[i] += 1
			i = 1
		else
			c[i] = 1
			i += 1
		end
	end
	local simplices = {}
	for a = 1, d do
		at[a] = 0
	end
	local ncells = 1
	for a = 1, d do
		ncells *= counts[a]
	end
	for _ = 1, ncells do
		local base = 1
		for a = 1, d do
			base += at[a] * strides[a]
		end
		for _, offs in orders do
			local s = table.create(d + 1, base) :: { number }
			for t = 1, d do
				s[t + 1] = base + offs[t]
			end
			table.insert(simplices, s)
		end
		for a = 1, d do
			at[a] += 1
			if at[a] < counts[a] then
				break
			end
			at[a] = 0
		end
	end
	return points, simplices
end

-- A rectangle width by height (default 1 by 1) cut into nx by ny cells, each cut into two
-- triangles: grid_mesh({ nx, ny }, { width, height }).
local function rect_mesh(nx: number, ny: number, width: number?, height: number?): ({ { number } }, { { number } })
	if not (whole(nx) and whole(ny)) then
		fail("rect_mesh", "the numbers of cells must be positive whole numbers")
	end
	local wd, ht = width or 1, height or 1
	if not (wd > 0 and ht > 0) then
		fail("rect_mesh", "the width and height must be positive")
	end
	return grid_mesh({ nx, ny }, { wd, ht })
end

export type Domain = {
	n: number, -- the number of points
	d: number, -- the number of coordinates each point has
	k: number, -- the dimension of the simplices (1 segments, 2 triangles, 3 tetrahedra, ...)
	coords: { { number } }, -- coords[a][p]: coordinate a of point p
	x: { number }, -- coords[1], coords[2], coords[3] (those there are)
	y: { number }, -- (nil with fewer coordinates)
	z: { number },
	simplices: { { number } },
	volume: number, -- the mesh's total length, area, volume, ...
	boundary: { number }, -- the points on the boundary
	interior: { number }, -- the others (the unknowns)
	stiffness: Sparse, -- K: u^T K u is the integral of |grad u|^2 (all points)
	mass: { number }, -- the lumped mass: each point's share of the volume
	[string]: any,
}

-- A domain for pde.poisson, pde.heat and pde.modes from a mesh of simplices of any
-- dimension k (segments, triangles, tetrahedra, ...) whose points have any number d >= k of
-- coordinates: points are tables of d numbers or vectors (d = 3), and simplices tables of
-- k + 1 indices of points. With k < d the mesh is a curve or surface in a larger space, and
-- the Laplacian is the surface's own (Laplace-Beltrami): triangles in 3-D give heat flow and
-- vibration on a surface. Assembled with linear finite elements: on a simplex with
-- vertices p0, ..., pk, E = [p1 - p0, ..., pk - p0] and G = E^T E, its volume is
-- sqrt(det G) / k!, its stiffness is volume * B^T G^-1 B (B maps the vertex values to their
-- differences from p0's), and each vertex gets volume / (k + 1) of the (lumped) mass. The
-- boundary is the points on faces (k - 1 of a simplex's vertices) that belong to only one
-- simplex; a closed surface has none. Errors on a flat simplex, an index outside the
-- points, or a point no simplex uses.
local function fem(points: { any }, simplices: { { number } }): Domain
	local n = #points
	local ns = #simplices
	if n < 2 or ns < 1 or type(simplices[1]) ~= "table" then
		fail("fem", "need at least 2 points and 1 simplex")
	end
	local first = points[1]
	local d = if type(first) == "vector" then 3 elseif type(first) == "table" then #first else 0
	local k = #simplices[1] - 1
	if d < 1 then
		fail("fem", "point 1 is neither a table of numbers nor a vector")
	end
	if k < 1 or k > d then
		fail("fem", string.format("simplex 1 has %d points: a simplex needs 2 to %d (one more than the points' %d coordinates)", k + 1, d + 1, d))
	end
	local coords: { { number } } = table.create(d) :: any
	for a = 1, d do
		coords[a] = table.create(n, 0) :: { number }
	end
	for p, pt in points do
		local ok = true
		if type(pt) == "vector" and d == 3 then
			local v: vector = pt
			coords[1][p], coords[2][p], coords[3][p] = v.x, v.y, v.z
		elseif type(pt) == "table" and #pt == d then
			for a = 1, d do
				local c = pt[a]
				if type(c) ~= "number" or c ~= c then
					ok = false
					break
				end
				coords[a][p] = c
			end
		else
			ok = false
		end
		if not ok then
			fail("fem", string.format("point %d is not like point 1 (a vector, or a table of %d numbers)", p, d))
		end
	end
	local what = if k == 1 then "segment" elseif k == 2 then "triangle" elseif k == 3 then "tetrahedron" else "simplex"
	local used = table.create(n, false)
	-- faces as numbers when they fit (sorted indices, base n + 1), else as strings
	local numeric = (n + 1) ^ k < 2 ^ 53
	local faces: { [any]: number } = {}
	local ii = table.create(ns * (k + 1) * (k + 1)) :: { number }
	local jj = table.create(ns * (k + 1) * (k + 1)) :: { number }
	local vv = table.create(ns * (k + 1) * (k + 1)) :: { number }
	local mass = table.create(n, 0) :: { number }
	local volume = 0
	local fact = 1
	for s = 2, k do
		fact *= s
	end
	local E = table.create(k * d, 0) :: { number } -- E[(s - 1) d + a]: coordinate a of p_s - p0
	local G = table.create(k * k, 0) :: { number } -- G, then its Cholesky factor, then G^-1
	local Ginv = table.create(k * k, 0) :: { number }
	local face = table.create(k, 0) :: { number }
	local cnt = 0
	for t, simplex in simplices do
		if type(simplex) ~= "table" or #simplex ~= k + 1 then
			fail("fem", string.format("simplex %d does not have %d points, as simplex 1 does", t, k + 1))
		end
		for s = 1, k + 1 do
			local p = simplex[s]
			if not whole(p) or p > n then
				fail("fem", string.format("%s %d refers to point %s, not one of the %d points", what, t, tostring(p), n))
			end
			used[p] = true
		end
		local p0 = simplex[1]
		for s = 1, k do
			local ps = simplex[s + 1]
			for a = 1, d do
				local ca = coords[a]
				E[(s - 1) * d + a] = ca[ps] - ca[p0]
			end
		end
		-- G = E^T E and its Cholesky factor (lower, in place)
		local scale = 0
		for s = 1, k do
			for r = 1, s do
				local sum = 0
				for a = 1, d do
					sum += E[(s - 1) * d + a] * E[(r - 1) * d + a]
				end
				G[(s - 1) * k + r] = sum
			end
			scale = math.max(scale, G[(s - 1) * k + s])
		end
		local det = 1
		for j = 1, k do
			local djj = G[(j - 1) * k + j]
			for r = 1, j - 1 do
				djj -= G[(j - 1) * k + r] ^ 2
			end
			if not (djj > 64 * EPS * scale) then
				fail("fem", string.format("%s %d is flat (zero %s)", what, t, if k == 1 then "length" elseif k == 2 then "area" else "volume"))
			end
			local ljj = math.sqrt(djj)
			G[(j - 1) * k + j] = ljj
			det *= djj
			for i = j + 1, k do
				local sum = G[(i - 1) * k + j]
				for r = 1, j - 1 do
					sum -= G[(i - 1) * k + r] * G[(j - 1) * k + r]
				end
				G[(i - 1) * k + j] = sum / ljj
			end
		end
		local vol = math.sqrt(det) / fact
		volume += vol
		-- G^-1 = L^-T L^-1, column by column: solve L y = e_j, then L^T x = y
		for j = 1, k do
			for i = 1, k do
				local sum = if i == j then 1 else 0
				for r = 1, i - 1 do
					sum -= G[(i - 1) * k + r] * Ginv[(r - 1) * k + j]
				end
				Ginv[(i - 1) * k + j] = sum / G[(i - 1) * k + i]
			end
			for i = k, 1, -1 do
				local sum = Ginv[(i - 1) * k + j]
				for r = i + 1, k do
					sum -= G[(r - 1) * k + i] * Ginv[(r - 1) * k + j]
				end
				Ginv[(i - 1) * k + j] = sum / G[(i - 1) * k + i]
			end
		end
		-- the stiffness: vol * B^T G^-1 B, B = [-1 | I]; row and column 0 are minus the
		-- sums of the others
		for s = 1, k + 1 do
			local ps = simplex[s]
			mass[ps] += vol / (k + 1)
			for r = 1, k + 1 do
				local v
				if s > 1 and r > 1 then
					v = Ginv[(s - 2) * k + r - 1]
				elseif s > 1 then
					v = 0
					for q = 1, k do
						v -= Ginv[(s - 2) * k + q]
					end
				elseif r > 1 then
					v = 0
					for q = 1, k do
						v -= Ginv[(q - 1) * k + r - 1]
					end
				else
					v = 0
					for q = 1, k * k do
						v += Ginv[q]
					end
				end
				cnt += 1
				ii[cnt], jj[cnt], vv[cnt] = ps, simplex[r], vol * v
			end
		end
		-- the faces: the simplex without vertex s, sorted
		for s = 1, k + 1 do
			local m = 0
			for r = 1, k + 1 do
				if r ~= s then
					local v = simplex[r]
					local q = m
					while q > 0 and face[q] > v do
						face[q + 1] = face[q]
						q -= 1
					end
					face[q + 1] = v
					m += 1
				end
			end
			local key: any
			if numeric then
				key = 0
				for q = k, 1, -1 do
					key = key * (n + 1) + face[q]
				end
			else
				key = table.concat(face, ",")
			end
			faces[key] = (faces[key] or 0) + 1
		end
	end
	for p = 1, n do
		if not used[p] then
			fail("fem", string.format("point %d belongs to no %s", p, what))
		end
	end
	-- the boundary: the points on faces of only one simplex
	local on_boundary = table.create(n, false)
	for key, count in faces do
		if count == 1 then
			if numeric then
				local rest = key :: number
				for _ = 1, k do
					on_boundary[rest % (n + 1)] = true
					rest = rest // (n + 1)
				end
			else
				for v in string.gmatch(key :: string, "%d+") do
					on_boundary[tonumber(v) :: number] = true
				end
			end
		end
	end
	local boundary, interior = {}, {}
	local index = table.create(n, 0) -- a point's place among the unknowns (0 on the boundary)
	for p = 1, n do
		if on_boundary[p] then
			table.insert(boundary, p)
		else
			table.insert(interior, p)
			index[p] = #interior
		end
	end
	local K = f64.sparse(n, n, ii, jj, f64.from(vv))
	K.symmetric = true
	-- the stiffness between unknowns only
	local ki, kj, kv = {}, {}, {}
	for e = 1, cnt do
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
		d = d,
		k = k,
		coords = coords,
		x = coords[1],
		y = coords[2],
		z = coords[3],
		simplices = simplices,
		volume = volume,
		boundary = boundary,
		interior = interior,
		stiffness = K,
		mass = mass,
		_index = index,
		_Kii = Kii,
	}
end

-- Data given as a number, a function of the coordinates (x, y, ...) or a table of one
-- number per point, as a table of values at the points.
local function field(D: Domain, data: any, who: string, what: string): { number }
	local out = table.create(D.n, 0) :: { number }
	if data == nil then
		return out
	elseif type(data) == "number" then
		for p = 1, D.n do
			out[p] = data
		end
	elseif type(data) == "function" then
		local f = data :: (...number) -> any
		local C = D.coords
		local d = D.d
		local pt = table.create(d, 0) :: { number }
		for p = 1, D.n do
			local v
			if d == 1 then
				v = f(C[1][p])
			elseif d == 2 then
				v = f(C[1][p], C[2][p])
			elseif d == 3 then
				v = f(C[1][p], C[2][p], C[3][p])
			else
				for a = 1, d do
					pt[a] = C[a][p]
				end
				v = f(table.unpack(pt))
			end
			if type(v) ~= "number" or v ~= v then
				fail(who, string.format("%s returned %s at point %d, not a number", what, tostring(v), p))
			end
			out[p] = v
		end
	elseif type(data) == "table" then
		if #data ~= D.n then
			fail(who, string.format("%s has %d values where the mesh has %d points", what, #data, D.n))
		end
		for p = 1, D.n do
			out[p] = data[p]
		end
	else
		fail(who, what .. " must be a number, a function of the coordinates or a table of one number per point")
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
-- (f and g: numbers, functions of the coordinates or tables per point; both default to 0).
-- Returns u (a table, one value per point) and the solver's info (linalg.solve). A mesh
-- without a boundary (a closed surface) is refused: u would be fixed only up to a constant.
local function poisson(D: Domain, f: any, g: any): ({ number }, linalg.Info)
	if #D.boundary == 0 then
		fail("poisson", "the mesh has no boundary (a closed surface): the solution would be fixed only up to a constant")
	end
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
-- function of the coordinates or a table per point), with u held at options.boundary on the
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
-- boundary (free, on a closed surface: the first eigenvalue is then 0, a constant):
-- -(the Laplacian of u) = lambda u, the drum's equation. Returns the eigenvalues
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
	grid_mesh = grid_mesh,
	rect_mesh = rect_mesh,
	fem = fem,
	poisson = poisson,
	heat = heat,
	modes = modes,
}
