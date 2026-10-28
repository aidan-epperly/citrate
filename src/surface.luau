--!strict

-- Surfaces from grids of control points: Bezier, B-spline and interpolating cubic spline
-- patches (tensor products), with points, partial derivatives, normals, and meshes for
-- drawing them.
--
--   local S = surface.new(grid, { kind = "spline" })  -- grid[i][j]: vectors, a rows by
--                                                      -- columns table
--   surface.eval(S, u, v)        -- the point at parameters (u, v)
--   surface.normal(S, u, v)
--   local points, triangles, normals = surface.mesh(S, 32, 32)
--
-- A tensor-product surface is a curve of curves: row i of the grid is a curve C_i in v, and
-- the surface at (u, v) is the curve through the points C_i(v) (of the same kind) at u. So
-- every surface here is built from citrate.curve, and evaluates as accurately as it does.
-- Points are vectors (or, with options.space, any curve space).

local types = require("./types")
local curve = require("./curve")

type Space<T> = types.Space<T>

export type SurfaceOptions = { kind: string?, degree: number?, space: Space<any>? }

export type Surface = {
	kind: string,
	rows: { any }, -- one curve per grid row, in v
	nu: number, -- grid rows
	nv: number, -- grid columns
	u0: number,
	u1: number,
	v0: number,
	v1: number,
	options: { [string]: any },
}

local function fail(who: string, message: string): never
	error("surface." .. who .. ": " .. message, 3)
end

-- A curve of the surface's kind through (or controlled by) the points.
local function make(kind: string, points: { any }, options: { [string]: any }): any
	if kind == "bezier" then
		return curve.bezier(points, options)
	elseif kind == "bspline" then
		return curve.bspline(points, options)
	end
	return curve.cubic_spline(points, options)
end

-- A surface from a grid of control points (grid[i][j]: rows i = 1..m, columns j = 1..n;
-- vectors by default). options.kind:
--   "bspline" (the default)  a uniform clamped B-spline surface of options.degree (default 3,
--                            lowered to fit the grid): smooth, passing through the four
--                            corners and following the grid loosely; domain [0, m - p] by
--                            [0, n - p];
--   "bezier"                 a single Bezier patch of degree (m - 1, n - 1); domain [0, 1]^2;
--   "spline"                 the natural cubic spline surface through every grid point,
--                            with u and v counting rows and columns from 0.
local function new(grid: { { any } }, options: SurfaceOptions?): Surface
	local o: SurfaceOptions = options or {}
	local kind = o.kind or "bspline"
	if kind ~= "bspline" and kind ~= "bezier" and kind ~= "spline" then
		fail("new", "kind must be \"bspline\", \"bezier\" or \"spline\"")
	end
	if type(grid) ~= "table" or #grid < 2 or type(grid[1]) ~= "table" or #grid[1] < 2 then
		fail("new", "need a grid of at least 2 by 2 points (a table of rows)")
	end
	local m, n = #grid, #grid[1]
	for i = 2, m do
		if type(grid[i]) ~= "table" or #grid[i] ~= n then
			fail("new", string.format("row %d has %s points where row 1 has %d", i, if type(grid[i]) == "table" then tostring(#grid[i]) else "no", n))
		end
	end
	local function curve_options(count: number): { [string]: any }
		local co: { [string]: any } = { space = o.space }
		if kind == "bspline" then
			co.degree = math.min(o.degree or 3, count - 1)
		elseif kind == "spline" then
			local breaks = table.create(count, 0)
			for k = 1, count do
				breaks[k] = k - 1
			end
			co.breaks = breaks
		end
		return co
	end
	local vopt, uopt = curve_options(n), curve_options(m)
	local rows = table.create(m) :: { any }
	for i = 1, m do
		rows[i] = make(kind, grid[i], vopt)
	end
	local v0, v1 = curve.domain(rows[1])
	-- the u domain: that of a curve of this kind through m points (the first column's)
	local col = table.create(m) :: { any }
	for i = 1, m do
		col[i] = grid[i][1]
	end
	local u0, u1 = curve.domain(make(kind, col, uopt))
	return { kind = kind, rows = rows, nu = m, nv = n, u0 = u0, u1 = u1, v0 = v0, v1 = v1, options = uopt }
end

local function check_uv(S: Surface, u: number, v: number, who: string)
	if type(u) ~= "number" or type(v) ~= "number" or u ~= u or v ~= v then
		fail(who, "u and v must be numbers")
	end
end

-- The points C_i(v) (or their k-th v-derivatives) of the row curves.
local function row_points(S: Surface, v: number, k: number): { any }
	local pts = table.create(S.nu)
	for i = 1, S.nu do
		pts[i] = if k == 0 then curve.eval(S.rows[i], v) else curve.derivative(S.rows[i], v, k)
	end
	return pts
end

-- The curves across the rows at v (k = 0: through the row curves' points; k = 1: their
-- v-derivatives), kept for the last v asked: evaluating along u at a fixed v builds them
-- once.
local function across(S: Surface, v: number, k: number): any
	local cache = S :: any
	local key = if k == 0 then "at" else "dv"
	if cache[key .. "_v"] == v then
		return cache[key]
	end
	local C = make(S.kind, row_points(S, v, k), S.options)
	cache[key .. "_v"], cache[key] = v, C
	return C
end

-- The point at parameters (u, v) (clamped to the domain; see surface.domain). Building the
-- curve across the rows dominates a single point's cost; it is kept for the last v, so
-- points along u at a fixed v (a scanline) cost much less.
local function eval(S: Surface, u: number, v: number): any
	check_uv(S, u, v, "eval")
	u = math.clamp(u, S.u0, S.u1)
	v = math.clamp(v, S.v0, S.v1)
	return curve.eval(across(S, v, 0), u)
end

-- The partial derivatives dS/du and dS/dv at (u, v).
local function derivatives(S: Surface, u: number, v: number): (any, any)
	check_uv(S, u, v, "derivatives")
	u = math.clamp(u, S.u0, S.u1)
	v = math.clamp(v, S.v0, S.v1)
	local su = curve.derivative(across(S, v, 0), u, 1)
	local sv = curve.eval(across(S, v, 1), u)
	return su, sv
end

-- The unit normal at (u, v): dS/du x dS/dv, normalized (zero where the surface is
-- degenerate). Vectors only.
local function normal(S: Surface, u: number, v: number): vector
	local su, sv = derivatives(S, u, v)
	local c = vector.cross(su :: vector, sv :: vector)
	local len = vector.magnitude(c)
	return if len > 0 then c / len else vector.zero
end

-- The parameter domain: u in [u0, u1], v in [v0, v1].
local function domain(S: Surface): (number, number, number, number)
	return S.u0, S.u1, S.v0, S.v1
end

-- A triangle mesh of the surface: (nu + 1) by (nv + 1) points evenly spaced in (u, v), the
-- triangles (each { a, b, c }, indices of points, two per cell, wound so their normals agree
-- with surface.normal), and the unit normal at each point. Vectors only. For drawing (an
-- EditableMesh) or for a heightfield's collision.
local function mesh(S: Surface, nu: number, nv: number): ({ vector }, { { number } }, { vector })
	if nu % 1 ~= 0 or nv % 1 ~= 0 or nu < 1 or nv < 1 then
		fail("mesh", "the numbers of cells must be positive whole numbers")
	end
	local points: { vector } = {}
	local normals: { vector } = {}
	for b = 0, nv do
		local v = S.v0 + (S.v1 - S.v0) * b / nv
		-- the curves across the rows at this v, built once for the whole row of the mesh
		local Cu = across(S, v, 0)
		local Cv = across(S, v, 1)
		for a = 0, nu do
			local u = S.u0 + (S.u1 - S.u0) * a / nu
			table.insert(points, curve.eval(Cu, u) :: vector)
			local c = vector.cross(curve.derivative(Cu, u, 1) :: vector, curve.eval(Cv, u) :: vector)
			local len = vector.magnitude(c)
			table.insert(normals, if len > 0 then c / len else vector.zero)
		end
	end
	local triangles: { { number } } = {}
	local W = nu + 1
	for b = 0, nv - 1 do
		for a = 0, nu - 1 do
			local p = b * W + a + 1
			table.insert(triangles, { p, p + 1, p + W + 1 })
			table.insert(triangles, { p, p + W + 1, p + W })
		end
	end
	return points, triangles, normals
end

return {
	new = new,
	eval = eval,
	derivatives = derivatives,
	normal = normal,
	domain = domain,
	mesh = mesh,
}
