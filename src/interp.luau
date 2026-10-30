--!strict
--!native
--!optimize 2

-- Interpolation in more than one dimension: values on a regular grid of any dimension (a
-- heightfield, a texture, a volume), and values at scattered points.
--
--   local H = interp.grid(heights, { dx = 4, dy = 4 })   -- heights[i][j] at (4 (i - 1), 4 (j - 1))
--   H:at(10.5, 7.2)                                      -- smooth between grid points
--   local fx, fy = H:gradient(10.5, 7.2)                 -- the slope there
--   local f, fx, fy = H:sample(10.5, 7.2)                -- both, for one evaluation
--   local V = interp.grid(density)                       -- density[i][j][k]: a volume
--   V:at(1.5, 2.5, 0.5)
--
--   local R, info = interp.rbf(points, values)           -- any points (vectors)
--   R:at(vector.create(1, 2, 0))
--   R:gradient(vector.create(1, 2, 0))
--
-- (Curves, one dimension, are in citrate.curve; surfaces through grids of points in
-- citrate.surface.)

local types = require("./types")
local f64 = require("./f64")
local hodlr = require("./hodlr")

type Matrix = types.Matrix

local function fail(who: string, message: string): never
	error("interp." .. who .. ": " .. message, 3)
end

-- Grids

export type GridOptions = {
	x0: number?,
	y0: number?,
	z0: number?,
	dx: number?,
	dy: number?,
	dz: number?,
	origin: { number }?,
	spacing: { number }?,
	shape: { number }?,
	method: string?,
}
export type Grid = {
	dims: number,
	shape: { number },
	origin: { number },
	spacing: { number },
	strides: { number },
	data: buffer, -- the values, the first axis fastest
	method: string,
	nx: number,
	ny: number,
	x0: number,
	y0: number,
	dx: number,
	dy: number,
	values: { { number } }?, -- 2-D: the rows, for the 2-D path
	at: (self: Grid, ...number) -> number,
	gradient: (self: Grid, ...number) -> ...number,
	sample: (self: Grid, ...number) -> ...number,
}

local Grid = {}
Grid.__index = Grid

-- An interpolant of values on a regular grid in any number of dimensions d. values is a
-- nested table d deep (values[i][j] in 2-D, values[i][j][k] in 3-D, ...), an f64 Matrix
-- (2-D), or an f64 array with options.shape = { n1, ..., nd } (the first index fastest).
-- The value with indices (i1, ..., id) is at origin[a] + (i_a - 1) spacing[a] on each axis
-- a: options.origin and options.spacing (default 0 and 1), or for the first three axes
-- x0, y0, z0 and dx, dy, dz. options.method:
--   "cubic" (the default)  Keys' cubic convolution (a = -1/2, as Catmull-Rom) along each
--                          axis (a tensor product: 4^d values a point): it passes through
--                          the values, its slope is continuous, and its error falls as the
--                          cube of the spacing; it can overshoot a little near sharp steps;
--   "linear"               multilinear (2^d values a point): never overshoots, but the slope
--                          jumps at grid lines.
-- Points outside the grid are clamped to its edge. Needs at least 2 points along each axis
-- (3 for cubic; with 2, linear is used). Evaluation takes the d coordinates: G:at(x, y),
-- G:at(x, y, z), ...; gradient and sample return the d partial derivatives. 2-D grids have
-- their own evaluation (the general one gives the same values: tested).
local function grid(values: any, options: GridOptions?): Grid
	local o: GridOptions = options or {}
	local method = o.method or "cubic"
	if method ~= "cubic" and method ~= "linear" then
		fail("grid", "method must be \"cubic\" or \"linear\"")
	end
	local shape: { number } = {}
	local data: buffer
	if type(values) == "buffer" then
		local sh = o.shape
		if type(sh) ~= "table" or #sh < 1 then
			fail("grid", "an f64 array of values needs options.shape, its size along each axis")
		end
		local count = 1
		for a, n in sh :: { number } do
			if type(n) ~= "number" or n < 1 or n % 1 ~= 0 then
				fail("grid", string.format("shape[%d] is not a positive whole number", a))
			end
			shape[a] = n
			count *= n
		end
		if f64.length(values) ~= count then
			fail("grid", string.format("the array has %d values where the shape has %d", f64.length(values), count))
		end
		data = buffer.create(8 * count)
		buffer.copy(data, 0, values)
	elseif type(values) == "table" and values.data ~= nil then
		local M = values :: Matrix
		shape = { M.rows, M.cols }
		data = buffer.create(8 * M.rows * M.cols)
		buffer.copy(data, 0, M.data, 0, 8 * M.rows * M.cols)
	elseif type(values) == "table" and (type(values[1]) == "table" or type(values[1]) == "number") then
		-- the shape from the first entries, then every entry read and checked
		local t: any = values
		while type(t) == "table" do
			table.insert(shape, #t)
			t = t[1]
		end
		local d = #shape
		local count = 1
		for a = 1, d do
			count *= shape[a]
		end
		data = buffer.create(8 * count)
		local stride = table.create(d, 1) :: { number }
		for a = 2, d do
			stride[a] = stride[a - 1] * shape[a - 1]
		end
		local function fill(node: any, axis: number, offset: number, where: string)
			if type(node) ~= "table" or #node ~= shape[axis] then
				fail("grid", string.format("values%s does not have %d entries, as values%s does", where, shape[axis], string.rep("[1]", axis - 1)))
			end
			for i = 1, shape[axis] do
				local v = node[i]
				local at = where .. "[" .. i .. "]"
				if axis == d then
					if type(v) ~= "number" or v ~= v or v == math.huge or v == -math.huge then
						fail("grid", string.format("value %s is not a finite number", at))
					end
					buffer.writef64(data, 8 * (offset + (i - 1) * stride[axis]), v)
				else
					fill(v, axis + 1, offset + (i - 1) * stride[axis], at)
				end
			end
		end
		fill(values, 1, 0, "")
	else
		fail("grid", "values must be nested tables, an f64 Matrix or an f64 array with options.shape")
	end
	local d = #shape
	local strides = table.create(d, 1) :: { number }
	for a = 2, d do
		strides[a] = strides[a - 1] * shape[a - 1]
	end
	local origin = table.create(d, 0) :: { number }
	local spacing = table.create(d, 1) :: { number }
	local named0 = { o.x0, o.y0, o.z0 }
	local namedh = { o.dx, o.dy, o.dz }
	for a = 1, d do
		local x0 = if o.origin then o.origin[a] else named0[a]
		local h = if o.spacing then o.spacing[a] else namedh[a]
		origin[a] = x0 or 0
		spacing[a] = h or 1
		if type(origin[a]) ~= "number" or type(spacing[a]) ~= "number" or not (spacing[a] > 0) then
			fail("grid", string.format("axis %d needs a number for its origin and a positive spacing", a))
		end
		if shape[a] < 2 then
			fail("grid", "need at least 2 values along each axis")
		end
		if shape[a] < 3 then
			method = "linear"
		end
	end
	for k = 0, buffer.len(data) // 8 - 1 do
		local v = buffer.readf64(data, 8 * k)
		if v ~= v or v == math.huge or v == -math.huge then
			fail("grid", string.format("value %d is not a finite number", k + 1))
		end
	end
	local G: any = {
		dims = d,
		shape = shape,
		origin = origin,
		spacing = spacing,
		strides = strides,
		data = data,
		method = method,
		nx = shape[1],
		ny = shape[2] or 1,
		x0 = origin[1],
		y0 = origin[2] or 0,
		dx = spacing[1],
		dy = spacing[2] or 1,
		_coords = table.create(d, 0),
		_grad = table.create(d, 0),
		_w = table.create(4 * d, 0),
		_dw = table.create(4 * d, 0),
		_idx = table.create(4 * d, 0),
		_len = table.create(d, 0),
	}
	if d == 2 then
		local rows = table.create(shape[1]) :: { { number } }
		for i = 1, shape[1] do
			local row = table.create(shape[2], 0) :: { number }
			for j = 1, shape[2] do
				row[j] = buffer.readf64(data, 8 * ((i - 1) + (j - 1) * shape[1]))
			end
			rows[i] = row
		end
		G.values = rows
	end
	return setmetatable(G, Grid) :: any
end

-- The value at (i, j), with Keys' boundary rule past the edges: f(-1) = 3 f(0) - 3 f(1) +
-- f(2), which keeps the third-order accuracy there.
local function value(G: Grid, i: number, j: number): number
	local nx, ny, v = G.nx, G.ny, G.values :: { { number } }
	if i < 1 then
		return 3 * value(G, 1, j) - 3 * value(G, 2, j) + value(G, 3, j)
	elseif i > nx then
		return 3 * value(G, nx, j) - 3 * value(G, nx - 1, j) + value(G, nx - 2, j)
	elseif j < 1 then
		return 3 * value(G, i, 1) - 3 * value(G, i, 2) + value(G, i, 3)
	elseif j > ny then
		return 3 * value(G, i, ny) - 3 * value(G, i, ny - 1) + value(G, i, ny - 2)
	end
	return v[i][j]
end

-- The cell (1-based lower corner) and the fraction t in [0, 1] across it, for coordinate
-- x on n points from x0 spaced h (clamped to the grid).
local function locate(x: number, x0: number, h: number, n: number): (number, number)
	local s = math.clamp((x - x0) / h, 0, n - 1)
	local k = math.min(math.floor(s), n - 2)
	return k + 1, s - k
end

-- Keys' cubic convolution weights for the points before, at, after and two after, and
-- their derivatives in t.
local function keys(t: number): (number, number, number, number, number, number, number, number)
	local t2, t3 = t * t, t * t * t
	return (-t3 + 2 * t2 - t) / 2,
		(3 * t3 - 5 * t2 + 2) / 2,
		(-3 * t3 + 4 * t2 + t) / 2,
		(t3 - t2) / 2,
		(-3 * t2 + 4 * t - 1) / 2,
		(9 * t2 - 10 * t) / 2,
		(-9 * t2 + 8 * t + 1) / 2,
		(3 * t2 - 2 * t) / 2
end

-- The value and both partial derivatives at (x, y).
local function evaluate(G: Grid, x: number, y: number): (number, number, number)
	local i, tx = locate(x, G.x0, G.dx, G.nx)
	local j, ty = locate(y, G.y0, G.dy, G.ny)
	if G.method == "linear" then
		local v = G.values :: { { number } }
		local a, b, c, d = v[i][j], v[i + 1][j], v[i][j + 1], v[i + 1][j + 1]
		local f = (1 - tx) * (1 - ty) * a + tx * (1 - ty) * b + (1 - tx) * ty * c + tx * ty * d
		local fx = ((1 - ty) * (b - a) + ty * (d - c)) / G.dx
		local fy = ((1 - tx) * (c - a) + tx * (d - b)) / G.dy
		return f, fx, fy
	end
	local a1: number, a2: number, a3: number, a4: number, da1: number, da2: number, da3: number, da4: number = keys(tx)
	local b1: number, b2: number, b3: number, b4: number, db1: number, db2: number, db3: number, db4: number = keys(ty)
	-- each of the four rows i - 1 .. i + 2 combined along y (value and y-derivative), then
	-- the rows combined along x; rows and columns past the edges by Keys' rule
	local f, fx, fy = 0, 0, 0
	local v = G.values :: { { number } }
	local inside = j >= 2 and j + 2 <= G.ny
	for a = 1, 4 do
		local r = i + a - 2
		local p1: number, p2: number, p3: number, p4: number
		if inside and r >= 1 and r <= G.nx then
			local row = v[r]
			p1 = row[j - 1]
			p2 = row[j]
			p3 = row[j + 1]
			p4 = row[j + 2]
		else
			p1 = value(G, r, j - 1)
			p2 = value(G, r, j)
			p3 = value(G, r, j + 1)
			p4 = value(G, r, j + 2)
		end
		local s = b1 * p1 + b2 * p2 + b3 * p3 + b4 * p4
		local ds = db1 * p1 + db2 * p2 + db3 * p3 + db4 * p4
		local wa = if a == 1 then a1 elseif a == 2 then a2 elseif a == 3 then a3 else a4
		local dwa = if a == 1 then da1 elseif a == 2 then da2 elseif a == 3 then da3 else da4
		f += wa * s
		fx += dwa * s
		fy += wa * ds
	end
	return f, fx / G.dx, fy / G.dy
end

-- Any dimension: along each axis the (up to 4) indices and their weights and derivative
-- weights, Keys' edge rule folded in (the value one past an edge is 3 f(edge) - 3 f(next)
-- + f(next but one), all within the stencil), then the tensor product of the axes'
-- weights. Returns the value; the gradient goes into G._grad.
local function evaluate_nd(G: any, xs: { number }): number
	local d = G.dims
	local W, DW, I, len = G._w, G._dw, G._idx, G._len
	local linear = G.method == "linear"
	for a = 1, d do
		local n, h = G.shape[a], G.spacing[a]
		local i, t = locate(xs[a], G.origin[a], h, n)
		local base = 4 * (a - 1)
		if linear then
			I[base + 1], W[base + 1], DW[base + 1] = i, 1 - t, -1 / h
			I[base + 2], W[base + 2], DW[base + 2] = i + 1, t, 1 / h
			len[a] = 2
		else
			local w1, w2, w3, w4, d1, d2, d3, d4 = keys(t)
			local w = { w1, w2, w3, w4 }
			local dw = { d1 / h, d2 / h, d3 / h, d4 / h }
			-- stencil i - 1 .. i + 2; a ghost at 0 or n + 1 folds into its neighbors
			if i - 1 < 1 then
				w[2] += 3 * w[1]
				w[3] -= 3 * w[1]
				w[4] += w[1]
				dw[2] += 3 * dw[1]
				dw[3] -= 3 * dw[1]
				dw[4] += dw[1]
				w[1], dw[1] = 0, 0
			end
			if i + 2 > n then
				w[3] += 3 * w[4]
				w[2] -= 3 * w[4]
				w[1] += w[4]
				dw[3] += 3 * dw[4]
				dw[2] -= 3 * dw[4]
				dw[1] += dw[4]
				w[4], dw[4] = 0, 0
			end
			local m = 0
			for q = 1, 4 do
				local idx = i - 2 + q
				if idx >= 1 and idx <= n then
					m += 1
					I[base + m], W[base + m], DW[base + m] = idx, w[q], dw[q]
				end
			end
			len[a] = m
		end
	end
	-- the tensor product, by an odometer over the axes' stencils
	local grad = G._grad
	for a = 1, d do
		grad[a] = 0
	end
	local pos = table.create(d, 1) :: { number }
	local strides, data = G.strides, G.data
	local f = 0
	while true do
		local offset, prod = 0, 1
		for a = 1, d do
			local q = 4 * (a - 1) + pos[a]
			offset += (I[q] - 1) * strides[a]
			prod *= W[q]
		end
		local v = buffer.readf64(data, 8 * offset)
		f += prod * v
		for a = 1, d do
			local g = v
			for b = 1, d do
				local q = 4 * (b - 1) + pos[b]
				g *= if b == a then DW[q] else W[q]
			end
			grad[a] += g
		end
		local a = 1
		while a <= d do
			pos[a] += 1
			if pos[a] <= len[a] then
				break
			end
			pos[a] = 1
			a += 1
		end
		if a > d then
			break
		end
	end
	return f
end

local function coordinates(G: any, who: string, x: number, y: number?, ...: number): { number }
	local xs = G._coords
	local d = G.dims
	local count = if y == nil then 1 else 2 + select("#", ...)
	if count ~= d then
		fail(who, string.format("the grid has %d dimensions: give %d coordinates", d, d))
	end
	xs[1] = x
	if y ~= nil then
		xs[2] = y
		for a = 3, d do
			xs[a] = (select(a - 2, ...))
		end
	end
	return xs
end

function Grid.at(self: Grid, x: number, y: number?, ...: number): number
	if self.dims == 2 and y then
		local f = evaluate(self, x, y)
		return f
	end
	return evaluate_nd(self, coordinates(self, "at", x, y, ...))
end

-- The d partial derivatives.
function Grid.gradient(self: Grid, x: number, y: number?, ...: number): ...number
	if self.dims == 2 and y then
		local _, fx, fy = evaluate(self, x, y)
		return fx, fy
	end
	evaluate_nd(self, coordinates(self, "gradient", x, y, ...))
	return table.unpack((self :: any)._grad, 1, self.dims)
end

-- The value and the d partial derivatives together (one evaluation), as a heightfield's
-- user often needs both.
function Grid.sample(self: Grid, x: number, y: number?, ...: number): ...number
	if self.dims == 2 and y then
		return evaluate(self, x, y)
	end
	local f = evaluate_nd(self, coordinates(self, "sample", x, y, ...))
	return f, table.unpack((self :: any)._grad, 1, self.dims)
end

-- Scattered points

export type RbfOptions = {
	kernel: string?,
	epsilon: number?,
	nu: number?,
	smoothing: (number | string)?,
	method: string?,
	tol: number?,
}
export type RbfInfo = {
	method: string,
	kernel: string,
	epsilon: number?,
	smoothing: number,
	residual: number,
	ranks: { number }?,
}
export type Rbf = {
	at: (self: Rbf, p: any) -> number,
	gradient: (self: Rbf, p: any) -> ...number,
	[string]: any,
}

local Rbf = {}
Rbf.__index = Rbf

local EPS = 2 ^ -52
-- Above this many points, positive definite kernels (gaussian, matern) in one or two
-- dimensions are solved by HODLR compression (citrate.hodlr) instead of densely: measured
-- about equal there, and HODLR pulls ahead beyond (3 to 20 times faster at 4000 points,
-- in 20 to 70 percent of the dense matrix's memory).
local HODLR_ABOVE = 1500

-- The number of coordinates of a point: a vector's 3, a table's or f64 array's d.
local function dimension(p: any): number
	return if type(p) == "vector" then 3
		elseif type(p) == "table" then #p
		elseif type(p) == "buffer" then buffer.len(p) // 8
		else 0
end

-- A point's d coordinates into q from q[base + 1] (a vector's in double precision, as
-- its single-precision values); false if it does not have d.
local function read_point(p: any, d: number, q: { number }, base: number): boolean
	if type(p) == "vector" and d == 3 then
		q[base + 1], q[base + 2], q[base + 3] = p.x, p.y, p.z
	elseif type(p) == "table" and #p == d then
		for a = 1, d do
			local v = p[a]
			if type(v) ~= "number" then
				return false
			end
			q[base + a] = v
		end
	elseif type(p) == "buffer" and buffer.len(p) == 8 * d then
		for a = 1, d do
			q[base + a] = buffer.readf64(p, 8 * (a - 1))
		end
	else
		return false
	end
	return true
end

-- The kernel phi(r) and phi'(r) / r (for gradients; 0 at r = 0, where the gradient of the
-- term vanishes or, for matern with nu = 1/2, does not exist), for scale e; and whether it
-- is positive definite and whether it has a scale.
local function kernel_functions(name: string, e: number, nu: number): ((number) -> number, (number) -> number, boolean)
	if name == "thin_plate" then
		return function(r: number): number
			return if r > 0 then r * r * math.log(r) else 0
		end, function(r: number): number
			return if r > 0 then 2 * math.log(r) + 1 else 0
		end, false
	elseif name == "cubic" then
		return function(r: number): number
			return r * r * r
		end, function(r: number): number
			return 3 * r
		end, false
	elseif name == "gaussian" then
		local e2 = e * e
		return function(r: number): number
			return math.exp(-e2 * r * r)
		end, function(r: number): number
			return -2 * e2 * math.exp(-e2 * r * r)
		end, true
	elseif name == "multiquadric" then
		local e2 = e * e
		return function(r: number): number
			return math.sqrt(1 + e2 * r * r)
		end, function(r: number): number
			return e2 / math.sqrt(1 + e2 * r * r)
		end, false
	elseif name == "matern" then
		if nu == 0.5 then
			return function(r: number): number
				return math.exp(-e * r)
			end, function(r: number): number
				return if r > 0 then -e * math.exp(-e * r) / r else 0
			end, true
		elseif nu == 1.5 then
			local a = math.sqrt(3) * e
			return function(r: number): number
				return (1 + a * r) * math.exp(-a * r)
			end, function(r: number): number
				return -a * a * math.exp(-a * r)
			end, true
		elseif nu == 2.5 then
			local a = math.sqrt(5) * e
			return function(r: number): number
				local s = a * r
				return (1 + s + s * s / 3) * math.exp(-s)
			end, function(r: number): number
				local s = a * r
				return -(a * a / 3) * (1 + s) * math.exp(-s)
			end, true
		end
		error("interp.rbf: nu must be 0.5, 1.5 or 2.5", 3)
	end
	error("interp.rbf: kernel must be \"thin_plate\", \"cubic\", \"gaussian\", \"multiquadric\" or \"matern\"", 3)
end

-- An interpolant of values at scattered points in any dimension d (vectors, d = 3, or tables
-- or f64 arrays of d numbers) by
-- radial basis functions: f(p) = sum of w_k phi(|p - p_k|) plus a linear polynomial in the
-- coordinates the points span. Returns the interpolant and info: method ("dense" or
-- "hodlr"), kernel, epsilon, smoothing (the lambda used) and residual (the largest misfit
-- at the points, relative to the largest value; check it when the system is
-- ill-conditioned). options.kernel:
--   "thin_plate" (the default)  r^2 log r: in 2-D the surface through the points that bends
--                               least (in 3-D that is phi = r); reproduces linear data;
--   "cubic"                     r^3;
--   "gaussian"                  exp(-(epsilon r)^2);
--   "matern"                    Matern with smoothness options.nu = 0.5, 1.5 or 2.5 (the
--                               default; a kink at each point for 0.5), scale 1 / epsilon;
--   "multiquadric"              sqrt(1 + (epsilon r)^2).
-- epsilon defaults to 1 / (3 h), h the points' typical spacing (scale-free kernels ignore
-- it). The weights solve a dense linear system (n^3 work: hundreds of points); for
-- gaussian and matern in one or two dimensions above 1500 points (or options.method =
-- "hodlr"), a HODLR compression (citrate.hodlr, accurate to options.tol, default 1e-10)
-- instead, which handles thousands.
-- options.smoothing: lambda added to the diagonal, fitting the values approximately
-- (larger: smoother), for noisy values; "auto" chooses it by maximum likelihood (the
-- values as a Gaussian process with this kernel; about 20 factorizations). Gaussian and
-- matern systems always get at least the lambda their factorization needs to stay
-- positive definite: n eps |K| dense, 2 tol |K| with HODLR.
local function rbf(points: { any }, values: { number }, options: RbfOptions?): (Rbf, RbfInfo)
	local o: RbfOptions = options or {}
	local name = o.kernel or "thin_plate"
	local n = #points
	if n < 1 or #values ~= n then
		fail("rbf", "need one value per point, and at least one point")
	end
	local d = dimension(points[1])
	if d < 1 then
		fail("rbf", "point 1 is not a vector, a table of numbers or an f64 array")
	end
	local X = table.create(n * d, 0) :: { number } -- point k's coordinates from (k - 1) d + 1
	local vmax = 0
	for k = 1, n do
		if not read_point(points[k], d, X, (k - 1) * d) then
			fail("rbf", string.format("point %d is not like point 1 (%d coordinates)", k, d))
		end
		local v = values[k]
		if type(v) ~= "number" or v ~= v or v == math.huge or v == -math.huge then
			fail("rbf", string.format("value %d is not a finite number", k))
		end
		vmax = math.max(vmax, math.abs(v))
	end
	-- The coordinates the points span (by rank, so points along a diagonal count as one
	-- dimension), for the polynomial, the spacing and the method.
	local mean = table.create(d, 0) :: { number }
	for a = 1, d do
		for k = 1, n do
			mean[a] += X[(k - 1) * d + a] / n
		end
	end
	local C = f64.matrix(n, d)
	local scale = 0
	for a = 1, d do
		for k = 1, n do
			local v = X[(k - 1) * d + a] - mean[a]
			buffer.writef64(C.data, 8 * ((a - 1) * n + k - 1), v)
			scale = math.max(scale, math.abs(v))
		end
	end
	local axes: { number } = {}
	if scale > 0 then
		local F = f64.qrp(C)
		local rank = f64.qrp_rank(F, 1e-10)
		for j = 1, rank do
			table.insert(axes, buffer.readi32(F.perm, 4 * (j - 1)))
		end
		table.sort(axes)
	end
	local dims = #axes
	local h = 0
	if dims > 0 then
		local vol = 1
		for _, a in axes do
			local lo, hi = X[a], X[a]
			for k = 2, n do
				local v: number = X[(k - 1) * d + a]
				if v < lo then
					lo = v
				elseif v > hi then
					hi = v
				end
			end
			vol *= hi - lo
		end
		h = (vol / n) ^ (1 / dims)
	end
	local eps = o.epsilon or (if h > 0 then 1 / (3 * h) else 1)
	local phi, dphi, definite = kernel_functions(name, eps, o.nu or 2.5)
	local np = if n > dims + 1 then dims + 1 else 0
	local method = o.method or "auto"
	if method == "auto" then
		method = if definite and dims <= 2 and n > HODLR_ABOVE then "hodlr" else "dense"
	elseif method ~= "dense" and method ~= "hodlr" then
		fail("rbf", "method must be \"auto\", \"dense\" or \"hodlr\"")
	end
	if method == "hodlr" and not definite then
		fail("rbf", "method \"hodlr\" needs a positive definite kernel (gaussian or matern)")
	end
	local smoothing = o.smoothing or 0
	if smoothing ~= "auto" and not (type(smoothing) == "number" and smoothing >= 0) then
		fail("rbf", "smoothing must be a nonnegative number or \"auto\"")
	end
	local P = f64.matrix(n, np) -- the polynomial's columns: 1, then the spanned coordinates
	for k = 1, n do
		if np > 0 then
			buffer.writef64(P.data, 8 * (k - 1), 1)
			for c, a in axes do
				buffer.writef64(P.data, 8 * (c * n + k - 1), X[(k - 1) * d + a])
			end
		end
	end
	local yv = f64.create(n)
	for k = 1, n do
		f64.set(yv, k, values[k])
	end

	-- A factored K + lambda I: factor(lambda) (false if not positive definite), solve(b),
	-- logdet(), apply(x) = K x, and |K|.
	local factor: (number) -> boolean, solve: (buffer) -> buffer, logdet: () -> number, apply: (buffer) -> buffer
	local normK: number
	local floor = 0
	local ranks: { number }? = nil
	if method == "hodlr" then
		local pts = table.create(n)
		for k = 1, n do
			pts[k] = table.move(X, (k - 1) * d + 1, k * d, 1, table.create(d, 0))
		end
		local tol = o.tol or 1e-10
		local H = hodlr.new(pts, phi, { tol = tol })
		ranks = H.ranks
		normK = H:norm()
		floor = 2 * tol * normK
		factor = function(lambda: number): boolean
			return (pcall(function()
				H:factor(lambda)
			end))
		end
		solve = function(b: buffer): buffer
			return H:solve(b)
		end
		logdet = function(): number
			return H:logdet()
		end
		apply = function(x: buffer): buffer
			return H:matvec(x)
		end
	else
		local K = f64.matrix(n, n)
		normK = 0
		for j = 1, n do
			local s = 0
			for i = 1, n do
				local r2 = 0
				for a = 1, d do
					r2 += (X[(i - 1) * d + a] - X[(j - 1) * d + a]) ^ 2
				end
				local v = phi(math.sqrt(r2))
				buffer.writef64(K.data, 8 * ((j - 1) * n + i - 1), v)
				s += math.abs(v)
			end
			normK = math.max(normK, s)
		end
		if definite then
			floor = n * EPS * normK
		end
		local F: any = nil
		local ld = 0
		factor = function(lambda: number): boolean
			if definite then
				local A = f64.matcopy(nil, K)
				for i = 0, n - 1 do
					local off = 8 * (i * n + i)
					buffer.writef64(A.data, off, buffer.readf64(A.data, off) + lambda)
				end
				F = f64.chol(A)
				if not F.definite then
					return false
				end
				ld = 0
				for i = 0, n - 1 do
					ld += 2 * math.log(buffer.readf64(A.data, 8 * (i * n + i)))
				end
				return true
			end
			-- not positive definite (thin_plate, cubic, multiquadric): the whole system
			-- [K + lambda I, P; P^T, 0] by LU, as K alone can be singular
			local N = n + np
			local A = f64.matrix(N, N)
			for j = 0, n - 1 do
				buffer.copy(A.data, 8 * j * N, K.data, 8 * j * n, 8 * n)
				local off = 8 * (j * N + j)
				buffer.writef64(A.data, off, buffer.readf64(A.data, off) + lambda)
				for c = 0, np - 1 do
					local v = buffer.readf64(P.data, 8 * (c * n + j))
					buffer.writef64(A.data, 8 * (j * N + n + c), v)
					buffer.writef64(A.data, 8 * ((n + c) * N + j), v)
				end
			end
			F = f64.lu(A)
			return not F.singular
		end
		solve = function(b: buffer): buffer
			return if definite then f64.chol_solve(nil, F, b) else f64.lu_solve(nil, F, b)
		end
		logdet = function(): number
			return ld
		end
		apply = function(x: buffer): buffer
			return f64.matvec(nil, K, x)
		end
	end

	-- The weights for one lambda: a = A^-1 y, Q = A^-1 P, c = (P^T Q)^-1 P^T a, w = a - Q c
	-- (A = K + lambda I, positive definite); for the other kernels, from the whole system.
	local function weights(): (buffer, buffer)
		local c = f64.create(np)
		if method == "dense" and not definite then
			local rhs = f64.create(n + np)
			buffer.copy(rhs, 0, yv, 0, 8 * n)
			local sol = solve(rhs)
			local w = f64.create(n)
			buffer.copy(w, 0, sol, 0, 8 * n)
			buffer.copy(c, 0, sol, 8 * n, 8 * np)
			return w, c
		end
		local a = solve(yv)
		if np > 0 then
			local Q = f64.matrix(n, np)
			for j = 0, np - 1 do
				local col = f64.create(n)
				buffer.copy(col, 0, P.data, 8 * j * n, 8 * n)
				buffer.copy(Q.data, 8 * j * n, solve(col), 0, 8 * n)
			end
			local PtQ = f64.gemm(nil, 1, P, Q, 0, "t")
			local Pta = f64.create(np)
			for j = 0, np - 1 do
				local s = 0
				for k = 0, n - 1 do
					s += buffer.readf64(P.data, 8 * (j * n + k)) * buffer.readf64(a, 8 * k)
				end
				buffer.writef64(Pta, 8 * j, s)
			end
			local G = f64.lu(PtQ)
			if G.singular then
				fail("rbf", "the polynomial part is singular: do two points coincide?")
			end
			c = f64.lu_solve(nil, G, Pta)
			for j = 0, np - 1 do
				local cj = buffer.readf64(c, 8 * j)
				for k = 0, n - 1 do
					buffer.writef64(a, 8 * k, buffer.readf64(a, 8 * k) - buffer.readf64(Q.data, 8 * (j * n + k)) * cj)
				end
			end
		end
		return a, c
	end

	local lambda: number
	if smoothing == "auto" then
		-- Maximum likelihood: y - (its least squares linear fit) as a sample of a Gaussian
		-- process with covariance s^2 (K + lambda I), s^2 at its best for each lambda:
		-- maximize -(n / 2) log(r^T (K + lambda I)^-1 r) - (1 / 2) log det(K + lambda I)
		-- over log lambda, from the floor (or 1e-12 |K|) to |K|.
		if not definite then
			fail("rbf", "smoothing = \"auto\" needs a positive definite kernel (gaussian or matern)")
		end
		local res = f64.copy(nil, yv)
		if np > 0 then
			local F = f64.qr_copy(P)
			local c = f64.qr_solve(nil, F, yv)
			for j = 0, np - 1 do
				local cj = buffer.readf64(c, 8 * j)
				for k = 0, n - 1 do
					buffer.writef64(res, 8 * k, buffer.readf64(res, 8 * k) - buffer.readf64(P.data, 8 * (j * n + k)) * cj)
				end
			end
		end
		local function likelihood(t: number): number
			if not factor(10 ^ t) then
				return -math.huge
			end
			local q = f64.dot(res, solve(res))
			return -(n / 2) * math.log(math.max(q, 1e-300)) - logdet() / 2
		end
		local lo, hi = math.log10(math.max(floor, 1e-12 * normK)), math.log10(normK)
		local best, bestt = -math.huge, lo
		local steps = 6
		local grid = {}
		for i = 0, steps do
			local t = lo + (hi - lo) * i / steps
			grid[i] = likelihood(t)
			if grid[i] > best then
				best, bestt = grid[i], t
			end
		end
		-- golden section between the best grid point's neighbors
		local a, b = math.max(lo, bestt - (hi - lo) / steps), math.min(hi, bestt + (hi - lo) / steps)
		local g = (math.sqrt(5) - 1) / 2
		local x1, x2 = b - g * (b - a), a + g * (b - a)
		local f1, f2 = likelihood(x1), likelihood(x2)
		for _ = 1, 8 do
			if f1 > f2 then
				b, x2, f2 = x2, x1, f1
				x1 = b - g * (b - a)
				f1 = likelihood(x1)
			else
				a, x1, f1 = x1, x2, f2
				x2 = a + g * (b - a)
				f2 = likelihood(x2)
			end
		end
		local t = if f1 > best or f2 > best then (if f1 > f2 then x1 else x2) else bestt
		lambda = 10 ^ t
	else
		lambda = math.max(smoothing :: number, floor)
	end
	if not factor(lambda) then
		if method == "dense" and not definite then
			fail("rbf", "the system is singular: do two points coincide?")
		end
		fail("rbf", string.format("K + %g I is not positive definite (points too close together?)", lambda))
	end
	local w, c = weights()
	-- the misfit at the points: K w + P c - y
	local fit = apply(w)
	local misfit = 0
	for k = 0, n - 1 do
		local v = buffer.readf64(fit, 8 * k)
		for j = 0, np - 1 do
			v += buffer.readf64(P.data, 8 * (j * n + k)) * buffer.readf64(c, 8 * j)
		end
		misfit = math.max(misfit, math.abs(v - buffer.readf64(yv, 8 * k)))
	end
	local info: RbfInfo = {
		method = method,
		kernel = name,
		epsilon = if name == "thin_plate" or name == "cubic" then nil else eps,
		smoothing = lambda,
		residual = if vmax > 0 then misfit / vmax else misfit,
		ranks = ranks,
	}
	local R = setmetatable({ X = X, d = d, weights = w, poly = c, axes = axes, n = n, phi = phi, dphi = dphi, _q = table.create(d, 0), _g = table.create(d, 0) }, Rbf) :: any
	return R, info
end

-- The value at p (in the points' form). Two and three coordinates have their own loops
-- (16.5 and 18.2 us a value at 1000 points natively, against 20.4 and 23.7 for the general
-- one).
function Rbf.at(self: Rbf, p: any): number
	local s = self :: any
	local d, X, w, phi, q = s.d, s.X, s.weights, s.phi, s._q
	if not read_point(p, d, q, 0) then
		fail("rbf", string.format("the interpolant's points have %d coordinates: so must p", d))
	end
	local f = 0
	if d == 3 then
		local qx, qy, qz = q[1], q[2], q[3]
		for k = 1, s.n do
			local b = 3 * (k - 1)
			local dx, dy, dz = qx - X[b + 1], qy - X[b + 2], qz - X[b + 3]
			f += buffer.readf64(w, 8 * (k - 1)) * phi(math.sqrt(dx * dx + dy * dy + dz * dz))
		end
	elseif d == 2 then
		local qx, qy = q[1], q[2]
		for k = 1, s.n do
			local b = 2 * (k - 1)
			local dx, dy = qx - X[b + 1], qy - X[b + 2]
			f += buffer.readf64(w, 8 * (k - 1)) * phi(math.sqrt(dx * dx + dy * dy))
		end
	else
		for k = 1, s.n do
			local b = d * (k - 1)
			local r2 = 0
			for a = 1, d do
				local e = q[a] - X[b + a]
				r2 += e * e
			end
			f += buffer.readf64(w, 8 * (k - 1)) * phi(math.sqrt(r2))
		end
	end
	local c = s.poly
	if buffer.len(c) > 0 then
		f += buffer.readf64(c, 0)
		for j, a in s.axes do
			f += buffer.readf64(c, 8 * j) * q[a]
		end
	end
	return f
end

-- The gradient at p: the d partial derivatives.
function Rbf.gradient(self: Rbf, p: any): ...number
	local s = self :: any
	local d, X, w, dphi, q, g = s.d, s.X, s.weights, s.dphi, s._q, s._g
	if not read_point(p, d, q, 0) then
		fail("rbf", string.format("the interpolant's points have %d coordinates: so must p", d))
	end
	for a = 1, d do
		g[a] = 0
	end
	for k = 1, s.n do
		local b = d * (k - 1)
		local r2 = 0
		for a = 1, d do
			local e = q[a] - X[b + a]
			r2 += e * e
		end
		local t = buffer.readf64(w, 8 * (k - 1)) * dphi(math.sqrt(r2))
		for a = 1, d do
			g[a] += t * (q[a] - X[b + a])
		end
	end
	local c = s.poly
	if buffer.len(c) > 0 then
		for j, a in s.axes do
			g[a] += buffer.readf64(c, 8 * j)
		end
	end
	return table.unpack(g, 1, d)
end

return {
	grid = grid,
	rbf = rbf,
}
