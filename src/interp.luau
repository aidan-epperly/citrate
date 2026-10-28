--!strict
--!native
--!optimize 2

-- Interpolation in more than one dimension: values on a regular grid (a heightfield, a
-- texture), and values at scattered points.
--
--   local H = interp.grid(heights, { dx = 4, dy = 4 })   -- heights[i][j] at (4 (i - 1), 4 (j - 1))
--   H:at(10.5, 7.2)                                      -- smooth between grid points
--   local fx, fy = H:gradient(10.5, 7.2)                 -- the slope there
--   local f, fx, fy = H:sample(10.5, 7.2)                -- both, for one evaluation
--
--   local R = interp.rbf(points, values)                 -- any points (vectors)
--   R:at(vector.create(1, 2, 0))
--
-- (Curves, one dimension, are in citrate.curve; surfaces through grids of points in
-- citrate.surface.)

local types = require("./types")
local f64 = require("./f64")

type Matrix = types.Matrix

local function fail(who: string, message: string): never
	error("interp." .. who .. ": " .. message, 3)
end

-- Grids

export type GridOptions = { x0: number?, y0: number?, dx: number?, dy: number?, method: string? }
export type Grid = {
	nx: number,
	ny: number,
	x0: number,
	y0: number,
	dx: number,
	dy: number,
	method: string,
	values: { { number } },
	at: (self: Grid, x: number, y: number) -> number,
	gradient: (self: Grid, x: number, y: number) -> (number, number),
	sample: (self: Grid, x: number, y: number) -> (number, number, number),
}

local Grid = {}
Grid.__index = Grid

-- An interpolant of values on a regular grid: values[i][j] (or a Matrix's entry (i, j)) is
-- the value at (x0 + (i - 1) dx, y0 + (j - 1) dy) (x0, y0 default 0; dx, dy default 1).
-- options.method:
--   "cubic" (the default)  Keys' cubic convolution (a = -1/2, as Catmull-Rom): it passes
--                          through the values, its slope is continuous, and its error falls
--                          as the cube of the spacing; it can overshoot a little near
--                          sharp steps;
--   "linear"               bilinear: never overshoots, but the slope jumps at grid lines.
-- Points outside the grid are clamped to its edge. Needs at least 2 points each way (3 for
-- cubic; with 2, linear is used).
local function grid(values: any, options: GridOptions?): Grid
	local o: GridOptions = options or {}
	local method = o.method or "cubic"
	if method ~= "cubic" and method ~= "linear" then
		fail("grid", "method must be \"cubic\" or \"linear\"")
	end
	local rows: { { number } } = {}
	if type(values) == "table" and values.data ~= nil then
		local M = values :: Matrix
		for i = 1, M.rows do
			rows[i] = {}
			for j = 1, M.cols do
				rows[i][j] = buffer.readf64(M.data, 8 * ((j - 1) * M.rows + i - 1))
			end
		end
	elseif type(values) == "table" and type(values[1]) == "table" then
		rows = values
	else
		fail("grid", "values must be a table of rows or an f64 Matrix")
	end
	local nx, ny = #rows, #rows[1]
	if nx < 2 or ny < 2 then
		fail("grid", "need at least 2 by 2 values")
	end
	local copy = table.create(nx) :: { { number } }
	for i = 1, nx do
		if type(rows[i]) ~= "table" or #rows[i] ~= ny then
			fail("grid", string.format("row %d has a different length from row 1", i))
		end
		copy[i] = table.create(ny, 0)
		for j = 1, ny do
			local v = rows[i][j]
			if type(v) ~= "number" or v ~= v or v == math.huge or v == -math.huge then
				fail("grid", string.format("value (%d, %d) is not a finite number", i, j))
			end
			copy[i][j] = v
		end
	end
	local dx, dy = o.dx or 1, o.dy or 1
	if not (dx > 0 and dy > 0) then
		fail("grid", "dx and dy must be positive")
	end
	if nx < 3 or ny < 3 then
		method = "linear"
	end
	return setmetatable({ nx = nx, ny = ny, x0 = o.x0 or 0, y0 = o.y0 or 0, dx = dx, dy = dy, method = method, values = copy }, Grid) :: any
end

-- The value at (i, j), with Keys' boundary rule past the edges: f(-1) = 3 f(0) - 3 f(1) +
-- f(2), which keeps the third-order accuracy there.
local function value(G: Grid, i: number, j: number): number
	local nx, ny, v = G.nx, G.ny, G.values
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
		local v = G.values
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
	local v = G.values
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

function Grid.at(self: Grid, x: number, y: number): number
	local f = evaluate(self, x, y)
	return f
end

function Grid.gradient(self: Grid, x: number, y: number): (number, number)
	local _, fx, fy = evaluate(self, x, y)
	return fx, fy
end

-- The value and slope together (one evaluation), as a heightfield's user often needs both.
function Grid.sample(self: Grid, x: number, y: number): (number, number, number)
	return evaluate(self, x, y)
end

-- Scattered points

export type RbfOptions = { kernel: string?, epsilon: number?, smoothing: number? }
export type Rbf = { at: (self: Rbf, p: any) -> number, [string]: any }

local Rbf = {}
Rbf.__index = Rbf

-- A point's coordinates, in double precision (a vector's are single).
local function point(p: any, k: number): (number, number, number)
	if type(p) == "vector" then
		return p.x, p.y, p.z
	elseif type(p) == "table" and type(p[1]) == "number" then
		return p[1], p[2] or 0, p[3] or 0
	end
	error(string.format("interp.rbf: point %d is not a vector or { x, y, z }", k), 4)
end

local KERNELS: { [string]: (number, number) -> number } = {
	thin_plate = function(r: number, _: number): number
		return if r > 0 then r * r * math.log(r) else 0
	end,
	cubic = function(r: number, _: number): number
		return r * r * r
	end,
	gaussian = function(r: number, e: number): number
		return math.exp(-(e * r) ^ 2)
	end,
	multiquadric = function(r: number, e: number): number
		return math.sqrt(1 + (e * r) ^ 2)
	end,
}

-- An interpolant of values at scattered points (vectors, or tables { x, y } / { x, y, z })
-- by radial basis functions: f(p) = sum of w_k phi(|p - p_k|) plus a linear polynomial,
-- with the weights solving a dense linear system (so it suits hundreds of points, not
-- many thousands: the solve costs n^3 / 3). options.kernel:
--   "thin_plate" (the default)  r^2 log r: the smoothest surface through the points (it
--                               bends least); exact for linear data;
--   "cubic"                     r^3;
--   "gaussian", "multiquadric"  exp(-(epsilon r)^2), sqrt(1 + (epsilon r)^2), with
--                               options.epsilon (default 1) setting the scale: smaller is
--                               smoother.
-- options.smoothing > 0 fits the data approximately instead (larger: smoother), for noisy
-- values. Errors if two points coincide.
local function rbf(points: { any }, values: { number }, options: RbfOptions?): Rbf
	local o: RbfOptions = options or {}
	local kernel = o.kernel or "thin_plate"
	local phi = KERNELS[kernel]
	if phi == nil then
		fail("rbf", "kernel must be \"thin_plate\", \"cubic\", \"gaussian\" or \"multiquadric\"")
	end
	local n = #points
	if n < 1 or #values ~= n then
		fail("rbf", "need one value per point, and at least one point")
	end
	local eps = o.epsilon or 1
	local smooth = o.smoothing or 0
	local X, Y, Z = table.create(n, 0) :: { number }, table.create(n, 0) :: { number }, table.create(n, 0) :: { number }
	local usey, usez = false, false
	for k = 1, n do
		X[k], Y[k], Z[k] = point(points[k], k)
		local v = values[k]
		if type(v) ~= "number" or v ~= v then
			fail("rbf", string.format("value %d is not a number", k))
		end
		usey = usey or Y[k] ~= Y[1]
		usez = usez or Z[k] ~= Z[1]
	end
	local coords = { x = X, y = Y, z = Z }
	-- the polynomial: 1, x and those of y, z that vary (fewer when the points lie on a line
	-- or in a plane), if there are points enough
	local cols = { "x" }
	if usey then
		table.insert(cols, "y")
	end
	if usez then
		table.insert(cols, "z")
	end
	local np = if n > #cols then 1 + #cols else 0
	local N = n + np
	local A = f64.matrix(N, N)
	local function set(i: number, j: number, v: number)
		buffer.writef64(A.data, 8 * ((j - 1) * N + i - 1), v)
	end
	for i = 1, n do
		for j = 1, n do
			local r = math.sqrt((X[i] - X[j]) ^ 2 + (Y[i] - Y[j]) ^ 2 + (Z[i] - Z[j]) ^ 2)
			set(i, j, phi(r, eps) + (if i == j then smooth else 0))
		end
		if np > 0 then
			set(i, n + 1, 1)
			set(n + 1, i, 1)
			for c, axis in cols do
				local v = coords[axis][i]
				set(i, n + 1 + c, v)
				set(n + 1 + c, i, v)
			end
		end
	end
	local rhs = f64.create(N)
	for k = 1, n do
		f64.set(rhs, k, values[k])
	end
	local F = f64.lu_copy(A)
	if F.singular then
		fail("rbf", "the system is singular: do two points coincide?")
	end
	local w = f64.lu_solve(nil, F, rhs)
	return setmetatable({ X = X, Y = Y, Z = Z, weights = w, n = n, np = np, cols = cols, phi = phi, eps = eps }, Rbf) :: any
end

function Rbf.at(self: Rbf, p: any): number
	local qx, qy, qz = point(p, 0)
	local s = self :: any
	local X, Y, Z = s.X, s.Y, s.Z
	local f = 0
	for k = 1, s.n do
		local r = math.sqrt((qx - X[k]) ^ 2 + (qy - Y[k]) ^ 2 + (qz - Z[k]) ^ 2)
		f += f64.get(s.weights, k) * s.phi(r, s.eps)
	end
	if s.np > 0 then
		local q: { [string]: number } = { x = qx, y = qy, z = qz }
		f += f64.get(s.weights, s.n + 1)
		for c, axis in s.cols do
			f += f64.get(s.weights, s.n + 1 + c) * q[axis]
		end
	end
	return f
end

return {
	grid = grid,
	rbf = rbf,
}
