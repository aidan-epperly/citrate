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
	gradient: (self: Rbf, p: any) -> (number, number, number),
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

-- A point's coordinates, in double precision (a vector's are single).
local function point(p: any, k: number): (number, number, number)
	if type(p) == "vector" then
		return p.x, p.y, p.z
	elseif type(p) == "table" and type(p[1]) == "number" then
		return p[1], p[2] or 0, p[3] or 0
	end
	error(string.format("interp.rbf: point %d is not a vector or { x, y, z }", k), 4)
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

-- An interpolant of values at scattered points (vectors, or tables { x, y } / { x, y, z }) by
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
	local X, Y, Z = table.create(n, 0) :: { number }, table.create(n, 0) :: { number }, table.create(n, 0) :: { number }
	local vmax = 0
	for k = 1, n do
		X[k], Y[k], Z[k] = point(points[k], k)
		local v = values[k]
		if type(v) ~= "number" or v ~= v or v == math.huge or v == -math.huge then
			fail("rbf", string.format("value %d is not a finite number", k))
		end
		vmax = math.max(vmax, math.abs(v))
	end
	-- The coordinates the points span (by rank, so points along a diagonal count as one
	-- dimension), for the polynomial, the spacing and the method.
	local coords = { X, Y, Z }
	local mean = { 0, 0, 0 }
	for a = 1, 3 do
		for k = 1, n do
			mean[a] += coords[a][k] / n
		end
	end
	local C = f64.matrix(n, 3)
	local scale = 0
	for a = 1, 3 do
		for k = 1, n do
			local v = coords[a][k] - mean[a]
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
			local col: { number } = coords[a]
			local lo, hi = col[1], col[1]
			for k = 2, n do
				local v: number = col[k]
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
				buffer.writef64(P.data, 8 * (c * n + k - 1), coords[a][k])
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
			pts[k] = { X[k], Y[k], Z[k] }
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
				local r = math.sqrt((X[i] - X[j]) ^ 2 + (Y[i] - Y[j]) ^ 2 + (Z[i] - Z[j]) ^ 2)
				local v = phi(r)
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
	local R = setmetatable({ X = X, Y = Y, Z = Z, weights = w, poly = c, axes = axes, n = n, phi = phi, dphi = dphi }, Rbf) :: any
	return R, info
end

function Rbf.at(self: Rbf, p: any): number
	local qx, qy, qz = point(p, 0)
	local s = self :: any
	local X, Y, Z, w, phi = s.X, s.Y, s.Z, s.weights, s.phi
	local f = 0
	for k = 1, s.n do
		local dx, dy, dz = qx - X[k], qy - Y[k], qz - Z[k]
		f += buffer.readf64(w, 8 * (k - 1)) * phi(math.sqrt(dx * dx + dy * dy + dz * dz))
	end
	local c = s.poly
	if buffer.len(c) > 0 then
		f += buffer.readf64(c, 0)
		for j, a in s.axes do
			f += buffer.readf64(c, 8 * j) * (if a == 1 then qx elseif a == 2 then qy else qz)
		end
	end
	return f
end

-- The gradient (df/dx, df/dy, df/dz) at p.
function Rbf.gradient(self: Rbf, p: any): (number, number, number)
	local qx, qy, qz = point(p, 0)
	local s = self :: any
	local X, Y, Z, w, dphi = s.X, s.Y, s.Z, s.weights, s.dphi
	local gx, gy, gz = 0, 0, 0
	for k = 1, s.n do
		local dx, dy, dz = qx - X[k], qy - Y[k], qz - Z[k]
		local d = buffer.readf64(w, 8 * (k - 1)) * dphi(math.sqrt(dx * dx + dy * dy + dz * dz))
		gx += d * dx
		gy += d * dy
		gz += d * dz
	end
	local c = s.poly
	if buffer.len(c) > 0 then
		for j, a in s.axes do
			local cj = buffer.readf64(c, 8 * j)
			if a == 1 then
				gx += cj
			elseif a == 2 then
				gy += cj
			else
				gz += cj
			end
		end
	end
	return gx, gy, gz
end

return {
	grid = grid,
	rbf = rbf,
}
