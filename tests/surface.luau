--!strict

-- Surfaces and multidimensional interpolation against exact answers: bilinear and planar
-- patches, Bezier corner derivatives, splines through their grids, derivatives against
-- differences, consistent mesh winding; grids exact on bilinear (linear method) and
-- quadratic (cubic method) data with third-order convergence; radial basis functions
-- through their data and exact on linear data.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local surface, interp = lib.surface, lib.interp

local function grid_of(m: number, n: number, f: (number, number) -> number): { { vector } }
	local g = {}
	for i = 1, m do
		g[i] = {}
		for j = 1, n do
			g[i][j] = vector.create(i - 1, j - 1, f(i - 1, j - 1))
		end
	end
	return g
end

local function errors(f: () -> (), pattern: string): boolean
	local ok, message = pcall(f)
	return not ok and string.find(tostring(message), pattern, 1, true) ~= nil
end

test("surface: patches", function()
	-- a bilinear Bezier patch is bilinear interpolation
	local c = { { vector.create(0, 0, 0), vector.create(0, 2, 1) }, { vector.create(3, 0, 2), vector.create(3, 2, 7) } }
	local B = surface.new(c, { kind = "bezier" })
	local p = surface.eval(B, 0.25, 0.5)
	local expect = c[1][1] * 0.75 * 0.5 + c[2][1] * 0.25 * 0.5 + c[1][2] * 0.75 * 0.5 + c[2][2] * 0.25 * 0.5
	check(vector.magnitude(p - expect) < 1e-5, "bilinear patch")
	-- a biquadratic Bezier patch: dS/du at a corner is 2 (P10 - P00)
	local q = grid_of(3, 3, function(x: number, y: number): number
		return x * y - y * y
	end)
	local Q = surface.new(q, { kind = "bezier" })
	local su, sv = surface.derivatives(Q, 0, 0)
	check(vector.magnitude(su - 2 * (q[2][1] - q[1][1])) < 1e-5 and vector.magnitude(sv - 2 * (q[1][2] - q[1][1])) < 1e-5, "corner derivatives")
	-- a B-spline of a flat grid stays flat, its corners on the grid's corners, its normal up
	local flat = grid_of(6, 5, function(): number
		return 0
	end)
	local S = surface.new(flat)
	local u0, u1, v0, v1 = surface.domain(S)
	check(u0 == 0 and u1 == 3 and v0 == 0 and v1 == 2, "B-spline domain")
	local inplane = true
	for _, uv in { { 0.3, 0.7 }, { 2.9, 1.2 }, { 1.5, 0 } } do
		inplane = inplane and math.abs(surface.eval(S, uv[1], uv[2]).z) < 1e-6
	end
	check(inplane and vector.magnitude(surface.eval(S, u1, v1) - flat[6][5]) < 1e-5, "flat B-spline")
	check(math.abs(surface.normal(S, 1, 1).z - 1) < 1e-6, "normal")
	-- the spline surface passes through every grid point and is accurate between them
	local f = function(x: number, y: number): number
		return math.sin(0.6 * x) * math.cos(0.4 * y)
	end
	local g = grid_of(9, 8, f)
	local P = surface.new(g, { kind = "spline" })
	local through = true
	for i = 1, 9 do
		for j = 1, 8 do
			through = through and vector.magnitude(surface.eval(P, i - 1, j - 1) - g[i][j]) < 1e-5
		end
	end
	check(through, "spline through its grid")
	local worst = 0
	for _, uv in { { 3.5, 2.5 }, { 4.25, 4.75 }, { 2.5, 5.5 } } do
		local pt = surface.eval(P, uv[1], uv[2])
		worst = math.max(worst, math.abs(pt.z - f(pt.x, pt.y)))
	end
	check(worst < 2e-3, string.format("spline between points (%.1e)", worst))
	-- derivatives against differences
	local du, dv = surface.derivatives(P, 3.3, 2.2)
	local e = 1e-3
	local fu = (surface.eval(P, 3.3 + e, 2.2) - surface.eval(P, 3.3 - e, 2.2)) / (2 * e)
	local fv = (surface.eval(P, 3.3, 2.2 + e) - surface.eval(P, 3.3, 2.2 - e)) / (2 * e)
	check(vector.magnitude(du - fu) < 1e-3 and vector.magnitude(dv - fv) < 1e-3, "derivatives")
	-- a mesh: sizes, unit normals, triangles wound with them
	local points, triangles, normals = surface.mesh(P, 6, 5)
	check(#points == 42 and #triangles == 60 and #normals == 42, "mesh sizes")
	local wound = true
	for _, t in triangles do
		local a, b, cc = points[t[1]], points[t[2]], points[t[3]]
		wound = wound and vector.dot(vector.cross(b - a, cc - a), normals[t[1]]) > 0
	end
	check(wound, "mesh winding")
	check(errors(function()
		surface.new({ { vector.zero, vector.zero }, { vector.zero } })
	end, "row 2 has 1 points"), "a ragged grid")
	check(errors(function()
		surface.new(flat, { kind = "nurbs" })
	end, "kind must be"), "a bad kind")
end)

test("interp: grids", function()
	-- bilinear data: the linear method is exact
	local function table_of(nx: number, ny: number, dx: number, dy: number, f: (number, number) -> number): { { number } }
		local t = {}
		for i = 1, nx do
			t[i] = {}
			for j = 1, ny do
				t[i][j] = f((i - 1) * dx, (j - 1) * dy)
			end
		end
		return t
	end
	local bil = function(x: number, y: number): number
		return 1 + 2 * x + 3 * y + 4 * x * y
	end
	local L = interp.grid(table_of(5, 4, 0.5, 2, bil), { dx = 0.5, dy = 2, method = "linear" })
	close(L:at(1.3, 3.7), bil(1.3, 3.7), 1e-12, "bilinear exact")
	local gx, gy = L:gradient(1.3, 3.7)
	close(gx, 2 + 4 * 3.7, 1e-12, "bilinear gradient x")
	close(gy, 3 + 4 * 1.3, 1e-12, "bilinear gradient y")
	-- quadratic data: cubic convolution is exact (Keys' edge rule included)
	local quad = function(x: number, y: number): number
		return x * x + x * y - 2 * y * y + 3
	end
	local C = interp.grid(table_of(7, 6, 1, 1, quad))
	local worst = 0
	for _, xy in { { 0.2, 0.3 }, { 5.7, 4.9 }, { 3.5, 2.25 }, { 0, 5 } } do
		worst = math.max(worst, math.abs(C:at(xy[1], xy[2]) - quad(xy[1], xy[2])))
	end
	check(worst < 1e-12, string.format("cubic exact on quadratics (%.1e)", worst))
	local cx, cy = C:gradient(2.5, 1.5)
	close(cx, 2 * 2.5 + 1.5, 1e-12, "cubic gradient x")
	close(cy, 2.5 - 4 * 1.5, 1e-12, "cubic gradient y")
	-- sample: value and slope together, inside and in the edge cells (Keys' rule)
	worst = 0
	for _, xy in { { 2.5, 1.5 }, { 0.2, 0.3 }, { 5.7, 4.9 }, { 0.4, 3.1 }, { 4.2, 4.6 } } do
		local x, y = xy[1], xy[2]
		local f, fx, fy = C:sample(x, y)
		worst = math.max(worst, math.abs(f - quad(x, y)), math.abs(fx - (2 * x + y)), math.abs(fy - (x - 4 * y)))
	end
	check(worst < 1e-12, string.format("cubic sample exact on quadratics (%.1e)", worst))
	-- third-order convergence on a smooth function
	local s = function(x: number, y: number): number
		return math.sin(3 * x) * math.cos(2 * y)
	end
	local function err(n: number): number
		local d = 1 / (n - 1)
		local G = interp.grid(table_of(n, n, d, d, s), { dx = d, dy = d })
		local e = 0
		for k = 1, 50 do
			local x, y = (k * 0.6180339) % 1, (k * 0.4142135) % 1
			e = math.max(e, math.abs(G:at(x, y) - s(x, y)))
		end
		return e
	end
	local e1, e2 = err(17), err(33)
	check(e1 / e2 > 6, string.format("third order: %.1e then %.1e", e1, e2))
	-- clamped outside; a Matrix works too
	close(L:at(-5, -5), bil(0, 0), 1e-12, "clamped")
	local M = lib.f64.matrix(3, 3)
	lib.f64.matset(M, 2, 3, 5)
	close(interp.grid(M, { method = "linear" }):at(1, 2), 5, 1e-14, "Matrix values")
end)

test("interp: grids in any dimension", function()
	-- values f(i, j, k, ...) at (i - 1) h_a, ... as nested tables d deep
	local function nested(shape: { number }, h: { number }, f: ({ number }) -> number): any
		local function build(axis: number, at: { number }): any
			local t = {}
			for i = 1, shape[axis] do
				at[axis] = (i - 1) * h[axis]
				t[i] = if axis == #shape then f(at) else build(axis + 1, at)
			end
			return t
		end
		return build(1, {})
	end
	-- the general evaluation against the 2-D one: a 3-D grid constant along z
	local g = function(x: number, y: number): number
		return math.sin(2 * x) * math.exp(-y) + x * y
	end
	for _, method in { "cubic", "linear" } do
		local G2 = interp.grid(nested({ 6, 5 }, { 0.5, 0.75 }, function(p)
			return g(p[1], p[2])
		end), { dx = 0.5, dy = 0.75, method = method })
		local G3 = interp.grid(nested({ 6, 5, 4 }, { 0.5, 0.75, 1 }, function(p)
			return g(p[1], p[2])
		end), { spacing = { 0.5, 0.75, 1 }, method = method })
		local worst = 0
		for k = 1, 40 do
			local x, y, z = 2.5 * ((k * 0.618) % 1), 3 * ((k * 0.414) % 1), 3 * ((k * 0.732) % 1)
			local f2, fx2, fy2 = G2:sample(x, y)
			local f3, fx3, fy3, fz3 = G3:sample(x, y, z)
			worst = math.max(worst, math.abs(f2 - f3), math.abs(fx2 - fx3), math.abs(fy2 - fy3), math.abs(fz3))
		end
		check(worst < 1e-13, string.format("%s: the general evaluation is the 2-D one (%.1e)", method, worst))
	end
	-- cubic convolution is exact on quadratics, multilinear on multilinear data, in 1 to 4-D
	for d = 1, 4 do
		local shape, h = {}, {}
		for a = 1, d do
			shape[a], h[a] = if d == 4 then 4 else 5, 0.5 + 0.25 * a
		end
		local quad = function(p: { number }): number
			local v = 1
			for a = 1, #p do
				v += a * p[a] + p[a] * p[a] / a + (if a > 1 then p[a] * p[a - 1] else 0)
			end
			return v
		end
		local multi = function(p: { number }): number
			local v = 1
			for a = 1, #p do
				v *= 1 + a * p[a]
			end
			return v
		end
		local C = interp.grid(nested(shape, h, quad), { spacing = h })
		local L = interp.grid(nested(shape, h, multi), { spacing = h, method = "linear" })
		local worst = 0
		for k = 1, 30 do
			local p = {}
			for a = 1, d do
				p[a] = (shape[a] - 1) * h[a] * ((k * (0.618 + 0.1 * a)) % 1)
			end
			local f, gx = C:sample(table.unpack(p))
			local ex = 1 + 2 * p[1] + (if d > 1 then p[2] else 0)
			worst = math.max(worst, math.abs(f - quad(p)), math.abs(gx - ex), math.abs(L:at(table.unpack(p)) - multi(p)))
		end
		check(worst < 1e-11, string.format("%d-D: exact on quadratics and multilinear data (%.1e)", d, worst))
	end
	-- third order in 3-D
	local s = function(p: { number }): number
		return math.sin(3 * p[1]) * math.cos(2 * p[2]) * math.exp(p[3])
	end
	local function err(n: number): number
		local h = 1 / (n - 1)
		local G = interp.grid(nested({ n, n, n }, { h, h, h }, s), { spacing = { h, h, h } })
		local e = 0
		for k = 1, 50 do
			local p = { (k * 0.6180339) % 1, (k * 0.4142135) % 1, (k * 0.7320508) % 1 }
			e = math.max(e, math.abs(G:at(p[1], p[2], p[3]) - s(p)))
		end
		return e
	end
	local e1, e2 = err(9), err(17)
	check(e1 / e2 > 6, string.format("3-D third order: %.1e then %.1e", e1, e2))
	-- an f64 array with a shape is the nested table
	local T = nested({ 4, 3, 3 }, { 1, 1, 1 }, function(p)
		return p[1] + 10 * p[2] + 100 * p[3] + p[1] * p[3]
	end)
	local flat = lib.f64.create(36)
	for k = 0, 2 do
		for j = 0, 2 do
			for i = 0, 3 do
				lib.f64.set(flat, 1 + i + 4 * j + 12 * k, T[i + 1][j + 1][k + 1])
			end
		end
	end
	close(interp.grid(flat, { shape = { 4, 3, 3 } }):at(1.3, 0.4, 1.7), interp.grid(T):at(1.3, 0.4, 1.7), 1e-14, "an f64 array with a shape")
	check(pcall(function()
		interp.grid(T):at(1, 2)
	end) == false, "the wrong number of coordinates")
end)

test("interp: radial basis functions", function()
	local pts, vals = {}, {}
	local f = function(x: number, y: number): number
		return math.exp(-(x * x + y * y)) + 0.5 * x
	end
	local k = 0
	for i = -4, 4 do
		for j = -4, 4 do
			k += 1
			local x, y = i / 2 + 0.1 * math.sin(k), j / 2 + 0.1 * math.cos(k)
			pts[k] = { x, y }
			vals[k] = f(x, y)
		end
	end
	for _, kernel in { "thin_plate", "cubic", "gaussian", "multiquadric" } do
		local R = interp.rbf(pts, vals, { kernel = kernel, epsilon = 1.5 })
		local through = 0
		for i = 1, #pts do
			through = math.max(through, math.abs(R:at(pts[i]) - vals[i]))
		end
		local between = math.abs(R:at({ 0.3, -0.4 }) - f(0.3, -0.4))
		check(through < 1e-9 and between < 0.05, string.format("%s: through %.1e, between %.1e", kernel, through, between))
	end
	-- linear data is reproduced exactly (the polynomial part), in 3-D too
	local p3, v3 = {}, {}
	for i = 1, 30 do
		local p = vector.create(math.sin(i), math.cos(2 * i), math.sin(3 * i))
		p3[i] = p
		v3[i] = 2 - p.x + 3 * p.y + 0.5 * p.z
	end
	local R3 = interp.rbf(p3, v3)
	local q = vector.create(0.2, -0.1, 0.4)
	close(R3:at(q), 2 - q.x + 3 * q.y + 0.5 * q.z, 1e-6, "linear data in 3-D")
	-- smoothing fits approximately
	local Rs = interp.rbf(pts, vals, { smoothing = 1 })
	check(math.abs(Rs:at(pts[41]) - vals[41]) > 1e-6, "smoothing does not interpolate")
	check(errors(function()
		interp.rbf({ { 0, 0 }, { 0, 0 }, { 1, 0 }, { 0, 1 } }, { 1, 2, 3, 4 })
	end, "coincide"), "coincident points")
end)

test("interp: rbf kernels, scales, gradients, HODLR and smoothing", function()
	local f = function(x: number, y: number): number
		return math.sin(x / 15) * math.cos(y / 20) + x / 100
	end
	-- points in studs: the default epsilon follows their spacing (about 10 here)
	local rng = lib.random.new(3)
	local pts, vals = {}, {}
	for k = 1, 100 do
		local x, y = rng:uniform(0, 100), rng:uniform(0, 100)
		pts[k], vals[k] = { x, y }, f(x, y)
	end
	local function err(R: any): number
		local e = 0
		for k = 1, 100 do
			local x, y = 10 + 80 * ((k * 0.618) % 1), 10 + 80 * ((k * 0.414) % 1)
			e = math.max(e, math.abs(R:at({ x, y }) - f(x, y)))
		end
		return e
	end
	for _, c: any in { { "thin_plate", 0.05 }, { "cubic", 0.05 }, { "gaussian", 0.01 }, { "multiquadric", 0.01 }, { "matern", 0.03 } } do
		local name: string, limit: number = c[1], c[2]
		local R, info = interp.rbf(pts, vals, { kernel = name })
		check(info.method == "dense" and info.residual < 1e-6 and err(R) < limit, string.format("%s at its default scale: error %.1e, residual %.1e", name, err(R), info.residual))
		-- the gradient against central differences
		local q = { 43.2, 61.7 }
		local gx, gy, gz = R:gradient(q)
		local d = 1e-4
		local fx = (R:at({ q[1] + d, q[2] }) - R:at({ q[1] - d, q[2] })) / (2 * d)
		local fy = (R:at({ q[1], q[2] + d }) - R:at({ q[1], q[2] - d })) / (2 * d)
		check(math.abs(gx - fx) < 1e-6 and math.abs(gy - fy) < 1e-6 and gz == nil, string.format("%s gradient: %.1e, %.1e", name, gx - fx, gy - fy))
	end
	for _, c in { { 0.5, 0.2 }, { 1.5, 0.05 } } do
		local R = interp.rbf(pts, vals, { kernel = "matern", nu = c[1] })
		check(err(R) < c[2], string.format("matern nu = %g: error %.1e", c[1], err(R)))
	end
	-- points along a diagonal span one dimension: the polynomial is 1 and x, not 1, x, y
	local line, lv = {}, {}
	for k = 1, 6 do
		line[k], lv[k] = { k, k }, 2 * k + 1
	end
	local L = interp.rbf(line, lv)
	close(L:at({ 3.5, 3.5 }), 8, 1e-9, "points on a diagonal: linear data along it")
	-- HODLR gives the dense answer
	local Rd = interp.rbf(pts, vals, { kernel = "matern" })
	local Rh, ih = interp.rbf(pts, vals, { kernel = "matern", method = "hodlr" })
	local worst = 0
	for k = 1, 50 do
		local q = { 100 * ((k * 0.618) % 1), 100 * ((k * 0.414) % 1) }
		worst = math.max(worst, math.abs(Rh:at(q) - Rd:at(q)))
	end
	check(ih.method == "hodlr" and worst < 1e-6, string.format("HODLR against dense: %.1e", worst))
	check(errors(function()
		interp.rbf(pts, vals, { kernel = "cubic", method = "hodlr" })
	end, "positive definite"), "HODLR needs a positive definite kernel")
	-- noisy values: smoothing = "auto" fits through the noise
	local noisy = {}
	for k = 1, 100 do
		noisy[k] = vals[k] + rng:normal(0, 0.05)
	end
	local R0 = interp.rbf(pts, noisy, { kernel = "matern" })
	local Ra, ia = interp.rbf(pts, noisy, { kernel = "matern", smoothing = "auto" })
	check(err(Ra) < err(R0) and ia.smoothing > 1e-4, string.format("auto smoothing: error %.3f (lambda %.1e) against %.3f", err(Ra), ia.smoothing, err(R0)))
end)

test("interp: rbf and hodlr in any dimension", function()
	-- a smooth function of 4 and 6 coordinates at Halton points, as tables and f64 arrays
	for _, d in { 1, 4, 6 } do
		local f = function(p: { number }): number
			local v = 0
			for a = 1, #p do
				v += math.sin(a * p[a]) / a
			end
			return v
		end
		local q = lib.random.halton(d)
		local pts, bufs, vals = {}, {}, {}
		local n = if d == 1 then 30 else 300
		for k = 1, n do
			local p = q:next()
			pts[k], bufs[k], vals[k] = p, lib.f64.from(p), f(p)
		end
		local R, info = interp.rbf(pts, vals, { kernel = "gaussian", smoothing = 1e-10 })
		local Rb = interp.rbf(bufs, vals, { kernel = "gaussian", smoothing = 1e-10 })
		local worst, same = 0, 0
		local test_point = {}
		for k = 1, 40 do
			for a = 1, d do
				test_point[a] = 0.2 + 0.6 * ((k * (0.618 + 0.13 * a)) % 1)
			end
			worst = math.max(worst, math.abs(R:at(test_point) - f(test_point)))
			same = math.max(same, math.abs(R:at(test_point) - Rb:at(lib.f64.from(test_point))))
		end
		-- the gradient against central differences, in every coordinate
		local g = { R:gradient(test_point) }
		local gerr = 0
		for a = 1, d do
			local up, down = table.clone(test_point), table.clone(test_point)
			up[a] += 1e-5
			down[a] -= 1e-5
			gerr = math.max(gerr, math.abs(g[a] - (R:at(up) - R:at(down)) / 2e-5))
		end
		check(#g == d and worst < (if d == 1 then 1e-6 else 0.05) and same == 0 and gerr < 1e-5 and info.residual < 1e-6,
			string.format("%d-D rbf: error %.1e, gradient %.1e", d, worst, gerr))
	end
	-- HODLR in 3-D on tables: its solves match the dense matrix's
	local q = lib.random.halton(3)
	local pts = {}
	for k = 1, 600 do
		pts[k] = q:next()
	end
	local kern = function(r: number): number
		return math.exp(-4 * r * r)
	end
	local H = lib.hodlr.new(pts, kern, { tol = 1e-10, leaf = 32 })
	H:factor(1e-6)
	local x = lib.f64.create(600)
	for k = 1, 600 do
		lib.f64.set(x, k, math.sin(k))
	end
	local y = H:matvec(x)
	local worst = 0
	for _, i in { 1, 77, 300, 599 } do
		local s = 0
		for j = 1, 600 do
			local r2 = 0
			for a = 1, 3 do
				r2 += (pts[i][a] - pts[j][a]) ^ 2
			end
			s += kern(math.sqrt(r2)) * lib.f64.get(x, j)
		end
		worst = math.max(worst, math.abs(s - lib.f64.get(y, i)))
	end
	local back = H:solve(lib.f64.axpy(nil, 1e-6, x, y))
	local err = lib.f64.amax(lib.f64.sub(nil, back, x))
	check(worst < 1e-8 and err < 1e-4, string.format("3-D HODLR on tables: product %.1e, solve %.1e", worst, err))
end)

return nil
