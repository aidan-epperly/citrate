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
		check(math.abs(gx - fx) < 1e-6 and math.abs(gy - fy) < 1e-6 and gz == 0, string.format("%s gradient: %.1e, %.1e", name, gx - fx, gy - fy))
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

return nil
