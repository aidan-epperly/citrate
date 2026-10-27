--!strict

-- Root finding and optimization: scalar roots and minima by every method, then least
-- squares, minimization and systems on the Moré-Garbow-Hillstrom problems (tests/mgh.luau),
-- with exact and finite-difference derivatives, on f64 and f32 arrays and vectors.

local h = require("./harness")
local lib = require("../src")
local mgh = require("./mgh")

local test, check = h.test, h.check
local opt, f64, f32, vec = lib.optimize, lib.f64, lib.f32, lib.vec

local function tab(b: buffer): { number }
	local t = {}
	for k = 1, buffer.len(b) // 8 do
		t[k] = buffer.readf64(b, 8 * (k - 1))
	end
	return t
end

local function solved(v: number, fstar: number): boolean
	return v <= fstar + 1e-5 * math.max(1e-3, fstar) + 1e-9
end

test("scalar roots: Brent, Chebyshev and AAA", function()
	local cases = {
		{ function(x: number): number return math.cos(x) - x end, 0, 1, 0.7390851332151607 },
		{ function(x: number): number return x ^ 3 - 2 * x - 5 end, 2, 3, 2.0945514815423265 },
		{ function(x: number): number return 1 / (x - 1.01) + 1.5 end, -1, 0.9, 1.01 - 1 / 1.5 },
	}
	for _, c in cases do
		local f, a, b, want = c[1] :: (number) -> number, c[2] :: number, c[3] :: number, c[4] :: number
		for _, m in { "brent", "chebyshev", "aaa" } do
			local x, info = opt.root_scalar(f, a, b, { method = m })
			check(x ~= nil and math.abs((x :: number) - want) <= 1e-10, `{m}: root {x} (want {want})`)
			check(info.evaluations > 0 and info.converged, `{m}: info`)
		end
	end
	-- every root of sin(10 x) in [0.1, 1]: 0.314.., 0.628.., 0.942..
	for _, m in { "chebyshev", "aaa" } do
		local rs = opt.roots(function(x: number): number
			return math.sin(10 * x)
		end, 0.1, 1, { method = m })
		local ok = #rs == 3
		for k = 1, 3 do
			ok = ok and math.abs((rs[k] or 0) - k * math.pi / 10) <= 1e-10
		end
		check(ok, `{m}: all three roots ({#rs} found)`)
	end
	-- Brent needs a bracket
	check(not pcall(opt.root_scalar, function(x: number): number
		return x * x + 1
	end, -1, 1, nil), "brent rejects an interval without a sign change")
	-- Newton (with f') and the secant method
	local x1, i1 = opt.newton(function(x: number): number
		return x * x - 2
	end, 1, { df = function(x: number): number
		return 2 * x
	end })
	check(math.abs(x1 - math.sqrt(2)) <= 1e-14 and i1.iterations <= 7, `newton: {x1} in {i1.iterations}`)
	local x2, i2 = opt.newton(function(x: number): number
		return x * x - 2
	end, 1)
	check(math.abs(x2 - math.sqrt(2)) <= 1e-12 and i2.method == "secant", `secant: {x2}`)
end)

test("scalar minima: Brent is local, the interpolants global", function()
	-- two minima; the global one at x = 0.7727 (right), Brent's start finds the left
	local f = function(x: number): number
		return x ^ 4 - x ^ 2 - 0.3 * x
	end
	local xb = opt.minimize_scalar(f, -1.5, 1.5)
	check(math.abs(xb - 0.7726979) <= 1e-5 or math.abs(xb + 0.6327) <= 1e-3, `brent: a local minimum ({xb})`)
	for _, m in { "chebyshev", "aaa" } do
		local x, fx = opt.minimize_scalar(f, -1.5, 1.5, { method = m })
		check(math.abs(x - 0.7726979) <= 1e-5 and fx <= f(0.7726979) + 1e-12, `{m}: the global minimum ({x})`)
	end
end)

-- the MGH problems on f64 arrays
local function adapt(p: any): (any, any, any, any)
	local function r(x: buffer, out: buffer?): buffer
		local v = p.r(tab(x))
		local o = out or buffer.create(8 * p.m)
		for i = 1, p.m do
			buffer.writef64(o, 8 * (i - 1), v[i])
		end
		return o
	end
	local function jac(x: buffer, J: any): any
		local v = p.jac(tab(x))
		for i = 1, p.m do
			for j = 1, p.n do
				buffer.writef64(J.data, 8 * ((j - 1) * p.m + i - 1), v[i][j])
			end
		end
		return J
	end
	local function f(x: buffer): number
		local v = p.r(tab(x))
		local s = 0
		for i = 1, p.m do
			s += v[i] ^ 2
		end
		return 0.5 * s
	end
	local function grad(x: buffer, out: buffer): buffer
		local xt = tab(x)
		local v, J = p.r(xt), p.jac(xt)
		for j = 1, p.n do
			local s = 0
			for i = 1, p.m do
				s += J[i][j] * v[i]
			end
			buffer.writef64(out, 8 * (j - 1), s)
		end
		return out
	end
	return r, jac, f, grad
end

test("least squares: Levenberg-Marquardt on the MGH problems", function()
	for _, p in mgh.problems do
		local r, jac = adapt(p)
		for _, lam in { "newton", "aaa" } do
			local _, info = opt.least_squares(f64.space, r, f64.from(p.x0), { jac = jac, lambda = lam })
			check(solved(info.cost, p.fstar) and info.converged, `{p.name} ({lam}): cost {info.cost}, {info.message}`)
		end
		-- finite-difference Jacobians
		local _, info = opt.least_squares(f64.space, r, f64.from(p.x0))
		check(solved(info.cost, p.fstar), `{p.name} (differences): cost {info.cost}`)
	end
end)

test("minimization: L-BFGS line searches and Nelder-Mead", function()
	-- every problem but Wood (Wolfe stops at its stationary point, as SciPy's does)
	-- reaches its minimum with the Wolfe search and tight tolerances
	for _, p in mgh.problems do
		local _, _, f, grad = adapt(p)
		if p.name ~= "Wood" then
			local _, fx, info = opt.minimize(f64.space, f, f64.from(p.x0), { grad = grad, gtol = 1e-9, ftol = 1e-15 })
			check(solved(fx, p.fstar), `{p.name} (wolfe): f {fx}, {info.message}`)
		end
	end
	-- the interpolating searches, on a few (Wood included: they see past the saddle)
	for _, name in { "Rosenbrock", "Wood", "Bard", "Helical valley" } do
		for _, p in mgh.problems do
			if p.name == name then
				local _, _, f, grad = adapt(p)
				for _, ls in { "chebyshev", "aaa" } do
					local _, fx = opt.minimize(f64.space, f, f64.from(p.x0), { grad = grad, line_search = ls })
					check(solved(fx, p.fstar), `{name} ({ls}): f {fx}`)
				end
			end
		end
	end
	-- finite-difference gradients
	local p = mgh.problems[1]
	local _, _, f = adapt(p)
	local _, fx, info = opt.minimize(f64.space, f, f64.from(p.x0))
	check(fx <= 1e-8 and info.gradients > 0, `Rosenbrock with differences: f {fx}`)
	-- Nelder-Mead, as SciPy's (159 evaluations on Rosenbrock)
	local _, fn, ni = opt.minimize(f64.space, f, f64.from(p.x0), { method = "nelder-mead" })
	check(fn <= 1e-8 and ni.evaluations <= 200, `Nelder-Mead: f {fn} in {ni.evaluations}`)
end)

test("systems: Newton's method", function()
	for _, p in mgh.problems do
		if p.root then
			local r, jac = adapt(p)
			for _, ls in { "backtracking", "chebyshev" } do
				local _, info = opt.root(f64.space, r, f64.from(p.x0), { jac = jac, line_search = ls })
				check(info.converged and info.norm <= 1e-9, `{p.name} ({ls}): |F| {info.norm}, {info.message}`)
			end
			local _, info = opt.root(f64.space, r, f64.from(p.x0))
			check(info.converged, `{p.name} (differences): |F| {info.norm}`)
		end
	end
end)

test("optimization on f32 arrays and vectors", function()
	-- vectors: the minimum of |x - (1, 2, 3)|^2 + (x . x)^2 / 100 with its gradient
	local c = vector.create(1, 2, 3)
	local function fv(x: vector): number
		local d = x - c
		return vector.dot(d, d) + vector.dot(x, x) ^ 2 / 100
	end
	local function gv(x: vector, _: vector): vector
		return 2 * (x - c) + x * (4 * vector.dot(x, x) / 100)
	end
	local xv, _, iv = opt.minimize(vec.space, fv, vector.zero, { grad = gv })
	check(iv.converged and vector.magnitude(gv(xv, vector.zero)) <= 1e-4, `vector minimize: {xv}`)
	-- vectors without a gradient, and a vector system (x^2 - 2, y - x, z^3 - 8)
	local xd = opt.minimize(vec.space, fv, vector.zero)
	check(vector.magnitude(xd - xv) <= 1e-3, `vector minimize, differences: {xd}`)
	local xs, is = opt.root(vec.space, function(x: vector, _: vector): vector
		return vector.create(x.x * x.x - 2, x.y - x.x, x.z ^ 3 - 8)
	end, vector.create(1, 1, 1))
	check(is.converged and math.abs(xs.x - math.sqrt(2)) <= 1e-6 and math.abs(xs.z - 2) <= 1e-6, `vector root: {xs}`)
	-- f32 arrays: least squares on Rosenbrock
	local p = mgh.problems[1]
	local function r32(x: buffer, out: buffer?): buffer
		local v = p.r({ buffer.readf32(x, 0), buffer.readf32(x, 4) })
		local o = out or buffer.create(16)
		buffer.writef64(o, 0, v[1])
		buffer.writef64(o, 8, v[2])
		return o
	end
	local x32, i32 = opt.least_squares(f32.space, r32, f32.from(p.x0))
	check(math.abs(buffer.readf32(x32, 0) - 1) <= 1e-3 and math.abs(buffer.readf32(x32, 4) - 1) <= 1e-3, `f32 least squares: cost {i32.cost}`)
end)

return nil
