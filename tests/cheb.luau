--!strict

-- Chebyshev technology: points, barycentric interpolation, transforms, adaptive
-- construction, calculus, roots, extrema, spectral differentiation and quadrature.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local cheb, f64 = lib.cheb, lib.f64
local EPS = 2 ^ -52

local function maxerr(F: lib.Chebfun, f: (number) -> number): number
	local e = 0
	for k = 0, 1000 do
		local x = F.a + (F.b - F.a) * k / 1000
		e = math.max(e, math.abs(cheb.eval(F, x) - f(x)))
	end
	return e
end

local function poly(cs: { number }): (number) -> number
	return function(x: number): number
		local s = 0
		for i = #cs, 1, -1 do
			s = s * x + cs[i]
		end
		return s
	end
end

test("Chebyshev points and weights", function()
	local x = f64.totable(cheb.chebpts(5))
	check(x[1] == -1 and x[5] == 1 and x[3] == 0, "second kind: endpoints and middle exact")
	close(x[2], -math.sqrt(0.5), 4 * EPS, "second kind: interior point")
	check(x[2] == -x[4], "second kind: exactly symmetric")
	local y = f64.totable(cheb.chebpts(3, 1))
	close(y[1], -math.sqrt(3) / 2, 4 * EPS, "first kind")
	check(y[2] == 0 and y[1] == -y[3], "first kind: symmetric")
	local z = f64.totable(cheb.chebpts(3, 2, 2, 6))
	check(z[1] == 2 and z[2] == 4 and z[3] == 6, "mapped to [2, 6]")
	check(f64.totable(cheb.chebpts(1))[1] == 0, "one point")
	local w = f64.totable(cheb.chebweights(4))
	check(w[1] == 0.5 and w[2] == -1 and w[3] == 1 and w[4] == -0.5, "second-kind weights")
end)

test("barycentric interpolation", function()
	-- Interpolation through n points reproduces polynomials of degree < n exactly.
	local p = poly({ 0.5, -2, 0, 3 })
	for _, kind in { 1, 2 } do
		local x = cheb.chebpts(6, kind)
		local v = f64.create(6)
		for j = 1, 6 do
			f64.set(v, j, p(f64.get(x, j)))
		end
		local w = cheb.chebweights(6, kind)
		local worst = 0
		for k = 0, 100 do
			local t = -1 + 2 * k / 100
			worst = math.max(worst, math.abs(cheb.bary(t, x, v, w) - p(t)))
		end
		check(worst <= 20 * EPS, `kind {kind}: cubic reproduced (error {worst})`)
		check(cheb.bary(f64.get(x, 3), x, v, w) == f64.get(v, 3), `kind {kind}: exact at a node`)
		-- The closed-form weights agree with the general formula (up to scaling).
		local g = cheb.bary_weights(x)
		local ratio = f64.get(g, 1) / f64.get(w, 1)
		local same = true
		for j = 1, 6 do
			same = same and math.abs(f64.get(g, j) - ratio * f64.get(w, j)) <= 1e-13
		end
		check(same, `kind {kind}: closed-form weights match bary_weights`)
	end
	-- Arbitrary (equispaced) nodes with general weights.
	local nodes = f64.from({ -1, -0.5, 0, 0.5, 1 })
	local vals = f64.create(5)
	for j = 1, 5 do
		f64.set(vals, j, p(f64.get(nodes, j)))
	end
	close(cheb.bary(0.3, nodes, vals), p(0.3), 50 * EPS, "equispaced nodes, default weights")
	check(not pcall(function()
		cheb.bary_weights(f64.from({ 0, 1, 1 }))
	end), "repeated nodes are an error")
end)

test("coefficient transforms and Clenshaw", function()
	math.randomseed(121)
	for _, n in { 1, 2, 5, 17, 40 } do
		local c = f64.create(n)
		for j = 1, n do
			f64.set(c, j, 2 * math.random() - 1)
		end
		local v = cheb.coeffs2vals(c)
		local back = cheb.vals2coeffs(v)
		check(f64.norm(f64.sub(nil, back, c)) <= 50 * EPS * math.sqrt(n), `n={n}: vals2coeffs(coeffs2vals(c)) = c`)
		-- The values agree with Clenshaw at the points, and barycentric agrees in between.
		local x = cheb.chebpts(n)
		local worst = 0
		for j = 1, n do
			worst = math.max(worst, math.abs(cheb.clenshaw(c, f64.get(x, j)) - f64.get(v, j)))
		end
		check(worst <= 50 * EPS * n, `n={n}: values match Clenshaw ({worst})`)
		if n > 1 then
			local w = cheb.chebweights(n)
			close(cheb.bary(0.123, x, v, w), cheb.clenshaw(c, 0.123), 100 * EPS * n, `n={n}: barycentric = Clenshaw`)
		end
	end
	-- T_3 sampled at 4 points has coefficients e_4.
	local x = cheb.chebpts(4)
	local v = f64.create(4)
	for j = 1, 4 do
		local t = f64.get(x, j)
		f64.set(v, j, 4 * t ^ 3 - 3 * t)
	end
	local c = f64.totable(cheb.vals2coeffs(v))
	check(math.abs(c[1]) + math.abs(c[2]) + math.abs(c[3]) <= 10 * EPS and math.abs(c[4] - 1) <= 10 * EPS, "T_3")
end)

test("standard_chop", function()
	-- Geometric decay to a noise floor at 1e-16: cut where the plateau starts (52, as the
	-- reference implementation of the algorithm gives).
	local c = f64.create(100)
	for j = 1, 100 do
		f64.set(c, j, math.max(0.5 ^ (j - 1), 1e-16 * (1 + (j % 3))))
	end
	check(cheb.standard_chop(c) == 52, `cut at the plateau ({cheb.standard_chop(c)})`)
	-- With too little of the plateau visible, nothing is chopped.
	local short = f64.create(60)
	buffer.copy(short, 0, c, 0, 8 * 60)
	check(cheb.standard_chop(short) == 60, "plateau not established: not chopped")
	check(cheb.standard_chop(f64.create(10)) == 10, "short series are not chopped")
	local z = f64.create(20)
	f64.set(z, 1, 3)
	check(cheb.standard_chop(z) == 1, "a constant chops to length 1")
end)

test("adaptive construction", function()
	-- Lengths as in Chebfun for its standard examples.
	local cases: { { any } } = {
		{ "exp", math.exp, 15 },
		{ "sin", math.sin, 14 },
		{ "x^3", poly({ 0, 0, 0, 1 }), 4 },
		{ "constant", function(_: number): number
			return 2.5
		end, 1 },
	}
	for _, case in cases do
		local F = cheb.fun(case[2])
		check(F.happy and cheb.length(F) == case[3], `{case[1]}: length {cheb.length(F)} (Chebfun: {case[3]})`)
		check(maxerr(F, case[2]) <= 10 * EPS, `{case[1]}: accurate to machine precision`)
	end
	local runge = function(x: number): number
		return 1 / (1 + 25 * x * x)
	end
	local R = cheb.fun(runge)
	check(R.happy and cheb.length(R) > 150 and cheb.length(R) < 220 and maxerr(R, runge) <= 10 * EPS, `Runge: length {cheb.length(R)}`)
	local E = cheb.fun(function(x: number): number
		return math.exp(-x)
	end, { 0, 10 })
	check(E.a == 0 and E.b == 10 and maxerr(E, function(x: number): number
		return math.exp(-x)
	end) <= 10 * EPS, "on [0, 10]")
	local loose = cheb.fun(math.exp, nil, { tol = 1e-6 })
	check(cheb.length(loose) < 15 and maxerr(loose, math.exp) <= 1e-5, `tol = 1e-6: length {cheb.length(loose)}`)
	local fixed = cheb.fun(math.exp, nil, { n = 7 })
	check(cheb.length(fixed) == 7, "fixed n")
	-- A function that is not smooth cannot be resolved; the result says so.
	local S = cheb.fun(math.abs, nil, { maxlength = 257 })
	check(not S.happy and cheb.length(S) == 257, "|x| is not resolved (happy = false)")
	check(not pcall(function()
		cheb.fun(function(x: number): number
			return 1 / x
		end, { 0, 1 })
	end), "a non-finite sample is an error")
end)

test("arithmetic and calculus", function()
	local S, Cs = cheb.fun(math.sin), cheb.fun(math.cos)
	check(maxerr(cheb.add(S, Cs), function(x: number): number
		return math.sin(x) + math.cos(x)
	end) <= 10 * EPS, "add")
	check(maxerr(cheb.sub(S, S), function(_: number): number
		return 0
	end) == 0 and cheb.length(cheb.sub(S, S)) == 1, "F - F is exactly zero, length 1")
	check(maxerr(cheb.mul(S, Cs), function(x: number): number
		return math.sin(2 * x) / 2
	end) <= 10 * EPS, "mul")
	check(maxerr(cheb.scale(S, 3), function(x: number): number
		return 3 * math.sin(x)
	end) <= 10 * EPS, "scale")
	check(maxerr(cheb.shift(S, 1), function(x: number): number
		return math.sin(x) + 1
	end) <= 10 * EPS, "shift")
	check(maxerr(cheb.diff(S), math.cos) <= 100 * EPS, "diff(sin) = cos")
	check(maxerr(cheb.diff(S, 2), function(x: number): number
		return -math.sin(x)
	end) <= 1e-11, "second derivative")
	check(maxerr(cheb.cumsum(Cs), function(x: number): number
		return math.sin(x) - math.sin(-1)
	end) <= 10 * EPS, "cumsum(cos) = sin - sin(-1)")
	close(cheb.sum(cheb.fun(math.exp)), math.exp(1) - math.exp(-1), 10 * EPS, "sum(exp)")
	local G = cheb.fun(function(x: number): number
		return math.exp(-x * x)
	end, { -6, 6 })
	close(cheb.sum(G), math.sqrt(math.pi), 1e-14, "Gaussian integral on [-6, 6]")
	local D = cheb.fun(math.exp, { 1, 3 })
	check(maxerr(cheb.diff(D), math.exp) <= 1e3 * EPS * math.exp(3), "diff on [1, 3]")
	close(cheb.sum(D), math.exp(3) - math.exp(1), 10 * EPS, "sum on [1, 3]")
	close(cheb.norm(cheb.fun(function(x: number): number
		return x
	end)), math.sqrt(2 / 3), 10 * EPS, "L2 norm of x")
	close(cheb.inner(S, Cs), 0, 10 * EPS, "sin and cos are orthogonal on [-1, 1]")
	check(not pcall(function()
		cheb.add(S, D)
	end), "different domains are an error")
end)

test("roots and extrema", function()
	for _, w in { 5, 100, 300 } do
		local F = cheb.fun(function(x: number): number
			return math.sin(w * x)
		end)
		local r = cheb.roots(F)
		local worst, seen, distinct = 0, {}, true
		for j = 1, f64.length(r) do
			local x = f64.get(r, j)
			local k = math.round(x * w / math.pi)
			distinct = distinct and not seen[k]
			seen[k] = true
			worst = math.max(worst, math.abs(x - k * math.pi / w))
		end
		local expected = 2 * math.floor(w / math.pi) + 1
		check(f64.length(r) == expected and distinct and worst <= 1e-14,
			`sin({w}x): {f64.length(r)} roots ({expected} expected), worst error {worst}`)
	end
	local C = cheb.fun(math.cos, { 0, 10 })
	local r = f64.totable(cheb.roots(C))
	check(#r == 3, "cos on [0, 10] has 3 roots")
	close(r[1], math.pi / 2, 1e-14)
	close(r[3], 5 * math.pi / 2, 1e-14, "cos roots")
	-- A double root.
	local Q = cheb.fun(function(x: number): number
		return (x - 0.3) ^ 2
	end)
	local q = f64.totable(cheb.roots(Q))
	check(#q == 1 and math.abs(q[1] - 0.3) <= 1e-7, `double root found once ({#q})`)
	local B = cheb.fun(function(x: number): number
		return x * (1 - x)
	end, { 0, 1 })
	local mx, at = cheb.max(B)
	close(mx, 0.25, 4 * EPS)
	close(at, 0.5, 1e-12, "max of x (1 - x)")
	local mn, where = cheb.min(cheb.fun(function(x: number): number
		return math.sin(3 * x) + 0.1 * x
	end))
	check(mn < -0.9 and math.abs(3 * math.cos(3 * where) + 0.1) <= 1e-10, `min at a critical point ({where})`)
	local lo, lox, hi, hix = cheb.minmax(cheb.fun(function(x: number): number
		return x
	end, { 2, 5 }))
	check(lo == 2 and lox == 2 and hi == 5 and hix == 5, "extrema at the endpoints")
end)

test("spectral differentiation and quadrature", function()
	local n = 12
	local x = cheb.chebpts(n, 2, 0, 2)
	local D = cheb.diffmat(n, 0, 2)
	local v = f64.create(n)
	for j = 1, n do
		f64.set(v, j, math.exp(f64.get(x, j)))
	end
	local dv = f64.matvec(nil, D, v)
	local worst = 0
	for j = 1, n do
		worst = math.max(worst, math.abs(f64.get(dv, j) - math.exp(f64.get(x, j))))
	end
	check(worst <= 1e-8, `diffmat differentiates exp on [0, 2] spectrally ({worst})`)
	local ones = f64.matvec(nil, D, f64.fill(f64.create(n), 1))
	check(f64.norm(ones) <= 1e-12, "rows sum to zero")

	for _, m in { 2, 5, 8, 9 } do
		local qx, qw = cheb.ccquad(m)
		local exact = true
		for k = 0, m - 1 do
			local s = 0
			for j = 1, m do
				s += f64.get(qw, j) * f64.get(qx, j) ^ k
			end
			exact = exact and math.abs(s - (1 - (-1) ^ (k + 1)) / (k + 1)) <= 1e-13
		end
		check(exact, `Clenshaw-Curtis with {m} points is exact to degree {m - 1}`)
		local gx, gw = cheb.legpts(m)
		local gexact = true
		for k = 0, 2 * m - 1 do
			local s = 0
			for j = 1, m do
				s += f64.get(gw, j) * f64.get(gx, j) ^ k
			end
			gexact = gexact and math.abs(s - (1 - (-1) ^ (k + 1)) / (k + 1)) <= 1e-13
		end
		check(gexact, `Gauss-Legendre with {m} points is exact to degree {2 * m - 1}`)
	end
	local gx, gw = cheb.legpts(3)
	close(f64.get(gx, 1), -math.sqrt(0.6), 1e-15)
	close(f64.get(gw, 2), 8 / 9, 1e-15, "3-point Gauss-Legendre")
	local cx, cw = cheb.ccquad(17, 0, math.pi)
	local s = 0
	for j = 1, 17 do
		s += f64.get(cw, j) * math.sin(f64.get(cx, j))
	end
	close(s, 2, 1e-13, "Clenshaw-Curtis on [0, pi]")
end)

return nil
