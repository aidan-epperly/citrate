--!strict

-- Special functions against SciPy's values (tests/special_data.luau), the Gauss-Kronrod
-- rule's exactness, adaptive integration on smooth, kinked, singular and infinite
-- integrals, and the normal distribution's functions.

local h = require("./harness")
local lib = require("../src")
local D = require("./special_data")

local test, check, close = h.test, h.check, h.close
local special, quad, stats = lib.special, lib.quad, lib.stats

local function rel(a: number, b: number): number
	return if b == 0 then math.abs(a) else math.abs(a - b) / math.abs(b)
end
local function scaled(a: number, b: number): number
	return math.abs(a - b) / math.max(1, math.abs(b))
end

test("special: gamma, erf and incomplete functions against SciPy", function()
	local tolerances: { [string]: number } = { gamma = 2e-13, lgamma = 1e-14, erf = 5e-15, erfc = 5e-14 }
	for name, tol in tolerances do
		local worst, at = 0, 0
		for _, r: any in (D :: any)[name] do
			local e = rel((special :: any)[name](r[1]), r[2])
			if e > worst then
				worst, at = e, r[1]
			end
		end
		check(worst < tol, string.format("%s: %.1e at %g", name, worst, at))
	end
	local w = 0
	for _, r in D.gammainc do
		w = math.max(w, scaled(special.gamma_inc(r[1], r[2]), r[3]), rel(special.gamma_inc(r[1], r[2], true), r[4]))
	end
	check(w < 1e-13, string.format("gamma_inc: %.1e", w))
	w = 0
	for _, r in D.betainc do
		w = math.max(w, scaled(special.beta_inc(r[1], r[2], r[3]), r[4]))
	end
	check(w < 1e-13, string.format("beta_inc: %.1e", w))
	check(special.gamma(5) == 24 and special.gamma(1) == 1, "factorials exact")
	close(special.beta(2, 3), 1 / 12, 1e-15, "beta")
	check(special.erf(-0.5) == -special.erf(0.5), "erf is odd")
	check(special.gamma(-2) ~= special.gamma(-2), "a pole gives NaN")
end)

test("special: Bessel functions against SciPy", function()
	for _, name in { "besselj", "bessely", "besseli", "besselk" } do
		local worst: number, at: string = 0, ""
		for _, r: any in (D :: any)[name] do
			local v = (special :: any)[name](r[1], r[2])
			-- J and Y oscillate: their error is measured against max(1, |value|)
			local e = if name == "besselj" or name == "bessely" then scaled(v, r[3]) else rel(v, r[3])
			if e > worst then
				worst, at = e, string.format("n = %d, x = %g", r[1], r[2])
			end
		end
		check(worst < 5e-14, string.format("%s: %.1e at %s", name, worst, at :: string))
	end
	-- tiny values keep their relative accuracy (J by its series)
	close(special.besselj(10, 1e-3) / 2.6911443943049774e-40, 1, 1e-12, "J10(0.001)")
	-- the first zero of J0, and symmetry
	check(math.abs(special.besselj(0, 2.404825557695773)) < 2e-15, "a zero of J0 (to about 1e-15 absolutely)")
	close(special.besselj(3, -2), -special.besselj(3, 2), 1e-16, "J3 is odd")
	check(not pcall(function()
		special.besselj(1.5, 2)
	end), "integer orders only")
end)

test("quad: rules and adaptive integration", function()
	-- the 15-point Kronrod rule is exact to degree 22, the 7-point Gauss rule to 13
	local function rule(degree: number, gauss: boolean): number
		local s = 0
		if gauss then
			s = quad.WG[4] * (if degree == 0 then 1 else 0)
			for j = 2, 6, 2 do
				local x = quad.XGK[j]
				s += quad.WG[j // 2] * (x ^ degree + (-x) ^ degree)
			end
		else
			s = quad.WGK[8] * (if degree == 0 then 1 else 0)
			for j = 1, 7 do
				local x = quad.XGK[j]
				s += quad.WGK[j] * (x ^ degree + (-x) ^ degree)
			end
		end
		return s - (if degree % 2 == 0 then 2 / (degree + 1) else 0)
	end
	local ek, eg = 0, 0
	for d = 0, 22 do
		ek = math.max(ek, math.abs(rule(d, false)))
	end
	for d = 0, 13 do
		eg = math.max(eg, math.abs(rule(d, true)))
	end
	check(ek < 1e-15 and eg < 1e-15, "exactness of the rules")
	local cases: { any } = {
		{ "Gaussian on the line", function(x: number): number
			return math.exp(-x * x)
		end, -math.huge, math.huge, math.sqrt(math.pi) },
		{ "a kink", function(x: number): number
			return math.abs(x - 1 / 3)
		end, 0, 1, 5 / 18 },
		{ "oscillation", function(x: number): number
			return math.cos(100 * x)
		end, 0, 1, math.sin(100) / 100 },
		{ "a half line", function(x: number): number
			return 1 / (1 + x * x)
		end, 0, math.huge, math.pi / 2 },
		{ "the other half line", function(x: number): number
			return math.exp(x)
		end, -math.huge, 0, 1 },
	}
	for _, c in cases do
		local v, info = quad.integrate(c[2], c[3], c[4])
		check(info.converged and rel(v, c[5]) < 1e-10, string.format("Gauss-Kronrod, %s: %.1e", c[1], rel(v, c[5])))
	end
	-- singularities at the ends: tanh-sinh, with a fraction of the evaluations
	for _, c: any in {
		{ "1 / sqrt(x)", function(x: number): number
			return 1 / math.sqrt(x)
		end, 2 },
		{ "log x", function(x: number): number
			return math.log(x)
		end, -1 },
	} do
		local v, info = quad.integrate(c[2], 0, 1, { method = "tanh-sinh" })
		local _, gk = quad.integrate(c[2], 0, 1)
		check(rel(v, c[3]) < 1e-10 and info.evaluations * 5 < gk.evaluations, string.format("tanh-sinh, %s: %.1e in %d evaluations (Gauss-Kronrod %d)", c[1], rel(v, c[3]), info.evaluations, gk.evaluations))
	end
	-- reversed limits, an empty interval, a bad integrand
	close(quad.integrate(function(x: number): number
		return x
	end, 1, 0), -0.5, 1e-15, "reversed limits")
	check(quad.integrate(math.sin, 2, 2) == 0, "empty interval")
	check(not pcall(function()
		quad.integrate(function(): any
			return "x"
		end, 0, 1)
	end), "an integrand that is not a number")
end)

test("stats: the normal distribution", function()
	close(stats.normal_cdf(1.96), 0.9750021048517795, 1e-15, "cdf")
	close(stats.normal_cdf(-10), 7.61985302416047e-24, 1e-13, "cdf, far tail")
	close(stats.normal_pdf(0), 1 / math.sqrt(2 * math.pi), 1e-15, "pdf")
	local q = { { 1e-300, -37.0470962993612 }, { 1e-10, -6.361340902404056 }, { 0.025, -1.9599639845400545 }, { 0.975, 1.959963984540054 }, { 1 - 1e-12, 7.0344869100478356 } }
	local worst = 0
	for _, r in q do
		worst = math.max(worst, rel(stats.normal_quantile(r[1]), r[2]))
	end
	check(worst < 1e-12, string.format("quantiles: %.1e", worst))
	check(stats.normal_quantile(0.5) == 0, "median")
	close(stats.normal_quantile(0.975, 100, 15), 100 + 15 * 1.959963984540054, 1e-13, "shifted and scaled")
end)

return nil
