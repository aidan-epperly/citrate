--!strict

-- Random numbers: the generator against xoshiro128**'s reference outputs, determinism,
-- each distribution's moments, sampling, points, the alias sampler, and the quasi-random
-- sequences (Halton's and Sobol's known points, Sobol's table: primitive polynomials, odd
-- direction numbers; stratification; faster integration than random points); and the
-- descriptive statistics.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local random, stats = lib.random, lib.stats

test("random: the generator", function()
	-- xoshiro128** from state (1, 2, 3, 4): the reference implementation's first outputs
	local r = random.new(1)
	r.s0, r.s1, r.s2, r.s3 = 1, 2, 3, 4
	local expect = { 11520, 0, 5927040, 70819200 }
	local ok = true
	for k = 1, 4 do
		-- two outputs per uniform: rebuild them from the state directly instead
		local s1 = r.s1
		local out = (bit32.lrotate((s1 * 5) % 4294967296, 7) * 9) % 4294967296
		ok = ok and out == expect[k]
		-- advance one output
		local t = bit32.lshift(r.s1, 9)
		r.s2 = bit32.bxor(r.s2, r.s0)
		r.s3 = bit32.bxor(r.s3, r.s1)
		r.s1 = bit32.bxor(r.s1, r.s2)
		r.s0 = bit32.bxor(r.s0, r.s3)
		r.s2 = bit32.bxor(r.s2, t)
		r.s3 = bit32.lrotate(r.s3, 11)
	end
	check(ok, "reference outputs")
	-- the library's step agrees: integer(0, 2^32 - 1) is the raw output
	local a = random.new(1)
	a.s0, a.s1, a.s2, a.s3 = 1, 2, 3, 4
	local seq = {}
	for k = 1, 4 do
		seq[k] = a:integer(0, 4294967295)
	end
	check(seq[1] == 11520 and seq[2] == 0 and seq[3] == 5927040 and seq[4] == 70819200, "next32 matches")
	-- the same seed, the same numbers; different seeds, different ones
	local x, y, z = random.new(42), random.new(42), random.new(43)
	local same, differ = true, false
	for _ = 1, 100 do
		local u = x:uniform()
		same = same and u == y:uniform()
		differ = differ or u ~= z:uniform()
	end
	check(same and differ, "seeding")
	check(not pcall(function()
		random.new(0 / 0)
	end), "a NaN seed")
end)

test("random: distributions", function()
	local r = random.new(7)
	local N = 200000
	local function moments(draw: () -> number): (number, number)
		local run = stats.running()
		for _ = 1, N do
			run:add(draw())
		end
		return run.mean, run:variance()
	end
	-- uniform: mean 1/2, variance 1/12, inside [0, 1)
	local lo, hi = 1, 0
	local m, v = moments(function(): number
		local u = r:uniform()
		lo, hi = math.min(lo, u), math.max(hi, u)
		return u
	end)
	check(math.abs(m - 0.5) < 0.003 and math.abs(v - 1 / 12) < 0.002 and lo >= 0 and hi < 1, "uniform")
	-- integer: a die, chi-square with 5 degrees of freedom (under 20.5 at the 0.001 level)
	local counts = table.create(6, 0)
	for _ = 1, 60000 do
		local k = r:integer(1, 6)
		counts[k] += 1
	end
	local chi = 0
	for k = 1, 6 do
		chi += (counts[k] - 10000) ^ 2 / 10000
	end
	check(chi < 20.5, string.format("die chi-square %.1f", chi))
	-- normal: mean, variance, and the fraction within one standard deviation
	local within = 0
	m, v = moments(function(): number
		local z = r:normal(3, 2)
		if math.abs(z - 3) < 2 then
			within += 1
		end
		return z
	end)
	check(math.abs(m - 3) < 0.02 and math.abs(v - 4) < 0.05 and math.abs(within / N - 0.6827) < 0.005, "normal")
	m, v = moments(function(): number
		return r:exponential(2)
	end)
	check(math.abs(m - 0.5) < 0.005 and math.abs(v - 0.25) < 0.01, "exponential")
	for _, shape in { 0.5, 3, 10 } do
		m, v = moments(function(): number
			return r:gamma(shape, 2)
		end)
		check(math.abs(m / (2 * shape) - 1) < 0.01 and math.abs(v / (4 * shape) - 1) < 0.03, "gamma, shape " .. shape)
	end
	m, v = moments(function(): number
		return r:beta(2, 5)
	end)
	check(math.abs(m - 2 / 7) < 0.003 and math.abs(v - 10 / (49 * 8)) < 0.002, "beta")
	for _, mean in { 3, 50, 1000 } do
		m, v = moments(function(): number
			return r:poisson(mean)
		end)
		check(math.abs(m / mean - 1) < 0.01 and math.abs(v / mean - 1) < 0.03, string.format("poisson %d (%.3f, %.3f)", mean, m, v))
	end
end)

test("random: sampling, points and weighted choice", function()
	local r = random.new(3)
	local list = {}
	for i = 1, 50 do
		list[i] = i
	end
	r:shuffle(list)
	local seen = {}
	for _, v in list do
		seen[v] = true
	end
	local all = true
	for i = 1, 50 do
		all = all and seen[i] == true
	end
	check(all, "shuffle is a permutation")
	local s = r:sample(list, 10)
	local distinct = {}
	for _, v in s do
		distinct[v] = true
	end
	local count = 0
	for _ in distinct do
		count += 1
	end
	check(#s == 10 and count == 10, "sample without replacement")
	-- points: unit directions; a quarter of the disk's points within half its radius, an
	-- eighth of the ball's
	local unit, quarter, eighth = true, 0, 0
	for _ = 1, 40000 do
		unit = unit and math.abs(vector.magnitude(r:direction()) - 1) < 1e-6 -- vectors are single precision
		if vector.magnitude(r:in_ball(2, true)) < 1 then
			quarter += 1
		end
		if vector.magnitude(r:in_ball(2)) < 1 then
			eighth += 1
		end
	end
	check(unit and math.abs(quarter / 40000 - 0.25) < 0.01 and math.abs(eighth / 40000 - 0.125) < 0.008, "points")
	-- weighted choice: frequencies follow the weights
	local W = random.weighted({ 70, 25, 5, 0 })
	local c = table.create(4, 0)
	for _ = 1, 100000 do
		local k = W:sample(r)
		c[k] += 1
	end
	check(math.abs(c[1] / 1e5 - 0.7) < 0.006 and math.abs(c[2] / 1e5 - 0.25) < 0.006 and math.abs(c[3] / 1e5 - 0.05) < 0.003 and c[4] == 0, "weighted")
	check(not pcall(function()
		random.weighted({ 0, 0 })
	end), "all-zero weights")
end)

-- whether the polynomial x^s + a_1 x^(s-1) + ... + a_(s-1) x + 1 over GF(2) is primitive:
-- x has order 2^s - 1 modulo it
local function primitive(s: number, a: number): boolean
	local poly = bit32.bor(bit32.lshift(1, s), bit32.lshift(a, 1), 1)
	local order = 2 ^ s - 1
	local x = 1
	for k = 1, order do
		x = bit32.lshift(x, 1)
		if bit32.band(x, bit32.lshift(1, s)) ~= 0 then
			x = bit32.bxor(x, poly)
		end
		if x == 1 then
			return k == order
		end
	end
	return false
end

test("random: quasi-random sequences", function()
	-- Halton: radical inverses in bases 2 and 3
	local H = random.halton(2)
	local p1, p2, p3 = H:next(), H:next(), H:next()
	check(p1[1] == 0.5 and p2[1] == 0.25 and p3[1] == 0.75, "Halton base 2")
	close(p1[2], 1 / 3, 1e-15, "Halton base 3")
	close(p3[2], 1 / 9, 1e-15, "Halton base 3, third point")
	-- Sobol: the known first points in two dimensions
	local S = random.sobol(2)
	local expect = { { 0.5, 0.5 }, { 0.75, 0.25 }, { 0.25, 0.75 }, { 0.375, 0.375 }, { 0.875, 0.875 } }
	local ok = true
	for k = 1, 5 do
		local p = S:next()
		ok = ok and p[1] == expect[k][1] and p[2] == expect[k][2]
	end
	check(ok, "Sobol's first points")
	-- the table: primitive polynomials and odd direction numbers below 2^k
	local table_ok = true
	for d, row in random.SOBOL_TABLE do
		local s, a, m = row[1] :: number, row[2] :: number, row[3] :: { number }
		table_ok = table_ok and primitive(s, a) and #m == s
		for k, mk in m do
			table_ok = table_ok and mk % 2 == 1 and mk < 2 ^ k
		end
		if not table_ok then
			print("bad Sobol row", d + 1)
			break
		end
	end
	check(table_ok, "Sobol table")
	-- every coordinate of the first 2^10 points (with the point 0) fills each of 2^10 slices once
	local dim = 21
	local Q = random.sobol(dim)
	local filled = {}
	for j = 1, dim do
		filled[j] = { [0] = true }
	end
	local strat = true
	for _ = 1, 1023 do
		local p = Q:next()
		for j = 1, dim do
			local slot = math.floor(p[j] * 1024)
			strat = strat and not filled[j][slot]
			filled[j][slot] = true
		end
	end
	check(strat, "Sobol stratification")
	-- integration of a smooth product over the 6-cube: Sobol's error far below random points'
	local function f(p: { number }): number
		local v = 1
		for j = 1, 6 do
			v *= 1 + 0.5 * (p[j] - 0.5)
		end
		return v
	end
	local Q6, R6 = random.sobol(6), random.new(11)
	local sq, sr = 0, 0
	for _ = 1, 4095 do
		sq += f(Q6:next())
		local p = {}
		for j = 1, 6 do
			p[j] = R6:uniform()
		end
		sr += f(p)
	end
	local eq, er = math.abs(sq / 4095 - 1), math.abs(sr / 4095 - 1)
	check(eq < 1e-3 and eq < er, string.format("integration: Sobol %.1e, random %.1e", eq, er))
end)

test("stats: descriptive statistics", function()
	local x = { 2, 4, 4, 4, 5, 5, 7, 9 }
	check(stats.mean(x) == 5 and stats.sum(x) == 40, "mean, sum")
	close(stats.variance(x, true), 4, 1e-15, "population variance")
	close(stats.std(x), math.sqrt(32 / 7), 1e-15, "sample std")
	check(stats.median(x) == 4.5 and stats.quantile(x, 0) == 2 and stats.quantile(x, 1) == 9, "quantiles")
	local q = stats.quantile({ 1, 2, 3, 4 }, { 0.25, 0.5 })
	check(q[1] == 1.75 and q[2] == 2.5, "type-7 quantiles")
	local lo, hi = stats.range(x)
	check(lo == 2 and hi == 9, "range")
	local counts, edges = stats.histogram({ 0, 0.5, 1, 1.5, 2 }, 2)
	check(counts[1] == 2 and counts[2] == 3 and edges[2] == 1, "histogram")
	close(stats.correlation({ 1, 2, 3 }, { 2, 4, 6 }), 1, 1e-15, "correlation")
	close(stats.covariance({ 1, 2, 3 }, { 1, 0, -1 }), -1, 1e-15, "covariance")
	-- compensated sums keep 1e8 copies of 0.1 accurate where plain addition drifts
	local r = stats.running()
	local big = {}
	for k = 1, 100000 do
		big[k] = 1e8 + (k % 3)
		r:add(big[k])
	end
	close(stats.mean(big), 1e8 + 1, 1e-15, "a mean of large similar numbers")
	close(r:variance(), stats.variance(big), 1e-9, "running variance agrees")
	check(r.count == 100000 and r.min == 1e8 and r.max == 1e8 + 2, "running min and max")
	-- arrays work too
	check(stats.mean(lib.f64.from({ 1, 2, 3 })) == 2, "f64 arrays")
	check(not pcall(function()
		stats.mean({ 1, 0 / 0 })
	end), "NaN rejected")
end)

return nil
