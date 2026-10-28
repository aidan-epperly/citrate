--!strict

-- Descriptive statistics of a list of numbers (a table, or an f64 array): mean, spread,
-- quantiles, histograms, correlation, and a running summary for numbers that arrive one
-- at a time (frame times, say).
--
--   mean({ 2, 4, 9 })          -- 5
--   quantile(times, 0.95)      -- the 95th percentile
--   local r = running()        -- then r:add(x) each frame; r.mean, r:std(), r.min
--
-- Sums are compensated (Kahan-Neumaier) and the variance is computed in two passes (or by
-- Welford's update, for the running summary), so long lists of similar numbers keep their
-- accuracy.

local special = require("./special")

-- The numbers of x (a table or an f64 array) as a table, checked.
local function values(x: any, who: string, least: number?): { number }
	local t: { number }
	if type(x) == "buffer" then
		t = table.create(buffer.len(x) // 8, 0)
		for k = 1, #t do
			t[k] = buffer.readf64(x, 8 * (k - 1))
		end
	elseif type(x) == "table" then
		t = x
	else
		error("stats." .. who .. ": expected a list of numbers (a table or an f64 array)", 3)
	end
	if #t < (least or 1) then
		error(string.format("stats.%s: need at least %d numbers, got %d", who, least or 1, #t), 3)
	end
	for k = 1, #t do
		local v = t[k]
		if type(v) ~= "number" or v ~= v then
			error(string.format("stats.%s: entry %d is %s, not a number", who, k, tostring(v)), 3)
		end
	end
	return t
end

-- A compensated sum (Neumaier's variant of Kahan's).
local function ksum(t: { number }): number
	local s, c = 0, 0
	for _, v in t do
		local u = s + v
		if math.abs(s) >= math.abs(v) then
			c += (s - u) + v
		else
			c += (v - u) + s
		end
		s = u
	end
	return s + c
end

-- The sum of the numbers (compensated: accurate even for many of mixed size).
local function sum(x: any): number
	return ksum(values(x, "sum", 0))
end

-- The mean (average).
local function mean(x: any): number
	local t = values(x, "mean")
	return ksum(t) / #t
end

-- The sample variance: the mean squared distance from the mean, dividing by n - 1 (pass
-- population = true to divide by n).
local function variance(x: any, population: boolean?): number
	local t = values(x, "variance", if population then 1 else 2)
	local m = ksum(t) / #t
	local d = table.create(#t, 0)
	for k, v in t do
		d[k] = (v - m) ^ 2
	end
	return ksum(d) / (if population then #t else #t - 1)
end

-- The sample standard deviation (the square root of the variance; population as there).
local function std(x: any, population: boolean?): number
	return math.sqrt(variance(x, population))
end

-- The smallest and largest numbers.
local function range(x: any): (number, number)
	local t = values(x, "range")
	local lo, hi = t[1], t[1]
	for _, v in t do
		lo, hi = math.min(lo, v), math.max(hi, v)
	end
	return lo, hi
end

-- The q-quantile (q in [0, 1]): the value below which a fraction q of the numbers lie, by
-- linear interpolation between the sorted numbers (the common definition, "type 7"; q = 0.5
-- is the median). q may be a table of fractions: then a table of quantiles.
local function quantile(x: any, q: number | { number }): any
	local t = table.clone(values(x, "quantile"))
	table.sort(t)
	local n = #t
	local function at(p: number): number
		if type(p) ~= "number" or not (p >= 0 and p <= 1) then
			error("stats.quantile: q must be in [0, 1]", 3)
		end
		local h = (n - 1) * p + 1
		local lo = math.floor(h)
		if lo >= n then
			return t[n]
		end
		return t[lo] + (h - lo) * (t[lo + 1] - t[lo])
	end
	if type(q) == "table" then
		local out = table.create(#q, 0)
		for k, p in q do
			out[k] = at(p)
		end
		return out
	end
	return at(q :: number)
end

-- The median (the middle value: half the numbers lie below it).
local function median(x: any): number
	return quantile(x, 0.5)
end

-- A histogram: how many numbers fall in each of `bins` equal intervals from lo to hi
-- (default: the numbers' range). Returns the counts and the bins' edges (bins + 1 of
-- them); numbers outside [lo, hi] are not counted, and hi itself counts in the last bin.
local function histogram(x: any, bins: number, lo: number?, hi: number?): ({ number }, { number })
	local t = values(x, "histogram")
	if bins % 1 ~= 0 or bins < 1 then
		error("stats.histogram: bins must be a positive whole number", 2)
	end
	local a, b = lo, hi
	if a == nil or b == nil then
		local mn, mx = range(t)
		a, b = a or mn, b or mx
	end
	local first: number, last: number = a :: number, b :: number
	if not (last > first) then
		last = first + 1
	end
	local counts = table.create(bins, 0)
	for _, v in t do
		if v >= first and v <= last then
			local k = math.min(math.floor((v - first) / (last - first) * bins) + 1, bins)
			counts[k] += 1
		end
	end
	local edges = table.create(bins + 1, 0)
	for k = 0, bins do
		edges[k + 1] = first + (last - first) * k / bins
	end
	return counts, edges
end

-- The sample covariance of two lists of the same length.
local function covariance(x: any, y: any): number
	local a, b = values(x, "covariance", 2), values(y, "covariance", 2)
	if #a ~= #b then
		error("stats.covariance: the lists have different lengths", 2)
	end
	local ma, mb = ksum(a) / #a, ksum(b) / #b
	local p = table.create(#a, 0)
	for k = 1, #a do
		p[k] = (a[k] - ma) * (b[k] - mb)
	end
	return ksum(p) / (#a - 1)
end

-- The (Pearson) correlation of two lists of the same length, in [-1, 1]: 1 when one is an
-- increasing linear function of the other. NaN if either list is constant.
local function correlation(x: any, y: any): number
	local c = covariance(x, y)
	return c / math.sqrt(variance(x) * variance(y))
end

-- The normal distribution

local function check_sd(who: string, sd: number)
	if not (sd > 0) then
		error("stats." .. who .. ": the standard deviation must be positive", 3)
	end
end

-- The normal (Gaussian) density at x, for the given mean (default 0) and standard
-- deviation (default 1).
local function normal_pdf(x: number, mean: number?, sd: number?): number
	local s = sd or 1
	check_sd("normal_pdf", s)
	local z = (x - (mean or 0)) / s
	return math.exp(-0.5 * z * z) / (s * math.sqrt(2 * math.pi))
end

-- The normal cumulative probability: the chance that a normal number (mean, sd: default 0
-- and 1) is at most x. Accurate far into the tails (special.erfc).
local function normal_cdf(x: number, mean: number?, sd: number?): number
	local s = sd or 1
	check_sd("normal_cdf", s)
	return 0.5 * special.erfc(-(x - (mean or 0)) / (s * math.sqrt(2)))
end

-- The normal quantile: the x with normal_cdf(x) = p, for p in (0, 1) (the inverse of
-- normal_cdf; p = 0.975 gives 1.96 standard deviations above the mean). Abramowitz and
-- Stegun's approximation 26.2.23, refined by Newton's method on the cdf to full accuracy.
local function normal_quantile(p: number, mean: number?, sd: number?): number
	local s = sd or 1
	check_sd("normal_quantile", s)
	if not (p > 0 and p < 1) then
		if p == 0 then
			return -math.huge
		elseif p == 1 then
			return math.huge
		end
		error("stats.normal_quantile: p must be in [0, 1]", 2)
	end
	if p == 0.5 then
		return mean or 0
	end
	local q = math.min(p, 1 - p)
	local t = math.sqrt(-2 * math.log(q))
	local z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
	if p < 0.5 then
		z = -z
	end
	for _ = 1, 4 do
		-- Newton on the cdf, by its lower or upper tail (whichever is smaller, so tiny
		-- probabilities keep their accuracy)
		local f = if p < 0.5 then 0.5 * special.erfc(-z / math.sqrt(2)) - p else (1 - p) - 0.5 * special.erfc(z / math.sqrt(2))
		local d = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
		if d == 0 then
			break
		end
		local step = f / d
		z -= step
		if math.abs(step) < 1e-15 * math.max(1, math.abs(z)) then
			break
		end
	end
	return (mean or 0) + s * z
end

-- A running summary

export type Running = {
	count: number,
	mean: number,
	min: number,
	max: number,
	m2: number,
	add: (self: Running, x: number) -> Running,
	variance: (self: Running) -> number,
	std: (self: Running) -> number,
}

local Running = {}
Running.__index = Running

-- A summary of numbers added one at a time (r:add(x)), in constant memory: r.count,
-- r.mean, r.min, r.max, r:variance() and r:std() (sample), by Welford's update.
local function running(): Running
	return setmetatable({ count = 0, mean = 0, min = math.huge, max = -math.huge, m2 = 0 }, Running) :: any
end

function Running.add(self: Running, x: number): Running
	if type(x) ~= "number" or x ~= x then
		error("running:add: expected a number", 2)
	end
	self.count += 1
	local d = x - self.mean
	self.mean += d / self.count
	self.m2 += d * (x - self.mean)
	self.min = math.min(self.min, x)
	self.max = math.max(self.max, x)
	return self
end

function Running.variance(self: Running): number
	return if self.count > 1 then self.m2 / (self.count - 1) else 0
end

function Running.std(self: Running): number
	return math.sqrt(Running.variance(self))
end

return {
	sum = sum,
	mean = mean,
	variance = variance,
	std = std,
	range = range,
	quantile = quantile,
	median = median,
	histogram = histogram,
	covariance = covariance,
	correlation = correlation,
	running = running,
	normal_pdf = normal_pdf,
	normal_cdf = normal_cdf,
	normal_quantile = normal_quantile,
}
