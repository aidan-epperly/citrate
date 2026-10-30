--!strict
--!native
--!optimize 2

-- Random numbers: a seeded generator (the same seed gives the same numbers on every
-- machine), distributions, sampling, random points, and quasi-random sequences.
--
--   local rng = new(42)
--   rng:uniform()            -- in [0, 1)
--   rng:integer(1, 6)        -- a die
--   rng:normal(0, 1)
--   rng:shuffle(list)
--   local loot = weighted({ 70, 25, 5 })   -- then loot:sample(rng): 1, 2 or 3
--   local q = sobol(2)                      -- then q:next(): evenly spread points
--
-- The generator is xoshiro128** (Blackman and Vigna, 2018): four 32-bit words of state,
-- period 2^128 - 1, passing the standard statistical test batteries; its arithmetic is
-- exact in Luau's numbers. Each uniform number takes two outputs (53 random bits).

-- The generator

export type Rng = {
	s0: number,
	s1: number,
	s2: number,
	s3: number,
	spare: number?, -- the second normal number of the last Box-Muller pair
	[string]: any,
}

local Rng = {}
Rng.__index = Rng

local band, bxor, lrotate, lshift, rshift = bit32.band, bit32.bxor, bit32.lrotate, bit32.lshift, bit32.rshift

-- The next 32-bit output (an integer in [0, 2^32)).
local function next32(r: Rng): number
	local s0, s1, s2, s3 = r.s0, r.s1, r.s2, r.s3
	local result = (lrotate((s1 * 5) % 4294967296, 7) * 9) % 4294967296
	local t = lshift(s1, 9)
	s2 = bxor(s2, s0)
	s3 = bxor(s3, s1)
	s1 = bxor(s1, s2)
	s0 = bxor(s0, s3)
	s2 = bxor(s2, t)
	s3 = lrotate(s3, 11)
	r.s0, r.s1, r.s2, r.s3 = s0, s1, s2, s3
	return result
end

-- splitmix32 (for seeding): the state and its next output.
local function splitmix(x: number): (number, number)
	x = (x + 0x9E3779B9) % 4294967296
	local z = x
	z = bxor(z, rshift(z, 16))
	-- z * 0x85EBCA6B mod 2^32, in two halves to stay exact
	z = ((band(z, 0xFFFF) * 0x85EBCA6B) + (rshift(z, 16) * 0xCA6B % 65536) * 65536) % 4294967296
	z = bxor(z, rshift(z, 13))
	z = ((band(z, 0xFFFF) * 0xC2B2AE35) + (rshift(z, 16) * 0xAE35 % 65536) * 65536) % 4294967296
	z = bxor(z, rshift(z, 16))
	return x, z
end

-- A generator seeded with any number (the same seed, the same numbers); without a seed,
-- one from the clock.
local function new(seed: number?): Rng
	local x = seed or (os.clock() * 1e9 + os.time())
	if x ~= x or x == math.huge or x == -math.huge then
		error("random.new: the seed must be a finite number", 2)
	end
	-- fold the seed (any number) into 32 bits, then expand it
	x = math.floor(math.abs(x)) % 4294967296 + (if x < 0 then 1 else 0) + math.floor((x % 1) * 2 ^ 31)
	x %= 4294967296
	local s = {}
	for k = 1, 4 do
		local out
		x, out = splitmix(x)
		s[k] = out
	end
	if s[1] == 0 and s[2] == 0 and s[3] == 0 and s[4] == 0 then
		s[1] = 1 -- the one state xoshiro cannot leave
	end
	return setmetatable({ s0 = s[1], s1 = s[2], s2 = s[3], s3 = s[4], spare = nil }, Rng) :: any
end

-- A uniform number in [0, 1) (or [a, b): rng:uniform(a, b)).
function Rng.uniform(self: Rng, a: number?, b: number?): number
	local hi = rshift(next32(self), 5) -- 27 bits
	local lo = rshift(next32(self), 6) -- 26 bits
	local u = (hi * 67108864 + lo) / 9007199254740992
	if a then
		local top = b or 1
		return a + (top - a) * u
	end
	return u
end

-- A uniform integer in lo..hi (both included), without bias.
function Rng.integer(self: Rng, lo: number, hi: number): number
	if lo % 1 ~= 0 or hi % 1 ~= 0 or hi < lo then
		error("rng:integer: need whole numbers lo <= hi", 2)
	end
	local range = hi - lo + 1
	if range <= 4294967296 then
		-- the largest multiple of range under 2^32: draws above it are redrawn
		local limit = 4294967296 - 4294967296 % range
		local x = next32(self)
		while x >= limit do
			x = next32(self)
		end
		return lo + x % range
	end
	return lo + math.floor(self:uniform() * range)
end

-- A normal (Gaussian) number with the given mean (default 0) and standard deviation
-- (default 1), by the Box-Muller transform (pairs: every other call is free).
function Rng.normal(self: Rng, mean: number?, sd: number?): number
	local z: number
	if self.spare then
		z = self.spare :: number
		self.spare = nil
	else
		local u1 = 1 - self:uniform() -- in (0, 1]
		local u2 = self:uniform()
		local r = math.sqrt(-2 * math.log(u1))
		z = r * math.cos(2 * math.pi * u2)
		self.spare = r * math.sin(2 * math.pi * u2)
	end
	return (mean or 0) + (sd or 1) * z
end

-- An exponential number with the given rate (default 1; mean 1 / rate): waiting times.
function Rng.exponential(self: Rng, rate: number?): number
	return -math.log(1 - self:uniform()) / (rate or 1)
end

-- A gamma number with the given shape (> 0) and scale (default 1), by Marsaglia and
-- Tsang's method (2000), boosted for shape < 1.
function Rng.gamma(self: Rng, shape: number, scale: number?): number
	if not (shape > 0) then
		error("rng:gamma: the shape must be positive", 2)
	end
	local boost = 1
	local a = shape
	if a < 1 then
		boost = (1 - self:uniform()) ^ (1 / a)
		a += 1
	end
	local d = a - 1 / 3
	local c = 1 / math.sqrt(9 * d)
	while true do
		local x, v
		repeat
			x = self:normal()
			v = 1 + c * x
		until v > 0
		v = v * v * v
		local u = 1 - self:uniform()
		if u < 1 - 0.0331 * x ^ 4 or math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)) then
			return d * v * boost * (scale or 1)
		end
	end
end

-- A beta number in (0, 1) with shapes a, b > 0 (from two gamma numbers).
function Rng.beta(self: Rng, a: number, b: number): number
	local x = self:gamma(a)
	local y = self:gamma(b)
	return x / (x + y)
end

local function log_factorial(k: number): number
	-- Stirling's series past 10, exact below
	if k < 10 then
		local s = 0
		for i = 2, k do
			s += math.log(i)
		end
		return s
	end
	local x = k + 1
	return (x - 0.5) * math.log(x) - x + 0.5 * math.log(2 * math.pi) + 1 / (12 * x) - 1 / (360 * x ^ 3)
end

-- A Poisson count with the given mean (> 0): events in a fixed time. By inversion for
-- means under 10, and Hormann's transformed rejection (PTRS, 1993) above.
function Rng.poisson(self: Rng, mean: number): number
	if not (mean >= 0) then
		error("rng:poisson: the mean must be nonnegative", 2)
	end
	if mean < 10 then
		local p = math.exp(-mean)
		local k, s = 0, p
		local u = self:uniform()
		while u > s do
			k += 1
			p *= mean / k
			s += p
			if k > 1000 then
				break
			end
		end
		return k
	end
	local slam, loglam = math.sqrt(mean), math.log(mean)
	local b = 0.931 + 2.53 * slam
	local a = -0.059 + 0.02483 * b
	local invalpha = 1.1239 + 1.1328 / (b - 3.4)
	local vr = 0.9277 - 3.6224 / (b - 2)
	while true do
		local u = self:uniform() - 0.5
		local v = self:uniform()
		local us = 0.5 - math.abs(u)
		local k = math.floor((2 * a / us + b) * u + mean + 0.43)
		if us >= 0.07 and v <= vr then
			return k
		end
		if k >= 0 and not (us < 0.013 and v > us) then
			if math.log(v * invalpha / (a / (us * us) + b)) <= -mean + k * loglam - log_factorial(k) then
				return k
			end
		end
	end
end

-- Sampling

-- Shuffles a list in place (every order equally likely: Fisher-Yates), and returns it.
function Rng.shuffle(self: Rng, list: { any }): { any }
	for i = #list, 2, -1 do
		local j = self:integer(1, i)
		list[i], list[j] = list[j], list[i]
	end
	return list
end

-- An entry of the list, each equally likely.
function Rng.pick(self: Rng, list: { any }): any
	if #list == 0 then
		error("rng:pick: the list is empty", 2)
	end
	return list[self:integer(1, #list)]
end

-- k distinct entries of the list, in random order (sampling without replacement).
function Rng.sample(self: Rng, list: { any }, k: number): { any }
	local n = #list
	if k % 1 ~= 0 or k < 0 or k > n then
		error("rng:sample: k must be a whole number from 0 to the list's length", 2)
	end
	local idx = table.create(n, 0)
	for i = 1, n do
		idx[i] = i
	end
	local out = table.create(k)
	for i = 1, k do
		local j = self:integer(i, n)
		idx[i], idx[j] = idx[j], idx[i]
		out[i] = list[idx[i]]
	end
	return out
end

-- Fills an f64 array with uniform numbers in [0, 1) (kind "uniform", the default) or
-- standard normal ones ("normal"), and returns it.
function Rng.fill(self: Rng, x: buffer, kind: string?): buffer
	local normal = kind == "normal"
	if not normal and kind ~= nil and kind ~= "uniform" then
		error("rng:fill: kind must be \"uniform\" or \"normal\"", 2)
	end
	for k = 0, buffer.len(x) // 8 - 1 do
		buffer.writef64(x, 8 * k, if normal then self:normal() else self:uniform())
	end
	return x
end

-- Random points

-- A random direction, uniform over the sphere. dim says which sphere: nil, the unit sphere
-- in 3-D, as a vector; true, the circle (a vector with z = 0); a number d, the sphere in d
-- dimensions, as a table of d numbers, or written into out (a table or an f64 buffer of d
-- numbers) when given. Any d is the normalized vector of d normal numbers (Muller's method);
-- the vectors' cases take an angle (and a height) directly.
function Rng.direction(self: Rng, dim: (boolean | number)?, out: any?): any
	if type(dim) == "number" then
		if dim < 1 or dim % 1 ~= 0 then
			error("rng:direction: the dimension must be a positive whole number", 2)
		end
		local x = out or table.create(dim, 0)
		local isbuffer = type(x) == "buffer"
		local norm2 = 0
		while norm2 == 0 do
			norm2 = 0
			for a = 1, dim do
				local g = self:normal()
				norm2 += g * g
				if isbuffer then
					buffer.writef64(x, 8 * (a - 1), g)
				else
					x[a] = g
				end
			end
		end
		local inv = 1 / math.sqrt(norm2)
		for a = 1, dim do
			if isbuffer then
				buffer.writef64(x, 8 * (a - 1), buffer.readf64(x, 8 * (a - 1)) * inv)
			else
				x[a] *= inv
			end
		end
		return x
	end
	local phi = 2 * math.pi * self:uniform()
	if dim then
		return vector.create(math.cos(phi), math.sin(phi), 0)
	end
	local z = 2 * self:uniform() - 1
	local r = math.sqrt(1 - z * z)
	return vector.create(r * math.cos(phi), r * math.sin(phi), z)
end

-- A random point inside the ball of the given radius (default 1) about the origin, uniform
-- by volume: a direction (dim and out as for rng:direction: nil the 3-D ball, true the disk,
-- a number d the ball in d dimensions) times radius u^(1/d).
function Rng.in_ball(self: Rng, radius: number?, dim: (boolean | number)?, out: any?): any
	local d = if type(dim) == "number" then dim elseif dim then 2 else 3
	local r = (radius or 1) * self:uniform() ^ (1 / d)
	local x = self:direction(dim, out)
	if type(dim) ~= "number" then
		return x * r
	end
	for a = 1, d do
		if type(x) == "buffer" then
			buffer.writef64(x, 8 * (a - 1), buffer.readf64(x, 8 * (a - 1)) * r)
		else
			x[a] *= r
		end
	end
	return x
end

-- Weighted choice

export type Weighted = { n: number, prob: { number }, alias: { number }, sample: (self: Weighted, rng: Rng) -> number }

local Weighted = {}
Weighted.__index = Weighted

-- A sampler choosing index i with probability weights[i] / (their sum): weighted:sample(rng)
-- costs the same however many weights there are (Vose's alias method, built in O(n)).
-- Weights must be nonnegative, not all zero.
local function weighted(weights: { number }): Weighted
	local n = #weights
	if n == 0 then
		error("random.weighted: no weights", 2)
	end
	local total = 0
	for i, w in weights do
		if type(w) ~= "number" or not (w >= 0) or w == math.huge then
			error(string.format("random.weighted: weight %d is not a finite nonnegative number", i), 2)
		end
		total += w
	end
	if total <= 0 then
		error("random.weighted: the weights are all zero", 2)
	end
	local prob, alias = table.create(n, 0) :: { number }, table.create(n, 0) :: { number }
	local scaled = table.create(n, 0) :: { number }
	local small: { number }, large: { number } = {}, {}
	for i = 1, n do
		scaled[i] = weights[i] * n / total
		if scaled[i] < 1 then
			table.insert(small, i)
		else
			table.insert(large, i)
		end
	end
	while #small > 0 and #large > 0 do
		local s = table.remove(small) :: number
		local l = table.remove(large) :: number
		prob[s], alias[s] = scaled[s], l
		scaled[l] = scaled[l] + scaled[s] - 1
		if scaled[l] < 1 then
			table.insert(small, l)
		else
			table.insert(large, l)
		end
	end
	for _, i in large do
		prob[i] = 1
	end
	for _, i in small do
		prob[i] = 1 -- only from rounding
	end
	return setmetatable({ n = n, prob = prob, alias = alias }, Weighted) :: any
end

function Weighted.sample(self: Weighted, rng: Rng): number
	local u = rng:uniform() * self.n
	local i = math.floor(u)
	if u - i < self.prob[i + 1] then
		return i + 1
	end
	return self.alias[i + 1]
end

-- Quasi-random sequences

export type Sequence = { dim: number, index: number, next: (self: Sequence) -> { number }, [string]: any }

local PRIMES = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131 }

local Halton = {}
Halton.__index = Halton

-- The Halton sequence in dim dimensions (at most 32): point k has coordinate j the
-- radical inverse of k in the j-th prime base. halton:next() gives the next point (a
-- table), starting from point 1. Simple and good in a few dimensions; its coordinates
-- become correlated in high ones (prefer sobol there).
local function halton(dim: number): Sequence
	if dim % 1 ~= 0 or dim < 1 or dim > #PRIMES then
		error(string.format("random.halton: dim must be a whole number from 1 to %d", #PRIMES), 2)
	end
	return setmetatable({ dim = dim, index = 0 }, Halton) :: any
end

function Halton.next(self: Sequence): { number }
	self.index += 1
	local p = table.create(self.dim, 0)
	for j = 1, self.dim do
		local base = PRIMES[j]
		local f, r, k = 1, 0, self.index
		while k > 0 do
			f /= base
			r += f * (k % base)
			k //= base
		end
		p[j] = r
	end
	return p
end

-- Sobol direction numbers (Joe and Kuo, 2008, new-joe-kuo-6.21201) for dimensions 2..21:
-- the degree s and coefficients a of each primitive polynomial x^s + ... + 1 over GF(2),
-- and the initial odd numbers m (m_k < 2^k).
local SOBOL: { { any } } = {
	{ 1, 0, { 1 } },
	{ 2, 1, { 1, 3 } },
	{ 3, 1, { 1, 3, 1 } },
	{ 3, 2, { 1, 1, 1 } },
	{ 4, 1, { 1, 1, 3, 3 } },
	{ 4, 4, { 1, 3, 5, 13 } },
	{ 5, 2, { 1, 1, 5, 5, 17 } },
	{ 5, 4, { 1, 1, 5, 5, 5 } },
	{ 5, 7, { 1, 1, 7, 11, 19 } },
	{ 5, 11, { 1, 1, 5, 1, 1 } },
	{ 5, 13, { 1, 1, 1, 3, 11 } },
	{ 5, 14, { 1, 3, 5, 5, 31 } },
	{ 6, 1, { 1, 3, 3, 9, 7, 49 } },
	{ 6, 13, { 1, 1, 1, 15, 21, 21 } },
	{ 6, 16, { 1, 3, 1, 13, 27, 49 } },
	{ 6, 19, { 1, 1, 1, 15, 7, 5 } },
	{ 6, 22, { 1, 3, 1, 15, 13, 25 } },
	{ 6, 25, { 1, 1, 5, 5, 19, 61 } },
	{ 7, 1, { 1, 3, 7, 11, 23, 15, 103 } },
	{ 7, 4, { 1, 3, 7, 13, 13, 15, 69 } },
}

local BITS = 32

local Sobol = {}
Sobol.__index = Sobol

-- The Sobol sequence in dim dimensions (at most 21), with Joe and Kuo's direction numbers:
-- sobol:next() gives the next point (a table), starting from point 1 (the point 0, all
-- zeros, is skipped). Any 2^k consecutive points from the start fill each of the 2^k
-- equal slices of every coordinate exactly once, and pairs of coordinates nearly as
-- evenly: integrals over the cube converge faster than with random points.
local function sobol(dim: number): Sequence
	if dim % 1 ~= 0 or dim < 1 or dim > #SOBOL + 1 then
		error(string.format("random.sobol: dim must be a whole number from 1 to %d", #SOBOL + 1), 2)
	end
	local v = table.create(dim)
	for j = 1, dim do
		local d = table.create(BITS, 0)
		if j == 1 then
			for k = 1, BITS do
				d[k] = 2 ^ (BITS - k)
			end
		else
			local s: number, a: number, m: { number } = SOBOL[j - 1][1], SOBOL[j - 1][2], SOBOL[j - 1][3]
			for k = 1, math.min(s, BITS) do
				d[k] = m[k] * 2 ^ (BITS - k)
			end
			for k = s + 1, BITS do
				local x = bxor(d[k - s], rshift(d[k - s], s))
				for i = 1, s - 1 do
					if band(rshift(a, s - 1 - i), 1) == 1 then
						x = bxor(x, d[k - i])
					end
				end
				d[k] = x
			end
		end
		v[j] = d
	end
	return setmetatable({ dim = dim, index = 0, v = v, x = table.create(dim, 0) }, Sobol) :: any
end

function Sobol.next(self: Sequence): { number }
	-- Gray code order: point i differs from point i - 1 in the direction number of the
	-- lowest zero bit of i - 1
	local i = self.index
	local c = 1
	while band(i, 1) == 1 do
		i = rshift(i, 1)
		c += 1
	end
	self.index += 1
	local p = table.create(self.dim, 0)
	for j = 1, self.dim do
		local x = bxor(self.x[j], self.v[j][c])
		self.x[j] = x
		p[j] = x / 2 ^ BITS
	end
	return p
end

return {
	new = new,
	weighted = weighted,
	halton = halton,
	sobol = sobol,
	SOBOL_TABLE = SOBOL,
}
