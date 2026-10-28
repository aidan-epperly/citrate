--!strict

-- Special functions: gamma and beta, their incomplete (regularized) forms, the error
-- function, and Bessel functions of integer order.
--
--   special.gamma(5)            -- 24
--   special.erf(1)              -- 0.8427...
--   special.besselj(0, 2.4048)  -- near a zero of J0
--
-- Accuracy: about 1e-15 relative for gamma, log-gamma (away from its zeros at 1 and 2,
-- where it is about 1e-15 absolute), erf and erfc (erfc keeps it far into the tail);
-- incomplete gamma and beta to about 1e-14; Bessel functions to about 1e-15 times the
-- larger of 1 and the value (so not relative near their zeros).

local cheb = require("./cheb")

local PI = math.pi
local EPS = 2 ^ -52
local TINY = 1e-300

local function fail(who: string, message: string): never
	error("special." .. who .. ": " .. message, 3)
end

-- Gamma and beta

-- Lanczos's approximation with g = 7 and 9 terms (as in Numerical Recipes and Press et
-- al.): Gamma(x + 1) = sqrt(2 pi) (x + g + 1/2)^(x + 1/2) e^-(x + g + 1/2) A_g(x).
local LANCZOS = {
	0.99999999999980993,
	676.5203681218851,
	-1259.1392167224028,
	771.32342877765313,
	-176.61502916214059,
	12.507343278686905,
	-0.13857109526572012,
	9.9843695780195716e-6,
	1.5056327351493116e-7,
}

local function lanczos_sum(x: number): number
	local a = LANCZOS[1]
	for i = 1, 8 do
		a += LANCZOS[i + 1] / (x + i)
	end
	return a
end

-- The gamma function: Gamma(n) = (n - 1)! for whole n; poles (infinity) at 0, -1, -2, ...
local function gamma(x: number): number
	if x ~= x then
		return x
	end
	if x <= 0 and x % 1 == 0 then
		return 0 / 0
	end
	if x < 0.5 then
		-- reflection: Gamma(x) Gamma(1 - x) = pi / sin(pi x)
		return PI / (math.sin(PI * x) * gamma(1 - x))
	end
	if x > 171.7 then
		return math.huge
	end
	if x % 1 == 0 and x <= 23 then
		local f = 1
		for k = 2, x - 1 do
			f *= k
		end
		return f
	end
	local z = x - 1
	local t = z + 7.5
	-- t^(z + 1/2) e^-t, as two halves so neither overflows first
	local p = t ^ ((z + 0.5) / 2)
	return math.sqrt(2 * PI) * lanczos_sum(z) * p * (p * math.exp(-t))
end

-- The natural logarithm of |Gamma(x)|.
local function lgamma(x: number): number
	if x ~= x then
		return x
	end
	if x <= 0 and x % 1 == 0 then
		return math.huge
	end
	if x < 0.5 then
		return math.log(PI / math.abs(math.sin(PI * x))) - lgamma(1 - x)
	end
	local z = x - 1
	local t = z + 7.5
	return 0.5 * math.log(2 * PI) + (z + 0.5) * math.log(t) - t + math.log(lanczos_sum(z))
end

-- The beta function B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b), for a, b > 0.
local function beta(a: number, b: number): number
	if not (a > 0 and b > 0) then
		fail("beta", "a and b must be positive")
	end
	return math.exp(lgamma(a) + lgamma(b) - lgamma(a + b))
end

-- Incomplete gamma

-- P(a, x) by its series (for x < a + 1).
local function gamma_series(a: number, x: number): number
	local ap, del = a, 1 / a
	local sum = del
	for _ = 1, 10000 do
		ap += 1
		del *= x / ap
		sum += del
		if math.abs(del) < math.abs(sum) * EPS then
			break
		end
	end
	return sum * math.exp(-x + a * math.log(x) - lgamma(a))
end

-- Q(a, x) by its continued fraction (for x >= a + 1), by Lentz's method.
local function gamma_fraction(a: number, x: number): number
	local b = x + 1 - a
	local c = 1 / TINY
	local d = 1 / b
	local h = d
	for i = 1, 10000 do
		local an = -i * (i - a)
		b += 2
		d = an * d + b
		if math.abs(d) < TINY then
			d = TINY
		end
		c = b + an / c
		if math.abs(c) < TINY then
			c = TINY
		end
		d = 1 / d
		local del = d * c
		h *= del
		if math.abs(del - 1) < EPS then
			break
		end
	end
	return math.exp(-x + a * math.log(x) - lgamma(a)) * h
end

-- The regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a), for
-- a > 0 and x >= 0: from 0 at x = 0 to 1 as x grows (the chi-square and Poisson
-- distributions' cumulative probabilities). upper = true gives Q(a, x) = 1 - P(a, x),
-- accurately even where P is near 1.
local function gamma_inc(a: number, x: number, upper: boolean?): number
	if not (a > 0) or not (x >= 0) then
		fail("gamma_inc", "need a > 0 and x >= 0")
	end
	if x == 0 then
		return if upper then 1 else 0
	end
	if x < a + 1 then
		local p = gamma_series(a, x)
		return if upper then 1 - p else p
	end
	local q = gamma_fraction(a, x)
	return if upper then q else 1 - q
end

-- The error function erf(x) = 2 / sqrt(pi) times the integral of e^-t^2 from 0 to x.
local function erf(x: number): number
	if x ~= x then
		return x
	end
	if x == 0 then
		return x
	end
	local p = gamma_inc(0.5, x * x)
	return if x > 0 then p else -p
end

-- The complementary error function erfc(x) = 1 - erf(x), accurate (relatively) even where
-- it is tiny: erfc(10) = 2.1e-45.
local function erfc(x: number): number
	if x ~= x then
		return x
	end
	if x < 0 then
		return 1 + gamma_inc(0.5, x * x)
	end
	return gamma_inc(0.5, x * x, true)
end

-- Incomplete beta

-- The continued fraction for I_x(a, b) (Lentz's method).
local function beta_fraction(a: number, b: number, x: number): number
	local qab, qap, qam = a + b, a + 1, a - 1
	local c = 1
	local d = 1 - qab * x / qap
	if math.abs(d) < TINY then
		d = TINY
	end
	d = 1 / d
	local h = d
	for m = 1, 10000 do
		local m2 = 2 * m
		local aa = m * (b - m) * x / ((qam + m2) * (a + m2))
		d = 1 + aa * d
		if math.abs(d) < TINY then
			d = TINY
		end
		c = 1 + aa / c
		if math.abs(c) < TINY then
			c = TINY
		end
		d = 1 / d
		h *= d * c
		aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
		d = 1 + aa * d
		if math.abs(d) < TINY then
			d = TINY
		end
		c = 1 + aa / c
		if math.abs(c) < TINY then
			c = TINY
		end
		d = 1 / d
		local del = d * c
		h *= del
		if math.abs(del - 1) < EPS then
			break
		end
	end
	return h
end

-- The regularized incomplete beta function I_x(a, b), for a, b > 0 and x in [0, 1]: from 0
-- at x = 0 to 1 at x = 1 (the beta, binomial, Student's t and F distributions' cumulative
-- probabilities).
local function beta_inc(a: number, b: number, x: number): number
	if not (a > 0 and b > 0) or not (x >= 0 and x <= 1) then
		fail("beta_inc", "need a, b > 0 and x in [0, 1]")
	end
	if x == 0 or x == 1 then
		return x
	end
	local front = math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1 - x))
	if x < (a + 1) / (a + b + 2) then
		return front * beta_fraction(a, b, x) / a
	end
	return 1 - front * beta_fraction(b, a, 1 - x) / b
end

-- Bessel functions of integer order

-- Gauss-Legendre nodes and weights on [0, 1], n of them, kept by n.
local legendre_cache: { [number]: { buffer } } = {}
local function legendre(n: number): (buffer, buffer)
	local c = legendre_cache[n]
	if c == nil then
		local x, w = cheb.legpts(n, 0, 1)
		c = { x, w }
		legendre_cache[n] = c
	end
	return c[1], c[2]
end

-- The integral of f over [lo, hi] by Gauss-Legendre with n points.
local function gl(f: (number) -> number, lo: number, hi: number, n: number): number
	local x, w = legendre(n)
	local s = 0
	for k = 0, n - 1 do
		s += buffer.readf64(w, 8 * k) * f(lo + (hi - lo) * buffer.readf64(x, 8 * k))
	end
	return s * (hi - lo)
end

-- Points for an oscillating integrand on [0, pi] of frequency about q: rounded up to a
-- multiple of 16 (fewer distinct node sets to keep).
local function points_for(q: number): number
	return 16 * math.ceil((math.ceil(q) + 40) / 16)
end

-- Hankel's asymptotic expansion for large x: P and Q with J_n = sqrt(2 / (pi x)) (P cos chi
-- - Q sin chi), Y_n = sqrt(2 / (pi x)) (P sin chi + Q cos chi), chi = x - (n / 2 + 1 / 4) pi.
local function hankel(n: number, x: number): (number, number, number)
	local mu = 4 * n * n
	local P, Q = 0, 0
	local term = 1
	local last = math.huge
	local k = 0
	while k < 60 do
		-- term_k = prod_{j=1..k} (mu - (2j - 1)^2) / (k! (8x)^k)
		if k > 0 then
			term *= (mu - (2 * k - 1) ^ 2) / (k * 8 * x)
		end
		if math.abs(term) > last then
			break -- the series has started to diverge: stop at its smallest term
		end
		last = math.abs(term)
		local sign = if (k // 2) % 2 == 0 then 1 else -1
		if k % 2 == 0 then
			P += sign * term
		else
			Q += sign * term
		end
		if math.abs(term) < EPS * math.max(math.abs(P), 1e-300) then
			break
		end
		k += 1
	end
	return P, Q, x - (n / 2 + 0.25) * PI
end

-- The power series sum over k of s^k (x/2)^(2k + n) / (k! (k + n)!), with s = -1 for J_n and
-- +1 for I_n: for small x its first term dominates, so tiny values keep their relative
-- accuracy (where an integral would only give them to about 1e-16 absolutely).
local function power_series(n: number, x: number, s: number): number
	local h = x / 2
	local term = math.exp(n * math.log(h) - lgamma(n + 1))
	local sum = term
	for k = 1, 500 do
		term *= s * h * h / (k * (k + n))
		sum += term
		if math.abs(term) < EPS * math.abs(sum) then
			break
		end
	end
	return sum
end

local function large(n: number, x: number): boolean
	return x > 25 + n * n / 2
end

local function check_order(who: string, n: number)
	if type(n) ~= "number" or n % 1 ~= 0 or n < 0 then
		fail(who, "the order n must be a whole number >= 0")
	end
end

-- The Bessel function of the first kind J_n(x), for whole n >= 0: the shape of a drum's
-- modes along a radius, and of waves in a cylinder. From its integral (1 / pi) times the
-- integral of cos(n t - x sin t) over [0, pi] by Gauss-Legendre (exponentially accurate),
-- or Hankel's expansion for large x.
local function besselj(n: number, x: number): number
	check_order("besselj", n)
	if x ~= x then
		return x
	end
	local sign = if x < 0 and n % 2 == 1 then -1 else 1
	x = math.abs(x)
	if x == 0 then
		return if n == 0 then 1 else 0
	end
	if large(n, x) then
		local P, Q, chi = hankel(n, x)
		return sign * math.sqrt(2 / (PI * x)) * (P * math.cos(chi) - Q * math.sin(chi))
	end
	if x * x < n + 1 then
		-- the series' terms fall from the first: no cancellation
		return sign * power_series(n, x, -1)
	end
	local v = gl(function(t: number): number
		return math.cos(n * t - x * math.sin(t))
	end, 0, PI, points_for(x + n)) / PI
	return sign * v
end

-- Y_0 or Y_1 from their integrals: (1 / pi) times the integral of sin(x sin t - n t) over
-- [0, pi], less (1 / pi) times that of (e^(n t) + (-1)^n e^(-n t)) e^(-x sinh t) over
-- [0, inf), which decays doubly exponentially (cut where it falls below e^-40).
local function y01(n: number, x: number): number
	local a = gl(function(t: number): number
		return math.sin(x * math.sin(t) - n * t)
	end, 0, PI, points_for(x + n)) / PI
	local y = (40 + n * 10) / x
	local T = math.log(y + math.sqrt(y * y + 1)) + 2 -- asinh
	local b = gl(function(t: number): number
		return (math.exp(n * t) + (if n % 2 == 0 then 1 else -1) * math.exp(-n * t)) * math.exp(-x * math.sinh(t))
	end, 0, T, 128) / PI
	return a - b
end

-- The Bessel function of the second kind Y_n(x), for whole n >= 0 and x > 0 (infinite at
-- 0). Y_0 and Y_1 from their integrals (or Hankel's expansion), the rest by the upward
-- recurrence Y_(k+1) = (2 k / x) Y_k - Y_(k-1), which is stable for Y.
local function bessely(n: number, x: number): number
	check_order("bessely", n)
	if x ~= x then
		return x
	end
	if not (x > 0) then
		return if x == 0 then -math.huge else 0 / 0
	end
	if large(n, x) then
		local P, Q, chi = hankel(n, x)
		return math.sqrt(2 / (PI * x)) * (P * math.sin(chi) + Q * math.cos(chi))
	end
	local y0 = if large(0, x) then bessely(0, x) else y01(0, x)
	if n == 0 then
		return y0
	end
	local y1 = if large(1, x) then bessely(1, x) else y01(1, x)
	for k = 1, n - 1 do
		y0, y1 = y1, (2 * k / x) * y1 - y0
	end
	return y1
end

-- The modified Bessel function of the first kind I_n(x), for whole n >= 0: (1 / pi) times
-- the integral of e^(x cos t) cos(n t) over [0, pi]. It grows like e^x / sqrt(2 pi x).
local function besseli(n: number, x: number): number
	check_order("besseli", n)
	if x ~= x then
		return x
	end
	local sign = if x < 0 and n % 2 == 1 then -1 else 1
	x = math.abs(x)
	if x > 700 then
		return sign * math.huge
	end
	if x == 0 then
		return if n == 0 then 1 else 0
	end
	if x < 2 * n + 10 then
		-- every term positive: accurate for any x, and short while x is not large
		return sign * power_series(n, x, 1)
	end
	-- e^(x cos t) is scaled by e^-x inside and restored outside, so it cannot overflow
	local v = gl(function(t: number): number
		return math.exp(x * (math.cos(t) - 1)) * math.cos(n * t)
	end, 0, PI, points_for(math.sqrt(x) * 4 + n)) / PI
	return sign * v * math.exp(x)
end

-- K_0 or K_1: the integral of e^(-x cosh t) cosh(n t) over [0, inf), cut where it falls
-- below e^-40 of its start.
local function k01(n: number, x: number): number
	local y = 1 + 40 / x
	local T = math.log(y + math.sqrt(y * y - 1)) + n -- acosh
	return gl(function(t: number): number
		return math.exp(-x * (math.cosh(t) - 1)) * math.cosh(n * t)
	end, 0, T, 128) * math.exp(-x)
end

-- The modified Bessel function of the second kind K_n(x), for whole n >= 0 and x > 0
-- (infinite at 0): decaying like e^-x. K_0 and K_1 from their integrals, the rest by the
-- upward recurrence K_(k+1) = K_(k-1) + (2 k / x) K_k (stable for K).
local function besselk(n: number, x: number): number
	check_order("besselk", n)
	if x ~= x then
		return x
	end
	if not (x > 0) then
		return if x == 0 then math.huge else 0 / 0
	end
	local k0 = k01(0, x)
	if n == 0 then
		return k0
	end
	local k1 = k01(1, x)
	for k = 1, n - 1 do
		k0, k1 = k1, k0 + (2 * k / x) * k1
	end
	return k1
end

return {
	gamma = gamma,
	lgamma = lgamma,
	beta = beta,
	gamma_inc = gamma_inc,
	beta_inc = beta_inc,
	erf = erf,
	erfc = erfc,
	besselj = besselj,
	bessely = bessely,
	besseli = besseli,
	besselk = besselk,
}
