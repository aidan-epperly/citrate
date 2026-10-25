--!strict
--!native
--!optimize 2

-- Chebyshev technology for approximating functions on an interval, in the spirit of
-- Chebfun (Driscoll, Hale and Trefethen): Chebyshev points, barycentric interpolation,
-- coefficient transforms, adaptive construction of Chebyshev interpolants to machine
-- precision, and calculus, roots and extrema of the results. Everything is in double
-- precision; arrays are f64 arrays (buffers) and points are in ascending order.
--
-- A Chebfun is { a, b, coeffs, happy }: the Chebyshev coefficients of a polynomial on
-- [a, b] (in the variable t = (2x - a - b) / (b - a)), and whether construction resolved
-- the function to the requested tolerance.
--
-- standard_chop is a translation of Chebfun's standardChop, and roots follows Chebfun's
-- roots (BSD-3-Clause; see licenses/Chebfun-LICENSE.txt).

local f64 = require("./f64")
local c64 = require("./c64")
local types = require("./types")

export type Chebfun = types.Chebfun

-- Options for fun: tol (relative accuracy, default machine epsilon), maxlength (default
-- 65537 samples), n (a fixed number of points instead of adaptive construction).
export type FunOptions = { tol: number?, maxlength: number?, n: number? }

local EPS = 2 ^ -52

local function count(x: buffer): number
	return buffer.len(x) // 8
end

local function get(x: buffer, i: number): number
	return buffer.readf64(x, 8 * i)
end

local function set(x: buffer, i: number, v: number)
	buffer.writef64(x, 8 * i, v)
end

-- Points and weights

-- n Chebyshev points on [a, b] (default [-1, 1]) in ascending order: of the second kind
-- (the extrema of T_n-1, including the endpoints; the default) or of the first kind (the
-- roots of T_n). The sine formulas make the points exactly symmetric.
local function chebpts(n: number, kind: number?, a: number?, b: number?): buffer
	local k = kind or 2
	assert(k == 1 or k == 2, "kind must be 1 or 2")
	local lo, hi = a or -1, b or 1
	local x = f64.create(n)
	if n == 1 then
		set(x, 0, (lo + hi) / 2)
		return x
	end
	for j = 0, n - 1 do
		local t = if k == 2
			then math.sin(math.pi * (2 * j - (n - 1)) / (2 * (n - 1)))
			else math.sin(math.pi * (2 * j + 1 - n) / (2 * n))
		set(x, j, (lo + hi) / 2 + (hi - lo) / 2 * t)
	end
	if k == 2 then
		set(x, 0, lo)
		set(x, n - 1, hi)
	end
	return x
end

-- Barycentric weights for n Chebyshev points of the given kind (closed forms, scaled to
-- have largest magnitude 1).
local function chebweights(n: number, kind: number?): buffer
	local k = kind or 2
	local w = f64.create(n)
	for j = 0, n - 1 do
		local sign = if j % 2 == 0 then 1 else -1
		if k == 2 then
			set(w, j, if n == 1 then 1 elseif j == 0 or j == n - 1 then sign / 2 else sign)
		else
			set(w, j, sign * math.sin((2 * j + 1) * math.pi / (2 * n)))
		end
	end
	return w
end

-- Barycentric weights w_j = 1 / prod_(k ~= j) (x_j - x_k) for arbitrary distinct nodes,
-- with each factor scaled by 4 / (max - min) against overflow and underflow, and the
-- result scaled to have largest magnitude 1 (the formula is invariant under scaling).
local function bary_weights(nodes: buffer): buffer
	local n = count(nodes)
	local lo, hi = math.huge, -math.huge
	for j = 0, n - 1 do
		lo = math.min(lo, get(nodes, j))
		hi = math.max(hi, get(nodes, j))
	end
	local c = if hi > lo then 4 / (hi - lo) else 1
	local w = f64.create(n)
	local big = 0
	for j = 0, n - 1 do
		local p = 1
		local xj = get(nodes, j)
		for k = 0, n - 1 do
			if k ~= j then
				p *= c * (xj - get(nodes, k))
			end
		end
		assert(p ~= 0, "nodes must be distinct")
		set(w, j, 1 / p)
		big = math.max(big, math.abs(1 / p))
	end
	for j = 0, n - 1 do
		set(w, j, get(w, j) / big)
	end
	return w
end

-- The barycentric interpolation formula (second, "true" form): the value at x of the
-- polynomial through (nodes[j], values[j]), given barycentric weights (default:
-- bary_weights(nodes); pass chebweights(n, kind) for Chebyshev points). Exact at the
-- nodes, and numerically stable for Chebyshev points (Higham 2004).
local function bary(x: number, nodes: buffer, values: buffer, weights: buffer?): number
	local n = count(nodes)
	assert(count(values) == n, "need one value per node")
	local w = weights or bary_weights(nodes)
	local num, den = 0, 0
	for j = 0, n - 1 do
		local d = x - get(nodes, j)
		if d == 0 then
			return get(values, j)
		end
		local q = get(w, j) / d
		num += q * get(values, j)
		den += q
	end
	return num / den
end

-- Coefficient transforms

-- Chebyshev coefficients of the polynomial taking the given values at the n Chebyshev
-- points of the second kind (ascending), by a type 1 DCT.
local function vals2coeffs(values: buffer): buffer
	local n = count(values)
	local c = f64.create(n)
	if n <= 1 then
		buffer.copy(c, 0, values)
		return c
	end
	-- The DCT wants the points in descending order.
	for j = 0, n - 1 do
		set(c, j, get(values, n - 1 - j))
	end
	f64.dct(c, 1)
	for k = 0, n - 1 do
		local v = get(c, k) / (n - 1)
		if k == 0 or k == n - 1 then
			v /= 2
		end
		set(c, k, v)
	end
	return c
end

-- Values at the n Chebyshev points of the second kind (ascending) of the Chebyshev series
-- with the given n coefficients.
local function coeffs2vals(coeffs: buffer): buffer
	local n = count(coeffs)
	local v = f64.create(n)
	if n <= 1 then
		buffer.copy(v, 0, coeffs)
		return v
	end
	local y = f64.dct_copy(coeffs, 1)
	local c0, cn = get(coeffs, 0), get(coeffs, n - 1)
	for j = 0, n - 1 do
		local sign = if j % 2 == 0 then 1 else -1
		set(v, n - 1 - j, (get(y, j) + c0 + sign * cn) / 2)
	end
	return v
end

-- Value at t in [-1, 1] of sum c_k T_k(t), by Clenshaw's recurrence.
local function clenshaw(coeffs: buffer, t: number): number
	local n = count(coeffs)
	if n == 0 then
		return 0
	end
	local b1, b2 = 0, 0
	for k = n - 1, 1, -1 do
		b1, b2 = get(coeffs, k) + 2 * t * b1 - b2, b1
	end
	return get(coeffs, 0) + t * b1 - b2
end

-- Chopping

-- How many leading Chebyshev coefficients to keep: Aurentz and Trefethen's standardChop
-- ("Chopping a Chebyshev series", 2017), translated from Chebfun's standardChop.m. It looks for a plateau in
-- the monotone envelope of the coefficients below tol (relative to the largest one) and
-- cuts where the envelope plus a gentle linear tilt is smallest. Series shorter than 17
-- are never chopped.
local function standard_chop(coeffs: buffer, tol: number?): number
	local t = tol or EPS
	if t >= 1 then
		return 1
	end
	local n = count(coeffs)
	if n < 17 then
		return n
	end
	-- Step 1: the monotone envelope, normalized.
	local env = table.create(n, 0)
	env[n] = math.abs(get(coeffs, n - 1))
	for j = n - 1, 1, -1 do
		env[j] = math.max(math.abs(get(coeffs, j - 1)), env[j + 1])
	end
	if env[1] == 0 then
		return 1
	end
	for j = n, 1, -1 do
		env[j] /= env[1]
	end
	-- Step 2: find a plateau.
	local plateau, j2 = 0, 0
	for j = 2, n do
		j2 = math.floor(1.25 * j + 5 + 0.5)
		if j2 > n then
			return n
		end
		local e1, e2 = env[j], env[j2]
		local r = if e1 > 0 then 3 * (1 - math.log(e1) / math.log(t)) else 0
		if e1 == 0 or e2 / e1 > r then
			plateau = j - 1
			break
		end
	end
	-- Step 3: the cutoff, where the tilted log-envelope is smallest.
	if env[plateau] == 0 then
		return plateau
	end
	local floor = t ^ (7 / 6)
	local j3 = 0
	for j = 1, n do
		if env[j] >= floor then
			j3 += 1
		end
	end
	if j3 < j2 then
		j2 = j3 + 1
		env[j2] = floor
	end
	local best, d = math.huge, 1
	for j = 1, j2 do
		local tilt = if j2 > 1 then (j - 1) / (j2 - 1) * (-1 / 3) * math.log10(t) else 0
		local cc = math.log10(env[j]) + tilt
		if cc < best then
			best, d = cc, j
		end
	end
	return math.max(d - 1, 1)
end

-- Constructors

local function make(a: number, b: number, coeffs: buffer, happy: boolean): Chebfun
	return { a = a, b = b, coeffs = coeffs, happy = happy }
end

local function prefix(c: buffer, m: number): buffer
	local out = f64.create(m)
	buffer.copy(out, 0, c, 0, 8 * math.min(m, count(c)))
	return out
end

-- Drops coefficients below the noise level (standard_chop at machine precision) and
-- trailing zeros, keeping at least one.
local function simplify_coeffs(c: buffer): buffer
	local m = standard_chop(c, EPS)
	while m > 1 and get(c, m - 1) == 0 do
		m -= 1
	end
	return if m == count(c) then c else prefix(c, m)
end

-- Deterministic "random" points in (-1, 1) for the sample test.
local PROBES = { -0.6264350394837611, 0.1874613957813, 0.8126457613049 }

-- A Chebyshev interpolant of f on domain = { a, b } (default { -1, 1 }). By default it is
-- adaptive, as in Chebfun: f is sampled at 17, 33, 65, ... Chebyshev points until the
-- coefficients reach a plateau at relative size options.tol (default machine epsilon),
-- the series is chopped there by standard_chop, and a check at a few extra points guards
-- against being fooled. If options.maxlength (default 65537) samples do not suffice, the
-- result keeps them all and has happy = false. options.n instead fixes the number of
-- points (the interpolant then has degree n - 1).
local function fun(f: (number) -> number, domain: { number }?, options: FunOptions?): Chebfun
	local a = if domain then domain[1] else -1
	local b = if domain then domain[2] else 1
	assert(b > a, "the domain must be an interval { a, b } with a < b")
	local o: FunOptions = options or {}
	local tol = o.tol or EPS
	local function sample(n: number): buffer
		local x = chebpts(n, 2, a, b)
		local v = f64.create(n)
		for j = 0, n - 1 do
			local y = f(get(x, j))
			assert(type(y) == "number" and y == y and math.abs(y) < math.huge, "the function returned a non-finite value")
			set(v, j, y)
		end
		return v
	end
	local fixed = o.n
	if fixed then
		return make(a, b, vals2coeffs(sample(fixed)), true)
	end
	local maxlength = o.maxlength or 65537
	local n = 17
	while true do
		local values = sample(n)
		local vscale = 0
		for j = 0, n - 1 do
			vscale = math.max(vscale, math.abs(get(values, j)))
		end
		local c = vals2coeffs(values)
		local cutoff = standard_chop(c, tol)
		if cutoff < n then
			local kept = prefix(c, cutoff)
			local ok = true
			for _, t in PROBES do
				local x = (a + b) / 2 + (b - a) / 2 * t
				if math.abs(f(x) - clenshaw(kept, t)) > 1e3 * math.max(tol, EPS) * math.max(vscale, EPS) then
					ok = false
				end
			end
			if ok then
				return make(a, b, kept, true)
			end
		end
		if n >= maxlength then
			return make(a, b, c, false)
		end
		n = 2 * n - 1
	end
end

-- The Chebfun with the given Chebyshev coefficients on domain (default { -1, 1 }).
local function from_coeffs(coeffs: buffer, domain: { number }?): Chebfun
	local a = if domain then domain[1] else -1
	local b = if domain then domain[2] else 1
	return make(a, b, f64.copy(nil, coeffs), true)
end

-- The polynomial interpolant of the given values at the Chebyshev points of the second
-- kind on domain (default { -1, 1 }).
local function from_values(values: buffer, domain: { number }?): Chebfun
	local a = if domain then domain[1] else -1
	local b = if domain then domain[2] else 1
	return make(a, b, vals2coeffs(values), true)
end

-- Basic queries

-- Number of coefficients (degree + 1).
local function length(F: Chebfun): number
	return count(F.coeffs)
end

-- F(x), by Clenshaw's recurrence.
local function eval(F: Chebfun, x: number): number
	return clenshaw(F.coeffs, (2 * x - F.a - F.b) / (F.b - F.a))
end

-- F at each entry of xs.
local function evalmany(F: Chebfun, xs: buffer): buffer
	local out = f64.create(count(xs))
	for j = 0, count(xs) - 1 do
		set(out, j, eval(F, get(xs, j)))
	end
	return out
end

-- The Chebyshev points of F (second kind, on its domain) and F's values there.
local function points(F: Chebfun): buffer
	return chebpts(length(F), 2, F.a, F.b)
end

-- F's values at its Chebyshev points (second kind, as chebpts gives them), as an f64 array.
local function values(F: Chebfun): buffer
	return coeffs2vals(F.coeffs)
end

-- Arithmetic

local function same_domain(F: Chebfun, G: Chebfun)
	assert(F.a == G.a and F.b == G.b, "the functions have different domains")
end

local function padded(c: buffer, m: number): buffer
	return if count(c) == m then c else prefix(c, m)
end

-- F + G (same domain).
local function add(F: Chebfun, G: Chebfun): Chebfun
	same_domain(F, G)
	local m = math.max(length(F), length(G))
	local c = f64.add(nil, padded(F.coeffs, m), padded(G.coeffs, m))
	return make(F.a, F.b, simplify_coeffs(c), F.happy and G.happy)
end

-- F - G (same domain).
local function sub(F: Chebfun, G: Chebfun): Chebfun
	same_domain(F, G)
	local m = math.max(length(F), length(G))
	local c = f64.sub(nil, padded(F.coeffs, m), padded(G.coeffs, m))
	return make(F.a, F.b, simplify_coeffs(c), F.happy and G.happy)
end

-- s F.
local function scale(F: Chebfun, s: number): Chebfun
	return make(F.a, F.b, f64.scale(nil, s, F.coeffs), F.happy)
end

-- F + s (adds a constant).
local function shift(F: Chebfun, s: number): Chebfun
	local c = f64.copy(nil, F.coeffs)
	set(c, 0, get(c, 0) + s)
	return make(F.a, F.b, c, F.happy)
end

-- F G (same domain): exact product on length(F) + length(G) - 1 points, then chopped.
local function mul(F: Chebfun, G: Chebfun): Chebfun
	same_domain(F, G)
	local m = length(F) + length(G) - 1
	local v = f64.mul(nil, coeffs2vals(padded(F.coeffs, m)), coeffs2vals(padded(G.coeffs, m)))
	return make(F.a, F.b, simplify_coeffs(vals2coeffs(v)), F.happy and G.happy)
end

-- Calculus

-- The k-th derivative of F (default k = 1), by the Chebyshev derivative recurrence.
local function diff(F: Chebfun, k: number?): Chebfun
	local c = F.coeffs
	local scalef = 2 / (F.b - F.a)
	for _ = 1, k or 1 do
		local n = count(c)
		if n <= 1 then
			c = f64.create(1)
			break
		end
		local d = table.create(n + 1, 0)
		for j = n - 1, 1, -1 do
			d[j] = d[j + 2] + 2 * j * get(c, j)
		end
		d[1] /= 2
		local out = f64.create(n - 1)
		for j = 0, n - 2 do
			set(out, j, d[j + 1] * scalef)
		end
		c = out
	end
	return make(F.a, F.b, c, F.happy)
end

-- The indefinite integral of F that vanishes at the left endpoint a.
local function cumsum(F: Chebfun): Chebfun
	local n = length(F)
	local function c(k: number): number
		return if k < n then get(F.coeffs, k) else 0
	end
	local half = (F.b - F.a) / 2
	local out = f64.create(n + 1)
	set(out, 1, (c(0) - c(2) / 2) * half)
	for k = 2, n do
		set(out, k, (c(k - 1) - c(k + 1)) / (2 * k) * half)
	end
	-- Choose the constant so that the value at t = -1 is 0.
	local s = 0
	for k = 1, n do
		s += (if k % 2 == 0 then 1 else -1) * get(out, k)
	end
	set(out, 0, -s)
	return make(F.a, F.b, simplify_coeffs(out), F.happy)
end

-- The definite integral of F over its domain (Clenshaw-Curtis, from the coefficients).
local function sum(F: Chebfun): number
	local s = 0
	for k = 0, length(F) - 1, 2 do
		s += 2 * get(F.coeffs, k) / (1 - k * k)
	end
	return s * (F.b - F.a) / 2
end

-- The L2 inner product of F and G (same domain).
local function inner(F: Chebfun, G: Chebfun): number
	return sum(mul(F, G))
end

-- The L2 norm of F.
local function norm(F: Chebfun): number
	return math.sqrt(math.max(inner(F, F), 0))
end

-- Roots

-- Roots in [-1, 1] of the Chebyshev series in the table c (c[1] is the T_0 coefficient),
-- appended to out (in the variable t). Degree up to 50: eigenvalues of the colleague
-- matrix; higher: split [-1, 1] at an arbitrary point and recurse, as Chebfun does.
local function roots_unit(c: { number }, out: { number })
	local big = 0
	for _, v in c do
		big = math.max(big, math.abs(v))
	end
	if big == 0 then
		return
	end
	local n = #c
	while n > 1 and math.abs(c[n]) <= EPS * big do
		n -= 1
	end
	local degree = n - 1
	if degree == 0 then
		return
	elseif degree == 1 then
		local r = -c[1] / c[2]
		if math.abs(r) <= 1 + 1e-12 then
			table.insert(out, math.clamp(r, -1, 1))
		end
		return
	elseif degree <= 50 then
		local A = f64.matrix(degree, degree)
		local function put(i: number, j: number, v: number)
			f64.matset(A, i, j, f64.matget(A, i, j) + v)
		end
		put(1, 2, 1)
		for k = 2, degree - 1 do
			put(k, k - 1, 0.5)
			put(k, k + 1, 0.5)
		end
		put(degree, degree - 1, 0.5)
		for k = 1, degree do
			put(degree, k, -c[k] / (2 * c[n]))
		end
		-- eig balances the matrix first, which matters here: with a small leading
		-- coefficient, the last row of the colleague matrix is huge.
		local re, im = c64.totable(f64.eig(A, false).values)
		for k = 1, degree do
			if math.abs(im[k]) <= 1e-8 and math.abs(re[k]) <= 1 + 1e-8 then
				table.insert(out, math.clamp(re[k], -1, 1))
			end
		end
		return
	end
	-- Split at Chebfun's (deliberately unremarkable) point and recurse.
	local split = -0.004849834917525
	local coeffs = f64.from(table.move(c, 1, n, 1, table.create(n, 0)))
	for _, side in { { -1, split }, { split, 1 } } do
		local lo, hi = side[1], side[2]
		local x = chebpts(n, 2, lo, hi)
		local v = f64.create(n)
		for j = 0, n - 1 do
			set(v, j, clenshaw(coeffs, get(x, j)))
		end
		local sub_coeffs = f64.totable(simplify_coeffs(vals2coeffs(v)))
		local found = {}
		roots_unit(sub_coeffs, found)
		for _, t in found do
			table.insert(out, (lo + hi) / 2 + (hi - lo) / 2 * t)
		end
	end
end

-- The real roots of F in its domain, in ascending order, polished by Newton's method.
-- Roots closer together than 1e-8 (relative to the domain) are merged, keeping the one
-- where |F| is smaller: the recursion can find a root twice near its split points.
local function roots(F: Chebfun): buffer
	local found = {}
	roots_unit(f64.totable(F.coeffs), found)
	local D = diff(make(-1, 1, F.coeffs, true))
	local polished: { { number } } = {}
	for _, t in found do
		-- Newton steps, while they keep reducing |F|.
		local p = clenshaw(F.coeffs, t)
		for _ = 1, 8 do
			local dp = clenshaw(D.coeffs, t)
			if dp == 0 or p == 0 then
				break
			end
			local tn = math.clamp(t - p / dp, -1, 1)
			local pn = clenshaw(F.coeffs, tn)
			if math.abs(pn) > math.abs(p) or tn == t then
				break
			end
			t, p = tn, pn
		end
		table.insert(polished, { t, math.abs(p) })
	end
	table.sort(polished, function(u: { number }, v: { number }): boolean
		return u[1] < v[1]
	end)
	local merged = {}
	for _, item in polished do
		local last = merged[#merged]
		if last and item[1] - last[1] <= 1e-8 then
			if item[2] < last[2] then
				merged[#merged] = item
			end
		else
			table.insert(merged, item)
		end
	end
	local out = f64.create(#merged)
	for j, item in merged do
		set(out, j - 1, (F.a + F.b) / 2 + (F.b - F.a) / 2 * item[1])
	end
	return out
end

-- Extrema

-- The global minimum and maximum of F on its domain, with their locations: min value,
-- where, max value, where. The candidates are the endpoints and the roots of F'.
local function minmax(F: Chebfun): (number, number, number, number)
	local candidates = { F.a, F.b }
	local r = roots(diff(F))
	for j = 0, count(r) - 1 do
		table.insert(candidates, get(r, j))
	end
	local lo, lox, hi, hix = math.huge, F.a, -math.huge, F.a
	for _, x in candidates do
		local v = eval(F, x)
		if v < lo then
			lo, lox = v, x
		end
		if v > hi then
			hi, hix = v, x
		end
	end
	return lo, lox, hi, hix
end

-- The maximum of F and where it is attained.
local function max(F: Chebfun): (number, number)
	local _, _, hi, hix = minmax(F)
	return hi, hix
end

-- The minimum of F and where it is attained.
local function min(F: Chebfun): (number, number)
	local lo, lox = minmax(F)
	return lo, lox
end

-- Spectral methods and quadrature

-- The n by n Chebyshev differentiation matrix on [a, b] (default [-1, 1]): D v holds the
-- derivative, at the n Chebyshev points of the second kind, of the interpolant of v
-- there. Off-diagonal entries from the barycentric weights; the diagonal makes each row
-- sum to zero (exact for constants, and more accurate than the closed form).
local function diffmat(n: number, a: number?, b: number?): types.Matrix
	local lo, hi = a or -1, b or 1
	local x = chebpts(n, 2, -1, 1)
	local w = chebweights(n, 2)
	local D = f64.matrix(n, n)
	local s = 2 / (hi - lo)
	for i = 0, n - 1 do
		local rowsum = 0
		for j = 0, n - 1 do
			if i ~= j then
				local v = (get(w, j) / get(w, i)) / (get(x, i) - get(x, j))
				f64.matset(D, i + 1, j + 1, v * s)
				rowsum += v
			end
		end
		f64.matset(D, i + 1, i + 1, -rowsum * s)
	end
	return D
end

-- Clenshaw-Curtis quadrature on [a, b] (default [-1, 1]): the n Chebyshev points of the
-- second kind and weights that integrate polynomials of degree n - 1 exactly. The
-- interpolant's coefficients are linear in the values, c = M f (see vals2coeffs: with
-- x_j = cos(pi j / N), c_k = (2 / N) g_k sum_j h_j f_j cos(pi j k / N), where g and h halve
-- the first and last terms), so its integral sum_k m_k c_k, with m_k the integral of T_k
-- (2 / (1 - k^2) for even k, 0 for odd), is w^T f for w = M^T m. The weights are symmetric,
-- so the order of the points does not matter.
local function ccquad(n: number, a: number?, b: number?): (buffer, buffer)
	local lo, hi = a or -1, b or 1
	local x = chebpts(n, 2, lo, hi)
	local w = f64.create(n)
	if n == 1 then
		set(w, 0, hi - lo)
		return x, w
	end
	local N = n - 1
	for j = 0, N do
		local s = 0
		for k = 0, N, 2 do
			local mk = 2 / (1 - k * k)
			local gk = if k == 0 or k == N then 0.5 else 1
			s += mk * gk * math.cos(math.pi * j * k / N)
		end
		local hj = if j == 0 or j == N then 0.5 else 1
		set(w, j, (2 / N) * hj * s * (hi - lo) / 2)
	end
	return x, w
end

-- Gauss-Legendre quadrature on [a, b] (default [-1, 1]): n nodes (ascending) and weights
-- integrating polynomials of degree 2n - 1 exactly, by Golub and Welsch (eigenvalues and
-- first eigenvector components of the Jacobi matrix). O(n^3); fine for n up to a few
-- hundred.
local function legpts(n: number, a: number?, b: number?): (buffer, buffer)
	local lo, hi = a or -1, b or 1
	local J = f64.matrix(n, n)
	for k = 1, n - 1 do
		local beta = k / math.sqrt(4 * k * k - 1)
		f64.matset(J, k + 1, k, beta)
		f64.matset(J, k, k + 1, beta)
	end
	local E = f64.eigh(J)
	local V = E.vectors :: types.Matrix
	local x, w = f64.create(n), f64.create(n)
	for j = 0, n - 1 do
		-- Exact symmetry: the nodes come in +- pairs.
		set(x, j, (lo + hi) / 2 + (hi - lo) / 2 * get(E.values, j))
		local v1 = f64.matget(V, 1, j + 1)
		set(w, j, 2 * v1 * v1 * (hi - lo) / 2)
	end
	return x, w
end

return {
	chebpts = chebpts,
	chebweights = chebweights,
	bary_weights = bary_weights,
	bary = bary,
	vals2coeffs = vals2coeffs,
	coeffs2vals = coeffs2vals,
	clenshaw = clenshaw,
	standard_chop = standard_chop,

	fun = fun,
	from_coeffs = from_coeffs,
	from_values = from_values,
	length = length,
	eval = eval,
	evalmany = evalmany,
	points = points,
	values = values,

	add = add,
	sub = sub,
	scale = scale,
	shift = shift,
	mul = mul,

	diff = diff,
	cumsum = cumsum,
	sum = sum,
	inner = inner,
	norm = norm,

	roots = roots,
	minmax = minmax,
	max = max,
	min = min,

	diffmat = diffmat,
	ccquad = ccquad,
	legpts = legpts,
}
