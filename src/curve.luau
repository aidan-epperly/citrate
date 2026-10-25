--!strict
--!native
--!optimize 2

-- Parametric curves with points in any Space<T>: Bézier curves of any degree (polynomial
-- or rational), cubic Hermite curves, Catmull-Rom and Kochanek-Bartels splines,
-- interpolating cubic splines, and B-splines and NURBS. Every constructor converts its
-- input to one form, a piecewise Bézier curve (see types.Curve), so evaluation,
-- derivatives, curvature, frames, splitting, closest points and arclength are written once.
--
-- Points can be `vector`s (detected automatically), f32/f64 buffers of any dimension, or
-- anything with a Space. For `vector`s there is a native fast path that accumulates in
-- double precision and rounds once per result, and constructors compute control points in
-- double precision and round each once. Scalars (speed, curvature, arclength, distances)
-- are always computed in double precision; dot products of derivatives come from the Gram
-- matrix of each segment's control points, so they are smooth functions of t whatever the
-- representation.
--
-- Arclength uses Chebyshev technology (see cheb): on each segment the speed |C'(t)| is a
-- Chebfun (split at cusps, where it vanishes), its indefinite integral is the arclength
-- s(t), and the inverse t(s) is a Chebfun too, so evaluating by arclength costs a binary
-- search and a short Clenshaw sum. These are built on first use and cached.
--
-- Parameters: a curve is defined on [breaks[1], breaks[#breaks]]; evaluating outside it
-- extends the end segments. Arclength arguments are clamped to [0, length].

local types = require("./types")
local vec = require("./vec")
local f64 = require("./f64")
local cheb = require("./cheb")

type Space<T> = types.Space<T>
type Chebfun = types.Chebfun
export type Curve<T> = types.Curve<T>
export type CurveOptions<T> = types.CurveOptions<T>

local EPS = 2 ^ -52

-- Typed table constructors (table.create returns {unknown}).
local function numbers(n: number, v: number?): { number }
	return table.create(n, v or 0) :: any
end

local function list(n: number): { any }
	return table.create(n) :: any
end

-- Per-curve data computed on demand: Gram matrices (generic spaces), bounding spheres for
-- closest-point searches, and the arclength tables.
type Piece = { a: number, b: number, s0: number, len: number, S: Chebfun, V: Chebfun, inv: Chebfun }
type Arc = { pieces: { Piece }, total: number, segs0: { number }, segpieces: { { number } }, tol: number }
type Aux = { gram: { [number]: { number } }, spheres: { { any } }?, arc: Arc? }

local auxes: { [any]: Aux } = setmetatable({}, { __mode = "k" }) :: any

local function aux(C: any): Aux
	local a = auxes[C]
	if a == nil then
		local fresh: Aux = { gram = {} }
		auxes[C] = fresh
		return fresh
	end
	return a
end

-- Construction helpers

-- The space of the points and whether they are vectors.
local function resolve<T>(points: { T }, space: Space<T>?): (Space<T>, boolean)
	assert(#points > 0, "need at least one point")
	local isvector = type(points[1]) == "vector"
	if space then
		return space, isvector
	end
	assert(isvector, "points that are not vectors need options.space")
	return (vec.space :: any) :: Space<T>, true
end

-- Constructors compute in a work space: for vectors, f64 buffers of length 3, so every
-- control point is computed in double precision and rounded once; otherwise the points'
-- own space.
local function to_work(isvector: boolean, p: any): any
	if isvector then
		local v = p :: vector
		return f64.from({ v.x, v.y, v.z })
	end
	return p
end

local function work_points<T>(points: { T }, space: Space<T>?): (Space<any>, { any }, Space<T>, boolean)
	local S, isvector = resolve(points, space)
	if not isvector then
		return S :: any, points :: any, S, false
	end
	local wp = list(#points)
	for i, p in points do
		wp[i] = to_work(true, p)
	end
	return f64.space :: any, wp, S, true
end

local function from_work(isvector: boolean, pts: { any }): { any }
	if not isvector then
		return pts
	end
	local out = list(#pts)
	for i, b in pts do
		out[i] = vector.create(f64.get(b, 1), f64.get(b, 2), f64.get(b, 3))
	end
	return out
end

-- sum of cs[k] * ps[k], as a new point of the work space.
local function lincomb(W: Space<any>, ps: { any }, cs: { number }): any
	local o = W.scale(W.similar(ps[1]), cs[1], ps[1])
	for k = 2, #ps do
		if cs[k] ~= 0 then
			o = W.axpy(o, cs[k], ps[k], o)
		end
	end
	return o
end

local function make<T>(
	space: Space<T>,
	isvector: boolean,
	degree: number,
	breaks: { number },
	points: { T },
	weights: { number }?,
	closed: boolean?
): Curve<T>
	assert(#points == (#breaks - 1) * (degree + 1), "curve: points and breaks do not match")
	for i = 1, #breaks - 1 do
		assert(breaks[i + 1] > breaks[i], "curve: breaks must increase")
	end
	if weights then
		assert(#weights == #points, "curve: one weight per control point")
		for _, w in weights do
			assert(w > 0, "curve: weights must be positive")
		end
	end
	return {
		space = space,
		vector = isvector,
		degree = degree,
		breaks = breaks,
		points = points,
		weights = weights,
		closed = closed == true,
	}
end

-- Bernstein basis

-- The Bernstein polynomials of degree m at u, into out[1 .. m + 1].
local function bernstein(m: number, u: number, out: { number })
	out[1] = 1
	local v = 1 - u
	for level = 1, m do
		out[level + 1] = 0
		for j = level + 1, 2, -1 do
			out[j] = v * out[j] + u * out[j - 1]
		end
		out[1] *= v
	end
end

-- Coefficients (over the d + 1 control points) of the k-th derivative with respect to t
-- of a polynomial Bézier segment of degree d and length h, at local parameter u, into c.
local function poly_coeffs(d: number, u: number, h: number, k: number, c: { number }): { number }
	for j = 1, d + 1 do
		c[j] = 0
	end
	if k > d then
		return c
	end
	bernstein(d - k, u, c)
	local len = d - k + 1
	for _ = 1, k do
		-- The adjoint of the forward difference: sum b_j (P_(j+1) - P_j) = sum P_j (b_(j-1) - b_j).
		for j = len + 1, 1, -1 do
			c[j] = (if j > 1 then c[j - 1] else 0) - (if j <= len then c[j] else 0)
		end
		len += 1
	end
	local f = 1
	for i = 0, k - 1 do
		f *= (d - i) / h
	end
	for j = 1, d + 1 do
		c[j] *= f
	end
	return c
end

local function binomial(n: number, k: number): number
	local r = 1
	for i = 1, k do
		r = r * (n - k + i) / i
	end
	return r
end

-- Scratch arrays for coeff_table, one per derivative order (it is called in every
-- evaluation, so it does not allocate for polynomial curves).
local coeff_scratch: { { number } } = {}

-- Coefficients of derivatives 0 .. K of segment seg at local parameter u (segment length
-- h), as a table of arrays over the segment's control points. Rational segments use
-- p^(k) = (N^(k) - sum_(m=1..k) binomial(k, m) D^(m) p^(k-m)) / D, which keeps every
-- derivative a linear combination of the control points. For polynomial curves the
-- arrays are reused by the next call: use them before calling again.
local function coeff_table(C: Curve<any>, seg: number, u: number, h: number, K: number): { { number } }
	local d = C.degree
	local a = coeff_scratch
	for k = 0, K do
		local arr = a[k + 1]
		if arr == nil then
			arr = numbers(d + 1)
			a[k + 1] = arr
		end
		poly_coeffs(d, u, h, k, arr)
	end
	local W = C.weights
	if not W then
		return a
	end
	local base = (seg - 1) * (d + 1)
	local D = numbers(K + 1)
	for k = 0, K do
		local s = 0
		for j = 1, d + 1 do
			s += a[k + 1][j] * W[base + j]
		end
		D[k + 1] = s
	end
	local c = list(K + 1)
	for k = 0, K do
		local ck = numbers(d + 1)
		for j = 1, d + 1 do
			local v = a[k + 1][j] * W[base + j]
			for m = 1, k do
				v -= binomial(k, m) * D[m + 1] * c[k - m + 1][j]
			end
			ck[j] = v / D[1]
		end
		c[k + 1] = ck
	end
	return c
end

-- The denominator D(u) = sum B_j(u) w_j of a rational segment (1 for polynomial ones).
local function denominator(C: Curve<any>, seg: number, u: number): number
	local W = C.weights
	if not W then
		return 1
	end
	local d = C.degree
	local b = numbers(d + 1)
	bernstein(d, u, b)
	local base = (seg - 1) * (d + 1)
	local s = 0
	for j = 1, d + 1 do
		s += b[j] * W[base + j]
	end
	return s
end

-- Segment containing t (the last one at or before it; end segments outside the domain),
-- the local parameter u and the segment length.
local function locate(C: Curve<any>, t: number): (number, number, number)
	local br = C.breaks
	local n = #br - 1
	local lo = 1
	if t >= br[n + 1] then
		lo = n
	elseif t > br[1] then
		local hi = n
		while hi > lo do
			local mid = (lo + hi + 1) // 2
			if br[mid] <= t then
				lo = mid
			else
				hi = mid - 1
			end
		end
	end
	local h = br[lo + 1] - br[lo]
	return lo, (t - br[lo]) / h, h
end

-- Combining control points

-- For vector curves: sum over j of c[j] P_j on segment seg, in double precision. With
-- relative, the coefficients sum to zero (a derivative) and P_j - P_1 is used instead,
-- which avoids cancellation far from the origin.
local function comb3(C: Curve<any>, seg: number, c: { number }, relative: boolean): (number, number, number)
	local d = C.degree
	local base = (seg - 1) * (d + 1)
	local pts = C.points :: { vector }
	local x, y, z = 0, 0, 0
	if relative then
		local P0 = pts[base + 1]
		local x0, y0, z0 = P0.x, P0.y, P0.z
		for j = 2, d + 1 do
			local P, cj = pts[base + j], c[j]
			x += cj * (P.x - x0)
			y += cj * (P.y - y0)
			z += cj * (P.z - z0)
		end
	else
		for j = 1, d + 1 do
			local P, cj = pts[base + j], c[j]
			x += cj * P.x
			y += cj * P.y
			z += cj * P.z
		end
	end
	return x, y, z
end

-- sum over j of c[j] P_j on segment seg, as a point (into out for buffer spaces).
local function combine<T>(C: Curve<T>, seg: number, c: { number }, relative: boolean, out: T?): T
	if C.vector then
		local x, y, z = comb3(C, seg, c, relative)
		return vector.create(x, y, z) :: any
	end
	local S, d = C.space, C.degree
	local base = (seg - 1) * (d + 1)
	local pts = C.points
	local target: T = if out ~= nil then out :: T else S.similar(pts[base + 1])
	local o = S.scale(target, c[1], pts[base + 1])
	for j = 2, d + 1 do
		o = S.axpy(o, c[j], pts[base + j], o)
	end
	return o
end

-- Gram matrix (P_j - P_1) . (P_k - P_1), j, k = 2 .. d + 1, of a segment (generic spaces).
local function gram(C: Curve<any>, seg: number): { number }
	local A = aux(C)
	local G = A.gram[seg]
	if G then
		return G
	end
	local S, d = C.space, C.degree
	local base = (seg - 1) * (d + 1)
	local P0 = C.points[base + 1]
	local diffs = list(d)
	for j = 1, d do
		diffs[j] = S.sub(S.similar(P0), C.points[base + 1 + j], P0)
	end
	local g = numbers(d * d)
	for j = 1, d do
		for k = j, d do
			local v = S.dot(diffs[j], diffs[k])
			g[(j - 1) * d + k] = v
			g[(k - 1) * d + j] = v
		end
	end
	A.gram[seg] = g
	return g
end

-- (sum a_j P_j) . (sum b_j P_j) for coefficient arrays that each sum to zero (derivatives).
local function dotc(C: Curve<any>, seg: number, a: { number }, b: { number }): number
	if C.vector then
		local ax, ay, az = comb3(C, seg, a, true)
		local bx, by, bz = comb3(C, seg, b, true)
		return ax * bx + ay * by + az * bz
	end
	local G = gram(C, seg)
	local d = C.degree
	local s = 0
	for j = 1, d do
		local aj = a[j + 1]
		if aj ~= 0 then
			local row = 0
			for k = 1, d do
				row += G[(j - 1) * d + k] * b[k + 1]
			end
			s += aj * row
		end
	end
	return s
end

-- Evaluation and differential geometry

local scratch: { number } = numbers(16)

-- C(t) (into out for buffer spaces).
local function eval<T>(C: Curve<T>, t: number, out: T?): T
	local seg, u = locate(C, t)
	local d = C.degree
	bernstein(d, u, scratch)
	local W = C.weights
	if W then
		local base = (seg - 1) * (d + 1)
		local s = 0
		for j = 1, d + 1 do
			local v = scratch[j] * W[base + j]
			scratch[j] = v
			s += v
		end
		for j = 1, d + 1 do
			scratch[j] /= s
		end
	end
	return combine(C, seg, scratch, false, out)
end

-- The k-th derivative C^(k)(t) (k >= 0; into out for buffer spaces).
local function derivative<T>(C: Curve<T>, t: number, k: number, out: T?): T
	if k == 0 then
		return eval(C, t, out)
	end
	local seg, u, h = locate(C, t)
	return combine(C, seg, coeff_table(C, seg, u, h, k)[k + 1], true, out)
end

-- |C'(t)|.
local function speed<T>(C: Curve<T>, t: number): number
	local seg, u, h = locate(C, t)
	local c = coeff_table(C, seg, u, h, 1)[2]
	return math.sqrt(math.max(dotc(C, seg, c, c), 0))
end

-- The unit tangent C'(t) / |C'(t)| (NaN where the speed is zero).
local function tangent<T>(C: Curve<T>, t: number, out: T?): T
	local seg, u, h = locate(C, t)
	local c = coeff_table(C, seg, u, h, 1)[2]
	if C.vector then
		local x, y, z = comb3(C, seg, c, true)
		local n = math.sqrt(x * x + y * y + z * z)
		return vector.create(x / n, y / n, z / n) :: any
	end
	local S = C.space
	local v = combine(C, seg, c, true, out)
	return S.scale(v, 1 / math.sqrt(dotc(C, seg, c, c)), v)
end

-- Curvature |C' x C''| / |C'|^3, in any dimension (from the Gram determinant of C' and
-- C''). NaN where the speed is zero.
local function curvature<T>(C: Curve<T>, t: number): number
	local seg, u, h = locate(C, t)
	local c = coeff_table(C, seg, u, h, 2)
	local g11 = dotc(C, seg, c[2], c[2])
	local g12 = dotc(C, seg, c[2], c[3])
	local g22 = dotc(C, seg, c[3], c[3])
	return math.sqrt(math.max(g11 * g22 - g12 * g12, 0)) / g11 ^ 1.5
end

-- Torsion (C' x C'') . C''' / |C' x C''|^2 of a curve of vectors. NaN where the curvature
-- is zero.
local function torsion(C: Curve<vector>, t: number): number
	assert(C.vector, "torsion needs vector points")
	local seg, u, h = locate(C, t)
	local c = coeff_table(C, seg, u, h, 3)
	local ax, ay, az = comb3(C, seg, c[2], true)
	local bx, by, bz = comb3(C, seg, c[3], true)
	local cx, cy, cz = comb3(C, seg, c[4], true)
	local nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
	return (nx * cx + ny * cy + nz * cz) / (nx * nx + ny * ny + nz * nz)
end

-- The Frenet frame at t: unit tangent, unit normal and (vector curves only) binormal.
-- The normal and binormal are nil where the curvature is zero.
local function frenet<T>(C: Curve<T>, t: number): (T, T?, vector?)
	local seg, u, h = locate(C, t)
	local c = coeff_table(C, seg, u, h, 2)
	if C.vector then
		local ax, ay, az = comb3(C, seg, c[2], true)
		local bx, by, bz = comb3(C, seg, c[3], true)
		local s = math.sqrt(ax * ax + ay * ay + az * az)
		local tx, ty, tz = ax / s, ay / s, az / s
		local p = bx * tx + by * ty + bz * tz
		local nx, ny, nz = bx - p * tx, by - p * ty, bz - p * tz
		local m = math.sqrt(nx * nx + ny * ny + nz * nz)
		local T = vector.create(tx, ty, tz)
		if not (m > 1e-12 * math.sqrt(bx * bx + by * by + bz * bz)) then
			return T :: any, nil, nil
		end
		nx, ny, nz = nx / m, ny / m, nz / m
		local N = vector.create(nx, ny, nz)
		return T :: any, N :: any, vector.create(ty * nz - tz * ny, tz * nx - tx * nz, tx * ny - ty * nx)
	end
	local S = C.space
	local g11 = dotc(C, seg, c[2], c[2])
	local g12 = dotc(C, seg, c[2], c[3])
	local g22 = dotc(C, seg, c[3], c[3])
	local d1 = combine(C, seg, c[2], true)
	local T = S.scale(d1, 1 / math.sqrt(g11), d1)
	-- N is C'' minus its component along C', normalized: |that|^2 = g22 - g12^2 / g11.
	local perp = g22 - g12 * g12 / g11
	if not (perp > 1e-24 * g22) then
		return T, nil, nil
	end
	local d2 = combine(C, seg, c[3], true)
	local N = S.axpy(d2, -g12 / g11, combine(C, seg, c[2], true), d2)
	N = S.scale(N, 1 / math.sqrt(perp), N)
	return T, N, nil
end

-- Arclength

-- Solves S(t) = s for t in [a, b] (S increasing, with derivative V): Newton's method,
-- safeguarded by bisection, from the given guess, until a Newton step is below stop
-- (relative to b - a; default 1e-10).
local function solve_length(S: Chebfun, V: Chebfun, a: number, b: number, s: number, guess: number, stop: number?): number
	local small = (stop or 1e-10) * (b - a)
	local lo, hi = a, b
	local t = math.clamp(guess, a, b)
	for _ = 1, 100 do
		local f = cheb.eval(S, t) - s
		if f == 0 then
			return t
		elseif f > 0 then
			hi = t
		else
			lo = t
		end
		local v = cheb.eval(V, t)
		local tn = if v > 0 then t - f / v else lo
		local newton = v > 0 and tn >= lo and tn <= hi
		if not newton then
			tn = (lo + hi) / 2
		end
		-- Newton converges quadratically, so after a Newton step this small tn is exact to
		-- rounding (waiting for a smaller step can stall on the rounding noise in S).
		if (newton and math.abs(tn - t) <= small) or hi - lo <= 4 * EPS * (math.abs(t) + (b - a)) then
			return tn
		end
		t = tn
	end
	return t
end

-- The inverse t(s) of the arclength S on a piece [a, b] of length len, as a Chebfun on
-- [0, len]: sampled at nested Chebyshev points (17, 33, 65, ... up to 1025) until the
-- coefficients have decayed to tol, as cheb.fun does. Every point of the previous level is
-- reused, and each new sample is solved by Newton's method starting from the previous
-- level's interpolant, so it converges in a step or two.
local function inverse_fun(S: Chebfun, V: Chebfun, a: number, b: number, len: number, tol: number): Chebfun
	local n = 17
	local old: buffer? = nil
	local oldcoeffs: buffer? = nil
	while true do
		local x = cheb.chebpts(n, 2, 0, len)
		local vals = buffer.create(8 * n)
		local vscale = 0
		for j = 0, n - 1 do
			local t: number
			if old and j % 2 == 0 then
				t = buffer.readf64(old, 4 * j)
			else
				local s = buffer.readf64(x, 8 * j)
				local guess = if oldcoeffs then cheb.clenshaw(oldcoeffs, 2 * s / len - 1) else a + (b - a) * s / len
				t = solve_length(S, V, a, b, s, guess)
			end
			buffer.writef64(vals, 8 * j, t)
			vscale = math.max(vscale, math.abs(t))
		end
		local c = cheb.vals2coeffs(vals)
		local cutoff = cheb.standard_chop(c, tol)
		if cutoff < n then
			local kept = buffer.create(8 * cutoff)
			buffer.copy(kept, 0, c, 0, 8 * cutoff)
			-- Guard against being fooled, as cheb.fun does.
			local ok = true
			for _, frac in { 0.3071, 0.7713 } do
				local s = frac * len
				local want = solve_length(S, V, a, b, s, cheb.clenshaw(kept, 2 * frac - 1))
				if math.abs(cheb.clenshaw(kept, 2 * frac - 1) - want) > 1e3 * math.max(tol, EPS) * math.max(vscale, EPS) then
					ok = false
				end
			end
			if ok then
				return { a = 0, b = len, coeffs = kept, happy = true }
			end
		end
		if n >= 1025 then
			return { a = 0, b = len, coeffs = c, happy = false }
		end
		old, oldcoeffs = vals, c
		n = 2 * n - 1
	end
end

-- The fast path for a polynomial segment's arclength: the speed sampled at m fixed
-- Chebyshev points (17, then 25 or 33 if those do not resolve it; 33 for tolerances below
-- 1e-10), its integral, and the inverse
-- t(s) at m points by Newton's method, each transformed to coefficients by a direct cosine
-- sum (cheaper than an FFT at these sizes). It gives up (nil, and the adaptive
-- construction is used) where the speed nearly vanishes (a cusp, which the adaptive one
-- splits at) or the coefficients have not decayed to tol by the end.
local COS: { [number]: buffer } = {}
local function cosine_table(m: number): buffer
	local t = COS[m]
	if t == nil then
		local nb = buffer.create(8 * m * m)
		for k = 0, m - 1 do
			for j = 0, m - 1 do
				-- T_k at the j-th point of the second kind, ascending: x_j = -cos(pi j / (m - 1))
				buffer.writef64(nb, 8 * (k * m + j), math.cos(math.pi * k * (m - 1 - j) / (m - 1)))
			end
		end
		COS[m] = nb
		t = nb
	end
	return t
end
-- Chebyshev coefficients of the values at m points of the second kind (ascending), chopped
-- after the last one above tol times the largest value; nil if the last three are not all
-- below that (not resolved).
local function fixed_coeffs(vals: { number }, m: number, tol: number): buffer?
	local T = cosine_table(m)
	local vscale = 0
	for j = 1, m do
		vscale = math.max(vscale, math.abs(vals[j]))
	end
	local c = table.create(m, 0)
	for k = 0, m - 1 do
		local acc = 0
		for j = 0, m - 1 do
			local w = if j == 0 or j == m - 1 then 0.5 else 1
			acc += w * vals[j + 1] * buffer.readf64(T, 8 * (k * m + j))
		end
		c[k + 1] = acc * 2 / (m - 1) * (if k == 0 or k == m - 1 then 0.5 else 1)
	end
	local limit = tol * math.max(vscale, EPS)
	local last = 0
	for k = 1, m do
		if math.abs(c[k]) > limit then
			last = k
		end
	end
	if last > m - 3 then
		return nil
	end
	last = math.max(last, 1)
	local out = buffer.create(8 * last)
	for k = 1, last do
		buffer.writef64(out, 8 * (k - 1), c[k])
	end
	return out
end
type FastPiece = { V: Chebfun, S: Chebfun, inv: Chebfun, len: number }
local function fast_piece(q: (number) -> number, a: number, b: number, tol: number): FastPiece?
	for _, m in (if tol < 1e-10 then { 33 } else { 17, 25, 33 }) do
		local vals = table.create(m, 0)
		local qmin, qmax = math.huge, 0
		for j = 0, m - 1 do
			local t = a + (b - a) * (1 - math.cos(math.pi * j / (m - 1))) / 2
			local qt = q(t)
			qmin = math.min(qmin, qt)
			qmax = math.max(qmax, qt)
			vals[j + 1] = math.sqrt(qt)
		end
		if not (qmax > 0) or qmin <= 1e-4 * qmax then
			return nil -- a cusp, or nearly: leave it to the adaptive construction
		end
		local vc = fixed_coeffs(vals, m, tol)
		if vc then
			local V: Chebfun = { a = a, b = b, coeffs = vc, happy = true }
			local S = cheb.cumsum(V)
			local len = cheb.eval(S, b)
			local tv = table.create(m, 0)
			-- each node from the last, by a first-order step; Newton converges quadratically,
			-- so stopping at a step of tol / 100 leaves an error far below tol
			local t, sprev = a, 0
			for j = 0, m - 1 do
				local sj = len * (1 - math.cos(math.pi * j / (m - 1))) / 2
				local guess = t + (sj - sprev) / math.max(cheb.eval(V, t), EPS)
				t = solve_length(S, V, a, b, sj, guess, math.max(tol / 100, 1e-14))
				tv[j + 1] = t
				sprev = sj
			end
			-- t - a, so that the tolerance is relative to the piece's width, not to how far
			-- its parameters are from 0
			for j = 1, m do
				tv[j] -= a
			end
			local ic = fixed_coeffs(tv, m, tol)
			if ic then
				buffer.writef64(ic, 0, buffer.readf64(ic, 0) + a)
				return { V = V, S = S, inv = { a = 0, b = len, coeffs = ic, happy = true }, len = len }
			end
		end
	end
	return nil
end

local function build_arc(C: Curve<any>, tol: number): Arc
	local br, d = C.breaks, C.degree
	local pieces: { Piece } = {}
	local segs0: { number }, segpieces: { { number } } = {}, {}
	local total = 0
	for seg = 1, #br - 1 do
		local a, b = br[seg], br[seg + 1]
		local h = b - a
		segs0[seg] = total
		local mine: { number } = {}
		segpieces[seg] = mine
		-- Squared speed: a polynomial of degree 2d - 2 for polynomial segments, so exactly
		-- a Chebfun on 2d - 1 points; adaptive for rational ones.
		local function q(t: number): number
			local c = coeff_table(C, seg, (t - a) / h, h, 1)[2]
			return math.max(dotc(C, seg, c, c), 0)
		end
		if not C.weights and h > 0 then
			local fast = fast_piece(q, a, b, tol)
			if fast then
				if fast.len > 0 then
					table.insert(pieces, { a = a, b = b, s0 = total, len = fast.len, S = fast.S, V = fast.V, inv = fast.inv })
					table.insert(mine, #pieces)
					total += fast.len
				end
				continue
			end
		end
		local Q = if C.weights
			then cheb.fun(q, { a, b }, { maxlength = 1025 })
			else cheb.fun(q, { a, b }, { n = math.max(2 * d - 1, 2) })
		local qmax = cheb.max(Q)
		if qmax > 0 then
			-- Split where the speed (nearly) vanishes: there |C'| has a kink, but on either
			-- side it is smooth.
			local cuts = { a }
			local dQ = cheb.diff(Q)
			if f64.amax(dQ.coeffs) > 0 then
				local r = cheb.roots(dQ)
				for k = 1, f64.length(r) do
					local x = f64.get(r, k)
					if x > cuts[#cuts] and x < b and cheb.eval(Q, x) <= 1e-6 * qmax then
						table.insert(cuts, x)
					end
				end
			end
			table.insert(cuts, b)
			for k = 1, #cuts - 1 do
				local p0, p1 = cuts[k], cuts[k + 1]
				local V = cheb.fun(function(t: number): number
					return math.sqrt(q(t))
				end, { p0, p1 }, { tol = tol, maxlength = 4097 })
				local S = cheb.cumsum(V)
				local len = cheb.eval(S, p1)
				if len > 0 then
					local inv = inverse_fun(S, V, p0, p1, len, tol)
					table.insert(pieces, { a = p0, b = p1, s0 = total, len = len, S = S, V = V, inv = inv })
					table.insert(mine, #pieces)
					total += len
				end
			end
		end
	end
	return { pieces = pieces, total = total, segs0 = segs0, segpieces = segpieces, tol = tol }
end

local function arc(C: Curve<any>): Arc
	local A = aux(C)
	local t = A.arc
	if t then
		return t
	end
	local built = build_arc(C, EPS)
	A.arc = built
	return built
end

-- Builds the arclength tables now with tolerance options.tol (default machine precision;
-- larger is faster to build) and returns the length. Later arclength calls use them.
local function arclength<T>(C: Curve<T>, options: { tol: number? }?): number
	local tol = if options and options.tol then options.tol else EPS
	local t = build_arc(C, tol)
	aux(C).arc = t
	return t.total
end

-- The length of the curve over its domain.
local function length<T>(C: Curve<T>): number
	return arc(C).total
end

-- The arclength s(t) from the start of the domain to t (t clamped to the domain).
local function length_at<T>(C: Curve<T>, t: number): number
	local A = arc(C)
	local br = C.breaks
	t = math.clamp(t, br[1], br[#br])
	local seg = locate(C, t)
	local s = A.segs0[seg]
	for _, k in A.segpieces[seg] do
		local p = A.pieces[k]
		if t >= p.b then
			s = p.s0 + p.len
		else
			if t > p.a then
				s = p.s0 + cheb.eval(p.S, t)
			end
			break
		end
	end
	return s
end

-- The parameter t with s(t) = s (s clamped to [0, length]).
local function param_at<T>(C: Curve<T>, s: number): number
	local A = arc(C)
	local P = A.pieces
	if #P == 0 then
		return C.breaks[1]
	end
	if s <= 0 then
		return P[1].a
	elseif s >= A.total then
		return P[#P].b
	end
	local lo, hi = 1, #P
	while hi > lo do
		local mid = (lo + hi + 1) // 2
		if P[mid].s0 <= s then
			lo = mid
		else
			hi = mid - 1
		end
	end
	local p = P[lo]
	local r = s - p.s0
	if r <= 0 then
		return p.a
	elseif r >= p.len then
		return p.b
	end
	local t = cheb.eval(p.inv, r)
	if not p.inv.happy then
		-- Near a cusp t(s) behaves like a square root and is not resolved; polish.
		t = solve_length(p.S, p.V, p.a, p.b, r, t)
	end
	return math.clamp(t, p.a, p.b)
end

-- The point at arclength s from the start (s clamped to [0, length]).
local function at_length<T>(C: Curve<T>, s: number, out: T?): T
	return eval(C, param_at(C, s), out)
end

-- n >= 2 points equally spaced in arclength, from the start to the end of the curve, and
-- their parameters.
local function sample<T>(C: Curve<T>, n: number): ({ T }, { number })
	assert(n >= 2, "sample needs at least 2 points")
	local L = length(C)
	local pts, ts = list(n), numbers(n)
	for i = 1, n do
		local t = param_at(C, L * (i - 1) / (n - 1))
		ts[i] = t
		pts[i] = eval(C, t)
	end
	return pts, ts
end

-- Frames

-- Rotation-minimizing frames at the parameters ts (ascending), by the double reflection
-- method of Wang, Jüttler, Zheng and Liu (2008): returns the unit tangents, the reference
-- vectors (normals) and, for vector curves, the binormals tangent x normal. The first
-- normal is r0 made perpendicular to the first tangent; by default some perpendicular
-- vector (vector curves) or the Frenet normal. Accuracy improves with denser ts (the
-- error is fourth order in the spacing).
local function frames<T>(C: Curve<T>, ts: { number }, r0: T?): { tangents: { T }, normals: { T }, binormals: { vector }? }
	local S = C.space
	local n = #ts
	assert(n >= 1, "frames needs at least one parameter")
	local xs, tg = list(n), list(n)
	for i, t in ts do
		xs[i] = eval(C, t)
		tg[i] = tangent(C, t)
	end
	local r: T
	if r0 ~= nil then
		r = S.axpy(S.similar(r0), -S.dot(r0, tg[1]), tg[1], r0)
	elseif C.vector then
		local t0 = tg[1] :: any
		local ax, ay, az = math.abs(t0.x), math.abs(t0.y), math.abs(t0.z)
		local e = if ax <= ay and ax <= az then vector.create(1, 0, 0) elseif ay <= az then vector.create(0, 1, 0) else vector.create(0, 0, 1)
		r = (e - t0 * vector.dot(e, t0)) :: any
	else
		local _, N = frenet(C, ts[1])
		assert(N ~= nil, "frames: the curve is straight at the start; pass r0")
		r = N
	end
	r = S.scale(r, 1 / S.norm(r), r)
	local rs = list(n)
	rs[1] = r
	for i = 1, n - 1 do
		local v1 = S.sub(S.similar(xs[i]), xs[i + 1], xs[i])
		local c1 = S.dot(v1, v1)
		local rL, tL = rs[i], tg[i]
		if c1 > 0 then
			rL = S.axpy(S.similar(rL), -2 / c1 * S.dot(v1, rs[i]), v1, rs[i])
			tL = S.axpy(S.similar(tL), -2 / c1 * S.dot(v1, tg[i]), v1, tg[i])
		end
		local v2 = S.sub(S.similar(tL), tg[i + 1], tL)
		local c2 = S.dot(v2, v2)
		local rn = if c2 > 0
			then S.axpy(S.similar(rL), -2 / c2 * S.dot(v2, rL), v2, rL)
			else S.copy(S.similar(rL), rL)
		-- Reflections preserve length; renormalize against rounding drift.
		rs[i + 1] = S.scale(rn, 1 / S.norm(rn), rn)
	end
	local binormals: { vector }? = nil
	if C.vector then
		local b = list(n)
		for i = 1, n do
			b[i] = vector.cross(tg[i] :: any, rs[i] :: any)
		end
		binormals = b
	end
	return { tangents = tg, normals = rs, binormals = binormals }
end

-- Closest point and bounds

-- Center and radius of a sphere around each segment's control points (which contain the
-- segment: both polynomial and positively weighted rational Bézier curves lie in the
-- convex hull of their control points).
local function spheres(C: Curve<any>): { { any } }
	local A = aux(C)
	if A.spheres then
		return A.spheres
	end
	local S, d = C.space, C.degree
	local out = {}
	local avg = numbers(d + 1, 1 / (d + 1))
	for seg = 1, #C.breaks - 1 do
		local center = combine(C, seg, avg, false)
		local base = (seg - 1) * (d + 1)
		local radius = 0
		for j = 1, d + 1 do
			radius = math.max(radius, S.norm(S.sub(S.similar(center), C.points[base + j], center)))
		end
		out[seg] = { center, radius * (1 + 1e-6) }
	end
	A.spheres = out
	return out
end

-- The point of the curve (over its domain) closest to q: returns the parameter, the point
-- and the distance. Each segment's candidates are its ends and the roots of
-- (C(t) - q) . C'(t), a polynomial of degree 2d - 1 (a Chebfun on 2d points; adaptive for
-- rational segments); segments whose bounding sphere is farther than the best distance so
-- far are skipped.
local function closest<T>(C: Curve<T>, q: T): (number, T, number)
	local rational = C.weights ~= nil
	local S, d = C.space, C.degree
	local br = C.breaks
	local nseg = #br - 1
	local sph = spheres(C)
	local order = list(nseg)
	local bound = numbers(nseg)
	for seg = 1, nseg do
		order[seg] = seg
		local center, radius = sph[seg][1], sph[seg][2]
		bound[seg] = math.max(S.norm(S.sub(S.similar(center), q, center)) - radius, 0)
	end
	table.sort(order, function(i: number, j: number): boolean
		return bound[i] < bound[j]
	end)
	local qv = q :: any
	local best, bestt = math.huge, br[1]
	local function dist(seg: number, t: number): number
		local c = coeff_table(C, seg, (t - br[seg]) / (br[seg + 1] - br[seg]), br[seg + 1] - br[seg], 0)[1]
		if C.vector then
			local x, y, z = comb3(C, seg, c, false)
			x, y, z = x - qv.x, y - qv.y, z - qv.z
			return math.sqrt(x * x + y * y + z * z)
		end
		local p = combine(C, seg, c, false)
		return S.norm(S.sub(p, p, q))
	end
	for _, seg in order do
		if bound[seg] >= best then
			break
		end
		local a, b = br[seg], br[seg + 1]
		local h = b - a
		local base = (seg - 1) * (d + 1)
		local g: (number) -> number
		if C.vector then
			g = function(t: number): number
				local c = coeff_table(C, seg, (t - a) / h, h, 1)
				local px, py, pz = comb3(C, seg, c[1], false)
				local vx, vy, vz = comb3(C, seg, c[2], true)
				return (px - qv.x) * vx + (py - qv.y) * vy + (pz - qv.z) * vz
			end
		else
			-- M[j][k] = (P_j - q) . (P_k - P_1), so g = sum c0_j c1_k M[j][k].
			local P0 = C.points[base + 1]
			local E, D = list(d + 1), list(d)
			for j = 1, d + 1 do
				E[j] = S.sub(S.similar(q), C.points[base + j], q)
			end
			for k = 1, d do
				D[k] = S.sub(S.similar(P0), C.points[base + 1 + k], P0)
			end
			local M = table.create((d + 1) * d, 0)
			for j = 1, d + 1 do
				for k = 1, d do
					M[(j - 1) * d + k] = S.dot(E[j], D[k])
				end
			end
			g = function(t: number): number
				local c = coeff_table(C, seg, (t - a) / h, h, 1)
				local c0, c1 = c[1], c[2]
				local s = 0
				for j = 1, d + 1 do
					local row = 0
					for k = 1, d do
						row += M[(j - 1) * d + k] * c1[k + 1]
					end
					s += c0[j] * row
				end
				return s
			end
		end
		-- For rational segments g D^3 is a polynomial of degree 3d - 1 with the same roots
		-- (D > 0), so the Chebfun is exact either way and never grows on rounding noise
		-- (when g vanishes identically, as on a circle seen from its axis).
		local gp = g
		if rational then
			gp = function(t: number): number
				return g(t) * denominator(C, seg, (t - a) / h) ^ 3
			end
		end
		local candidates = { a, b }
		local G = cheb.fun(gp, { a, b }, { n = if rational then 3 * d else 2 * d })
		if f64.amax(G.coeffs) > 0 then
			local r = cheb.roots(G)
			for k = 1, f64.length(r) do
				table.insert(candidates, f64.get(r, k))
			end
		end
		for _, t in candidates do
			local e = dist(seg, t)
			if e < best then
				best, bestt = e, t
			end
		end
	end
	return bestt, eval(C, bestt), best
end

-- What Newton's method for the distance to q needs at t: g = (C - q) . C', |C'|^2,
-- (C - q) . C'' and |C - q|^2.
local function distance_terms<T>(C: Curve<T>, q: T, t: number): (number, number, number, number)
	local seg, u, h = locate(C, t)
	local c = coeff_table(C, seg, u, h, 2)
	if C.vector then
		local qv = q :: any
		local px, py, pz = comb3(C, seg, c[1], false)
		px, py, pz = px - qv.x, py - qv.y, pz - qv.z
		local ax, ay, az = comb3(C, seg, c[2], true)
		local bx, by, bz = comb3(C, seg, c[3], true)
		return px * ax + py * ay + pz * az, ax * ax + ay * ay + az * az, px * bx + py * by + pz * bz, px * px + py * py + pz * pz
	end
	local S = C.space
	local p = combine(C, seg, c[1], false)
	p = S.sub(p, p, q)
	return S.dot(p, combine(C, seg, c[2], true)), dotc(C, seg, c[2], c[2]), S.dot(p, combine(C, seg, c[3], true)), S.dot(p, p)
end

-- The locally closest point to q, by Newton's method on g(t) = (C(t) - q) . C'(t) = 0
-- starting from t0: returns the parameter, the point and the distance. It costs a few
-- evaluations, so it suits tracking something that moves a little between calls (pass the
-- previous answer as t0). It finds a local minimum of the distance, which need not be the
-- global one; closest finds that. Safeguards: where the squared distance is not locally
-- convex a Gauss-Newton step (dropping the curvature term) replaces Newton's; steps are
-- halved until the distance does not increase; t is clamped to the domain, or wraps
-- around it on a closed curve.
local function closest_near<T>(C: Curve<T>, q: T, t0: number): (number, T, number)
	local br = C.breaks
	local a, b = br[1], br[#br]
	local span = b - a
	local function wrap(t: number): number
		if C.closed then
			return a + (t - a) % span
		end
		return math.clamp(t, a, b)
	end
	local t = wrap(t0)
	local g, v2, pc, d2 = distance_terms(C, q, t)
	for _ = 1, 50 do
		if not (v2 > 0) then
			break -- a cusp: no direction to move in
		end
		local curv = v2 + pc
		local newton = curv > 0.1 * v2
		local step = if newton then -g / curv else -g / v2
		if newton and math.abs(step) <= 1e-8 * span then
			-- Converged: this last step leaves an error of order its square. (Testing it
			-- for descent would only compare rounding errors.)
			t = wrap(t + step)
			_, _, _, d2 = distance_terms(C, q, t)
			break
		end
		local accepted = false
		local tn, gn, v2n, pcn, d2n = t, g, v2, pc, d2
		for attempt = 1, 40 do
			tn = wrap(t + step)
			gn, v2n, pcn, d2n = distance_terms(C, q, tn)
			if d2n <= d2 then
				accepted = true
				newton = newton and attempt == 1
				break
			end
			step /= 2
		end
		if not accepted then
			break
		end
		local moved = math.min(math.abs(tn - t), math.abs(step))
		t, g, v2, pc, d2 = tn, gn, v2n, pcn, d2n
		if moved <= 1e-12 * span then
			break
		end
	end
	return t, eval(C, t), math.sqrt(d2)
end

local roundbuf = buffer.create(4)

-- x rounded to single precision, toward -inf (dir < 0) or +inf (dir > 0).
local function round_f32(x: number, dir: number): number
	buffer.writef32(roundbuf, 0, x)
	local r = buffer.readf32(roundbuf, 0)
	if (dir < 0 and r > x) or (dir > 0 and r < x) then
		-- Step one unit in the last place toward x's side.
		local bits = buffer.readu32(roundbuf, 0)
		local positive = r > 0 or (r == 0 and dir > 0)
		if r == 0 then
			buffer.writeu32(roundbuf, 0, if dir > 0 then 1 else 0x80000001)
		elseif (dir > 0) == positive then
			buffer.writeu32(roundbuf, 0, bits + 1)
		else
			buffer.writeu32(roundbuf, 0, bits - 1)
		end
		r = buffer.readf32(roundbuf, 0)
	end
	return r
end

-- The tight axis-aligned bounding box of a vector curve over its domain: (min, max), rounded
-- outward to single precision. Each coordinate's extremes are at the segment ends or at
-- roots of its derivative x'(t), found from x'(t) D(t)^2, a polynomial of degree 2d - 2
-- (d - 1 for polynomial segments).
local function bounds(C: Curve<vector>): (vector, vector)
	assert(C.vector, "bounds needs vector points")
	local d = C.degree
	local br = C.breaks
	local lo, hi = { math.huge, math.huge, math.huge }, { -math.huge, -math.huge, -math.huge }
	for seg = 1, #br - 1 do
		local a, b = br[seg], br[seg + 1]
		local h = b - a
		for axis = 1, 3 do
			local function f(t: number): number
				local x, y, z = comb3(C, seg, coeff_table(C, seg, (t - a) / h, h, 0)[1], false)
				return if axis == 1 then x elseif axis == 2 then y else z
			end
			local function df(t: number): number
				local u = (t - a) / h
				local x, y, z = comb3(C, seg, coeff_table(C, seg, u, h, 1)[2], true)
				local v = if axis == 1 then x elseif axis == 2 then y else z
				return if C.weights then v * denominator(C, seg, u) ^ 2 else v
			end
			local candidates = { a, b }
			if d >= 2 then
				local F = cheb.fun(df, { a, b }, { n = if C.weights then 2 * d - 1 else d })
				if f64.amax(F.coeffs) > 0 then
					local r = cheb.roots(F)
					for k = 1, f64.length(r) do
						table.insert(candidates, f64.get(r, k))
					end
				end
			end
			for _, t in candidates do
				local v = f(t)
				lo[axis] = math.min(lo[axis], v)
				hi[axis] = math.max(hi[axis], v)
			end
		end
	end
	return vector.create(round_f32(lo[1], -1), round_f32(lo[2], -1), round_f32(lo[3], -1)),
		vector.create(round_f32(hi[1], 1), round_f32(hi[2], 1), round_f32(hi[3], 1))
end

-- Splitting

-- Splits C at t (strictly inside the domain) into the curves over [start, t] and [t, end],
-- by de Casteljau subdivision of the segment containing t (in homogeneous coordinates for
-- rational curves).
local function split<T>(C: Curve<T>, t: number): (Curve<T>, Curve<T>)
	local br = C.breaks
	local n = #br - 1
	assert(t > br[1] and t < br[n + 1], "split: t must be inside the domain")
	local seg, u = locate(C, t)
	local d = C.degree
	local m1 = d + 1
	local W = C.weights
	local pts = C.points
	local lp: { T }, rp: { T } = {}, {}
	local lw: { number }, rw: { number } = {}, {}
	local lbr: { number }, rbr: { number } = {}, {}
	for k = 1, seg do
		table.insert(lbr, br[k])
	end
	for k = 1, (seg - 1) * m1 do
		table.insert(lp, pts[k])
		if W then
			table.insert(lw, W[k])
		end
	end
	local first = seg -- first segment copied unchanged to the right part
	if u > 0 then
		table.insert(lbr, t)
		table.insert(rbr, t)
		first = seg + 1
		local base = (seg - 1) * m1
		local bvals = numbers(m1)
		local function control(cs: { number }): (T, number)
			local w = 1
			if W then
				w = 0
				for j = 1, m1 do
					w += cs[j] * W[base + j]
				end
				for j = 1, m1 do
					cs[j] = cs[j] * W[base + j] / w
				end
			end
			return combine(C, seg, cs, false), w
		end
		for m = 0, d do
			-- Left control point m: the Bernstein polynomials of degree m on points 0 .. m.
			bernstein(m, u, bvals)
			local cs = numbers(m1)
			for j = 1, m + 1 do
				cs[j] = bvals[j]
			end
			local p, w = control(cs)
			table.insert(lp, p)
			table.insert(lw, w)
		end
		for m = 0, d do
			-- Right control point m: degree d - m on points m .. d.
			bernstein(d - m, u, bvals)
			local cs = numbers(m1)
			for j = 1, d - m + 1 do
				cs[m + j] = bvals[j]
			end
			local p, w = control(cs)
			table.insert(rp, p)
			table.insert(rw, w)
		end
	end
	for k = first, n + 1 do
		table.insert(rbr, br[k])
	end
	for k = (first - 1) * m1 + 1, n * m1 do
		table.insert(rp, pts[k])
		if W then
			table.insert(rw, W[k])
		end
	end
	local L = make(C.space, C.vector, d, lbr, lp, if W then lw else nil)
	local R = make(C.space, C.vector, d, rbr, rp, if W then rw else nil)
	return L, R
end

-- Constructors

-- Default breaks 0, 1, ..., count.
local function integers(count: number): { number }
	local b = numbers(count + 1)
	for i = 0, count do
		b[i + 1] = i
	end
	return b
end

local function copy_points(W: Space<any>, isvector: boolean, pts: { any }): { any }
	if isvector then
		return pts
	end
	local out = list(#pts)
	for i, p in pts do
		out[i] = W.copy(W.similar(p), p)
	end
	return out
end

-- A Bézier curve of degree #points - 1 on options.breaks (default { 0, 1 }), rational if
-- options.weights (positive, one per point) is given. Reads options.space and .breaks.
local function bezier<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
	local o: CurveOptions<T> = options or {}
	local S, isvector = resolve(points, o.space)
	assert(#points >= 2, "a Bézier curve needs at least 2 points")
	local breaks = o.breaks or { 0, 1 }
	assert(#breaks == 2, "a single Bézier curve has breaks { a, b }")
	return make(S, isvector, #points - 1, table.clone(breaks), copy_points(S :: any, isvector, points), if o.weights then table.clone(o.weights) else nil)
end

-- A composite Bézier curve of the given degree (options.degree, default 3): segment i uses
-- points (i - 1) * degree + 1 .. i * degree + 1, so neighbors share their end points.
-- Reads options.space, .degree, .breaks (default 0 .. number of segments), .weights (one
-- per point, making it rational) and .closed (marks the curve closed; the last point
-- should equal the first).
local function bezier_path<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
	local o: CurveOptions<T> = options or {}
	local S, isvector = resolve(points, o.space)
	local d = o.degree or 3
	assert(d >= 1 and (#points - 1) % d == 0 and #points > 1, "bezier_path needs k * degree + 1 points")
	local nseg = (#points - 1) // d
	local pts, ws = {}, if o.weights then {} else nil
	local copied = copy_points(S :: any, isvector, points)
	for seg = 1, nseg do
		for j = 0, d do
			local k = (seg - 1) * d + j + 1
			table.insert(pts, copied[k])
			if ws and o.weights then
				table.insert(ws, o.weights[k])
			end
		end
	end
	return make(S, isvector, d, if o.breaks then table.clone(o.breaks) else integers(nseg), pts, ws, o.closed)
end

-- Cubic Bézier control points of Hermite segments in the work space: P_i, P_i + h/3 m_i^out,
-- P_(i+1) - h/3 m_(i+1)^in, P_(i+1).
local function hermite_points(W: Space<any>, P: { any }, nxt: (number) -> number, breaks: { number }, mout: { any }, min: { any }): { any }
	local out = {}
	for i = 1, #breaks - 1 do
		local j = nxt(i)
		local h = breaks[i + 1] - breaks[i]
		table.insert(out, W.copy(W.similar(P[i]), P[i]))
		table.insert(out, W.axpy(W.similar(P[i]), h / 3, mout[i], P[i]))
		table.insert(out, W.axpy(W.similar(P[j]), -h / 3, min[j], P[j]))
		table.insert(out, W.copy(W.similar(P[j]), P[j]))
	end
	return out
end

-- The cubic Hermite curve through points[i] with derivatives tangents[i] (with respect to
-- t) at breaks[i]. Reads options.space and .breaks (default 0 .. #points - 1).
local function hermite<T>(points: { T }, tangents: { T }, options: CurveOptions<T>?): Curve<T>
	local o: CurveOptions<T> = options or {}
	local W, P, S, isvector = work_points(points, o.space)
	local n = #P
	assert(n >= 2 and #tangents == n, "hermite needs at least 2 points and one tangent per point")
	local breaks = if o.breaks then table.clone(o.breaks) else integers(n - 1)
	assert(#breaks == n, "hermite: one break per point")
	local m = list(n)
	for i, v in tangents do
		m[i] = to_work(isvector, v)
	end
	local pts = from_work(isvector, hermite_points(W, P, function(i)
		return i + 1
	end, breaks, m, m))
	return make(S, isvector, 3, breaks, pts)
end

-- Drops consecutive repeated points (zero chord length), including a last point equal to
-- the first on a closed curve.
local function distinct(W: Space<any>, P: { any }, closed: boolean): { any }
	local out = { P[1] }
	for i = 2, #P do
		if W.norm(W.sub(W.similar(P[i]), P[i], out[#out])) > 0 then
			table.insert(out, P[i])
		end
	end
	if closed and #out > 1 and W.norm(W.sub(W.similar(out[1]), out[#out], out[1])) == 0 then
		table.remove(out)
	end
	return out
end

-- Knots t_1 = 0, t_(i+1) = t_i + |P_(i+1) - P_i|^alpha (alpha = 0 uniform, 0.5 centripetal,
-- 1 chordal); for a closed curve one more, back to the first point.
local function chord_knots(W: Space<any>, P: { any }, alpha: number, closed: boolean): { number }
	local n = #P
	local t = { 0 }
	local count = if closed then n else n - 1
	for i = 1, count do
		local j = i % n + 1
		local dist = W.norm(W.sub(W.similar(P[i]), P[j], P[i]))
		table.insert(t, t[i] + dist ^ alpha)
	end
	return t
end

-- The Catmull-Rom spline through the points: a C^1 cubic with knots spaced by chord
-- length to the power alpha (options.alpha: 0 uniform, 0.5 centripetal (the default,
-- which never forms cusps or self-intersections within a segment), 1 chordal). The
-- tangent at P_i is that of the Barry-Goldman construction:
--   (P_i - P_(i-1)) / h0 - (P_(i+1) - P_(i-1)) / (h0 + h1) + (P_(i+1) - P_i) / h1.
-- Open curves use the chord direction at the ends. Consecutive repeated points are
-- dropped. The parameter runs over the knots. Reads options.space, .alpha and .closed.
local function catmull_rom<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
	local o: CurveOptions<T> = options or {}
	local W, P0, S, isvector = work_points(points, o.space)
	local closed = o.closed == true
	local P = distinct(W, P0, closed)
	local n = #P
	assert(n >= 2 and (not closed or n >= 3), "catmull_rom needs at least 2 distinct points (3 if closed)")
	local t = chord_knots(W, P, o.alpha or 0.5, closed)
	local period = t[#t]
	local function knot(i: number): number
		if closed then
			if i < 1 then
				return t[n + i] - period
			elseif i > n then
				return t[i - n] + period
			end
		end
		return t[i]
	end
	local function at(i: number): any
		return P[(i - 1) % n + 1]
	end
	local m = list(n)
	for i = 1, n do
		if not closed and i == 1 then
			local h1 = knot(2) - knot(1)
			m[i] = lincomb(W, { P[1], P[2] }, { -1 / h1, 1 / h1 })
		elseif not closed and i == n then
			local h0 = knot(n) - knot(n - 1)
			m[i] = lincomb(W, { P[n - 1], P[n] }, { -1 / h0, 1 / h0 })
		else
			local h0, h1 = knot(i) - knot(i - 1), knot(i + 1) - knot(i)
			m[i] = lincomb(W, { at(i - 1), P[i], at(i + 1) }, {
				-1 / h0 + 1 / (h0 + h1),
				1 / h0 - 1 / h1,
				-1 / (h0 + h1) + 1 / h1,
			})
		end
	end
	local pts = from_work(isvector, hermite_points(W, P, function(i)
		return i % n + 1
	end, t, m, m))
	return make(S, isvector, 3, t, pts, nil, closed)
end

-- A per-point parameter: a number for every point, or a table with one per point.
local function param(v: (number | { number })?, i: number): number
	if v == nil then
		return 0
	elseif type(v) == "number" then
		return v
	end
	return (v :: { number })[i]
end

-- The Kochanek-Bartels (TCB) spline through the points, with uniform parameter spacing:
-- options.tension, .continuity and .bias (numbers, or tables with one per point; all 0
-- gives the uniform Catmull-Rom spline). The tangent leaving P_i is
--   (1-T)(1+B)(1-C)/2 (P_i - P_(i-1)) + (1-T)(1-B)(1+C)/2 (P_(i+1) - P_i)
-- and the one arriving swaps (1-C) and (1+C). Open ends reflect the neighboring chord.
-- Reads options.space and .closed as well.
local function kochanek_bartels<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
	local o: CurveOptions<T> = options or {}
	local W, P, S, isvector = work_points(points, o.space)
	local closed = o.closed == true
	local n = #P
	assert(n >= 2 and (not closed or n >= 3), "kochanek_bartels needs at least 2 points (3 if closed)")
	local mout, min = list(n), list(n)
	for i = 1, n do
		local T, C, B = param(o.tension, i), param(o.continuity, i), param(o.bias, i)
		-- Incoming and outgoing chords as combinations of (P_(i-1), P_i, P_(i+1)).
		local prev = if closed then (i - 2) % n + 1 elseif i > 1 then i - 1 else nil
		local nxt = if closed then i % n + 1 elseif i < n then i + 1 else nil
		local idx: { number }, din: { number }, dout: { number }
		if prev and nxt then
			idx, din, dout = { prev, i, nxt }, { -1, 1, 0 }, { 0, -1, 1 }
		elseif nxt then
			idx, din, dout = { i, nxt }, { -1, 1 }, { -1, 1 }
		else
			idx, din, dout = { prev :: number, i }, { -1, 1 }, { -1, 1 }
		end
		local function tangent_coeffs(a: number, b: number): { number }
			local cs = {}
			for k = 1, #idx do
				cs[k] = a * din[k] + b * dout[k]
			end
			return cs
		end
		local ps = {}
		for k, j in idx do
			ps[k] = P[j]
		end
		local f = (1 - T) / 2
		mout[i] = lincomb(W, ps, tangent_coeffs(f * (1 + B) * (1 - C), f * (1 - B) * (1 + C)))
		min[i] = lincomb(W, ps, tangent_coeffs(f * (1 + B) * (1 + C), f * (1 - B) * (1 - C)))
	end
	local breaks = integers(if closed then n else n - 1)
	local pts = from_work(isvector, hermite_points(W, P, function(i)
		return i % n + 1
	end, breaks, mout, min))
	return make(S, isvector, 3, breaks, pts, nil, closed)
end

-- Solves the tridiagonal system sub_i x_(i-1) + dia_i x_i + sup_i x_(i+1) = rhs_i with
-- scalar coefficients and points as unknowns (Thomas algorithm; the spline systems are
-- diagonally dominant or, for not-a-knot ends, safe without pivoting).
local function tridiagonal_points(W: Space<any>, sub: { number }, dia: { number }, sup: { number }, rhs: { any }): { any }
	local n = #dia
	local d, r = table.clone(dia), list(n)
	r[1] = rhs[1]
	for i = 2, n do
		local w = sub[i] / d[i - 1]
		d[i] -= w * sup[i - 1]
		r[i] = W.axpy(W.similar(rhs[i]), -w, r[i - 1], rhs[i])
	end
	local x = list(n)
	x[n] = W.scale(W.similar(r[n]), 1 / d[n], r[n])
	for i = n - 1, 1, -1 do
		local v = W.axpy(W.similar(r[i]), -sup[i], x[i + 1], r[i])
		x[i] = W.scale(v, 1 / d[i], v)
	end
	return x
end

-- The C^2 cubic spline through the points. Parameters: options.breaks, or knots spaced by
-- chord length to the power options.alpha (default 0.5, centripetal; consecutive repeated
-- points are then dropped). End conditions, options.ends: "natural" (zero second
-- derivative; the default), "clamped" (first derivatives options.start and .finish) or
-- "not-a-knot" (the third derivative is continuous at the second and second-to-last
-- points; with 3 points that is the parabola through them). options.closed gives the
-- periodic spline, C^2 everywhere including where it closes. Reads options.space as well.
local function cubic_spline<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
	local o: CurveOptions<T> = options or {}
	local W, P0, S, isvector = work_points(points, o.space)
	local closed = o.closed == true
	local P = if o.breaks then P0 else distinct(W, P0, closed)
	local n = #P
	assert(n >= 2 and (not closed or n >= 3), "cubic_spline needs at least 2 points (3 if closed)")
	local x = if o.breaks then table.clone(o.breaks) else chord_knots(W, P, o.alpha or 0.5, closed)
	assert(#x == (if closed then n + 1 else n), "cubic_spline: one break per point (one more if closed)")
	local nseg = #x - 1
	local h, delta = numbers(nseg), list(nseg)
	for i = 1, nseg do
		h[i] = x[i + 1] - x[i]
		assert(h[i] > 0, "cubic_spline: breaks must increase")
		local j = i % n + 1
		local dv = W.sub(W.similar(P[i]), P[j], P[i])
		delta[i] = W.scale(dv, 1 / h[i], dv)
	end
	-- Interior row for the derivative m_i at point i (between segments i - 1 and i):
	-- h_i m_(i-1) + 2 (h_(i-1) + h_i) m_i + h_(i-1) m_(i+1) = 3 (h_i d_(i-1) + h_(i-1) d_i).
	local function interior_rhs(hp: number, hn: number, dp: any, dn: any): any
		local v = W.scale(W.similar(dp), 3 * hn, dp)
		return W.axpy(v, 3 * hp, dn, v)
	end
	local m: { any }
	if closed then
		-- Cyclic tridiagonal, by bordering: row i reads x[i - 1], x[i], x[i + 1] cyclically.
		-- The first n - 1 unknowns are x = p + x_n q, where p solves the leading n - 1 rows
		-- with x_n = 0 and q gives the response to x_n (it enters row 1 through sub[1] and
		-- row n - 1 through sup[n - 1]); the last row then fixes x_n.
		local sub, dia, sup, rhs = numbers(n), numbers(n), numbers(n), list(n)
		for i = 1, n do
			local ip = (i - 2) % n + 1 -- previous segment index
			local hp, hn = h[ip], h[i]
			sub[i], dia[i], sup[i] = hn, 2 * (hp + hn), hp
			rhs[i] = interior_rhs(hp, hn, delta[ip], delta[i])
		end
		local k = n - 1
		local ksub, kdia, ksup, krhs = numbers(k), numbers(k), numbers(k), list(k)
		for i = 1, k do
			ksub[i], kdia[i], ksup[i], krhs[i] = sub[i], dia[i], sup[i], rhs[i]
		end
		local p = tridiagonal_points(W, ksub, kdia, ksup, krhs)
		-- q solves the same rows for the right-hand side -(sub[1] e_1 + sup[k] e_k), in
		-- scalars.
		local q, dd = numbers(k), table.clone(kdia)
		q[1] = -sub[1]
		q[k] -= sup[k]
		for i = 2, k do
			local w = ksub[i] / dd[i - 1]
			dd[i] -= w * ksup[i - 1]
			q[i] -= w * q[i - 1]
		end
		q[k] /= dd[k]
		for i = k - 1, 1, -1 do
			q[i] = (q[i] - ksup[i] * q[i + 1]) / dd[i]
		end
		-- Row n: sub[n] x[k] + dia[n] x_n + sup[n] x[1] = rhs[n].
		local num = W.axpy(W.similar(rhs[n]), -sub[n], p[k], rhs[n])
		num = W.axpy(num, -sup[n], p[1], num)
		local xn = W.scale(num, 1 / (dia[n] + sub[n] * q[k] + sup[n] * q[1]), num)
		m = list(n)
		for i = 1, k do
			m[i] = W.axpy(W.similar(p[i]), q[i], xn, p[i])
		end
		m[n] = xn
	else
		local ends = o.ends or "natural"
		m = list(n)
		if n == 2 then
			if ends == "clamped" then
				m[1], m[2] = to_work(isvector, o.start), to_work(isvector, o.finish)
			else
				m[1], m[2] = delta[1], delta[1]
			end
		elseif n == 3 and ends == "not-a-knot" then
			-- The parabola through the three points.
			local h0, h1 = h[1], h[2]
			local dd = W.sub(W.similar(delta[2]), delta[2], delta[1])
			dd = W.scale(dd, 1 / (h0 + h1), dd)
			m[1] = W.axpy(W.similar(dd), -h0, dd, delta[1])
			m[2] = W.axpy(W.similar(dd), h0, dd, delta[1])
			m[3] = W.axpy(W.similar(dd), h0 + 2 * h1, dd, delta[1])
		else
			local sub, dia, sup, rhs = numbers(n), numbers(n), numbers(n), list(n)
			for i = 2, n - 1 do
				local hp, hn = h[i - 1], h[i]
				sub[i], dia[i], sup[i] = hn, 2 * (hp + hn), hp
				rhs[i] = interior_rhs(hp, hn, delta[i - 1], delta[i])
			end
			if ends == "natural" then
				dia[1], sup[1] = 2, 1
				rhs[1] = W.scale(W.similar(delta[1]), 3, delta[1])
				sub[n], dia[n] = 1, 2
				rhs[n] = W.scale(W.similar(delta[n - 1]), 3, delta[n - 1])
			elseif ends == "clamped" then
				assert(o.start ~= nil and o.finish ~= nil, "clamped ends need options.start and options.finish")
				dia[1], sup[1] = 1, 0
				rhs[1] = to_work(isvector, o.start)
				sub[n], dia[n] = 0, 1
				rhs[n] = to_work(isvector, o.finish)
			elseif ends == "not-a-knot" then
				local h0, h1 = h[1], h[2]
				dia[1], sup[1] = h1, h0 + h1
				local r1 = W.scale(W.similar(delta[1]), (h0 + 2 * (h0 + h1)) * h1 / (h0 + h1), delta[1])
				rhs[1] = W.axpy(r1, h0 * h0 / (h0 + h1), delta[2], r1)
				local ha, hb = h[nseg - 1], h[nseg] -- h_(n-2), h_(n-1) in 0-based terms
				sub[n], dia[n] = ha + hb, ha
				local rn = W.scale(W.similar(delta[nseg - 1]), hb * hb / (ha + hb), delta[nseg - 1])
				rhs[n] = W.axpy(rn, (2 * (ha + hb) + hb) * ha / (ha + hb), delta[nseg], rn)
			else
				error(`cubic_spline: unknown ends {ends}`)
			end
			m = tridiagonal_points(W, sub, dia, sup, rhs)
		end
	end
	local pts = from_work(isvector, hermite_points(W, P, function(i)
		return i % n + 1
	end, x, m, m))
	return make(S, isvector, 3, x, pts, nil, closed)
end

-- A B-spline (NURBS if options.weights is given, one positive weight per point) of degree
-- options.degree (default 3) with control points `points`. Knots: options.knots (#points
-- + degree + 1 nondecreasing values; the domain is [knots[degree + 1], knots[#points +
-- 1]]), or by default clamped and uniform, so the curve starts at the first point, ends at
-- the last, and has domain [0, #points - degree]. options.closed instead makes a periodic
-- curve (uniform knots, domain [0, #points]). The curve is converted to Bézier segments by
-- blossoming. Reads options.space as well.
local function bspline<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
	local o: CurveOptions<T> = options or {}
	local W, P, S, isvector = work_points(points, o.space)
	local p = o.degree or 3
	local closed = o.closed == true
	local weights = o.weights
	assert(p >= 1, "bspline: degree must be at least 1")
	assert(#P >= p + 1, "bspline needs at least degree + 1 points")
	assert(weights == nil or #weights == #P, "bspline: one weight per point")
	local ctrl, w = P, weights
	local knots: { number }
	if closed then
		-- Wrap the first p points around; uniform knots -p, ..., n + p.
		ctrl = table.clone(P)
		local ww = if weights then table.clone(weights) else nil
		for i = 1, p do
			table.insert(ctrl, P[i])
			if ww and weights then
				table.insert(ww, weights[i])
			end
		end
		w = ww
		knots = {}
		for i = 0, #ctrl + p do
			table.insert(knots, i - p)
		end
	elseif o.knots then
		knots = table.clone(o.knots)
		assert(#knots == #P + p + 1, "bspline: need #points + degree + 1 knots")
		for i = 2, #knots do
			assert(knots[i] >= knots[i - 1], "bspline: knots must not decrease")
		end
	else
		knots = {}
		local inner = #P - p
		for i = 0, #P + p do
			table.insert(knots, math.clamp(i - p, 0, inner))
		end
	end
	local N = #ctrl
	local function k(i: number): number -- 0-based knot index
		return knots[i + 1]
	end
	local pts, ws, breaks = {}, if w then {} else nil, {}
	for r = p, N - 1 do
		local a, b = k(r), k(r + 1)
		if b > a then
			if #breaks == 0 then
				table.insert(breaks, a)
			end
			table.insert(breaks, b)
			local local_pts = list(p + 1)
			for j = 0, p do
				local_pts[j + 1] = ctrl[r - p + j + 1]
			end
			for m = 0, p do
				-- Bézier point m is the blossom at (a^(p-m), b^m): de Boor's algorithm with
				-- these arguments, run on the coefficients of the p + 1 control points.
				local dcoef = list(p + 1)
				for j = 0, p do
					local e = numbers(p + 1)
					e[j + 1] = 1
					dcoef[j + 1] = e
				end
				for level = 1, p do
					local x = if level <= p - m then a else b
					for j = p, level, -1 do
						local i = r - p + j
						local alpha = (x - k(i)) / (k(i + p + 1 - level) - k(i))
						local left, right = dcoef[j], dcoef[j + 1]
						local out = numbers(p + 1)
						for q = 1, p + 1 do
							out[q] = (1 - alpha) * left[q] + alpha * right[q]
						end
						dcoef[j + 1] = out
					end
				end
				local cs = dcoef[p + 1]
				if w then
					local total = 0
					for j = 1, p + 1 do
						total += cs[j] * w[r - p + j]
					end
					for j = 1, p + 1 do
						cs[j] = cs[j] * w[r - p + j] / total
					end
					table.insert(ws :: { number }, total)
				end
				table.insert(pts, lincomb(W, local_pts, cs))
			end
		end
	end
	assert(#breaks >= 2, "bspline: the knots leave no nonempty span")
	return make(S, isvector, p, breaks, from_work(isvector, pts), ws, closed)
end

-- The domain [a, b] of C.
local function domain<T>(C: Curve<T>): (number, number)
	return C.breaks[1], C.breaks[#C.breaks]
end

return {
	bezier = bezier,
	bezier_path = bezier_path,
	hermite = hermite,
	catmull_rom = catmull_rom,
	kochanek_bartels = kochanek_bartels,
	cubic_spline = cubic_spline,
	bspline = bspline,

	domain = domain,
	eval = eval,
	derivative = derivative,
	speed = speed,
	tangent = tangent,
	curvature = curvature,
	torsion = torsion,
	frenet = frenet,
	frames = frames,

	arclength = arclength,
	length = length,
	length_at = length_at,
	param_at = param_at,
	at_length = at_length,
	sample = sample,

	closest = closest,
	closest_near = closest_near,
	bounds = bounds,
	split = split,
}
