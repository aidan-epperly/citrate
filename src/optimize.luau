--!strict
--!native
--!optimize 2

-- Root finding and optimization: scalar roots and minima (classical methods, and global
-- ones through Chebyshev and AAA interpolants), roots of systems (Newton), nonlinear least
-- squares (Levenberg-Marquardt) and minimization (L-BFGS, Nelder-Mead), on any
-- representation (a Space; Jacobians and finite differences need buffers or vectors).
--
-- Each one returns its answer and an info table: evaluations (of the function), iterations,
-- converged, and method-specific counts. Wherever a method solves a one-dimensional
-- subproblem (a scalar problem, a line search, Levenberg-Marquardt's damping), a
-- Chebyshev or AAA alternative can be chosen, so they can be compared; the defaults are
-- the ones that measured best (bench/optimize.luau).

local types = require("./types")
local cheb = require("./cheb")
local rational = require("./rational")
local c64 = require("./c64")
local f64 = require("./f64")
local fdjac = require("./fdjac")

type Space<T> = types.Space<T>

local EPS = 2 ^ -52

export type ScalarInfo = { evaluations: number, iterations: number, converged: boolean, method: string }

-- f with its evaluations counted (into info.evaluations).
local function counted(f: (number) -> number, info: ScalarInfo): (number) -> number
	return function(x: number): number
		info.evaluations += 1
		return f(x)
	end
end

local function new_info(method: string): ScalarInfo
	return { evaluations = 0, iterations = 0, converged = false, method = method }
end

-- Scalar roots

-- Brent's method (zeroin): bisection, the secant method and inverse quadratic
-- interpolation, keeping a bracket, so it converges whenever f(a) and f(b) differ in sign,
-- and superlinearly near a simple root. xtol is the absolute tolerance (plus 4 eps |x|).
local function brent_root(f: (number) -> number, a: number, b: number, fa: number, fb: number, xtol: number, maxiter: number, info: ScalarInfo): number
	local c, fc = a, fa
	local d = b - a
	local e = d
	for iter = 1, maxiter do
		info.iterations = iter
		if (fb > 0) == (fc > 0) then
			c, fc = a, fa
			d = b - a
			e = d
		end
		if math.abs(fc) < math.abs(fb) then
			a, b, c = b, c, b
			fa, fb, fc = fb, fc, fb
		end
		local tol = 2 * EPS * math.abs(b) + 0.5 * xtol
		local m = 0.5 * (c - b)
		if math.abs(m) <= tol or fb == 0 then
			info.converged = true
			return b
		end
		if math.abs(e) >= tol and math.abs(fa) > math.abs(fb) then
			-- interpolation: secant if a = c, else inverse quadratic
			local s = fb / fa
			local p, q
			if a == c then
				p = 2 * m * s
				q = 1 - s
			else
				local qa, r = fa / fc, fb / fc
				p = s * (2 * m * qa * (qa - r) - (b - a) * (r - 1))
				q = (qa - 1) * (r - 1) * (s - 1)
			end
			if p > 0 then
				q = -q
			else
				p = -p
			end
			if 2 * p < math.min(3 * m * q - math.abs(tol * q), math.abs(e * q)) then
				e = d
				d = p / q
			else
				d, e = m, m
			end
		else
			d, e = m, m
		end
		a, fa = b, fb
		b += if math.abs(d) > tol then d elseif m > 0 then tol else -tol
		fb = f(b)
	end
	return b
end

-- A Chebyshev interpolant of f on [a, b] (adaptive, to relative tolerance tol), its
-- evaluations counted.
local function interpolant(f: (number) -> number, a: number, b: number, tol: number?, maxlength: number?): any
	return cheb.fun(f, { a, b }, { tol = tol, maxlength = maxlength or 4097 })
end

-- An AAA rational approximation of f on [a, b], from samples at Chebyshev points doubled
-- (17, 33, 65, ...) until it matches f at the midpoints between samples to tol (relative
-- to the largest |f|). Returns the rational, its sample count and whether it settled.
local function aaa_fit(f: (number) -> number, a: number, b: number, tol: number, maxpoints: number): (any, boolean)
	local n = 17
	local best: any = nil
	while true do
		local xs = cheb.chebpts(n, 2, a, b)
		local Z, F = c64.create(n), c64.create(n)
		local big = 0
		for k = 1, n do
			local x = buffer.readf64(xs, 8 * (k - 1))
			local v = f(x)
			big = math.max(big, math.abs(v))
			c64.set(Z, k, x, 0)
			c64.set(F, k, v, 0)
		end
		local R = rational.aaa(Z, F, { tol = math.max(tol, 1e-13) })
		best = R
		-- check between the samples
		local worst = 0
		for k = 1, n - 1, math.max(1, (n - 1) // 8) do
			local x = 0.5 * (buffer.readf64(xs, 8 * (k - 1)) + buffer.readf64(xs, 8 * k))
			local re = R.eval(x)
			worst = math.max(worst, math.abs(re - f(x)))
		end
		if worst <= tol * math.max(big, 1e-300) then
			return best, true
		end
		if 2 * n - 1 > maxpoints then
			return best, false
		end
		n = 2 * n - 1
	end
end

-- The real zeros of a rational from AAA in [a, b] (imaginary part below tol relative to
-- the interval), ascending.
local function aaa_real_zeros(R: any, a: number, b: number): { number }
	local out = {}
	local z = R.zeros
	local span = b - a
	for k = 1, buffer.len(z) // 16 do
		local re, im = buffer.readf64(z, 16 * (k - 1)), buffer.readf64(z, 16 * (k - 1) + 8)
		if math.abs(im) <= 1e-6 * span and re >= a - 1e-9 * span and re <= b + 1e-9 * span then
			table.insert(out, math.clamp(re, a, b))
		end
	end
	table.sort(out)
	return out
end

-- Polishes a root estimate x of f by the secant method (up to 4 steps, staying in [a, b]).
local function polish(f: (number) -> number, x: number, a: number, b: number): number
	local h = 1e-7 * math.max(b - a, 1e-300)
	local x0, x1 = x, math.clamp(x + h, a, b)
	if x1 == x0 then
		x1 = math.clamp(x - h, a, b)
	end
	local f0: number = f(x0)
	local f1: number = f(x1)
	for _ = 1, 4 do
		if f1 == f0 then
			break
		end
		local x2 = math.clamp(x1 - f1 * (x1 - x0) / (f1 - f0), a, b)
		x0, f0 = x1, f1
		x1 = x2
		f1 = f(x2)
		if math.abs(x1 - x0) <= 4 * EPS * math.max(math.abs(x1), 1e-300) then
			break
		end
	end
	return if math.abs(f1) <= math.abs(f0) then x1 else x0
end

export type ScalarOptions = {
	method: string?, -- "brent" (default), "chebyshev" or "aaa"
	xtol: number?, -- Brent's absolute tolerance on x (default 1e-12 (b - a))
	tol: number?, -- the interpolants' relative tolerance (default 1e-13)
	maxiter: number?,
}

-- A root of f in [a, b]. "brent" (the default) needs f(a) and f(b) of opposite signs and
-- finds a root between them, with few evaluations; "chebyshev" and "aaa" interpolate f on
-- [a, b] (Chebyshev polynomial or AAA rational) and return its first root there, which
-- needs no bracket but more evaluations (all roots: see roots). Returns the root (nil if
-- none was found) and info.
local function root_scalar(f: (number) -> number, a: number, b: number, options: ScalarOptions?): (number?, ScalarInfo)
	local o: ScalarOptions = options or {}
	local method = o.method or "brent"
	local info = new_info(method)
	local g = counted(f, info)
	assert(b > a, "root_scalar: need a < b")
	if method == "brent" then
		local fa, fb = g(a), g(b)
		if fa == 0 then
			info.converged = true
			return a, info
		elseif fb == 0 then
			info.converged = true
			return b, info
		end
		assert((fa > 0) ~= (fb > 0), "root_scalar: f(a) and f(b) must have opposite signs for brent")
		local x = brent_root(g, a, b, fa, fb, o.xtol or 1e-12 * (b - a), o.maxiter or 200, info)
		return x, info
	end
	local list = (nil :: any) :: { number }
	if method == "chebyshev" then
		local F = interpolant(g, a, b, o.tol)
		local r = cheb.roots(F)
		list = {}
		for k = 1, buffer.len(r) // 8 do
			list[k] = buffer.readf64(r, 8 * (k - 1))
		end
		info.converged = F.happy ~= false
	else
		assert(method == "aaa", `root_scalar: unknown method {method}`)
		local R, settled = aaa_fit(g, a, b, o.tol or 1e-13, 2049)
		list = aaa_real_zeros(R, a, b)
		for k, x in list do
			list[k] = polish(g, x, a, b)
		end
		info.converged = settled
	end
	return list[1], info
end

-- Every root of f in [a, b], ascending, from a Chebyshev ("chebyshev", the default) or
-- AAA ("aaa") interpolant of f there, and info.
local function roots(f: (number) -> number, a: number, b: number, options: ScalarOptions?): ({ number }, ScalarInfo)
	local o: ScalarOptions = options or {}
	local method = o.method or "chebyshev"
	local info = new_info(method)
	local g = counted(f, info)
	if method == "aaa" then
		local R, settled = aaa_fit(g, a, b, o.tol or 1e-13, 2049)
		local list = aaa_real_zeros(R, a, b)
		for k, x in list do
			list[k] = polish(g, x, a, b)
		end
		info.converged = settled
		return list, info
	end
	assert(method == "chebyshev", `roots: unknown method {method}`)
	local F = interpolant(g, a, b, o.tol)
	local r = cheb.roots(F)
	local list = {}
	for k = 1, buffer.len(r) // 8 do
		list[k] = buffer.readf64(r, 8 * (k - 1))
	end
	info.converged = F.happy ~= false
	return list, info
end

export type NewtonOptions = {
	df: ((number) -> number)?, -- f'; without it, the secant method
	tol: number?, -- stop when the step is below tol (1 + |x|) (default 1e-12)
	maxiter: number?, -- default 50
}

-- A root of f near x0: Newton's method with f' (options.df), else the secant method
-- (from x0 and a small step). Fast from a good guess (such as last frame's root), but
-- local: it may diverge or find another root. Returns the root and info (converged
-- false if it did not settle).
local function newton(f: (number) -> number, x0: number, options: NewtonOptions?): (number, ScalarInfo)
	local o: NewtonOptions = options or {}
	local df = o.df
	local info = new_info(if df then "newton" else "secant")
	local g = counted(f, info)
	local tol = o.tol or 1e-12
	local x, fx = x0, g(x0)
	local xp, fp = x0, fx
	if not df then
		xp = x0 + 1e-4 * (1 + math.abs(x0))
		fp = g(xp)
	end
	for iter = 1, o.maxiter or 50 do
		info.iterations = iter
		if fx == 0 then
			info.converged = true
			return x, info
		end
		local slope = if df then df(x) else (fx - fp) / (x - xp)
		if slope == 0 or slope ~= slope then
			return x, info
		end
		local step = fx / slope
		xp, fp = x, fx
		x -= step
		fx = g(x)
		if math.abs(step) <= tol * (1 + math.abs(x)) then
			info.converged = true
			return x, info
		end
	end
	return x, info
end

-- Scalar minima

-- Brent's minimization (fmin): golden-section search with parabolic steps; a local
-- minimum in [a, b], superlinear near a smooth minimum.
local function brent_min(f: (number) -> number, a: number, b: number, xtol: number, maxiter: number, info: ScalarInfo): (number, number)
	local golden = 0.5 * (3 - math.sqrt(5))
	local x = a + golden * (b - a)
	local w, v = x, x
	local fx = f(x)
	local fw, fv = fx, fx
	local d, e = 0, 0
	for iter = 1, maxiter do
		info.iterations = iter
		local xm = 0.5 * (a + b)
		local tol1 = math.sqrt(EPS) * math.abs(x) + xtol / 3
		local tol2 = 2 * tol1
		if math.abs(x - xm) <= tol2 - 0.5 * (b - a) then
			info.converged = true
			return x, fx
		end
		local golden_step = true
		if math.abs(e) > tol1 then
			local r = (x - w) * (fx - fv)
			local q = (x - v) * (fx - fw)
			local p = (x - v) * q - (x - w) * r
			q = 2 * (q - r)
			if q > 0 then
				p = -p
			else
				q = -q
			end
			if math.abs(p) < math.abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x) then
				e = d
				d = p / q
				local u = x + d
				if u - a < tol2 or b - u < tol2 then
					d = if xm >= x then tol1 else -tol1
				end
				golden_step = false
			end
		end
		if golden_step then
			e = if x >= xm then a - x else b - x
			d = golden * e
		end
		local u = if math.abs(d) >= tol1 then x + d elseif d > 0 then x + tol1 else x - tol1
		local fu = f(u)
		if fu <= fx then
			if u >= x then
				a = x
			else
				b = x
			end
			v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
		else
			if u < x then
				a = u
			else
				b = u
			end
			if fu <= fw or w == x then
				v, fv, w, fw = w, fw, u, fu
			elseif fu <= fv or v == x or v == w then
				v, fv = u, fu
			end
		end
	end
	return x, fx
end

-- The minimum of an AAA rational R on [a, b]: a fine grid, then golden section on the
-- best cell (evaluating R only, not f).
local function rational_min(R: any, a: number, b: number): number
	local grid = 400
	local best, bx = math.huge, a
	for k = 0, grid do
		local x = a + (b - a) * k / grid
		local v = R.eval(x)
		if v < best then
			best, bx = v, x
		end
	end
	local lo, hi = math.max(a, bx - (b - a) / grid), math.min(b, bx + (b - a) / grid)
	local golden = 0.5 * (math.sqrt(5) - 1)
	local c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
	local fc, fd = R.eval(c), R.eval(d)
	for _ = 1, 60 do
		if fc < fd then
			hi, d, fd = d, c, fc
			c = hi - golden * (hi - lo)
			fc = R.eval(c)
		else
			lo, c, fc = c, d, fd
			d = lo + golden * (hi - lo)
			fd = R.eval(d)
		end
	end
	local x = 0.5 * (lo + hi)
	return if R.eval(x) <= best then x else bx
end

-- A minimum of f on [a, b]: "brent" (the default) finds a local minimum with few
-- evaluations; "chebyshev" and "aaa" interpolate f on [a, b] and return the interpolant's
-- global minimum there. Returns x, f(x) (as evaluated) and info.
local function minimize_scalar(f: (number) -> number, a: number, b: number, options: ScalarOptions?): (number, number, ScalarInfo)
	local o: ScalarOptions = options or {}
	local method = o.method or "brent"
	local info = new_info(method)
	local g = counted(f, info)
	assert(b > a, "minimize_scalar: need a < b")
	if method == "brent" then
		local x, fx = brent_min(g, a, b, o.xtol or 1e-10 * (b - a), o.maxiter or 500, info)
		return x, fx, info
	elseif method == "chebyshev" then
		local F = interpolant(g, a, b, o.tol)
		local _, x = cheb.min(F)
		info.converged = F.happy ~= false
		return x, g(x), info
	end
	assert(method == "aaa", `minimize_scalar: unknown method {method}`)
	local R, settled = aaa_fit(g, a, b, o.tol or 1e-13, 2049)
	local x = rational_min(R, a, b)
	info.converged = settled
	return x, g(x), info
end

-- Coordinates

-- A representation's values as f64 arrays, for finite differences, Jacobians and simplexes:
-- f64 buffers as they are, f32 buffers converted, vectors as 3 components. load(x, out)
-- writes x's coordinates into the f64 array out; store(b, out) makes (or writes into out)
-- the value with coordinates b.
type Coords<T> = {
	n: number,
	load: (x: T, out: buffer) -> buffer,
	store: (b: buffer, out: T?) -> T,
	R: any, -- f64, as fdjac takes it, with the representation's precision (for the steps)
}

-- f64 with another unit roundoff, for fdjac's steps (single-precision values need steps of
-- sqrt(2^-24), not sqrt(2^-53), or a step rounds away when stored)
local function f64_with_eps(eps: number): any
	if eps <= 1e-15 then
		return f64
	end
	return setmetatable({ space = setmetatable({ eps = eps }, { __index = f64.space }) }, { __index = f64 })
end

local function coords<T>(S: Space<T>, x0: T, what: string): Coords<T>
	if type(x0) == "vector" then
		return {
			n = 3,
			R = f64_with_eps(S.eps or 2 ^ -24),
			load = function(x: any, out: buffer): buffer
				buffer.writef64(out, 0, x.x)
				buffer.writef64(out, 8, x.y)
				buffer.writef64(out, 16, x.z)
				return out
			end,
			store = function(b: buffer, _: any?): any
				return vector.create(buffer.readf64(b, 0), buffer.readf64(b, 8), buffer.readf64(b, 16))
			end,
		} :: any
	end
	assert(type(x0) == "buffer" and S.real ~= false, `optimize: {what} needs real buffers or vectors`)
	local single = (S.eps or EPS) > 1e-10
	if not single then
		return {
			n = buffer.len(x0 :: any) // 8,
			R = f64,
			load = function(x: any, out: buffer): buffer
				buffer.copy(out, 0, x)
				return out
			end,
			store = function(b: buffer, out: any?): any
				local o = out or buffer.create(buffer.len(b))
				buffer.copy(o, 0, b)
				return o
			end,
		} :: any
	end
	local n = buffer.len(x0 :: any) // 4
	return {
		n = n,
		R = f64_with_eps(S.eps or 2 ^ -24),
		load = function(x: any, out: buffer): buffer
			for k = 0, n - 1 do
				buffer.writef64(out, 8 * k, buffer.readf32(x, 4 * k))
			end
			return out
		end,
		store = function(b: buffer, out: any?): any
			local o = out or buffer.create(4 * n)
			for k = 0, n - 1 do
				buffer.writef32(o, 4 * k, buffer.readf64(b, 8 * k))
			end
			return o
		end,
	} :: any
end

-- Line searches
--
-- phi(a) = f(x + a p) along a descent direction p, phi(0) = phi0 and phi'(0) = dphi0 < 0.
-- "wolfe": the strong Wolfe conditions (Nocedal and Wright, algorithms 3.5 and 3.6, with
-- cubic interpolation in the zoom), evaluating f and its gradient together. "chebyshev" and
-- "aaa": f alone at k Chebyshev points of [0, amax], then the step to the interpolant's
-- (polynomial or AAA rational) global minimum there, accepted if it decreases f enough
-- (Armijo), else the interval is shrunk. "hybrid": Wolfe, but when its first trial fails
-- the Armijo condition, the interpolant on [0, that trial] replaces the zoom.

local C1, C2 = 1e-4, 0.9

-- The minimizer of the cubic through (a, fa) with slope da and (b, fb) with slope db, kept
-- inside [lo + 0.1 (hi - lo), hi - 0.1 (hi - lo)]; bisection if it is not defined.
local function cubic_min(a: number, fa: number, da: number, b: number, fb: number, db: number): number
	local lo, hi = math.min(a, b), math.max(a, b)
	local d1 = da + db - 3 * (fa - fb) / (a - b)
	local disc = d1 * d1 - da * db
	local x = 0.5 * (lo + hi)
	if disc >= 0 then
		local d2 = math.sqrt(disc) * (if b > a then 1 else -1)
		local den = db - da + 2 * d2
		if den ~= 0 then
			x = b - (b - a) * (db + d2 - d1) / den
		end
	end
	if not (x >= lo + 0.1 * (hi - lo) and x <= hi - 0.1 * (hi - lo)) then
		x = 0.5 * (lo + hi)
	end
	return x
end

-- Strong Wolfe search. phig(a) evaluates (phi(a), phi'(a)) and makes a the current trial.
-- Returns the accepted step and its phi, phi' (the last trial evaluated is the accepted
-- one unless ok is false).
local function wolfe(phig: (number) -> (number, number), phi0: number, dphi0: number, a1: number, amax: number, maxiter: number): (number, number, number, boolean, boolean)
	-- (the fifth result says whether the first trial already failed Armijo, for hybrid)
	local ap, fp, dp = 0, phi0, dphi0
	local a = a1
	local function zoom(alo: number, flo: number, dlo: number, ahi: number, fhi: number, dhi: number): (number, number, number, boolean)
		-- (alo satisfies Armijo and has the lower f; ahi brackets a step meeting both)
		for _ = 1, maxiter do
			local aj = cubic_min(alo, flo, dlo, ahi, fhi, dhi)
			local fj, dj = phig(aj)
			if fj > phi0 + C1 * aj * dphi0 or fj >= flo then
				ahi, fhi, dhi = aj, fj, dj
			else
				if math.abs(dj) <= -C2 * dphi0 then
					return aj, fj, dj, true
				end
				if dj * (ahi - alo) >= 0 then
					ahi, fhi, dhi = alo, flo, dlo
				end
				alo, flo, dlo = aj, fj, dj
			end
			if math.abs(ahi - alo) <= 1e-14 * math.max(alo, 1) then
				break
			end
		end
		-- the best point satisfies Armijo; the caller re-evaluates it if it was not the last
		return alo, flo, dlo, alo > 0
	end
	for i = 1, maxiter do
		local fa, da = phig(a)
		if fa > phi0 + C1 * a * dphi0 or (i > 1 and fa >= fp) then
			local s, fs, ds, ok = zoom(ap, fp, dp, a, fa, da)
			return s, fs, ds, ok, i == 1
		end
		if math.abs(da) <= -C2 * dphi0 then
			return a, fa, da, true, false
		end
		if da >= 0 then
			local s, fs, ds, ok = zoom(a, fa, da, ap, fp, dp)
			return s, fs, ds, ok, false
		end
		ap, fp, dp = a, fa, da
		if a >= amax then
			return a, fa, da, true, false
		end
		a = math.min(2 * a, amax)
	end
	return ap, fp, dp, ap > 0, false
end

-- The interpolating search: phi at k Chebyshev points of [0, amax] (phi(0) known), the
-- interpolant's minimum (method "chebyshev" or "aaa"), Armijo, else [0, amax / 4] (down to
-- 1e-14 of the first interval). Returns the step (0 if none was found) and phi there.
local function interpolating(phi: (number) -> number, phi0: number, dphi0: number, amax: number, method: string, k: number): (number, number)
	local a0 = amax
	while amax > 1e-14 * a0 do
		local function g(a: number): number
			return if a == 0 then phi0 else phi(a)
		end
		local a: number
		if method == "aaa" then
			local xs = cheb.chebpts(k, 2, 0, amax)
			local Z, F = c64.create(k), c64.create(k)
			for j = 1, k do
				local x = buffer.readf64(xs, 8 * (j - 1))
				c64.set(Z, j, x, 0)
				c64.set(F, j, g(x), 0)
			end
			local R = rational.aaa(Z, F, { tol = 1e-10 })
			a = rational_min(R, 0, amax)
		else
			local Fc = cheb.fun(g, { 0, amax }, { n = k })
			local _, am = cheb.min(Fc)
			a = am
		end
		if a > 0 then
			local fa = phi(a)
			if fa <= phi0 + C1 * a * dphi0 then
				return a, fa
			end
			amax = math.min(a, amax / 4)
		else
			amax /= 4
		end
	end
	return 0, phi0
end

-- Minimization

export type MinimizeOptions<T> = {
	method: string?, -- "lbfgs" (default) or "nelder-mead"
	grad: ((x: T, out: T) -> T)?, -- the gradient; without it, forward differences
	line_search: string?, -- "wolfe" (default), "chebyshev", "aaa" or "hybrid"
	points: number?, -- the interpolating searches' sample count (default 9)
	memory: number?, -- L-BFGS's history length (default 10)
	gtol: number?, -- stop when max |g_i| <= gtol (default 1e-5)
	ftol: number?, -- or when the relative decrease of f is below ftol (default 2.2e-9)
	xatol: number?, -- Nelder-Mead: simplex size (default 1e-4)
	fatol: number?, -- Nelder-Mead: spread of f over the simplex (default 1e-4)
	maxiter: number?,
}

export type Info = {
	evaluations: number, -- of the function
	gradients: number, -- gradient evaluations (user gradients, or finite-difference ones)
	iterations: number,
	converged: boolean,
	message: string,
	[string]: any,
}

-- L-BFGS: two-loop recursion over the last `memory` steps, the initial Hessian scaled by
-- s^T y / y^T y, and the chosen line search.
local function lbfgs<T>(S: Space<T>, f: (T) -> number, x0: T, o: MinimizeOptions<T>, info: Info): (T, number)
	local ls = o.line_search or "wolfe"
	local k = o.points or 9
	local m = o.memory or 10
	local gtol, ftol = o.gtol or 1e-5, o.ftol or 2.2e-9
	local maxiter = o.maxiter or 15000
	local function fc(x: T): number
		info.evaluations += 1
		return f(x)
	end
	local grad: (T, T) -> T
	if o.grad then
		local user = o.grad :: (T, T) -> T
		grad = function(x: T, out: T): T
			info.gradients += 1
			return user(x, out)
		end
	else
		local C = coords(S, x0, "a finite-difference gradient")
		local n = C.n
		local xb, gb = buffer.create(8 * n), buffer.create(8 * n)
		local xt: T? = nil
		grad = function(x: T, out: T): T
			info.gradients += 1
			local f0 = fc(x)
			xb = C.load(x, xb)
			for j = 0, n - 1 do
				local xj = buffer.readf64(xb, 8 * j)
				local h = math.sqrt(S.eps or EPS) * math.max(math.abs(xj), 1)
				buffer.writef64(xb, 8 * j, xj + h)
				local hh = buffer.readf64(xb, 8 * j) - xj
				xt = C.store(xb, xt)
				buffer.writef64(gb, 8 * j, (fc(xt :: T) - f0) / hh)
				buffer.writef64(xb, 8 * j, xj)
			end
			return C.store(gb, out)
		end
	end
	local x = S.copy(S.similar(x0), x0)
	local fx = fc(x)
	local g = grad(x, S.similar(x0))
	local p, xt, gt = S.similar(x0), S.similar(x0), S.similar(x0)
	local sk, yk, rho = {}, {}, {}
	local alpha = table.create(m, 0)
	local count, head = 0, 0
	local single = (S.eps or EPS) > 1e-10
	-- max |v_i| (the 2-norm for representations without coordinates)
	local function gmax(v: T): number
		local w: any = v
		if type(w) == "vector" then
			return math.max(math.abs(w.x), math.abs(w.y), math.abs(w.z))
		elseif type(w) == "buffer" and S.real ~= false then
			local big = 0
			if single then
				for o2 = 0, buffer.len(w) - 4, 4 do
					big = math.max(big, math.abs(buffer.readf32(w, o2)))
				end
			else
				for o2 = 0, buffer.len(w) - 8, 8 do
					big = math.max(big, math.abs(buffer.readf64(w, o2)))
				end
			end
			return big
		end
		return S.norm(v)
	end
	for iter = 1, maxiter do
		info.iterations = iter
		if gmax(g) <= gtol then
			info.converged, info.message = true, "the gradient is small"
			return x, fx
		end
		-- p = -H g by the two-loop recursion
		p = S.scale(p, -1, g)
		for j = 0, count - 1 do
			local i = (head - j - 1) % m + 1
			alpha[i] = rho[i] * S.dot(sk[i], p)
			p = S.axpy(p, -alpha[i], yk[i], p)
		end
		if count > 0 then
			local i = (head - 1) % m + 1
			p = S.scale(p, S.dot(sk[i], yk[i]) / S.dot(yk[i], yk[i]), p)
		end
		for j = count - 1, 0, -1 do
			local i = (head - j - 1) % m + 1
			local beta = rho[i] * S.dot(yk[i], p)
			p = S.axpy(p, alpha[i] - beta, sk[i], p)
		end
		local dphi0 = S.dot(g, p)
		if not (dphi0 < 0) then
			-- not a descent direction (rounding): restart from steepest descent
			p = S.scale(p, -1, g)
			dphi0 = -S.dot(g, g)
			count = 0
		end
		-- the first step: unit for quasi-Newton, scaled on the first iteration
		local a1 = if count == 0 then math.min(1, 1 / math.max(S.norm(g), 1e-300)) else 1
		local function trial(a: number): T
			xt = S.axpy(xt, a, p, x)
			return xt
		end
		local a, fa = 0, fx
		local have_g = false
		if ls == "hybrid" then
			-- the unit step if f decreases enough there (no gradient needed to decide), else
			-- the interpolant's minimum on [0, a1]
			local f1 = fc(trial(a1))
			if f1 <= fx + C1 * a1 * dphi0 then
				a, fa = a1, f1
			else
				a, fa = interpolating(function(t: number): number
					return fc(trial(t))
				end, fx, dphi0, a1, "chebyshev", k)
			end
		elseif ls == "wolfe" then
			local last = -1
			local function phig(s: number): (number, number)
				local xs = trial(s)
				local v = fc(xs)
				gt = grad(xs, gt)
				last = s
				return v, S.dot(gt, p)
			end
			local s, fs, _, ok = wolfe(phig, fx, dphi0, a1, 1e10, 30)
			if ok then
				a, fa = s, fs
				have_g = last == s
			end
		else
			a, fa = interpolating(function(t: number): number
				return fc(trial(t))
			end, fx, dphi0, a1 * 2, ls, k)
		end
		if a == 0 then
			info.message = "the line search found no decrease"
			return x, fx
		end
		xt = trial(a)
		if not have_g then
			gt = grad(xt, gt)
		end
		-- the new pair s = xt - x, y = gt - g (skipped unless y^T s > 0)
		local slot = head % m + 1
		local s_new = S.sub((sk[slot] or S.similar(x0)) :: T, xt, x)
		local y_new = S.sub((yk[slot] or S.similar(x0)) :: T, gt, g)
		local ys = S.dot(y_new, s_new)
		if ys > 1e-10 * S.norm(y_new) * S.norm(s_new) then
			sk[slot], yk[slot], rho[slot] = s_new, y_new, 1 / ys
			head = slot
			count = math.min(count + 1, m)
		end
		local fold = fx
		x, xt = xt, x
		g, gt = gt, g
		fx = fa
		if (fold - fx) <= ftol * math.max(math.abs(fold), math.abs(fx), 1) then
			info.converged, info.message = true, "the decrease of f is small"
			return x, fx
		end
	end
	info.message = "too many iterations"
	return x, fx
end

-- Nelder-Mead (SciPy's: reflection 1, expansion 2, contraction 1/2, shrink 1/2; the
-- initial simplex steps each nonzero coordinate by 5% and zero ones by 0.00025).
local function nelder_mead<T>(S: Space<T>, f: (T) -> number, x0: T, o: MinimizeOptions<T>, info: Info): (T, number)
	local C = coords(S, x0, "Nelder-Mead")
	local n = C.n
	local xatol, fatol = o.xatol or 1e-4, o.fatol or 1e-4
	local maxiter = o.maxiter or 200 * n
	local pts: { buffer } = {}
	local vals: { number } = {}
	local function value(b: buffer): number
		info.evaluations += 1
		return f(C.store(b))
	end
	local base = C.load(x0, buffer.create(8 * n))
	pts[1], vals[1] = base, value(base)
	for j = 1, n do
		local b = buffer.create(8 * n)
		buffer.copy(b, 0, base)
		local v = buffer.readf64(b, 8 * (j - 1))
		buffer.writef64(b, 8 * (j - 1), if v ~= 0 then 1.05 * v else 0.00025)
		pts[j + 1], vals[j + 1] = b, value(b)
	end
	local order = table.create(n + 1, 0)
	for i = 1, n + 1 do
		order[i] = i
	end
	local centroid, xr, xe, xc = buffer.create(8 * n), buffer.create(8 * n), buffer.create(8 * n), buffer.create(8 * n)
	-- out = c + t (c - worst)
	local function along(out: buffer, t: number, worst: buffer)
		for k = 0, 8 * (n - 1), 8 do
			local c = buffer.readf64(centroid, k)
			buffer.writef64(out, k, c + t * (c - buffer.readf64(worst, k)))
		end
	end
	for iter = 1, maxiter do
		info.iterations = iter
		table.sort(order, function(a: number, b: number): boolean
			return vals[a] < vals[b]
		end)
		local best, worst = order[1], order[n + 1]
		-- converged: the simplex and its values are small
		local size, spread = 0, 0
		for i = 2, n + 1 do
			local q = order[i]
			spread = math.max(spread, math.abs(vals[q] - vals[best]))
			for k = 0, 8 * (n - 1), 8 do
				size = math.max(size, math.abs(buffer.readf64(pts[q], k) - buffer.readf64(pts[best], k)))
			end
		end
		if size <= xatol and spread <= fatol then
			info.converged, info.message = true, "the simplex is small"
			return C.store(pts[best]), vals[best]
		end
		buffer.fill(centroid, 0, 0)
		for i = 1, n do
			local q = pts[order[i]]
			for k = 0, 8 * (n - 1), 8 do
				buffer.writef64(centroid, k, buffer.readf64(centroid, k) + buffer.readf64(q, k) / n)
			end
		end
		local W = pts[worst]
		along(xr, 1, W)
		local fr = value(xr)
		local second_worst = vals[order[n]]
		if fr < vals[best] then
			along(xe, 2, W)
			local fe = value(xe)
			if fe < fr then
				buffer.copy(W, 0, xe)
				vals[worst] = fe
			else
				buffer.copy(W, 0, xr)
				vals[worst] = fr
			end
		elseif fr < second_worst then
			buffer.copy(W, 0, xr)
			vals[worst] = fr
		else
			-- contraction (outside if the reflection improved on the worst, else inside)
			local outside = fr < vals[worst]
			along(xc, if outside then 0.5 else -0.5, W)
			local fcv = value(xc)
			if fcv < (if outside then fr else vals[worst]) then
				buffer.copy(W, 0, xc)
				vals[worst] = fcv
			else
				-- shrink toward the best
				local B = pts[best]
				for i = 2, n + 1 do
					local q = pts[order[i]]
					for k = 0, 8 * (n - 1), 8 do
						local b0 = buffer.readf64(B, k)
						buffer.writef64(q, k, b0 + 0.5 * (buffer.readf64(q, k) - b0))
					end
					vals[order[i]] = value(q)
				end
			end
		end
	end
	table.sort(order, function(a: number, b: number): boolean
		return vals[a] < vals[b]
	end)
	info.message = "too many iterations"
	return C.store(pts[order[1]]), vals[order[1]]
end

-- A local minimum of f (a number from a value of the space) from x0: L-BFGS (the default;
-- options.grad gives the gradient, else forward differences) or Nelder-Mead
-- (options.method = "nelder-mead": no derivatives, for small or noisy problems).
-- Returns x, f(x) and info (evaluations, gradients, iterations, converged, message).
local function minimize<T>(S: Space<T>, f: (T) -> number, x0: T, options: MinimizeOptions<T>?): (T, number, Info)
	local o: MinimizeOptions<T> = options or {}
	local method = o.method or "lbfgs"
	local info: Info = { evaluations = 0, gradients = 0, iterations = 0, converged = false, message = "", method = method }
	if method == "nelder-mead" then
		local x, fx = nelder_mead(S, f, x0, o, info)
		return x, fx, info
	end
	assert(method == "lbfgs", `minimize: unknown method {method}`)
	info.line_search = o.line_search or "wolfe"
	local x, fx = lbfgs(S, f, x0, o, info)
	return x, fx, info
end

-- Systems of equations

export type RootOptions<T> = {
	jac: ((x: T, J: any) -> any)?, -- the Jacobian, into the n by n f64 matrix J; else differences
	line_search: string?, -- "backtracking" (default), "chebyshev" or "aaa"
	points: number?, -- the interpolating searches' sample count (default 9)
	tol: number?, -- stop when max |F_i| <= tol (default 1e-10)
	xtol: number?, -- or when the step is below xtol (1 + |x|) (default 1e-14)
	maxiter: number?, -- default 100
}

-- A root of F (F(x, out) writes F(x), of x's size, into out) near x0: Newton's method,
-- the Jacobian from options.jac or forward differences, its step damped by a line search
-- on |F|^2 / 2. Returns x and info (evaluations, jacobians, iterations, converged, norm).
local function root<T>(S: Space<T>, F: (T, T) -> T, x0: T, options: RootOptions<T>?): (T, Info)
	local o: RootOptions<T> = options or {}
	local ls = o.line_search or "backtracking"
	local kpts = o.points or 9
	local info: Info = { evaluations = 0, gradients = 0, jacobians = 0, iterations = 0, converged = false, message = "", line_search = ls }
	local C = coords(S, x0, "root")
	local n = C.n
	local tol, xtol = o.tol or 1e-10, o.xtol or 1e-14
	-- everything in f64 coordinates: x, F(x), the step, a trial point and its F
	local xb, fb, pb, tb, ftb = buffer.create(8 * n), buffer.create(8 * n), buffer.create(8 * n), buffer.create(8 * n), buffer.create(8 * n)
	local xT: T?, fT: T? = nil, nil
	local function Fc(b: buffer, out: buffer): buffer
		info.evaluations += 1
		xT = C.store(b, xT)
		fT = F(xT :: T, (fT or S.similar(x0)) :: T)
		return C.load(fT :: T, out)
	end
	local J = f64.matrix(n, n)
	local yp, fp, col = buffer.create(8 * n), buffer.create(8 * n), buffer.create(8 * n)
	local LU: any = nil
	xb = C.load(x0, xb)
	fb = Fc(xb, fb)
	local function half_sq(b: buffer): number
		local d = f64.norm(b)
		return 0.5 * d * d
	end
	for iter = 1, o.maxiter or 100 do
		info.iterations = iter
		if f64.amax(fb) <= tol then
			info.converged, info.message = true, "|F| is small"
			break
		end
		info.jacobians += 1
		if o.jac then
			xT = C.store(xb, xT)
			J = (o.jac :: any)(xT, J)
		else
			fdjac.dense(C.R, Fc, xb, fb, J, yp, fp, col)
		end
		LU = f64.lu_copy(J, LU)
		if LU.singular then
			info.message = "the Jacobian is singular"
			break
		end
		pb = f64.lu_solve(pb, LU, fb)
		pb = f64.scale(pb, -1, pb)
		local m0 = half_sq(fb)
		local dm0 = -2 * m0 -- the merit's slope along the Newton step
		local function merit(a: number): number
			tb = f64.axpy(tb, a, pb, xb)
			ftb = Fc(tb, ftb)
			return half_sq(ftb)
		end
		local a = 1
		local ma = merit(1)
		if not (ma <= m0 + C1 * dm0) then
			if ls == "backtracking" then
				-- quadratic interpolation, the step kept in [0.1, 0.5] of the last
				a = 1
				for _ = 1, 30 do
					local q = -dm0 * a * a / (2 * (ma - m0 - dm0 * a))
					a = math.clamp(if q == q then q else 0.5 * a, 0.1 * a, 0.5 * a)
					ma = merit(a)
					if ma <= m0 + C1 * a * dm0 then
						break
					end
				end
			else
				local s, ms = interpolating(merit, m0, dm0, 1, ls, kpts)
				if s > 0 then
					a, ma = s, ms
				else
					a = 0
				end
			end
		end
		if a == 0 then
			info.message = "the line search found no decrease"
			break
		end
		local step = a * f64.norm(pb)
		xb, tb = tb, xb
		fb, ftb = ftb, fb
		if step <= xtol * (1 + f64.norm(xb)) then
			info.converged = f64.amax(fb) <= math.sqrt(tol)
			info.message = "the step is small"
			break
		end
	end
	info.norm = f64.norm(fb)
	return C.store(xb), info
end

-- Nonlinear least squares

export type LeastSquaresOptions<T> = {
	jac: ((x: T, J: any) -> any)?, -- the m by n Jacobian, into the f64 matrix J; else differences
	lambda: string?, -- the damping's method: "newton" (Moré's, the default), "aaa" or "chebyshev"
	ftol: number?, -- default 1e-8 (as SciPy's least_squares)
	xtol: number?, -- default 1e-8
	gtol: number?, -- default 1e-8
	maxiter: number?, -- default 100 n
}

-- Minimizes |r(x)|^2 / 2 (r(x, out) writes the m residuals into the f64 array out, a new
-- one when out is nil) from x0: Levenberg-Marquardt as Moré's trust-region form (MINPACK's
-- lmder), with variables scaled by the Jacobian's column norms. Each step solves
-- (J^T J + lambda D^2) p = -J^T r with |D p| = Delta, through an SVD of J D^-1, which
-- makes each trial lambda cost O(n): Moré's Newton iteration on 1/|D p| (the default),
-- or an AAA or Chebyshev root of |D p| - Delta (|D p|^2 is rational in lambda). Returns x
-- and info (evaluations, jacobians, iterations, converged, cost, lambda_trials).
local function least_squares<T>(S: Space<T>, r: (T, buffer?) -> buffer, x0: T, options: LeastSquaresOptions<T>?): (T, Info)
	local o: LeastSquaresOptions<T> = options or {}
	local lam_method = o.lambda or "newton"
	local info: Info = { evaluations = 0, gradients = 0, jacobians = 0, iterations = 0, converged = false, message = "", lambda = lam_method, lambda_trials = 0 }
	local C = coords(S, x0, "least_squares")
	local n = C.n
	local ftol, xtol, gtol = o.ftol or 1e-8, o.xtol or 1e-8, o.gtol or 1e-8
	local xT: T? = nil
	local function R(b: buffer, out: buffer?): buffer
		info.evaluations += 1
		xT = C.store(b, xT)
		return r(xT :: T, out)
	end
	local xb = C.load(x0, buffer.create(8 * n))
	local rb = R(xb, nil)
	local m = buffer.len(rb) // 8
	local J = f64.matrix(m, n)
	local yp, fp, col = buffer.create(8 * n), buffer.create(8 * m), buffer.create(8 * m)
	local xt, rt = buffer.create(8 * n), buffer.create(8 * m)
	local D = table.create(n, 0)
	local cost = 0.5 * f64.dot(rb, rb)
	local delta = 0
	local pq = table.create(n, 0) -- sigma_i c_i (the scaled step's numerators)
	local sig2 = table.create(n, 0)
	for iter = 1, o.maxiter or 100 * n do
		info.iterations = iter
		info.jacobians += 1
		if o.jac then
			xT = C.store(xb, xT)
			J = (o.jac :: any)(xT, J)
		else
			fdjac.dense(C.R, function(b: buffer, out: buffer): buffer
				local v = R(b, out)
				return v
			end, xb, rb, J, yp, fp, col)
		end
		-- column scaling D (the largest column norms seen), the gradient J^T r
		local gnorm = 0
		for j = 1, n do
			local s = 0
			local g = 0
			for i = 1, m do
				local v = buffer.readf64(J.data, 8 * ((j - 1) * m + i - 1))
				s += v * v
				g += v * buffer.readf64(rb, 8 * (i - 1))
			end
			local cn = math.sqrt(s)
			D[j] = math.max(D[j], cn)
			if D[j] == 0 then
				D[j] = 1
			end
			if cost > 0 then
				gnorm = math.max(gnorm, math.abs(g) / (D[j] * math.sqrt(2 * cost)))
			end
		end
		if gnorm <= gtol then
			info.converged, info.message = true, "the gradient is small"
			break
		end
		if iter == 1 then
			local dx = 0
			for j = 1, n do
				dx += (D[j] * buffer.readf64(xb, 8 * (j - 1))) ^ 2
			end
			delta = 100 * math.sqrt(dx)
			if delta == 0 then
				delta = 100
			end
		end
		-- SVD of J D^-1: the scaled step q(lambda) = -sum sigma_i c_i / (sigma_i^2 + lambda) v_i
		local Js = f64.matrix(m, n)
		for j = 1, n do
			for i = 1, m do
				local o2 = 8 * ((j - 1) * m + i - 1)
				buffer.writef64(Js.data, o2, buffer.readf64(J.data, o2) / D[j])
			end
		end
		local F = f64.svd_copy(Js)
		local k = buffer.len(F.values) // 8
		local U, V = F.u :: any, F.v :: any
		for i = 1, k do
			local s = buffer.readf64(F.values, 8 * (i - 1))
			local c = 0
			for l = 1, m do
				c += buffer.readf64(U.data, 8 * ((i - 1) * m + l - 1)) * buffer.readf64(rb, 8 * (l - 1))
			end
			pq[i], sig2[i] = s * c, s * s
		end
		local smax = if k > 0 then math.sqrt(sig2[1]) else 0
		local rank_tol = smax * math.max(m, n) * EPS
		local function qnorm(lam: number): number
			info.lambda_trials += 1
			local s = 0
			for i = 1, k do
				if sig2[i] > rank_tol * rank_tol or lam > 0 then
					local t = pq[i] / (sig2[i] + lam)
					s += t * t
				end
			end
			return math.sqrt(s)
		end
		local function dqnorm(lam: number, qn: number): number
			-- d|q|/d lambda = -sum (sigma c)^2 / (sigma^2 + lambda)^3 / |q|
			local s = 0
			for i = 1, k do
				local d = sig2[i] + lam
				if d > 0 then
					s += pq[i] * pq[i] / (d * d * d)
				end
			end
			return -s / qn
		end
		-- the trial loop: lambda for the current Delta, the step, the ratio
		local accepted = false
		local actred, prered, pnorm, lam = 0, 0, 0, 0
		for _ = 1, 30 do
			lam = 0
			local q0 = qnorm(0)
			if q0 > 1.1 * delta then
				-- find lambda with |q(lambda)| = Delta (to 10%): bounds [lo, hi]
				local g2 = 0
				for i = 1, k do
					g2 += pq[i] * pq[i]
				end
				local hi = math.sqrt(g2) / delta
				local lo = 0
				if lam_method == "newton" then
					-- Moré: Newton on 1/|q| - 1/Delta, safeguarded in [lo, hi]
					lam = math.max(0.001 * hi, math.sqrt(lo * hi))
					for _ = 1, 30 do
						local qn = qnorm(lam)
						local phi = qn - delta
						if math.abs(phi) <= 0.1 * delta then
							break
						end
						if phi < 0 then
							hi = lam
						else
							lo = lam
						end
						local step = (qn / delta) * (phi / -dqnorm(lam, qn))
						lam = lam + step
						if not (lam > lo and lam < hi) then
							lam = math.max(0.001 * hi, math.sqrt(lo * hi))
						end
					end
				else
					-- an interpolant of psi(lambda) = |q(lambda)| - Delta on [0, hi], its root
					local psi = function(l: number): number
						return qnorm(l) - delta
					end
					local x: number? = nil
					if lam_method == "aaa" then
						local Rr = aaa_fit(psi, 0, hi, 1e-6, 129)
						local zs = aaa_real_zeros(Rr, 0, hi)
						x = zs[1]
					else
						assert(lam_method == "chebyshev", `least_squares: unknown lambda method {lam_method}`)
						local Fc = cheb.fun(psi, { 0, hi }, { tol = 1e-6, maxlength = 1025 })
						local rts = cheb.roots(Fc)
						if buffer.len(rts) >= 8 then
							x = buffer.readf64(rts, 0)
						end
					end
					lam = x or hi
				end
			end
			-- the step p = D^-1 V q(lambda), and its predicted reduction
			buffer.fill(xt, 0, 0)
			for i = 1, k do
				local d = sig2[i] + lam
				if d > 0 and (sig2[i] > rank_tol * rank_tol or lam > 0) then
					local t = -pq[i] / d
					for j = 1, n do
						local o2 = 8 * (j - 1)
						buffer.writef64(xt, o2, buffer.readf64(xt, o2) + t * buffer.readf64(V.data, 8 * ((i - 1) * n + j - 1)))
					end
				end
			end
			pnorm = 0
			for j = 1, n do
				local qj = buffer.readf64(xt, 8 * (j - 1))
				pnorm += qj * qj
				buffer.writef64(xt, 8 * (j - 1), qj / D[j])
			end
			pnorm = math.sqrt(pnorm)
			-- predicted: |r|^2/2 - |r + J p|^2/2
			local jp = f64.matvec(fp, J, xt)
			fp = jp
			local lin = 0
			for i = 1, m do
				local v = buffer.readf64(rb, 8 * (i - 1)) + buffer.readf64(jp, 8 * (i - 1))
				lin += v * v
			end
			prered = cost - 0.5 * lin
			-- the trial point
			for j = 0, 8 * (n - 1), 8 do
				buffer.writef64(xt, j, buffer.readf64(xb, j) + buffer.readf64(xt, j))
			end
			rt = R(xt, rt)
			local cnew = 0.5 * f64.dot(rt, rt)
			actred = if cnew == cnew then cost - cnew else -math.huge
			local ratio = if prered > 0 then actred / prered else 0
			if ratio < 0.25 then
				delta = 0.5 * math.min(delta, pnorm)
			elseif ratio > 0.75 or lam == 0 then
				delta = 2 * pnorm
			end
			if ratio > 1e-4 then
				xb, xt = xt, xb
				rb, rt = rt, rb
				cost = cnew
				accepted = true
				break
			end
			if delta <= xtol * pnorm then
				break
			end
		end
		info.cost = cost
		if not accepted then
			info.message = "no step decreases the cost"
			break
		end
		if math.abs(actred) <= ftol * (cost + actred) and prered <= ftol * (cost + actred) then
			info.converged, info.message = true, "the cost's decrease is small"
			break
		end
		local dx = 0
		for j = 1, n do
			dx += (D[j] * buffer.readf64(xb, 8 * (j - 1))) ^ 2
		end
		if delta <= xtol * math.sqrt(dx) then
			info.converged, info.message = true, "the step is small"
			break
		end
	end
	info.cost = cost
	return C.store(xb), info
end

return {
	root_scalar = root_scalar,
	roots = roots,
	newton = newton,
	minimize_scalar = minimize_scalar,
	minimize = minimize,
	root = root,
	least_squares = least_squares,
}
