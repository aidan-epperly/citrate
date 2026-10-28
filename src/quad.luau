--!strict
--!native
--!optimize 2

-- Adaptive integration of a function over an interval, finite or infinite.
--
--   local value, info = quad.integrate(function(x) return math.exp(-x * x) end, -math.huge, math.huge)
--   -- sqrt(pi); info.error estimates the error, info.evaluations counts calls of f
--
-- Two methods:
--   "gauss-kronrod" (the default)  Gauss-Kronrod 7-15 on each interval, splitting the one
--                                  with the largest error estimate until the total meets
--                                  the tolerance (QUADPACK's QAG strategy, with its error
--                                  estimate): robust for smooth integrands and ones with
--                                  kinks or steep spots inside;
--   "tanh-sinh"                    double exponential substitution x = tanh(pi/2 sinh t),
--                                  which clusters points at the ends so fast that
--                                  integrable singularities there (1 / sqrt(x), log x)
--                                  converge as fast as smooth integrands.
-- Infinite limits are mapped to finite ones (x = a + t / (1 - t)). For a smooth function
-- integrated many times on one interval, cheb.sum (Clenshaw-Curtis) or a fixed Gauss rule
-- (cheb.legpts) is cheaper.

local function fail(who: string, message: string): never
	error("quad." .. who .. ": " .. message, 3)
end

local EPS = 2 ^ -52

-- Gauss-Kronrod 7-15 (QUADPACK's qk15): the Kronrod nodes (the Gauss nodes are the 2nd,
-- 4th and 6th, and 0) and weights, and the Gauss weights. Checked in the tests: exact for
-- polynomials of degree 22 and 13.
local XGK = {
	0.991455371120812639206854697526329,
	0.949107912342758524526189684047851,
	0.864864423359769072789712788640926,
	0.741531185599394439863864773280788,
	0.586087235467691130294144845693013,
	0.405845151377397166906606412076961,
	0.207784955007898467600689403773245,
	0,
}
local WGK = {
	0.022935322010529224963732008058970,
	0.063092092629978553290700663189204,
	0.104790010322250183839876322541518,
	0.140653259715525918745189590510238,
	0.169004726639267902826583426598550,
	0.190350578064785409913256402421014,
	0.204432940075298892414161999234649,
	0.209482141084727828012999174891714,
}
local WG = {
	0.129484966168869693270611432679082,
	0.279705391489276667901467771423780,
	0.381830050505118944950369775488975,
	0.417959183673469387755102040816327,
}

-- The Kronrod estimate on [a, b] and its error estimate (QUADPACK's: the Kronrod-Gauss
-- difference, scaled by how far f varies from its mean).
local function kronrod(f: (number) -> number, a: number, b: number): (number, number)
	local c, h = (a + b) / 2, (b - a) / 2
	local fc = f(c)
	local resg = fc * WG[4]
	local resk = fc * WGK[8]
	local resabs = math.abs(resk)
	local f1, f2 = table.create(7, 0), table.create(7, 0)
	for j = 1, 7 do
		local dx = h * XGK[j]
		local u, v = f(c - dx), f(c + dx)
		f1[j], f2[j] = u, v
		resk += WGK[j] * (u + v)
		resabs += WGK[j] * (math.abs(u) + math.abs(v))
		if j % 2 == 0 then
			resg += WG[j // 2] * (u + v)
		end
	end
	local mean = resk / 2
	local resasc = WGK[8] * math.abs(fc - mean)
	for j = 1, 7 do
		resasc += WGK[j] * (math.abs(f1[j] - mean) + math.abs(f2[j] - mean))
	end
	local result = resk * h
	resabs *= math.abs(h)
	resasc *= math.abs(h)
	local err = math.abs((resk - resg) * h)
	if resasc ~= 0 and err ~= 0 then
		err = resasc * math.min(1, (200 * err / resasc) ^ 1.5)
	end
	if resabs > 1e-290 then
		err = math.max(50 * EPS * resabs, err)
	end
	return result, err
end

export type QuadOptions = { tol: number?, method: string?, maxintervals: number? }
export type QuadInfo = { error: number, evaluations: number, converged: boolean, method: string, intervals: number? }

local function adaptive(f: (number) -> number, a: number, b: number, tol: number, maxint: number, info: QuadInfo): number
	local lo, hi, val, err = { a }, { b }, {}, {}
	val[1], err[1] = kronrod(f, a, b)
	local total, totalerr = val[1], err[1]
	while totalerr > math.max(tol * math.abs(total), 1e-300) and #lo < maxint do
		-- the interval with the largest error, split in two
		local k, worst = 1, err[1]
		for i = 2, #lo do
			if err[i] > worst then
				k, worst = i, err[i]
			end
		end
		local m = (lo[k] + hi[k]) / 2
		if m <= lo[k] or m >= hi[k] then
			break -- too narrow to split
		end
		local v1, e1 = kronrod(f, lo[k], m)
		local v2, e2 = kronrod(f, m, hi[k])
		total += v1 + v2 - val[k]
		totalerr += e1 + e2 - err[k]
		table.insert(lo, m)
		table.insert(hi, hi[k])
		table.insert(val, v2)
		table.insert(err, e2)
		hi[k], val[k], err[k] = m, v1, e1
	end
	-- the sums again from scratch (the running ones drift)
	total, totalerr = 0, 0
	for i = 1, #lo do
		total += val[i]
		totalerr += err[i]
	end
	info.error = totalerr
	info.intervals = #lo
	info.converged = totalerr <= math.max(tol * math.abs(total), 1e-300)
	return total
end

-- tanh-sinh on [a, b]: levels halving the step until two agree.
local function tanh_sinh(f: (number) -> number, a: number, b: number, tol: number, info: QuadInfo): number
	local c, h = (a + b) / 2, (b - a) / 2
	local half = math.pi / 2
	-- the sum over the nodes t = k step (k odd past the first level), each node's weight
	-- half cosh t / cosh^2(half sinh t) and its distance from the nearer end h (1 - |tanh|)
	-- computed directly (2 / (1 + e^(2u))), so points near the ends keep their accuracy
	local function point(t: number): number
		local u = half * math.sinh(t)
		local e = math.exp(-2 * math.abs(u))
		local gap = 2 * e / (1 + e) -- 1 - tanh |u|
		local w = half * math.cosh(t) * 4 * e / ((1 + e) * (1 + e))
		if gap == 0 or w == 0 then
			return 0
		end
		local s = 0
		local d = h * gap
		local x1, x2 = b - d, a + d
		if t == 0 then
			return w * f(c)
		end
		if x1 > a and x1 < b then
			s += f(x1)
		end
		if x2 > a and x2 < b then
			s += f(x2)
		end
		return w * s
	end
	local tmax = 3.5
	local step = 1
	local sum = point(0)
	for k = 1, math.floor(tmax / step) do
		sum += point(k * step)
	end
	local est = sum * step * h
	local prev = math.huge
	for _ = 1, 12 do
		step /= 2
		for k = 1, math.floor(tmax / step), 2 do
			sum += point(k * step)
		end
		prev = est
		est = sum * step * h
		if math.abs(est - prev) <= tol * math.abs(est) then
			break
		end
	end
	info.error = math.abs(est - prev)
	info.converged = info.error <= math.max(tol * math.abs(est), 1e-300)
	return est
end

-- The integral of f over [a, b] (a or b may be infinite: -math.huge, math.huge), to
-- relative tolerance options.tol (default 1e-10). options.method: "gauss-kronrod" (the
-- default) or "tanh-sinh" (for singularities at the ends). Returns the value and info:
-- error (an estimate), evaluations, converged (false if the tolerance was not met within
-- options.maxintervals, default 1000, or 12 tanh-sinh levels), the method and, for
-- Gauss-Kronrod, the number of intervals.
local function integrate(f: (number) -> number, a: number, b: number, options: QuadOptions?): (number, QuadInfo)
	local o: QuadOptions = options or {}
	if type(f) ~= "function" then
		fail("integrate", "f must be a function")
	end
	if a ~= a or b ~= b then
		fail("integrate", "the limits are NaN")
	end
	local method = o.method or "gauss-kronrod"
	if method ~= "gauss-kronrod" and method ~= "tanh-sinh" then
		fail("integrate", "method must be \"gauss-kronrod\" or \"tanh-sinh\"")
	end
	local tol = o.tol or 1e-10
	local info: QuadInfo = { error = 0, evaluations = 0, converged = false, method = method }
	if a == b then
		info.converged = true
		return 0, info
	end
	if a > b then
		local v, i2 = integrate(f, b, a, o)
		return -v, i2
	end
	local g = function(x: number): number
		info.evaluations += 1
		local y = f(x)
		if type(y) ~= "number" then
			error(string.format("quad.integrate: f(%g) returned %s, not a number", x, typeof(y)), 0)
		end
		return y
	end
	-- infinite limits mapped to [0, 1]
	local h: (number) -> number = g
	local lo, hi = a, b
	if a == -math.huge and b == math.huge then
		h = function(t: number): number
			local x = t / (1 - t * t)
			return g(x) * (1 + t * t) / (1 - t * t) ^ 2
		end
		lo, hi = -1, 1
	elseif b == math.huge then
		h = function(t: number): number
			return g(a + t / (1 - t)) / (1 - t) ^ 2
		end
		lo, hi = 0, 1
	elseif a == -math.huge then
		h = function(t: number): number
			return g(b - t / (1 - t)) / (1 - t) ^ 2
		end
		lo, hi = 0, 1
	end
	local value
	if method == "tanh-sinh" then
		value = tanh_sinh(h, lo, hi, tol, info)
	else
		value = adaptive(h, lo, hi, tol, o.maxintervals or 1000, info)
	end
	return value, info
end

return {
	integrate = integrate,
	XGK = XGK,
	WGK = WGK,
	WG = WG,
}
