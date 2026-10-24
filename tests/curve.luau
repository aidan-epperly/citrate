--!strict

-- Curves: every family against independent constructions, the differential geometry,
-- arclength, closest points, bounds, splitting and frames, on vectors and on buffers.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local curve, f64, cheb = lib.curve, lib.f64, lib.cheb
local V = vector.create

-- |a - b| for vectors, in double precision.
local function vdist(a: vector, b: vector): number
	local x, y, z = a.x - b.x, a.y - b.y, a.z - b.z
	return math.sqrt(x * x + y * y + z * z)
end

-- The same points as f64 buffers.
local function buffers(pts: { vector }): { buffer }
	local out = {}
	for i, p in pts do
		out[i] = f64.from({ p.x, p.y, p.z })
	end
	return out
end

local function bdist(a: buffer, b: buffer): number
	return f64.norm(f64.sub(nil, a, b))
end

local function bvec(b: buffer): vector
	return V(f64.get(b, 1), f64.get(b, 2), f64.get(b, 3))
end

-- Points on t -> (t, t^2, t^3) (the twisted cubic), and other polynomial curves.
local function twisted(t: number): vector
	return V(t, t * t, t * t * t)
end

local function central_diff(C: any, t: number, step: number): vector
	return (curve.eval(C, t + step) - curve.eval(C, t - step)) / (2 * step)
end

local PTS = { V(0, 0, 0), V(1, 2, 0), V(3, 3, 1), V(4, 0, 2), V(6, 1, 0), V(7, 3, -1) }

-- The NURBS unit circle: 9 control points on a square, degree 2.
local S2 = math.sqrt(2) / 2
local function circle(): any
	return curve.bspline({
		V(1, 0, 0),
		V(1, 1, 0),
		V(0, 1, 0),
		V(-1, 1, 0),
		V(-1, 0, 0),
		V(-1, -1, 0),
		V(0, -1, 0),
		V(1, -1, 0),
		V(1, 0, 0),
	}, {
		degree = 2,
		knots = { 0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1 },
		weights = { 1, S2, 1, S2, 1, S2, 1, S2, 1 },
	})
end

-- The exact cubic (t^2, t^3) with a cusp at t = 0, as a not-a-knot spline through samples
-- (which reproduces cubics exactly when the breaks are the sample parameters).
local function cusp(): any
	local ts = { -1, -0.5, 0.25, 1 }
	local pts: { buffer } = {}
	for i, t in ts do
		pts[i] = f64.from({ t * t, t * t * t })
	end
	return curve.cubic_spline(pts, { breaks = ts, ends = "not-a-knot", space = f64.space })
end

test("Bézier curves", function()
	local P = { V(0, 0, 0), V(1, 3, 1), V(4, 3, -1), V(5, 0, 2) }
	local C = curve.bezier(P)
	-- de Casteljau, independently.
	local function casteljau(u: number): vector
		local q = table.clone(P)
		for level = 1, 3 do
			for j = 1, 4 - level do
				q[j] = q[j] * (1 - u) + q[j + 1] * u
			end
		end
		return q[1]
	end
	local worst = 0
	for k = 0, 20 do
		worst = math.max(worst, vdist(curve.eval(C, k / 20), casteljau(k / 20)))
	end
	check(worst <= 1e-5, `eval matches de Casteljau ({worst})`)
	check(curve.eval(C, 0) == P[1] and curve.eval(C, 1) == P[4], "interpolates its end points")
	check(vdist(curve.derivative(C, 0, 1), (P[2] - P[1]) * 3) <= 1e-6, "C'(0) = 3 (P1 - P0)")
	check(vdist(curve.derivative(C, 1, 2), (P[4] - P[3] * 2 + P[2]) * 6) <= 1e-5, "C''(1) = 6 (P3 - 2 P2 + P1)")
	check(vdist(curve.derivative(C, 0.3, 3), (P[4] - P[3] * 3 + P[2] * 3 - P[1]) * 6) <= 1e-5, "C''' is constant")
	check(curve.derivative(C, 0.3, 4) == V(0, 0, 0), "C'''' = 0")
	check(vdist(curve.derivative(C, 0.37, 1), central_diff(C, 0.37, 1e-3)) <= 1e-4, "C' matches finite differences")
	-- Other domains: the derivative scales with the parameter.
	local D = curve.bezier(P, { breaks = { 2, 4 } })
	check(vdist(curve.eval(D, 3), curve.eval(C, 0.5)) == 0, "breaks { 2, 4 }: same points")
	check(vdist(curve.derivative(D, 3, 1), curve.derivative(C, 0.5, 1) / 2) <= 1e-6, "breaks { 2, 4 }: chain rule")

	-- A rational quadratic is a conic: weights (1, sqrt(2)/2, 1) give a quarter circle.
	local Q = curve.bezier({ V(1, 0, 0), V(1, 1, 0), V(0, 1, 0) }, { weights = { 1, S2, 1 } })
	local radius = 0
	for k = 0, 20 do
		radius = math.max(radius, math.abs(vector.magnitude(curve.eval(Q, k / 20)) - 1))
	end
	check(radius <= 1e-6, `rational quarter circle has radius 1 ({radius})`)
	close(curve.length(Q), math.pi / 2, 1e-14, "quarter circle length")
	close(curve.curvature(Q, 0.3), 1, 1e-6, "quarter circle curvature")

	local path = curve.bezier_path({ V(0, 0, 0), V(1, 1, 0), V(2, 1, 0), V(3, 0, 0), V(4, -1, 0), V(5, -1, 0), V(6, 0, 0) })
	check(#path.breaks == 3 and curve.eval(path, 1) == V(3, 0, 0), "bezier_path: segments share end points")
	check(not pcall(function()
		curve.bezier_path({ V(0, 0, 0), V(1, 1, 0), V(2, 1, 0) })
	end), "bezier_path needs k * degree + 1 points")
	check(not pcall(function()
		curve.bezier({ V(0, 0, 0), V(1, 1, 0) }, { weights = { 1, -1 } })
	end), "weights must be positive")
	check(not pcall(function()
		curve.bezier({ f64.from({ 0 }), f64.from({ 1 }) })
	end), "buffers need options.space")
end)

test("vector and buffer curves agree", function()
	-- Every constructor computes in double precision and rounds once, and vector
	-- evaluation accumulates in double precision: so a vector curve gives exactly the
	-- single-precision rounding of the same curve on f64 buffers.
	local bp = buffers(PTS)
	local makers: { { any } } = {
		{ "catmull_rom", {} },
		{ "kochanek_bartels", { tension = 0.2, bias = -0.3, continuity = 0.1 } },
		{ "cubic_spline", { ends = "not-a-knot" } },
		{ "bspline", { degree = 3 } },
		{ "bezier_path", { degree = 5 } },
	}
	for _, mk in makers do
		local name, opts = mk[1], mk[2]
		local Cv = (curve :: any)[name](PTS, opts)
		local bopts = table.clone(opts)
		bopts.space = f64.space
		local Cb = (curve :: any)[name](bp, bopts)
		-- Control points: computed in double precision, rounded once.
		local rounded = #Cv.points == #Cb.points
		for i, p in Cv.points do
			rounded = rounded and p == bvec(Cb.points[i])
		end
		check(rounded, `{name}: vector control points are the rounded double ones`)
		-- Evaluation: the same curve on f64 buffers (with the vector curve's control points),
		-- rounded once.
		local Cr: any = {
			space = f64.space,
			vector = false,
			degree = Cv.degree,
			breaks = Cv.breaks,
			points = buffers(Cv.points),
			weights = Cv.weights,
			closed = Cv.closed,
		}
		local same = true
		local a, b = curve.domain(Cv)
		for k = 0, 30 do
			local t = a + (b - a) * k / 30
			same = same and curve.eval(Cv, t) == bvec(curve.eval(Cr, t))
		end
		check(same, `{name}: vector eval is the rounded double eval`)
		close(curve.length(Cv), curve.length(Cb), 1e-6, `{name}: lengths agree`)
		close(curve.curvature(Cv, (a + b) / 3), curve.curvature(Cb, (a + b) / 3), 1e-5, `{name}: curvatures agree`)
	end
end)

test("Hermite, Catmull-Rom and Kochanek-Bartels", function()
	local tangents = { V(1, 0, 0), V(0, 1, 0), V(-1, 0, 1), V(0, 0, 1), V(1, 1, 1), V(0, -1, 0) }
	local H = curve.hermite(PTS, tangents, { breaks = { 0, 1, 3, 4, 7, 8 } })
	local ok = true
	for i, t in { 0, 1, 3, 4, 7, 8 } do
		ok = ok and vdist(curve.eval(H, t), PTS[i]) <= 1e-6 and vdist(curve.derivative(H, t, 1), tangents[i]) <= 1e-5
	end
	check(ok, "hermite interpolates points and derivatives")

	for _, alpha in { 0, 0.5, 1 } do
		local C = curve.catmull_rom(PTS, { alpha = alpha })
		local interp, c1 = true, true
		for i = 1, #PTS do
			local t = C.breaks[i]
			interp = interp and vdist(curve.eval(C, t), PTS[i]) <= 1e-5
			if i > 1 and i < #PTS then
				-- C^1 at the joints: left and right derivatives agree.
				local left = curve.derivative(C, t - 1e-9, 1)
				local right = curve.derivative(C, t, 1)
				c1 = c1 and vdist(left, right) <= 1e-4 * vector.magnitude(right)
			end
		end
		check(interp, `alpha {alpha}: interpolates`)
		check(c1, `alpha {alpha}: C^1`)
		local gap = C.breaks[2] - C.breaks[1]
		close(gap, vdist(PTS[2], PTS[1]) ^ alpha, 1e-6, `alpha {alpha}: knot spacing |chord|^alpha`)
	end
	local U = curve.catmull_rom(PTS, { alpha = 0 })
	check(vdist(curve.derivative(U, 2, 1), (PTS[4] - PTS[2]) / 2) <= 1e-5, "uniform: tangent (P_(i+1) - P_(i-1)) / 2")
	local Dup = curve.catmull_rom({ V(0, 0, 0), V(1, 0, 0), V(1, 0, 0), V(2, 1, 0) })
	check(#Dup.breaks == 3, "repeated points are dropped")

	local closed = curve.catmull_rom({ V(1, 0, 0), V(0, 1, 0), V(-1, 0, 0), V(0, -1, 0) }, { closed = true })
	local a, b = curve.domain(closed)
	check(#closed.breaks == 5 and closed.closed, "closed: one segment per point")
	check(vdist(curve.eval(closed, b), V(1, 0, 0)) <= 1e-6, "closed: returns to the start")
	check(vdist(curve.derivative(closed, a, 1), curve.derivative(closed, b, 1)) <= 1e-5, "closed: C^1 where it closes")

	local K0 = curve.kochanek_bartels(PTS)
	local same = true
	for k = 0, 25 do
		same = same and vdist(curve.eval(K0, k / 5), curve.eval(U, U.breaks[1] + k / 5)) <= 1e-5
	end
	check(same, "TCB with zero parameters is the uniform Catmull-Rom spline")
	local K1 = curve.kochanek_bartels(PTS, { tension = 1 })
	check(vector.magnitude(curve.derivative(K1, 2, 1)) <= 1e-6, "tension 1: zero tangents at the points")
	local Kc = curve.kochanek_bartels(PTS, { continuity = 0.5 })
	local jump = vdist(curve.derivative(Kc, 2 - 1e-9, 1), curve.derivative(Kc, 2, 1))
	check(jump > 0.1, `continuity 0.5: the tangent jumps ({jump})`)
	local Kt = curve.kochanek_bartels(PTS, { tension = { 0, 0, 1, 0, 0, 0 } })
	check(vector.magnitude(curve.derivative(Kt, 2, 1)) <= 1e-6 and vector.magnitude(curve.derivative(Kt, 1, 1)) > 0.1, "per-point tension")
end)

test("interpolating cubic splines", function()
	-- With breaks at the sample parameters, clamped and not-a-knot splines reproduce a
	-- cubic exactly.
	local ts = { -1, -0.4, 0.1, 0.7, 1.5 }
	local pts: { vector } = {}
	for i, t in ts do
		pts[i] = twisted(t)
	end
	local nak = curve.cubic_spline(pts, { breaks = ts, ends = "not-a-knot" })
	local cl = curve.cubic_spline(pts, { breaks = ts, ends = "clamped", start = V(1, -2, 3), finish = V(1, 3, 3 * 1.5 * 1.5) })
	local w1, w2 = 0, 0
	for k = 0, 40 do
		local t = -1 + 2.5 * k / 40
		w1 = math.max(w1, vdist(curve.eval(nak, t), twisted(t)))
		w2 = math.max(w2, vdist(curve.eval(cl, t), twisted(t)))
	end
	check(w1 <= 1e-5 and w2 <= 1e-5, `cubics reproduced (not-a-knot {w1}, clamped {w2})`)

	local C = curve.cubic_spline(PTS)
	local interp, c2 = true, true
	for i = 1, #PTS do
		local t = C.breaks[i]
		interp = interp and vdist(curve.eval(C, t), PTS[i]) <= 1e-5
		if i > 1 and i < #PTS then
			local l2, r2 = curve.derivative(C, t - 1e-12, 2), curve.derivative(C, t, 2)
			c2 = c2 and vdist(l2, r2) <= 1e-3 * (1 + vector.magnitude(r2))
		end
	end
	check(interp, "natural: interpolates")
	check(c2, "natural: C^2")
	local a, b = curve.domain(C)
	check(vector.magnitude(curve.derivative(C, a, 2)) <= 1e-4 and vector.magnitude(curve.derivative(C, b, 2)) <= 1e-4, "natural: zero second derivative at the ends")
	local N = curve.cubic_spline(PTS, { ends = "not-a-knot" })
	local t2 = N.breaks[2]
	check(vdist(curve.derivative(N, t2 - 1e-9, 3), curve.derivative(N, t2, 3)) <= 1e-3, "not-a-knot: C^3 at the second point")
	local three = curve.cubic_spline({ V(0, 0, 0), V(1, 1, 0), V(2, 0, 0) }, { breaks = { 0, 1, 2 }, ends = "not-a-knot" })
	check(vdist(curve.eval(three, 0.5), V(0.5, 0.75, 0)) <= 1e-6, "not-a-knot with 3 points: the parabola")
	local two = curve.cubic_spline({ V(0, 0, 0), V(2, 2, 0) })
	check(vdist(curve.eval(two, two.breaks[2] / 2), V(1, 1, 0)) <= 1e-6, "2 points: a straight line")

	local ring = { V(2, 0, 0), V(0, 1, 0), V(-2, 0, 1), V(0, -1, 0), V(1, -1, -1) }
	local P = curve.cubic_spline(ring, { closed = true })
	local pa, pb = curve.domain(P)
	check(#P.breaks == 6, "periodic: one segment per point")
	local d1 = vdist(curve.derivative(P, pa, 1), curve.derivative(P, pb, 1))
	local d2 = vdist(curve.derivative(P, pa, 2), curve.derivative(P, pb, 2))
	check(d1 <= 1e-4 and d2 <= 1e-3, `periodic: C^2 where it closes ({d1}, {d2})`)
	local ok = true
	for i = 2, #ring do
		local t = P.breaks[i]
		ok = ok and vdist(curve.derivative(P, t - 1e-12, 2), curve.derivative(P, t, 2)) <= 1e-3 * (1 + vector.magnitude(curve.derivative(P, t, 2)))
	end
	check(ok, "periodic: C^2 at every point")
	check(not pcall(function()
		curve.cubic_spline(PTS, { ends = "clamped" })
	end), "clamped needs start and finish")
end)

test("B-splines and NURBS", function()
	-- Cox-de Boor basis functions, independently.
	local function basis(knots: { number }, i: number, p: number, t: number): number
		if p == 0 then
			return if knots[i] <= t and t < knots[i + 1] then 1 else 0
		end
		local a, b = 0, 0
		if knots[i + p] > knots[i] then
			a = (t - knots[i]) / (knots[i + p] - knots[i]) * basis(knots, i, p - 1, t)
		end
		if knots[i + p + 1] > knots[i + 1] then
			b = (knots[i + p + 1] - t) / (knots[i + p + 1] - knots[i + 1]) * basis(knots, i + 1, p - 1, t)
		end
		return a + b
	end
	local knots = { 0, 0, 0, 0, 0.5, 1.5, 1.5, 3, 3, 3, 3 }
	local P = { V(0, 0, 0), V(1, 2, 0), V(2, -1, 1), V(3, 3, 0), V(4, 0, -1), V(5, 2, 2), V(6, 0, 0) }
	local w = { 1, 2, 0.5, 1, 3, 1, 1 }
	local B = curve.bspline(P, { knots = knots })
	local R = curve.bspline(P, { knots = knots, weights = w })
	local wb, wr = 0, 0
	for k = 0, 29 do
		local t = 3 * k / 30
		local sum, wsum = V(0, 0, 0), V(0, 0, 0)
		local den = 0
		for i = 1, #P do
			local N = basis(knots, i, 3, t)
			sum += P[i] * N
			wsum += P[i] * (N * w[i])
			den += N * w[i]
		end
		wb = math.max(wb, vdist(curve.eval(B, t), sum))
		wr = math.max(wr, vdist(curve.eval(R, t), wsum / den))
	end
	check(wb <= 1e-5, `B-spline matches Cox-de Boor ({wb})`)
	check(wr <= 1e-5, `NURBS matches the rational Cox-de Boor sum ({wr})`)
	check(#B.breaks == 4, "a double knot gives no empty segment")
	check(curve.eval(B, 0) == P[1] and vdist(curve.eval(B, 3), P[7]) == 0, "clamped: starts and ends at the end points")
	-- A double knot in a cubic leaves it C^1 but not C^2 there.
	local jump = vdist(curve.derivative(B, 1.5 - 1e-12, 2), curve.derivative(B, 1.5, 2))
	check(jump > 1e-3, "double knot: the second derivative jumps")

	local U = curve.bspline(P)
	local ua, ub = curve.domain(U)
	check(ua == 0 and ub == 4, "default knots: clamped uniform, domain [0, n - p]")
	local L = curve.bspline(P, { degree = 1 })
	check(vdist(curve.eval(L, 2.5), (P[3] + P[4]) / 2) <= 1e-6, "degree 1: the control polygon")

	local closed = curve.bspline({ V(1, 0, 0), V(0, 1, 0), V(-1, 0, 0), V(0, -1, 0) }, { closed = true })
	local ca, cb = curve.domain(closed)
	check(ca == 0 and cb == 4, "closed: domain [0, n]")
	local ok = vdist(curve.eval(closed, ca), curve.eval(closed, cb)) <= 1e-6
	for k = 1, 2 do
		ok = ok and vdist(curve.derivative(closed, ca, k), curve.derivative(closed, cb, k)) <= 1e-5
	end
	check(ok, "closed cubic: C^2 where it closes")

	local O = circle()
	local radius = 0
	for k = 0, 100 do
		radius = math.max(radius, math.abs(vector.magnitude(curve.eval(O, k / 100)) - 1))
	end
	check(radius <= 1e-6, `NURBS circle has radius 1 ({radius})`)
	close(curve.length(O), 2 * math.pi, 1e-14, "NURBS circle: length 2 pi")
	close(curve.curvature(O, 0.3), 1, 1e-6, "NURBS circle: curvature 1")
	close(curve.torsion(O, 0.3), 0, 1e-12, "NURBS circle: torsion 0")
end)

test("curvature, torsion and Frenet frames", function()
	-- The twisted cubic (t, t^2, t^3), exactly, as a not-a-knot spline.
	local ts = { -1, -0.3, 0.4, 1 }
	local pts: { vector } = {}
	for i, t in ts do
		pts[i] = twisted(t)
	end
	local C = curve.cubic_spline(pts, { breaks = ts, ends = "not-a-knot" })
	for _, t in { -0.8, 0, 0.5 } do
		local q = 1 + 4 * t * t + 9 * t ^ 4
		local kappa = 2 * math.sqrt(9 * t ^ 4 + 9 * t * t + 1) / q ^ 1.5
		local tau = 3 / (9 * t ^ 4 + 9 * t * t + 1)
		close(curve.curvature(C, t), kappa, 1e-5, `curvature at {t}`)
		close(curve.torsion(C, t), tau, 1e-5, `torsion at {t}`)
		close(curve.speed(C, t), math.sqrt(q), 1e-6, `speed at {t}`)
		local T, N, Bn = curve.frenet(C, t)
		local n, b = N :: vector, Bn :: vector
		local orth = math.max(math.abs(vector.dot(T, n)), math.abs(vector.dot(T, b)), math.abs(vector.dot(n, b)))
		check(orth <= 1e-6 and math.abs(vector.magnitude(n) - 1) <= 1e-6, `Frenet frame orthonormal at {t}`)
		check(vdist(vector.cross(T, n), b) <= 1e-6, `binormal is T x N at {t}`)
	end
	local line = curve.bezier({ V(0, 0, 0), V(1, 1, 1), V(2, 2, 2) })
	local T, N = curve.frenet(line, 0.5)
	check(N == nil and math.abs(vector.magnitude(T) - 1) <= 1e-6, "straight line: no normal")
	close(curve.curvature(line, 0.5), 0, 1e-12, "straight line: curvature 0")
	check(not pcall(function()
		curve.torsion(curve.bezier(buffers({ V(0, 0, 0), V(1, 1, 0) }), { space = f64.space }) :: any, 0.5)
	end), "torsion needs vectors")
	-- Curvature in any dimension: a circle of radius 2 in R^4, via a rational quadratic.
	local Q = curve.bezier(
		{ f64.from({ 2, 0, 0, 0 }), f64.from({ 2, 0, 2, 0 }), f64.from({ 0, 0, 2, 0 }) },
		{ space = f64.space, weights = { 1, S2, 1 } }
	)
	close(curve.curvature(Q, 0.4), 0.5, 1e-12, "R^4 circle of radius 2: curvature 1/2")
	local Tq, Nq = curve.frenet(Q, 0.4)
	local p = curve.eval(Q, 0.4)
	-- The normal points to the center.
	local toward = f64.scale(nil, -1 / f64.norm(p), p)
	check(bdist(Nq :: buffer, toward) <= 1e-12 and math.abs(f64.dot(Tq, Nq :: buffer)) <= 1e-12, "R^4 circle: normal points to the center")
end)

test("arclength", function()
	-- A straight segment with a nonuniform parameter: the length is exact.
	local line = curve.bezier({ V(0, 0, 0), V(0.75, 1, 0), V(3, 4, 0) })
	close(curve.length(line), 5, 1e-15, "segment of length 5")
	-- Reference values by Gauss-Legendre quadrature on many subintervals.
	-- (The speed is smooth within each segment but not across breaks.)
	local function reference(C: any): number
		local x, wts = cheb.legpts(20, 0, 1)
		local total = 0
		for seg = 1, #C.breaks - 1 do
			local a, b = C.breaks[seg], C.breaks[seg + 1]
			local pieces = 40
			for k = 0, pieces - 1 do
				local lo = a + (b - a) * k / pieces
				local hi = a + (b - a) * (k + 1) / pieces
				for j = 1, 20 do
					total += f64.get(wts, j) * (hi - lo) * curve.speed(C, lo + (hi - lo) * f64.get(x, j))
				end
			end
		end
		return total
	end
	for _, name in { "catmull_rom", "cubic_spline", "bspline", "kochanek_bartels" } do
		local C = (curve :: any)[name](PTS)
		close(curve.length(C), reference(C), 1e-13, `{name}: length`)
	end
	-- The cusp curve (t^2, t^3): length 2 (13^1.5 - 8) / 27.
	local K = cusp()
	local exact = 2 * (13 ^ 1.5 - 8) / 27
	close(curve.length(K), exact, 1e-13, "cusp: length")
	close(curve.length_at(K, 0), exact / 2, 1e-13, "cusp: half the length at the cusp")
	close(curve.param_at(K, exact / 2), 0, 1e-7, "cusp: the cusp at half the length")
	-- s(t) = (13^1.5 - (4 + 9 t^2)^1.5) / 27 for t <= 0.
	local s = (13 ^ 1.5 - (4 + 9 * 0.25) ^ 1.5) / 27
	close(curve.param_at(K, s), -0.5, 1e-12, "cusp: t(s) before the cusp")

	for _, name in { "catmull_rom", "cubic_spline", "bspline" } do
		local C = (curve :: any)[name](PTS)
		local L = curve.length(C)
		local a, b = curve.domain(C)
		local worst, unit = 0, 0
		for k = 0, 50 do
			local t = a + (b - a) * k / 50
			worst = math.max(worst, math.abs(curve.param_at(C, curve.length_at(C, t)) - t))
			local sk = L * k / 50
			worst = math.max(worst, math.abs(curve.length_at(C, curve.param_at(C, sk)) - sk) / L)
			-- d/ds C(t(s)) has length 1.
			if k > 0 and k < 50 then
				local step = 1e-4 * L
				local v = (curve.at_length(C, sk + step) - curve.at_length(C, sk - step)) / (2 * step)
				unit = math.max(unit, math.abs(vector.magnitude(v) - 1))
			end
		end
		check(worst <= 1e-12, `{name}: t(s(t)) = t and s(t(s)) = s ({worst})`)
		check(unit <= 1e-3, `{name}: unit speed by arclength ({unit})`)
		check(curve.param_at(C, -1) == a and curve.param_at(C, 2 * L) == b, `{name}: s is clamped`)
		local pts, params = curve.sample(C, 11)
		local even = 0
		for i = 1, 11 do
			even = math.max(even, math.abs(curve.length_at(C, params[i]) - L * (i - 1) / 10))
		end
		check(#pts == 11 and even <= 1e-12 * L, `{name}: sample is evenly spaced ({even})`)
	end
	-- A looser tolerance gives a shorter table and the same length to that accuracy.
	local C = curve.cubic_spline(PTS)
	local precise = curve.length(C)
	close(curve.arclength(C, { tol = 1e-6 }), precise, 1e-6, "tol = 1e-6")
	-- Generic spaces, in 4 dimensions.
	local B = curve.cubic_spline({ f64.from({ 0, 0, 0, 0 }), f64.from({ 1, 0, 1, 0 }), f64.from({ 1, 1, 0, 1 }), f64.from({ 0, 2, 1, 1 }) }, { space = f64.space })
	close(curve.length(B), reference(B), 1e-13, "R^4: length")
	local degenerate = curve.bezier({ V(1, 1, 1), V(1, 1, 1) })
	check(curve.length(degenerate) == 0 and curve.param_at(degenerate, 0.5) == 0, "a point: length 0")
end)

test("closest points and bounds", function()
	local O = circle()
	local t, p, dist = curve.closest(O, V(3, 4, 0))
	check(math.abs(dist - 4) <= 1e-6 and vdist(p, V(0.6, 0.8, 0)) <= 1e-6, `circle: closest to (3, 4) ({dist})`)
	check(vdist(curve.eval(O, t), p) == 0, "circle: the parameter gives the point")
	local _, _, d0 = curve.closest(O, V(0, 0, 1))
	close(d0, math.sqrt(2), 1e-6, "circle: from the axis")
	-- Against dense sampling plus local refinement.
	local C = curve.catmull_rom(PTS)
	local a, b = curve.domain(C)
	math.randomseed(201)
	local ok = true
	for _ = 1, 10 do
		local q = V(8 * math.random() - 1, 5 * math.random() - 1, 4 * math.random() - 2)
		local _, _, d = curve.closest(C, q)
		local brute = math.huge
		for k = 0, 4000 do
			brute = math.min(brute, vdist(curve.eval(C, a + (b - a) * k / 4000), q))
		end
		ok = ok and d <= brute + 1e-9 and d >= brute - 1e-3
	end
	check(ok, "random points: no closer point on a dense sampling")
	local G = curve.cubic_spline(buffers(PTS), { space = f64.space })
	local tg, pg, dg = curve.closest(G, f64.from({ 3, 3, 3 }))
	local tv, _, dv = curve.closest(curve.cubic_spline(PTS), V(3, 3, 3))
	check(math.abs(tg - tv) <= 1e-5 and math.abs(dg - dv) <= 1e-5 and f64.length(pg) == 3, "buffers agree with vectors")

	local lo, hi = curve.bounds(C)
	local inside, tight = true, { math.huge, math.huge, math.huge, math.huge, math.huge, math.huge }
	for k = 0, 2000 do
		local q = curve.eval(C, a + (b - a) * k / 2000)
		inside = inside and q.x >= lo.x and q.y >= lo.y and q.z >= lo.z and q.x <= hi.x and q.y <= hi.y and q.z <= hi.z
		tight = {
			math.min(tight[1], q.x - lo.x),
			math.min(tight[2], q.y - lo.y),
			math.min(tight[3], q.z - lo.z),
			math.min(tight[4], hi.x - q.x),
			math.min(tight[5], hi.y - q.y),
			math.min(tight[6], hi.z - q.z),
		}
	end
	check(inside, "bounds contain the curve")
	check(math.max(table.unpack(tight)) <= 1e-4, "bounds are tight")
	local clo, chi = curve.bounds(O)
	check(clo.x <= -1 and clo.x >= -1 - 1e-6 and chi.y >= 1 and chi.y <= 1 + 1e-6 and clo.z == 0, "circle bounds")
end)

test("closest_near: local Newton search", function()
	local C = curve.catmull_rom(PTS)
	local a, b = curve.domain(C)
	-- Started near the global answer, it finds the same point.
	math.randomseed(202)
	local agree = true
	for _ = 1, 20 do
		local q = V(8 * math.random() - 1, 5 * math.random() - 1, 4 * math.random() - 2)
		local tg, _, dg = curve.closest(C, q)
		if tg > a and tg < b then
			local tn, pn, dn = curve.closest_near(C, q, math.clamp(tg + 0.05 * (b - a) * (2 * math.random() - 1), a, b))
			agree = agree and math.abs(tn - tg) <= 1e-9 * (b - a) and math.abs(dn - dg) <= 1e-12 and vdist(pn, curve.eval(C, tn)) == 0
		end
	end
	check(agree, "from a nearby start: the global answer")

	-- Tracking a moving point, each search starting from the last answer.
	local track = true
	local t = a
	for k = 0, 200 do
		local s = curve.length(C) * k / 200
		local q = curve.at_length(C, s) + V(0, 0, 0.05)
		local tn, _, dn = curve.closest_near(C, q, t)
		local tg, _, dg = curve.closest(C, q)
		track = track and math.abs(dn - dg) <= 1e-9 and math.abs(tn - tg) <= 1e-6 * (b - a)
		t = tn
	end
	check(track, "tracking a moving point")

	-- Local, not global: a U-shaped curve, the point near one arm, the start on the other.
	local U = curve.bezier({ V(0, 0, 0), V(0, -4, 0), V(4, -4, 0), V(4, 0, 0) })
	local q = V(3.6, -0.2, 0)
	local tl, _, dl = curve.closest_near(U, q, 0.02)
	local _, _, dglobal = curve.closest(U, q)
	local g = vector.dot(curve.eval(U, tl) - q, curve.derivative(U, tl, 1))
	check(dl > dglobal + 1 and (math.abs(g) <= 1e-9 or tl == 0), `a local minimum on the other arm ({dl} vs {dglobal})`)

	-- From anywhere it ends at a critical point (or an end) no farther away than where it
	-- started: steps that would increase the distance are shortened.
	math.randomseed(203)
	local descends, critical = true, true
	for _ = 1, 200 do
		local qr = V(8 * math.random() - 1, 5 * math.random() - 1, 4 * math.random() - 2)
		local t0 = a + (b - a) * math.random()
		local start = vdist(curve.eval(C, t0), qr)
		local tn, pn, dn = curve.closest_near(C, qr, t0)
		descends = descends and dn <= start + 1e-12
		local gn = vector.dot(pn - qr, curve.derivative(C, tn, 1))
		critical = critical and (math.abs(gn) <= 1e-5 or tn == a or tn == b)
	end
	check(descends, "never ends farther away than it started")
	check(critical, "ends at a critical point of the distance or at an end")

	-- Open curves clamp at the ends; closed ones wrap around.
	local te = curve.closest_near(C, PTS[#PTS] + V(5, 0, 0), b - 0.1)
	check(te == b, "beyond the end: the end point")
	local ring = curve.catmull_rom({ V(1, 0, 0), V(0, 1, 0), V(-1, 0, 0), V(0, -1, 0) }, { closed = true })
	local ra, rb = curve.domain(ring)
	local tw = curve.closest_near(ring, V(1, 0.3, 0), rb - 0.05)
	local tg2 = curve.closest(ring, V(1, 0.3, 0))
	check(tw > ra and tw < ra + 0.5 * (rb - ra) and math.abs(tw - tg2) <= 1e-9, "closed: wraps past the end")
	local on, _, d0 = curve.closest_near(C, curve.eval(C, 1.3), 1.0)
	check(math.abs(on - 1.3) <= 1e-7 and d0 <= 1e-6, "a point on the curve")

	-- Buffers agree with vectors.
	local B = curve.catmull_rom(buffers(PTS), { space = f64.space })
	local tb, _, db = curve.closest_near(B, f64.from({ 3, 3, 3 }), 2)
	local tv, _, dv = curve.closest_near(C, V(3, 3, 3), 2)
	check(math.abs(tb - tv) <= 1e-5 and math.abs(db - dv) <= 1e-5, "buffers agree with vectors")
end)

test("splitting", function()
	local O = circle()
	local exact = curve.bspline(buffers({
		V(1, 0, 0), V(1, 1, 0), V(0, 1, 0), V(-1, 1, 0), V(-1, 0, 0), V(-1, -1, 0), V(0, -1, 0), V(1, -1, 0), V(1, 0, 0),
	}), {
		space = f64.space,
		degree = 2,
		knots = { 0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1 },
		weights = { 1, S2, 1, S2, 1, S2, 1, S2, 1 },
	})
	-- On f64 buffers splitting is exact up to rounding; on vectors the new control points
	-- are rounded to single precision, which moves the curve by about 1e-7.
	local cases: { { any } } = {
		{ curve.cubic_spline(buffers(PTS), { space = f64.space }), 1e-13 },
		{ exact, 1e-13 },
		{ curve.cubic_spline(PTS), 1e-6 },
		{ O, 1e-6 },
	}
	for _, case in cases do
		local C: any, tol: number = case[1], case[2]
		local a, b = curve.domain(C)
		local t = a + 0.37 * (b - a)
		local L, R = curve.split(C, t)
		local la, lb = curve.domain(L)
		local ra, rb = curve.domain(R)
		check(la == a and lb == t and ra == t and rb == b, "domains")
		local same = true
		for k = 0, 20 do
			local s = a + (b - a) * k / 20
			local part = if s <= t then L else R
			local p, q = curve.eval(part, s), curve.eval(C, s)
			local d = if type(p) == "vector" then vdist(p :: any, q :: any) else bdist(p :: any, q :: any)
			same = same and d <= 10 * tol
		end
		check(same, "the halves trace the curve")
		close(curve.length(L) + curve.length(R), curve.length(C), tol, "lengths add up")
		local Lb, Rb = curve.split(C, C.breaks[2])
		check(#Lb.breaks == 2 and #Rb.breaks == #C.breaks - 1, "at a break: whole segments")
	end
	check(not pcall(function()
		curve.split(curve.bezier({ V(0, 0, 0), V(1, 0, 0) }), 1)
	end), "t must be inside")
end)

test("rotation-minimizing frames", function()
	-- On a planar curve the rotation-minimizing normal can stay the plane's normal.
	local planar = curve.cubic_spline({ V(0, 0, 0), V(1, 2, 0), V(3, 1, 0), V(4, 3, 0), V(6, 0, 0) })
	local _, ts = curve.sample(planar, 200)
	local F = curve.frames(planar, ts, V(0, 0, 1))
	local stays = 0
	for i = 1, #ts do
		stays = math.max(stays, vdist(F.normals[i], V(0, 0, 1)))
	end
	check(stays <= 1e-5, `planar: the normal stays (0, 0, 1) ({stays})`)
	-- On a space curve: orthonormal, and the frame does not rotate about the tangent:
	-- dr/ds . b = 0.
	local C = curve.catmull_rom(PTS)
	local _, ts2 = curve.sample(C, 400)
	local G = curve.frames(C, ts2)
	local orth, twist = 0, 0
	local bn = G.binormals :: { vector }
	for i = 1, #ts2 do
		local T, r, b = G.tangents[i], G.normals[i], bn[i]
		orth = math.max(orth, math.abs(vector.dot(T, r)), math.abs(vector.magnitude(r) - 1), vdist(vector.cross(T, r), b))
		if i > 1 and i < #ts2 then
			local dr = G.normals[i + 1] - G.normals[i - 1]
			twist = math.max(twist, math.abs(vector.dot(dr, b)) / vector.magnitude(G.tangents[i + 1] - G.tangents[i - 1] + V(1e-9, 0, 0)))
		end
	end
	check(orth <= 1e-5, `orthonormal frames ({orth})`)
	check(twist <= 1e-2, `no rotation about the tangent ({twist})`)
	-- Buffers give the same frames.
	local Cb = curve.catmull_rom(buffers(PTS), { space = f64.space })
	local Gb = curve.frames(Cb, ts2, f64.from({ G.normals[1].x, G.normals[1].y, G.normals[1].z }))
	local diff = 0
	for i = 1, #ts2 do
		diff = math.max(diff, vdist(bvec(Gb.normals[i]), G.normals[i]))
	end
	check(diff <= 1e-4 and Gb.binormals == nil, `buffers agree ({diff})`)
end)

return nil
