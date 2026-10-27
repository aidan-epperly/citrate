--!strict
--!native
--!optimize 2

-- Pieces shared by the ODE solvers (ode.luau and odestiff.luau): error norms, the
-- continuous extension of a step, Hairer's starting step and event location.

local types = require("./types")

type Space<T> = types.Space<T>
type OdeFunction<T> = types.OdeFunction<T>

local EPS = 2 ^ -52

-- Workspace: a value of the space to write into, or a fresh one.
local function similar<T>(S: Space<T>, x: T, w: T?): T
	if w ~= nil then
		return w :: T
	end
	return S.similar(x)
end

-- y + h sum_i coeffs[i] K[i] (over the nonzero coefficients), into out.
local function combine<T>(S: Space<T>, out: T, y: T, h: number, K: { T }, coeffs: { [number]: number }, count: number): T
	local o = S.copy(out, y)
	for i = 1, count do
		local c = coeffs[i]
		if c and c ~= 0 then
			o = S.axpy(o, h * c, K[i], o)
		end
	end
	return o
end

-- h sum_i coeffs[i] K[i] (over the nonzero coefficients), into out: combine without the
-- base value (out's old contents, NaN or not, do not matter).
local function combine_zero<T>(S: Space<T>, out: T, h: number, K: { T }, coeffs: { [number]: number }, count: number): T
	local o: T? = nil
	for i = 1, count do
		local c = coeffs[i]
		if c and c ~= 0 then
			o = if o == nil then S.scale(out, h * c, K[i]) else S.axpy(o, h * c, K[i], o)
		end
	end
	if o == nil then
		return S.scale(out, 0, K[1])
	end
	return o :: T
end

-- The weighted error norm |e / (atol + rtol max(|y|, |yn|))| (Space.errnorm, or
-- whole-vector norms without it).
local function weighted<T>(S: Space<T>, e: T, y: T, yn: T, atol: number, rtol: number): number
	if S.errnorm then
		return S.errnorm(e, y, yn, atol, rtol)
	end
	return S.norm(e) / (atol + rtol * math.max(S.norm(y), S.norm(yn)))
end

-- One step's continuous extension, from t0 to t0 + h: kind "poly" (y0 + h sum_j q_j
-- theta^j), "dop853" (Hairer's nested form in theta and 1 - theta), "hermite" (cubic
-- Hermite from q = { y1, f0, f1 }), or any other kind with an eval function.
export type Piece<T> = { t0: number, h: number, y0: T, kind: string, q: { T }, eval: ((t: number, out: T?) -> T)? }

local function piece_eval<T>(S: Space<T>, P: Piece<T>, t: number, out: T?): T
	if P.eval then
		return P.eval(t, out)
	end
	local x = if P.h == 0 then 0 else (t - P.t0) / P.h
	local q = P.q
	local o = S.copy(similar(S, P.y0, out), P.y0)
	if P.kind == "poly" then
		local pw = x
		for j = 1, #q do
			o = S.axpy(o, P.h * pw, q[j], o)
			pw *= x
		end
		return o
	elseif P.kind == "dop853" then
		local acc = S.scale(S.similar(P.y0), 0, P.y0)
		local k = 0
		for i = #q, 1, -1 do
			acc = S.add(acc, acc, q[i])
			acc = S.scale(acc, if k % 2 == 0 then x else 1 - x, acc)
			k += 1
		end
		return S.add(o, o, acc)
	end
	local y1, f0, f1 = q[1], q[2], q[3]
	local h00 = 2 * x ^ 3 - 3 * x ^ 2 + 1
	local h10 = x ^ 3 - 2 * x ^ 2 + x
	local h01 = -2 * x ^ 3 + 3 * x ^ 2
	local h11 = x ^ 3 - x ^ 2
	o = S.scale(o, h00, o)
	o = S.axpy(o, h01, y1, o)
	o = S.axpy(o, h10 * P.h, f0, o)
	o = S.axpy(o, h11 * P.h, f1, o)
	return o
end

-- Hairer's initial step size (Solving ODEs I, II.4) for a method whose error estimate
-- has order q, from (t0, y0) with f0 = f(t0, y0). Returns the step and the number of f
-- evaluations it used (one).
local function initial_step<T>(
	S: Space<T>,
	f: OdeFunction<T>,
	t0: number,
	y0: T,
	f0: T,
	direction: number,
	span: number,
	atol: number,
	rtol: number,
	maxstep: number,
	q: number
): (number, number)
	local d0 = weighted(S, y0, y0, y0, atol, rtol)
	local d1 = weighted(S, f0, y0, y0, atol, rtol)
	local h0 = if d0 < 1e-5 or d1 < 1e-5 then 1e-6 else 0.01 * d0 / d1
	h0 = math.min(h0, span)
	local y1 = S.axpy(S.similar(y0), direction * h0, f0, y0)
	local f1 = f(t0 + direction * h0, y1, S.similar(y0))
	local d2 = weighted(S, S.sub(S.similar(y0), f1, f0), y0, y0, atol, rtol) / h0
	local h1 = if math.max(d1, d2) <= 1e-15 then math.max(1e-6, h0 * 1e-3) else (0.01 / math.max(d1, d2)) ^ (1 / (q + 1))
	return math.min(100 * h0, h1, span, maxstep), 1
end

-- A zero of g in [a, b], where g(a) and g(b) have opposite signs: the Illinois variant
-- of regula falsi.
local function bracket_root(g: (number) -> number, a: number, b: number, ga: number, gb: number): number
	local side = 0
	for _ = 1, 100 do
		local t = (a * gb - b * ga) / (gb - ga)
		if not (t > math.min(a, b) and t < math.max(a, b)) then
			t = (a + b) / 2
		end
		local gt = g(t)
		if gt == 0 or math.abs(b - a) <= 4 * EPS * math.max(math.abs(a), math.abs(b), 1e-300) then
			return t
		end
		if (gt > 0) == (gb > 0) then
			b, gb = t, gt
			if side == -1 then
				ga /= 2
			end
			side = -1
		else
			a, ga = t, gt
			if side == 1 then
				gb /= 2
			end
			side = 1
		end
	end
	return (a + b) / 2
end

return {
	EPS = EPS,
	similar = similar,
	combine = combine,
	combine_zero = combine_zero,
	weighted = weighted,
	piece_eval = piece_eval,
	initial_step = initial_step,
	bracket_root = bracket_root,
}
