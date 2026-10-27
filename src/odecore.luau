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
-- Hermite from q = { y1, f0, f1 }), "rodas" (Rodas5P's: (1 - theta) y0 + theta (y1 +
-- (1 - theta)(k1 + theta (k2 + theta k3))) from q = { y1, k1, k2, k3 }), "ndf" (the BDF
-- interpolant at the step's end t1 = t0 + h: q = { D_0 .. D_order }, y(t) = D_0 + sum_j D_j
-- prod_(m <= j) (t - (t1 - (m - 1) h)) / (m h)), or any other kind with an eval function.
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
	elseif P.kind == "rodas" then
		-- (1 - x) y0 + x y1 + x (1 - x) k1 + x^2 (1 - x) k2 + x^3 (1 - x) k3
		local w = 1 - x
		o = S.scale(o, w, o)
		o = S.axpy(o, x, q[1], o)
		o = S.axpy(o, x * w, q[2], o)
		o = S.axpy(o, x * x * w, q[3], o)
		return S.axpy(o, x * x * x * w, q[4], o)
	elseif P.kind == "ndf" then
		local t1 = P.t0 + P.h
		o = S.copy(o, q[1])
		local pj = 1
		for j = 1, #q - 1 do
			pj *= (t - (t1 - (j - 1) * P.h)) / (j * P.h)
			o = S.axpy(o, pj, q[j + 1], o)
		end
		return o
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

-- Dense output storage: the pieces of a solution, kept compactly. A piece's start value
-- y0 is the solution's stored state at the start of its step (states(k) for piece k), so
-- only its coefficients q are kept. With buffer values they go, byte for byte, into one
-- growing buffer; with vectors, as 3 doubles each; other values (and pieces with an eval
-- function) are kept as they are. A table and a closure per step were most of a solution's
-- memory (about 1.4 KB a step for dp5 on 3 unknowns, against 120 bytes of coefficients).
local KINDS = { "poly", "dop853", "hermite", "rodas", "ndf" }
local KIND_CODE: { [string]: number } = {}
for i, k in KINDS do
	KIND_CODE[k] = i
end
local META = 24 -- per piece: t0, h (f64), the data offset (u32), kind and count (u16)

export type PieceStore<T> = {
	count: number,
	add: (self: PieceStore<T>, P: Piece<T>) -> (),
	finish: (self: PieceStore<T>) -> (),
	eval: (self: PieceStore<T>, t: number, direction: number, out: T?) -> T,
}

local function piece_store<T>(S: Space<T>, sample: T, states: (number) -> T): PieceStore<T>
	local mode = if type(sample) == "buffer" then "bytes" elseif type(sample) == "vector" then "vector" else "tables"
	local width = if mode == "bytes" then buffer.len(sample :: any) else 24
	local meta, data = buffer.create(META * 16), buffer.create(width * 64)
	local used = 0
	local kept: { [number]: Piece<T> } = {} -- pieces stored as they are
	local scratch: Piece<T> = { t0 = 0, h = 0, y0 = sample, kind = "poly", q = {} }
	local store: any = { count = 0 }
	local function grow(b: buffer, need: number): buffer
		if buffer.len(b) >= need then
			return b
		end
		local nb = buffer.create(math.max(need, 2 * buffer.len(b)))
		buffer.copy(nb, 0, b)
		return nb
	end
	function store.add(self: any, P: Piece<T>)
		local k = self.count + 1
		self.count = k
		meta = grow(meta, META * k)
		local m = META * (k - 1)
		buffer.writef64(meta, m, P.t0)
		buffer.writef64(meta, m + 8, P.h)
		local code = KIND_CODE[P.kind]
		if mode == "tables" or P.eval ~= nil or code == nil then
			-- kept as a piece: a copy unless it computes itself (steppers refill theirs)
			if P.eval == nil then
				local q = {}
				for i, v in P.q do
					q[i] = S.copy(S.similar(v), v)
				end
				P = { t0 = P.t0, h = P.h, y0 = S.copy(S.similar(P.y0), P.y0), kind = P.kind, q = q } :: Piece<T>
			end
			kept[k] = P
			buffer.writeu16(meta, m + 20, 0)
			return
		end
		local q = P.q
		buffer.writeu32(meta, m + 16, used)
		buffer.writeu16(meta, m + 20, code)
		buffer.writeu16(meta, m + 22, #q)
		data = grow(data, used + width * #q)
		for _, v in q do
			if mode == "bytes" then
				buffer.copy(data, used, v :: any)
			else
				local w = v :: any
				buffer.writef64(data, used, w.x)
				buffer.writef64(data, used + 8, w.y)
				buffer.writef64(data, used + 16, w.z)
			end
			used += width
		end
	end
	-- Trims the buffers to their contents (once the solution is complete).
	function store.finish(self: any)
		local m = buffer.create(META * self.count)
		buffer.copy(m, 0, meta, 0, META * self.count)
		meta = m
		local d = buffer.create(used)
		buffer.copy(d, 0, data, 0, used)
		data = d
	end
	-- The solution at t from the piece whose step contains it.
	function store.eval(self: any, t: number, direction: number, out: T?): T
		local a, b = 1, self.count
		while b > a do
			local mid = (a + b) // 2
			local m = META * (mid - 1)
			if direction * (t - (buffer.readf64(meta, m) + buffer.readf64(meta, m + 8))) > 0 then
				a = mid + 1
			else
				b = mid
			end
		end
		local P = kept[a]
		if P then
			return piece_eval(S, P, t, out)
		end
		local m = META * (a - 1)
		local off, code, count = buffer.readu32(meta, m + 16), buffer.readu16(meta, m + 20), buffer.readu16(meta, m + 22)
		local Q = scratch
		Q.t0, Q.h, Q.kind, Q.y0 = buffer.readf64(meta, m), buffer.readf64(meta, m + 8), KINDS[code], states(a)
		local q = Q.q
		for i = 1, count do
			if mode == "bytes" then
				local v = q[i] :: any
				if v == nil then
					v = S.similar(sample)
					q[i] = v
				end
				buffer.copy(v, 0, data, off, width)
			else
				q[i] = vector.create(buffer.readf64(data, off), buffer.readf64(data, off + 8), buffer.readf64(data, off + 16)) :: any
			end
			off += width
		end
		for i = count + 1, #q do
			q[i] = nil
		end
		return piece_eval(S, Q, t, out)
	end
	return store
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
	piece_store = piece_store,
	initial_step = initial_step,
	bracket_root = bracket_root,
}
