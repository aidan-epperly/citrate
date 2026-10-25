--!strict
--!native
--!optimize 2

-- Rational approximation by the AAA algorithm (Nakatsukasa, Sete and Trefethen, "The AAA
-- algorithm for rational approximation", 2018).
--
-- The approximation is in barycentric form,
--   r(z) = sum_j w_j f_j / (z - z_j)  /  sum_j w_j / (z - z_j),
-- which interpolates f at the support points z_j (where w_j ~= 0). Starting from none, each
-- step adds the sample point where the error is largest as a support point, then chooses
-- the weights w (a unit vector) to minimize the linearized residual |f D - N| on the other
-- samples: the right singular vector of the Loewner matrix (F_i - f_j) / (Z_i - z_j) for
-- its smallest singular value. It stops when the error on the samples is within the
-- tolerance. Poles and zeros are the finite eigenvalues of (m + 1) by (m + 1) arrowhead
-- pencils; poles with tiny residues (Froissart doublets, artifacts of rounding) are
-- removed afterwards, with their nearest support points, and the weights recomputed.
--
-- The sample points can be anywhere in the complex plane: an interval, a circle, the
-- boundary of a region. The result is accurate near the samples, and its poles show where
-- the function's singularities are.

local c64 = require("./c64")
local types = require("./types")

type RationalOptions = types.RationalOptions
type Rational = types.Rational

-- Complex helpers on (re, im) pairs.
local function cdiv(ar: number, ai: number, br: number, bi: number): (number, number)
	if math.abs(br) >= math.abs(bi) then
		local t = bi / br
		local d = br + bi * t
		return (ar + ai * t) / d, (ai - ar * t) / d
	end
	local t = br / bi
	local d = bi + br * t
	return (ar * t + ai) / d, (ai * t - ar) / d
end

local function cabs(re: number, im: number): number
	return math.sqrt(re * re + im * im)
end

-- The barycentric r at (x, y) from support points (zr, zi), values (fr, fi) and weights
-- (wr, wi); at a support point, its value.
local function barycentric(zr: { number }, zi: { number }, fr: { number }, fi: { number }, wr: { number }, wi: { number }, x: number, y: number): (number, number)
	local nr, ni, dr, di = 0, 0, 0, 0
	for j = 1, #zr do
		local ur, ui = x - zr[j], y - zi[j]
		if ur == 0 and ui == 0 then
			return fr[j], fi[j]
		end
		-- c = w / (z - z_j)
		local cr, ci = cdiv(wr[j], wi[j], ur, ui)
		dr += cr
		di += ci
		nr += cr * fr[j] - ci * fi[j]
		ni += cr * fi[j] + ci * fr[j]
	end
	return cdiv(nr, ni, dr, di)
end

-- The weights: the right singular vector, for the smallest singular value, of the Loewner
-- matrix on the sample points not in the support (rows) and the support points (columns).
local function weights(Zr: { number }, Zi: { number }, Fr: { number }, Fi: { number }, inSupport: { boolean }, zr: { number }, zi: { number }, fr: { number }, fi: { number }): ({ number }, { number })
	local m = #zr
	local rows: { number } = {}
	for i = 1, #Zr do
		if not inSupport[i] then
			table.insert(rows, i)
		end
	end
	local wr, wi = table.create(m, 0), table.create(m, 0)
	if #rows == 0 then
		wr[m] = 1
		return wr, wi
	end
	local A = c64.matrix(#rows, m)
	for r, i in rows do
		for j = 1, m do
			-- (F_i - f_j) / (Z_i - z_j)
			local ar, ai = cdiv(Fr[i] - fr[j], Fi[i] - fi[j], Zr[i] - zr[j], Zi[i] - zi[j])
			c64.matset(A, r, j, ar, ai)
		end
	end
	if #rows < m then
		-- more unknowns than equations: pad with zero rows so the SVD has a null vector
		local P = c64.matrix(m, m)
		for r = 1, #rows do
			for j = 1, m do
				local ar, ai = c64.matget(A, r, j)
				c64.matset(P, r, j, ar, ai)
			end
		end
		A = P
	end
	local S = c64.svd_copy(A, true)
	local V = S.v :: any
	for j = 1, m do
		wr[j], wi[j] = c64.matget(V, j, m)
	end
	return wr, wi
end

-- Finite eigenvalues of the arrowhead pencil ([0 c^T; 1 diag(z)], diag(0, 1, ..., 1)):
-- the roots of sum_j c_j / (z - z_j) (poles for c = w, zeros for c = w f). The pencil has
-- at least two infinite eigenvalues, which are dropped.
local function arrowhead(zr: { number }, zi: { number }, cr: { number }, ci: { number }): ({ number }, { number })
	local m = #zr
	local E, B = c64.matrix(m + 1, m + 1), c64.matrix(m + 1, m + 1)
	for j = 1, m do
		c64.matset(E, 1, j + 1, cr[j], ci[j])
		c64.matset(E, j + 1, 1, 1, 0)
		c64.matset(E, j + 1, j + 1, zr[j], zi[j])
		c64.matset(B, j + 1, j + 1, 1, 0)
	end
	local G = c64.eig_gen(E, B)
	type Eigen = { k: number, a: number, b: number, ar: number, ai: number, br: number, bi: number }
	local order: { Eigen } = {}
	for k = 1, m + 1 do
		local ar, ai = c64.get(G.alpha, k)
		local br, bi = c64.get(G.beta, k)
		local a, b = cabs(ar, ai), cabs(br, bi)
		table.insert(order, { k = k, a = a, b = b, ar = ar, ai = ai, br = br, bi = bi })
	end
	-- largest |beta| / |alpha| first; at most m - 1 finite eigenvalues
	table.sort(order, function(p: Eigen, q: Eigen): boolean
		return p.b * q.a > q.b * p.a
	end)
	-- An eigenvalue counts as infinite beyond 1e10 times the support's scale: the pencil's
	-- infinite eigenvalues come out of QZ with |beta| at the rounding level, not exactly 0,
	-- and a pole that far away has no visible effect near the samples.
	local zscale = 1
	for j = 1, m do
		zscale = math.max(zscale, cabs(zr[j], zi[j]))
	end
	local outr, outi = {}, {}
	for idx = 1, math.min(m - 1, #order) do
		local e = order[idx]
		if e.a < 1e10 * zscale * e.b then
			local vr, vi = cdiv(e.ar, e.ai, e.br, e.bi)
			table.insert(outr, vr)
			table.insert(outi, vi)
		end
	end
	return outr, outi
end

-- Residues at the poles: N(p) / D'(p), with D'(p) = -sum w_j / (p - z_j)^2.
local function residues(zr: { number }, zi: { number }, fr: { number }, fi: { number }, wr: { number }, wi: { number }, pr: { number }, pi: { number }): ({ number }, { number })
	local rr, ri = table.create(#pr, 0), table.create(#pr, 0)
	for k = 1, #pr do
		local nr, ni, dr, di = 0, 0, 0, 0
		for j = 1, #zr do
			local ur, ui = pr[k] - zr[j], pi[k] - zi[j]
			local cr, ci = cdiv(wr[j], wi[j], ur, ui) -- w / (p - z)
			nr += cr * fr[j] - ci * fi[j]
			ni += cr * fi[j] + ci * fr[j]
			local er, ei = cdiv(cr, ci, ur, ui) -- w / (p - z)^2
			dr -= er
			di -= ei
		end
		rr[k], ri[k] = cdiv(nr, ni, dr, di)
	end
	return rr, ri
end

local function pack(re: { number }, im: { number }): buffer
	return c64.from(re, im)
end

-- The AAA rational approximation of the values F (a c64 array) at the sample points Z (a
-- c64 array of distinct points; for real data, c64.complexify(x)). options.tol is the
-- relative tolerance on the samples (default 1e-13), options.mmax the most support
-- points (default 100), options.cleanup (default true) removes Froissart doublets.
local function aaa(Z: buffer, F: buffer, options: RationalOptions?): Rational
	local o: RationalOptions = options or {}
	local tol = o.tol or 1e-13
	local mmax = o.mmax or 100
	local M = c64.length(Z)
	assert(c64.length(F) == M and M > 0, "rational.aaa: need as many values as points")
	local Zr, Zi, Fr, Fi = table.create(M, 0), table.create(M, 0), table.create(M, 0), table.create(M, 0)
	local fmax = 0
	local meanr, meani = 0, 0
	for i = 1, M do
		Zr[i], Zi[i] = c64.get(Z, i)
		Fr[i], Fi[i] = c64.get(F, i)
		fmax = math.max(fmax, cabs(Fr[i], Fi[i]))
		meanr += Fr[i] / M
		meani += Fi[i] / M
	end
	local inSupport = table.create(M, false)
	local zr, zi, fr, fi = {}, {}, {}, {}
	local wr: { number }, wi: { number } = {}, {}
	local Rr, Ri = table.create(M, meanr), table.create(M, meani)
	local errors = {}
	local function evaluate_all()
		for i = 1, M do
			if inSupport[i] then
				Rr[i], Ri[i] = Fr[i], Fi[i]
			else
				Rr[i], Ri[i] = barycentric(zr, zi, fr, fi, wr, wi, Zr[i], Zi[i])
			end
		end
	end
	local function max_error(): (number, number)
		local worst, at = -1, 1
		for i = 1, M do
			local e = cabs(Fr[i] - Rr[i], Fi[i] - Ri[i])
			if e ~= e then
				e = math.huge
			end
			if e > worst then
				worst, at = e, i
			end
		end
		return worst, at
	end
	local worst, at = max_error()
	for _ = 1, math.min(mmax, M) do
		if worst <= tol * fmax then
			break
		end
		inSupport[at] = true
		table.insert(zr, Zr[at])
		table.insert(zi, Zi[at])
		table.insert(fr, Fr[at])
		table.insert(fi, Fi[at])
		wr, wi = weights(Zr, Zi, Fr, Fi, inSupport, zr, zi, fr, fi)
		evaluate_all()
		worst, at = max_error()
		table.insert(errors, worst / math.max(fmax, 1e-300))
	end

	local function analyze(): ({ number }, { number }, { number }, { number }, { number }, { number })
		local pr, pi = arrowhead(zr, zi, wr, wi)
		local cr, ci = table.create(#zr, 0), table.create(#zr, 0)
		for j = 1, #zr do
			cr[j] = wr[j] * fr[j] - wi[j] * fi[j]
			ci[j] = wr[j] * fi[j] + wi[j] * fr[j]
		end
		local qr, qi = arrowhead(zr, zi, cr, ci)
		local rr, ri = residues(zr, zi, fr, fi, wr, wi, pr, pi)
		return pr, pi, rr, ri, qr, qi
	end
	local pr, pi, rr, ri, qr, qi = {}, {}, {}, {}, {}, {}
	if #zr > 1 then
		pr, pi, rr, ri, qr, qi = analyze()
	end

	-- Froissart doublets: poles with residues below 1e-13 |F| (by default), each with its
	-- nearest support point removed; then the weights are recomputed by least squares.
	local removed = 0
	if o.cleanup ~= false and #pr > 0 then
		local drop = {}
		for k = 1, #pr do
			if cabs(rr[k], ri[k]) < 1e-13 * fmax then
				local best, bj = math.huge, 0
				for j = 1, #zr do
					local d = cabs(pr[k] - zr[j], pi[k] - zi[j])
					if d < best and not drop[j] then
						best, bj = d, j
					end
				end
				if bj > 0 then
					drop[bj] = true
				end
			end
		end
		local nzr, nzi, nfr, nfi = {}, {}, {}, {}
		for j = 1, #zr do
			if drop[j] then
				removed += 1
			else
				table.insert(nzr, zr[j])
				table.insert(nzi, zi[j])
				table.insert(nfr, fr[j])
				table.insert(nfi, fi[j])
			end
		end
		if removed > 0 and #nzr > 0 then
			zr, zi, fr, fi = nzr, nzi, nfr, nfi
			for i = 1, M do
				inSupport[i] = false
			end
			for j = 1, #zr do
				for i = 1, M do
					if Zr[i] == zr[j] and Zi[i] == zi[j] then
						inSupport[i] = true
					end
				end
			end
			wr, wi = weights(Zr, Zi, Fr, Fi, inSupport, zr, zi, fr, fi)
			evaluate_all()
			if #zr > 1 then
				pr, pi, rr, ri, qr, qi = analyze()
			else
				pr, pi, rr, ri, qr, qi = {}, {}, {}, {}, {}, {}
			end
		end
	end
	local final = max_error()

	return {
		support = pack(zr, zi),
		values = pack(fr, fi),
		weights = pack(wr, wi),
		poles = pack(pr, pi),
		residues = pack(rr, ri),
		zeros = pack(qr, qi),
		errors = errors,
		error = final / math.max(fmax, 1e-300),
		removed = removed,
		eval = function(x: number, y: number?): (number, number)
			if #zr == 0 then
				return meanr, meani
			end
			return barycentric(zr, zi, fr, fi, wr, wi, x, y or 0)
		end,
	}
end

return {
	aaa = aaa,
}
