--!strict
--!native
--!optimize 2

-- Iterative (Krylov subspace) solvers for A x = b, and eigensolvers (see eigsh and eigs
-- below), generic over how vectors are stored:
-- they touch vectors only through a Space (CG, MINRES) or ComplexSpace (GMRES, BiCGSTAB)
-- and A only through an Operator, so A can be sparse, dense, structured or matrix-free,
-- and the vectors can be buffers of any element type (or vectors, for CG and MINRES).
--
-- Every solver takes (space, A, b, options) and returns (x, info); see types.KrylovOptions
-- and types.KrylovInfo. info.residual is the true relative residual |b - A x| / |b| of the
-- returned x, recomputed at the end, and info.converged says whether it is within tol.

local types = require("./types")
local complex64 = require("./complex64")
local f64 = require("./f64")
local c64 = require("./c64")

type Space<T> = types.Space<T>
type ComplexSpace<T> = types.ComplexSpace<T>
type Operator<T> = types.Operator<T>
type Options<T> = types.KrylovOptions<T>
type Info = types.KrylovInfo

local DEFAULT_TOL = 1e-5
local DEFAULT_MAXITER = 1000

local function hypot(a: number, b: number): number
	a, b = math.abs(a), math.abs(b)
	if a < b then
		a, b = b, a
	end
	if a == 0 or a == math.huge then
		return a
	end
	local r = b / a
	return a * math.sqrt(1 + r * r)
end

local function zero<T>(S: Space<T>, like: T): T
	return S.scale(S.similar(like), 0, like)
end

-- The starting point: x0 (overwritten later) or zero.
local function start<T>(S: Space<T>, b: T, o: Options<T>): T
	return (if o.x0 ~= nil then o.x0 else zero(S, b)) :: T
end

-- The returned info, with the true residual of x.
local function finish<T>(S: Space<T>, A: Operator<T>, b: T, x: T, bnorm: number, tol: number, iterations: number): (T, Info)
	local t = A(S.similar(b), x)
	t = S.sub(t, b, t)
	local residual = S.norm(t) / bnorm
	return x, { converged = residual <= tol, iterations = iterations, residual = residual }
end

-- A zero right-hand side has the solution zero.
local function trivial<T>(S: Space<T>, x: T): (T, Info)
	return S.scale(x, 0, x), { converged = true, iterations = 0, residual = 0 }
end

-- Conjugate gradients, for symmetric (Hermitian) positive definite A. The optional
-- preconditioner must be symmetric positive definite too. Stops early if A turns out
-- not to be positive definite (p^H A p <= 0).
local function cg<T>(S: Space<T>, A: Operator<T>, b: T, options: Options<T>?): (T, Info)
	local o: Options<T> = options or {}
	local tol, maxiter, M = o.tol or DEFAULT_TOL, o.maxiter or DEFAULT_MAXITER, o.precond
	local x = start(S, b, o)
	local bnorm = S.norm(b)
	if bnorm == 0 then
		return trivial(S, x)
	end
	local Ap = A(S.similar(b), x)
	local r = S.sub(S.similar(b), b, Ap)
	local z = if M then M(S.similar(b), r) else r
	local p = S.copy(S.similar(b), z)
	-- For Hermitian A and preconditioner these inner products are real, so the real
	-- inner product loses nothing.
	local rz = S.dot(r, z)
	local iterations = 0
	local done = S.norm(r) <= tol * bnorm
	while not done and iterations < maxiter do
		iterations += 1
		Ap = A(Ap, p)
		local pAp = S.dot(p, Ap)
		if not (pAp > 0) then
			break
		end
		local alpha = rz / pAp
		-- One fused pass when the space has it (rr = |r|^2), else two axpys.
		local rr: number? = nil
		if S.cg_update then
			rr = S.cg_update(x, r, p, Ap, alpha)
		else
			x = S.axpy(x, alpha, p, x)
			r = S.axpy(r, -alpha, Ap, r)
		end
		-- sqrt(rr) unless it may have overflowed or lost digits to underflow.
		local exact = rr ~= nil and rr > 2 ^ -500 and rr < math.huge
		local rnorm = if exact then math.sqrt(rr :: number) else S.norm(r)
		if rnorm <= tol * bnorm then
			break
		end
		z = if M then M(z, r) else r
		local rz_new = if M == nil and exact then rr :: number else S.dot(r, z)
		p = S.axpy(p, rz_new / rz, p, z)
		rz = rz_new
	end
	return finish(S, A, b, x, bnorm, tol, iterations)
end

-- MINRES (Paige and Saunders, as in SciPy), for symmetric (Hermitian) A, which may be
-- indefinite. It minimizes the residual over the Krylov subspace. The optional
-- preconditioner must be symmetric (Hermitian) positive definite; MINRES then minimizes
-- the residual in the M^-1 norm, so when that estimate meets tol the true residual is
-- checked as well and the iteration continues if it is not yet small enough.
local function minres<T>(S: Space<T>, A: Operator<T>, b: T, options: Options<T>?): (T, Info)
	local o: Options<T> = options or {}
	local tol, maxiter, M = o.tol or DEFAULT_TOL, o.maxiter or DEFAULT_MAXITER, o.precond
	local x = start(S, b, o)
	local bnorm = S.norm(b)
	if bnorm == 0 then
		return trivial(S, x)
	end
	-- Lanczos vectors: r2 is the current residual direction and r1 the previous one; y is
	-- M^-1 r2 (r2 itself without a preconditioner); work rotates through the buffers.
	local r1 = A(S.similar(b), x)
	r1 = S.sub(r1, b, r1)
	local r2 = S.copy(S.similar(b), r1)
	local ybuf = if M then S.similar(b) else nil
	local y = if M then M(ybuf :: T, r2) else r2
	local work = S.similar(b)
	local v = S.similar(b)
	local w, w1, w2 = zero(S, b), zero(S, b), zero(S, b)
	local beta1sq = S.dot(r2, y)
	assert(beta1sq >= 0, "minres: the preconditioner is not positive definite")
	local beta = math.sqrt(beta1sq)
	local beta1 = beta
	local oldb, dbar, epsln, phibar = 0, 0, 0, beta
	local cs, sn = -1, 0
	local iterations = 0
	local rtrue = S.similar(b)
	-- Is the true residual within tol?
	local function truly_converged(): boolean
		rtrue = A(rtrue, x)
		rtrue = S.sub(rtrue, b, rtrue)
		return S.norm(rtrue) <= tol * bnorm
	end
	if beta == 0 or (M == nil and beta <= tol * bnorm) then
		return finish(S, A, b, x, bnorm, tol, iterations)
	end
	while iterations < maxiter do
		iterations += 1
		-- Lanczos step: v = y / beta, then A v - (beta / oldb) r1 - (alpha / beta) r2.
		v = S.scale(v, 1 / beta, y)
		local z = A(work, v)
		if iterations >= 2 then
			z = S.axpy(z, -(beta / oldb), r1, z)
		end
		local alpha = S.dot(v, z)
		z = S.axpy(z, -(alpha / beta), r2, z)
		work, r1, r2 = r1, r2, z
		y = if M then M(ybuf :: T, r2) else r2
		oldb = beta
		local betasq = S.dot(r2, y)
		assert(betasq >= 0, "minres: the preconditioner is not positive definite")
		beta = math.sqrt(betasq)

		-- Apply the previous rotation, then make the new one.
		local oldeps = epsln
		local delta = cs * dbar + sn * alpha
		local gbar = sn * dbar - cs * alpha
		epsln = sn * beta
		dbar = -cs * beta
		local gamma = math.max(hypot(gbar, beta), 1e-300)
		cs, sn = gbar / gamma, beta / gamma
		local phi = cs * phibar
		phibar = sn * phibar

		-- Search direction w = (v - oldeps w1 - delta w2) / gamma, then the update.
		local free = w1
		w1, w2 = w2, w
		w = S.axpy(free, -oldeps, w1, v)
		w = S.axpy(w, -delta, w2, w)
		w = S.scale(w, 1 / gamma, w)
		x = S.axpy(x, phi, w, x)

		-- phibar is the residual norm (in the M^-1 norm with a preconditioner); beta = 0
		-- means the Krylov space is exhausted and x is exact.
		if beta == 0 then
			break
		end
		if M == nil then
			if math.abs(phibar) <= tol * bnorm then
				break
			end
		elseif math.abs(phibar) <= tol * beta1 and truly_converged() then
			break
		end
	end
	return finish(S, A, b, x, bnorm, tol, iterations)
end

-- Restarted GMRES(m), m = options.restart (default 30), for general A, with optional right
-- preconditioning. Arnoldi uses modified Gram-Schmidt run twice, so the basis stays
-- orthonormal to working precision. maxiter counts inner iterations (matrix-vector
-- products), over all restarts.
local function gmres<T>(S: ComplexSpace<T>, A: Operator<T>, b: T, options: Options<T>?): (T, Info)
	local o: Options<T> = options or {}
	local tol, maxiter, M = o.tol or DEFAULT_TOL, o.maxiter or DEFAULT_MAXITER, o.precond
	local m = o.restart or 30
	assert(m >= 1, "restart must be at least 1")
	local x = start(S, b, o)
	local bnorm = S.norm(b)
	if bnorm == 0 then
		return trivial(S, x)
	end
	local V: { T } = {}
	local r, t, z = S.similar(b), S.similar(b), S.similar(b)
	local total = 0
	local done = false
	while not done and total < maxiter do
		t = A(t, x)
		r = S.sub(r, b, t)
		local beta = S.norm(r)
		if beta <= tol * bnorm then
			break
		end
		V[1] = S.scale((V[1] or S.similar(b)) :: T, 1 / beta, r)
		-- g = beta e_1 rotated along; H is kept column by column (hr[j][i] = H_ij).
		local gr, gi = { beta }, { 0 }
		local hr, hi = {}, {}
		local cs, snr, sni = {}, {}, {}
		local k = 0
		for j = 1, m do
			if total >= maxiter then
				break
			end
			total += 1
			local zj = if M then M(z, V[j]) else V[j]
			local w = A((V[j + 1] or S.similar(b)) :: T, zj)
			local colr, coli = table.create(j + 1, 0), table.create(j + 1, 0)
			for _ = 1, 2 do
				for i = 1, j do
					local cr, ci = S.cdot(V[i], w)
					colr[i] += cr
					coli[i] += ci
					w = S.caxpy(w, -cr, -ci, V[i], w)
				end
			end
			local hn = S.norm(w)
			colr[j + 1], coli[j + 1] = hn, 0
			V[j + 1] = if hn ~= 0 then S.scale(w, 1 / hn, w) else w
			k = j

			-- Earlier rotations: [x; y] := [c x + s y; -conj(s) x + c y].
			for i = 1, j - 1 do
				local xr, xi, yr, yi = colr[i], coli[i], colr[i + 1], coli[i + 1]
				local c, sr, si = cs[i], snr[i], sni[i]
				colr[i], coli[i] = c * xr + sr * yr - si * yi, c * xi + sr * yi + si * yr
				colr[i + 1], coli[i + 1] = -(sr * xr + si * xi) + c * yr, -(sr * xi - si * xr) + c * yi
			end
			-- A new rotation zeroes the (real) subdiagonal entry h against the diagonal a:
			-- c = |a| / r, s = (a / |a|) h / r, r = sqrt(|a|^2 + h^2).
			local ar, ai = colr[j], coli[j]
			local amag = hypot(ar, ai)
			local c, sr, si = 1, 0, 0
			if hn ~= 0 then
				if amag == 0 then
					c, sr, si = 0, 1, 0
				else
					local rr = hypot(amag, hn)
					c = amag / rr
					sr, si = (ar / amag) * hn / rr, (ai / amag) * hn / rr
				end
			end
			cs[j], snr[j], sni[j] = c, sr, si
			colr[j], coli[j] = c * ar + sr * hn, c * ai + si * hn
			colr[j + 1], coli[j + 1] = 0, 0
			local gjr, gji = gr[j], gi[j]
			gr[j], gi[j] = c * gjr, c * gji
			gr[j + 1], gi[j + 1] = -(sr * gjr + si * gji), -(sr * gji - si * gjr)
			hr[j], hi[j] = colr, coli

			-- |g_j+1| is the residual norm of the current iterate.
			if hypot(gr[j + 1], gi[j + 1]) <= tol * bnorm or hn == 0 then
				done = true
				break
			end
		end
		if k == 0 then
			break
		end
		-- Solve the triangular system H y = g, then x += M (V y).
		local yr, yi = table.create(k, 0), table.create(k, 0)
		for i = k, 1, -1 do
			local sr, si = gr[i], gi[i]
			for l = i + 1, k do
				local hlr, hli = hr[l][i], hi[l][i]
				sr -= hlr * yr[l] - hli * yi[l]
				si -= hlr * yi[l] + hli * yr[l]
			end
			yr[i], yi[i] = complex64.div(sr, si, hr[i][i], hi[i][i])
		end
		local u = S.scale(t, 0, V[1])
		for i = 1, k do
			u = S.caxpy(u, yr[i], yi[i], V[i], u)
		end
		if M then
			u = M(z, u)
		end
		x = S.axpy(x, 1, u, x)
	end
	return finish(S, A, b, x, bnorm, tol, total)
end

-- BiCGSTAB (van der Vorst) for general A, with optional right preconditioning. Stops
-- early on a breakdown (a zero inner product).
local function bicgstab<T>(S: ComplexSpace<T>, A: Operator<T>, b: T, options: Options<T>?): (T, Info)
	local o: Options<T> = options or {}
	local tol, maxiter, M = o.tol or DEFAULT_TOL, o.maxiter or DEFAULT_MAXITER, o.precond
	local x = start(S, b, o)
	local bnorm = S.norm(b)
	if bnorm == 0 then
		return trivial(S, x)
	end
	local t = A(S.similar(b), x)
	local r = S.sub(S.similar(b), b, t)
	local rhat = S.copy(S.similar(b), r)
	local p, v = zero(S, b), zero(S, b)
	local s = S.similar(b)
	local phat, shat = S.similar(b), S.similar(b)
	local rhor, rhoi, alr, ali, omr, omi = 1, 0, 1, 0, 1, 0
	local iterations = 0
	local done = S.norm(r) <= tol * bnorm
	while not done and iterations < maxiter do
		iterations += 1
		local nr, ni = S.cdot(rhat, r)
		if nr == 0 and ni == 0 then
			break
		end
		if iterations == 1 then
			p = S.copy(p, r)
		else
			-- beta = (rho_new / rho) (alpha / omega); p = r + beta (p - omega v)
			local q1r, q1i = complex64.div(nr, ni, rhor, rhoi)
			local q2r, q2i = complex64.div(alr, ali, omr, omi)
			local br, bi = complex64.mul(q1r, q1i, q2r, q2i)
			p = S.caxpy(p, -omr, -omi, v, p)
			p = S.caxpy(p, br, bi, p, r)
		end
		rhor, rhoi = nr, ni
		local ph = if M then M(phat, p) else p
		v = A(v, ph)
		local dr, di = S.cdot(rhat, v)
		if dr == 0 and di == 0 then
			break
		end
		alr, ali = complex64.div(rhor, rhoi, dr, di)
		s = S.caxpy(s, -alr, -ali, v, r)
		if S.norm(s) <= tol * bnorm then
			x = S.caxpy(x, alr, ali, ph, x)
			break
		end
		local sh = if M then M(shat, s) else s
		t = A(t, sh)
		local tsr, tsi = S.cdot(t, s)
		local tt = S.dot(t, t)
		if tt == 0 then
			break
		end
		omr, omi = tsr / tt, tsi / tt
		x = S.caxpy(x, alr, ali, ph, x)
		x = S.caxpy(x, omr, omi, sh, x)
		r = S.caxpy(r, -omr, -omi, t, s)
		if S.norm(r) <= tol * bnorm or (omr == 0 and omi == 0) then
			break
		end
	end
	return finish(S, A, b, x, bnorm, tol, iterations)
end

-- Eigenvalues of operators
--
-- eigsh (Hermitian operators) and eigs (general operators) find a few eigenvalues and
-- eigenvectors of A from the Krylov space of a starting vector v0, restarting to keep the
-- basis small: thick-restart Lanczos (Wu and Simon) and the Krylov-Schur method (Stewart).
-- The basis is kept orthonormal by modified Gram-Schmidt run twice. The small projected
-- problems are solved in double precision (f64/c64) whatever the space. A Space cannot
-- make vectors on its own, so v0 is required (a random vector is a good choice).

type EigOptions = types.EigOptions
type EigInfo = types.EigInfo

-- A complex number's ordering key for `which`: larger keys are wanted first.
local function eig_key(which: string, re: number, im: number): number
	if which == "LM" then
		return math.sqrt(re * re + im * im)
	elseif which == "SM" then
		return -math.sqrt(re * re + im * im)
	elseif which == "LA" or which == "LR" then
		return re
	elseif which == "SA" or which == "SR" then
		return -re
	elseif which == "LI" then
		return im
	elseif which == "SI" then
		return -im
	end
	error(`unknown which {which}`)
end

-- A rotation (c real, s complex) with [c s; -conj(s) c] [f; g] = [r; 0] (LAPACK's lartg).
local function rotation(fr: number, fi: number, gr: number, gi: number): (number, number, number)
	if gr == 0 and gi == 0 then
		return 1, 0, 0
	end
	local nf = complex64.abs(fr, fi)
	if nf == 0 then
		return 0, 1, 0
	end
	local r = hypot(nf, complex64.abs(gr, gi))
	local ur, ui = fr / nf, fi / nf
	return nf / r, (ur * gr + ui * gi) / r, (ui * gr - ur * gi) / r
end

local function cget(M: types.Matrix, i: number, j: number): (number, number)
	local o = 16 * ((j - 1) * M.rows + i - 1)
	return buffer.readf64(M.data, o), buffer.readf64(M.data, o + 8)
end

local function cset(M: types.Matrix, i: number, j: number, re: number, im: number)
	local o = 16 * ((j - 1) * M.rows + i - 1)
	buffer.writef64(M.data, o, re)
	buffer.writef64(M.data, o + 8, im)
end

-- Swaps the diagonal entries i and i+1 (1-based) of the complex upper triangular T by a
-- unitary similarity, updating the Schur vectors Z (LAPACK's ztrexc step).
local function swap_schur(T: types.Matrix, Z: types.Matrix, i: number)
	local n = T.rows
	local ar, ai = cget(T, i, i)
	local br, bi = cget(T, i + 1, i + 1)
	local fr, fi = cget(T, i, i + 1)
	local c, sr, si = rotation(fr, fi, br - ar, bi - ai)
	-- zrot(x, y, c, s): x := c x + s y, y := c y - conj(s) x.
	local function zrot(M: types.Matrix, i1: number, j1: number, i2: number, j2: number, s1: number, s2: number)
		local xr, xi = cget(M, i1, j1)
		local yr, yi = cget(M, i2, j2)
		cset(M, i1, j1, c * xr + s1 * yr - s2 * yi, c * xi + s1 * yi + s2 * yr)
		cset(M, i2, j2, c * yr - (s1 * xr + s2 * xi), c * yi - (s1 * xi - s2 * xr))
	end
	for col = i + 2, n do
		zrot(T, i, col, i + 1, col, sr, si)
	end
	for row = 1, i - 1 do
		zrot(T, row, i, row, i + 1, sr, -si)
	end
	cset(T, i, i, br, bi)
	cset(T, i + 1, i + 1, ar, ai)
	for row = 1, Z.rows do
		zrot(Z, row, i, row, i + 1, sr, -si)
	end
end

-- A real Space made complex: elements are { re, im } pairs of its vectors, so complex
-- coefficients work (eigs uses this for real spaces, whose eigenvectors may be complex).
local function pair_space<T>(S: Space<T>): ComplexSpace<any>
	local function similar(x: any): any
		return { re = S.similar(x.re), im = S.similar(x.im) }
	end
	local function set(out: any, re: T, im: T): any
		out.re = S.copy(out.re, re)
		out.im = S.copy(out.im, im)
		return out
	end
	local function lin(ar: number, ai: number, x: any, y: any?): (T, T)
		-- (ar + i ai) x (+ y), into new vectors.
		local re = S.scale(S.similar(x.re), ar, x.re)
		re = S.axpy(re, -ai, x.im, re)
		local im = S.scale(S.similar(x.im), ar, x.im)
		im = S.axpy(im, ai, x.re, im)
		if y then
			re = S.add(re, re, y.re)
			im = S.add(im, im, y.im)
		end
		return re, im
	end
	local function dot(x: any, y: any): number
		return S.dot(x.re, y.re) + S.dot(x.im, y.im)
	end
	return {
		similar = similar,
		copy = function(out: any, x: any): any
			return set(out, x.re, x.im)
		end,
		add = function(out: any, x: any, y: any): any
			local re, im = S.add(S.similar(x.re), x.re, y.re), S.add(S.similar(x.im), x.im, y.im)
			return set(out, re, im)
		end,
		sub = function(out: any, x: any, y: any): any
			local re, im = S.sub(S.similar(x.re), x.re, y.re), S.sub(S.similar(x.im), x.im, y.im)
			return set(out, re, im)
		end,
		scale = function(out: any, a: number, x: any): any
			return set(out, lin(a, 0, x))
		end,
		axpy = function(out: any, a: number, x: any, y: any): any
			return set(out, lin(a, 0, x, y))
		end,
		dot = dot,
		norm = function(x: any): number
			return math.sqrt(math.max(dot(x, x), 0))
		end,
		cdot = function(x: any, y: any): (number, number)
			return S.dot(x.re, y.re) + S.dot(x.im, y.im), S.dot(x.re, y.im) - S.dot(x.im, y.re)
		end,
		cscale = function(out: any, ar: number, ai: number, x: any): any
			return set(out, lin(ar, ai, x))
		end,
		caxpy = function(out: any, ar: number, ai: number, x: any, y: any): any
			return set(out, lin(ar, ai, x, y))
		end,
	}
end

-- A few eigenvalues (as an f64 array, in the order `which` prefers) and eigenvectors of a
-- Hermitian (symmetric) operator A, by thick-restart Lanczos: options.which is "LM"
-- (largest magnitude, the default), "SM", "LA" (largest algebraic) or "SA"; options.ncv
-- (default max(2k + 1, 20)) is the basis size, options.tol (default 1000 eps of the
-- space) the relative residual |A x - lambda x| / |lambda| to reach, options.maxiter
-- (default 300) the number of restarts. Works on any Space (complex ones need cdot and
-- caxpy); for interior eigenvalues use a shift-invert operator. Returns values, vectors
-- (unit, as a table) and info.
local function eigsh<T>(S: Space<T>, A: Operator<T>, v0: T, k: number, options: EigOptions?): (buffer, { T }, EigInfo)
	local o: EigOptions = options or {}
	local which = o.which or "LM"
	local CS = S :: any
	local complex = S.real ~= true and CS.cdot ~= nil
	local eps = S.eps or 2 ^ -53
	local tol = o.tol or 1000 * eps
	local m = o.ncv or math.max(2 * k + 1, 20)
	local maxiter = o.maxiter or 300
	assert(k >= 1 and m > k, "eigsh needs 1 <= k < ncv")
	local nv = S.norm(v0)
	assert(nv > 0, "eigsh: v0 is zero")
	local V: { T } = { S.scale(S.similar(v0), 1 / nv, v0) }
	-- Lower triangle of the Hermitian projected matrix: hr[i][j], hi[i][j] for i >= j.
	local hr: { { number } }, hi: { { number } } = {}, {}
	for i = 1, m + 1 do
		hr[i], hi[i] = table.create(m, 0) :: { number }, table.create(m, 0) :: { number }
	end
	local p, restarts, hmax = 0, 0, 0
	while true do
		local mm, broke = m, false
		for j = p + 1, m do
			local w = A(S.similar(V[j]), V[j])
			local cr, ci = table.create(j, 0) :: { number }, table.create(j, 0) :: { number }
			for _ = 1, 2 do
				for i = 1, j do
					if complex then
						local dr, di = CS.cdot(V[i], w)
						cr[i] += dr
						ci[i] += di
						w = CS.caxpy(w, -dr, -di, V[i], w)
					else
						local d = S.dot(V[i], w)
						cr[i] += d
						w = S.axpy(w, -d, V[i], w)
					end
				end
			end
			for i = 1, j - 1 do
				hr[j][i], hi[j][i] = cr[i], -ci[i]
			end
			hr[j][j], hi[j][j] = cr[j], 0
			hmax = math.max(hmax, math.abs(cr[j]))
			local beta = S.norm(w)
			hmax = math.max(hmax, beta)
			if beta <= 64 * eps * hmax then
				mm, broke = j, true
				break
			end
			hr[j + 1][j], hi[j + 1][j] = beta, 0
			V[j + 1] = S.scale(w, 1 / beta, w)
		end
		-- Rayleigh-Ritz on the mm by mm projected matrix.
		local re: { { number } }, im: { { number } } = {}, {}
		for i = 1, mm do
			re[i], im[i] = table.create(mm, 0) :: { number }, table.create(mm, 0) :: { number }
			for j = 1, i do
				re[i][j], im[i][j] = hr[i][j], hi[i][j]
			end
		end
		local E = if complex then c64.eigh_copy(c64.matfrom(re, im)) else f64.eigh_copy(f64.matfrom(re))
		local theta = f64.totable(E.values)
		local Y = E.vectors :: types.Matrix
		local function yget(i: number, j: number): (number, number)
			if complex then
				return cget(Y, i, j)
			end
			return f64.matget(Y, i, j), 0
		end
		local order = {}
		for i = 1, mm do
			order[i] = i
		end
		table.sort(order, function(a: number, b: number): boolean
			return eig_key(which, theta[a], 0) > eig_key(which, theta[b], 0)
		end)
		local beta = if broke then 0 else hr[mm + 1][mm]
		local want = math.min(k, mm)
		local nconv = 0
		for q = 1, want do
			local yr, yi = yget(mm, order[q])
			local res = beta * math.sqrt(yr * yr + yi * yi)
			if res <= tol * math.max(math.abs(theta[order[q]]), eps ^ (2 / 3) * hmax) then
				nconv += 1
			end
		end
		restarts += 1
		local keep = if nconv >= want or restarts > maxiter or broke then want else math.min(mm - 1, want + (mm - want) // 2)
		-- New basis: the Ritz vectors of the kept values.
		local W: { T } = {}
		for q = 1, keep do
			local col = order[q]
			local x = S.scale(S.similar(V[1]), 0, V[1])
			for j = 1, mm do
				local yr, yi = yget(j, col)
				if complex then
					x = CS.caxpy(x, yr, yi, V[j], x)
				else
					x = S.axpy(x, yr, V[j], x)
				end
			end
			W[q] = x
		end
		if nconv >= want or restarts > maxiter or broke then
			local values = f64.create(want)
			for q = 1, want do
				f64.set(values, q, theta[order[q]])
			end
			return values, W, { converged = nconv >= want, restarts = restarts, nconv = nconv }
		end
		-- Thick restart: A W = W diag(theta) + beta v_(mm+1) (last row of Y). (The next
		-- expansion step recomputes that coupling row from the basis itself.)
		local last = V[mm + 1]
		V = W
		V[keep + 1] = last
		for i = 1, m + 1 do
			for j = 1, m do
				hr[i][j], hi[i][j] = 0, 0
			end
		end
		for q = 1, keep do
			hr[q][q] = theta[order[q]]
			local yr, yi = yget(mm, order[q])
			hr[keep + 1][q], hi[keep + 1][q] = beta * yr, beta * yi
		end
		p = keep
	end
end

-- Krylov-Schur on a ComplexSpace (complex coefficients allowed).
local function krylov_schur<T>(S: ComplexSpace<T>, A: Operator<T>, v0: T, k: number, o: EigOptions, eps: number): (buffer, { T }, EigInfo)
	local which = o.which or "LM"
	local tol = o.tol or 1000 * eps
	local m = o.ncv or math.max(2 * k + 1, 20)
	local maxiter = o.maxiter or 300
	assert(k >= 1 and m > k, "eigs needs 1 <= k < ncv")
	local nv = S.norm(v0)
	assert(nv > 0, "eigs: v0 is zero")
	local V: { T } = { S.scale(S.similar(v0), 1 / nv, v0) }
	-- The (m + 1) by m projected matrix: hr[i][j], hi[i][j].
	local hr: { { number } }, hi: { { number } } = {}, {}
	for i = 1, m + 1 do
		hr[i], hi[i] = table.create(m, 0) :: { number }, table.create(m, 0) :: { number }
	end
	local p, restarts, hmax = 0, 0, 0
	while true do
		local mm, broke = m, false
		for j = p + 1, m do
			local w = A(S.similar(V[j]), V[j])
			for _ = 1, 2 do
				for i = 1, j do
					local dr, di = S.cdot(V[i], w)
					hr[i][j] += dr
					hi[i][j] += di
					w = S.caxpy(w, -dr, -di, V[i], w)
				end
			end
			for i = 1, j do
				hmax = math.max(hmax, complex64.abs(hr[i][j], hi[i][j]))
			end
			local beta = S.norm(w)
			hmax = math.max(hmax, beta)
			if beta <= 64 * eps * hmax then
				mm, broke = j, true
				break
			end
			hr[j + 1][j], hi[j + 1][j] = beta, 0
			V[j + 1] = S.scale(w, 1 / beta, w)
		end
		-- Schur form of the mm by mm projected matrix, wanted values moved to the front.
		local re: { { number } }, im: { { number } } = {}, {}
		for i = 1, mm do
			re[i], im[i] = table.create(mm, 0) :: { number }, table.create(mm, 0) :: { number }
			for j = 1, mm do
				re[i][j], im[i][j] = hr[i][j], hi[i][j]
			end
		end
		local F = c64.schur(c64.matfrom(re, im))
		local T, Z = F.schur, F.z :: types.Matrix
		local want = math.min(k, mm)
		local keep = math.min(mm - 1, want + (mm - want) // 2)
		local sorted = if broke then mm else math.max(keep, want)
		for pos = 1, sorted do
			-- Selection: bring the most wanted remaining value to position pos.
			local best, at = -math.huge, pos
			for q = pos, mm do
				local key = eig_key(which, cget(T, q, q))
				if key > best then
					best, at = key, q
				end
			end
			for q = at - 1, pos, -1 do
				swap_schur(T, Z, q)
			end
		end
		-- b = beta (last row of Z): A (V Z) = (V Z) T + v_(mm+1) b.
		local beta = if broke then 0 else hr[mm + 1][mm]
		local nconv = 0
		for q = 1, want do
			local zr, zi = cget(Z, mm, q)
			local tr, ti = cget(T, q, q)
			if beta * math.sqrt(zr * zr + zi * zi) <= tol * math.max(complex64.abs(tr, ti), eps ^ (2 / 3) * hmax) then
				nconv += 1
			else
				break
			end
		end
		restarts += 1
		if nconv >= want or restarts > maxiter or broke then
			-- Ritz vectors: V Z_k Y with Y the eigenvectors of the leading k by k block.
			local Tk, Yk = c64.matrix(want, want), c64.identity(want)
			for j = 1, want do
				for i = 1, j do
					cset(Tk, i, j, cget(T, i, j))
				end
			end
			c64.eigvecs_from_schur(Tk, Yk)
			local Zk = c64.matrix(mm, want)
			buffer.copy(Zk.data, 0, Z.data, 0, 16 * mm * want)
			local Wm = c64.matmul(nil, Zk, Yk)
			local values = c64.create(want)
			local X: { T } = {}
			for q = 1, want do
				c64.set(values, q, cget(T, q, q))
				local x = S.scale(S.similar(V[1]), 0, V[1])
				for j = 1, mm do
					local wr, wi = cget(Wm, j, q)
					x = S.caxpy(x, wr, wi, V[j], x)
				end
				X[q] = S.scale(x, 1 / S.norm(x), x)
			end
			return values, X, { converged = nconv >= want, restarts = restarts, nconv = nconv }
		end
		-- Truncate to the kept Schur vectors: V := V Z(:, 1..keep).
		local W: { T } = {}
		for q = 1, keep do
			local x = S.scale(S.similar(V[1]), 0, V[1])
			for j = 1, mm do
				local zr, zi = cget(Z, j, q)
				x = S.caxpy(x, zr, zi, V[j], x)
			end
			W[q] = x
		end
		W[keep + 1] = V[mm + 1]
		V = W
		for i = 1, m + 1 do
			for j = 1, m do
				hr[i][j], hi[i][j] = 0, 0
			end
		end
		for j = 1, keep do
			for i = 1, j do
				hr[i][j], hi[i][j] = cget(T, i, j)
			end
			local zr, zi = cget(Z, mm, j)
			hr[keep + 1][j], hi[keep + 1][j] = beta * zr, beta * zi
		end
		p = keep
	end
end

-- A few eigenvalues (a c64 array, in the order options.which prefers: "LM" largest
-- magnitude (the default), "SM", "LR"/"SR" (real part), "LI"/"SI" (imaginary part)) and
-- unit eigenvectors of a general operator A, by the Krylov-Schur method; the options are
-- as for eigsh. On a complex space the vectors are elements of it; on a real space
-- (Space.real) the work runs on (re, im) pairs, since eigenvectors of real operators can
-- be complex, and each eigenvector is returned as { re = T, im = T }.
local function eigs<T>(S: ComplexSpace<T>, A: Operator<T>, v0: T, k: number, options: EigOptions?): (buffer, { any }, EigInfo)
	local o: EigOptions = options or {}
	local eps = S.eps or 2 ^ -53
	if S.real then
		local P = pair_space(S)
		local Ap: Operator<any> = function(out: any, x: any): any
			out.re = A(out.re, x.re)
			out.im = A(out.im, x.im)
			return out
		end
		local start = { re = v0, im = S.scale(S.similar(v0), 0, v0) }
		return krylov_schur(P, Ap, start, k, o, eps)
	end
	return krylov_schur(S, A, v0, k, o, eps)
end

return {
	cg = cg,
	minres = minres,
	gmres = gmres,
	bicgstab = bicgstab,
	eigsh = eigsh,
	eigs = eigs,
}
