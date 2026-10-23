--!strict
--!optimize 2

-- Iterative (Krylov subspace) solvers for A x = b, generic over how vectors are stored:
-- they touch vectors only through a Space (CG, MINRES) or ComplexSpace (GMRES, BiCGSTAB)
-- and A only through an Operator, so A can be sparse, dense, structured or matrix-free,
-- and the vectors can be buffers of any element type (or vectors, for CG and MINRES).
--
-- Every solver takes (space, A, b, options) and returns (x, info); see types.KrylovOptions
-- and types.KrylovInfo. info.residual is the true relative residual |b - A x| / |b| of the
-- returned x, recomputed at the end, and info.converged says whether it is within tol.

local types = require("./types")
local complex64 = require("./complex64")

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
		x = S.axpy(x, alpha, p, x)
		r = S.axpy(r, -alpha, Ap, r)
		if S.norm(r) <= tol * bnorm then
			break
		end
		z = if M then M(z, r) else r
		local rz_new = S.dot(r, z)
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

return {
	cg = cg,
	minres = minres,
	gmres = gmres,
	bicgstab = bicgstab,
}
