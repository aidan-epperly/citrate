--!strict
--!native
--!optimize 2

-- Implicit mechanics: M q'' = F(t, q, v) with stiff forces (springs, cloth, ropes, soft
-- bodies), stepped at a fixed h with one linearized backward Euler step per call (Baraff
-- and Witkin 1998). Each step solves
--   (M - h D - h^2 K) dv = h (F + h K v),   v += dv,   q += h v,
-- where K = dF/dq and D = dF/dv, by conjugate gradients preconditioned with the
-- diagonal: the cost is fixed (no Newton loop, no step-size control), and the step stays
-- stable however stiff the forces are, which explicit methods only manage with steps
-- below the stiffest period. Backward Euler damps the motion somewhat (more at larger h);
-- the price of that robustness.
--
-- The system matrix must be symmetric positive definite, which holds when K and D are
-- symmetric and negative semidefinite (conservative forces from a convex energy and
-- dissipative damping). For springs, drop the part of K that makes compressed springs
-- push sideways (Choi and Ko 2002); see demos/cloth. States are f64 or f32 arrays.

local types = require("./types")
local krylov = require("./krylov")
local stiff = require("./odestiff")
local c64 = require("./c64")
local c32 = require("./c32")

type Sparse = types.Sparse
type BlockSparse = types.BlockSparse

-- A Jacobian: a sparse or block sparse matrix, or an operator (out, x) -> out applying it.
type Jacobian = Sparse | BlockSparse | (buffer, buffer) -> buffer

export type ImplicitOptions = {
	jacobian: ((t: number, q: buffer, v: buffer) -> (Jacobian, Jacobian?))?,
	system: ((t: number, q: buffer, v: buffer, h: number) -> ((buffer, buffer) -> buffer, buffer | (buffer, buffer) -> buffer, buffer))?,
	mass: (number | buffer)?,
	fixed: { number }?,
	tol: number?,
	maxiter: number?,
	amg: (types.AMGOptions & { refresh: number? })?,
	rho_inf: number?,
	solver: string?,
}

-- An implicit mechanics integrator for M q'' = F(t, q, v): it.t, it.q, it.v, advanced by
-- it:step(h) (one step) or it:advance(dt, h?) (steps of h, default dt). force(t, q, v, out)
-- returns F. options:
--   jacobian(t, q, v) -> K, D?   dF/dq and dF/dv (sparse or block sparse matrices, or
--                                operators; D may be omitted when F does not depend on v).
--                                K and D of one format with the same pattern (with the
--                                diagonal stored) are combined into one matrix, so each CG
--                                iteration makes a single product; block sparse ones are
--                                preconditioned by block Jacobi. Mark K and D symmetric
--                                (as a conservative force's are) for the faster product.
--   system(t, q, v, h) -> apply, preconditioner, Kv   instead of jacobian, for models
--                                that can apply A = M - h D - h^2 K faster themselves
--                                (cloth, from its springs): the operator (out, x) -> out,
--                                A's diagonal (for the Jacobi preconditioner) or a
--                                preconditioner (out, r) -> out (symmetric positive
--                                definite, such as block Jacobi), and K v
--   mass      a number or an array (per component; default 1)
--   fixed     components (1-based) held in place: their velocity stays zero
--   tol, maxiter  for CG (default 1e-5 relative residual, 200 iterations)
--   amg       with block sparse K and D: precondition by algebraic multigrid (bsparse_amg,
--             these are its options, fixed added) instead of block Jacobi. Its hierarchy is
--             refilled every `refresh` steps (default 1); in between it is stale, which
--             stays a valid (positive definite) preconditioner but can cost many iterations
--             when the system changes quickly (swinging cloth: 9 iterations refilled every
--             step, 27 every 5th, 133 every 20th). For refills every step, plain
--             aggregation (smooth = false) was fastest on cloth.
--   rho_inf   in [0, 1]: instead of backward Euler, a third-order rational step that
--             damps slow modes far less (see the rational scheme below), with |R| =
--             rho_inf for the fastest: 0 removes them (L-stable), 1 keeps all energy. Needs
--             block sparse K and D; costs one complex solve per step (COCG).
--   solver    "auto" (default), "direct" or "cg": with block sparse K and D on one pattern,
--             "direct" factors the system in banded storage (Cholesky, or complex LU for
--             rho_inf) instead of iterating. "auto" does so when that is cheap (half
--             bandwidth kl <= 48 and n kl^2 <= 2e5: chains and ropes, not cloth) and amg is
--             not set. On a 6-link tail a rational step took 14 us natively against 35
--             by COCG.
-- it.stats counts steps, force and Jacobian evaluations and CG iterations; it.path says how
-- the last step applied the system ("sparse" or "block": K and D combined into one
-- matrix, "operators", or "system").
local function implicit_mechanics(S: any, force: (number, buffer, buffer, buffer) -> buffer, t0: number, q0: buffer, v0: buffer, options: ImplicitOptions): any
	local R = stiff.array_module(S)
	assert(R ~= nil and S.real, "ode.implicit_mechanics needs f32 or f64 arrays")
	local o = options
	assert(type(o.jacobian) == "function" or type(o.system) == "function", "ode.implicit_mechanics needs options.jacobian or options.system")
	local n = R.length(q0)
	local mass = o.mass or 1
	local mdiag: buffer
	if type(mass) == "number" then
		mdiag = R.scale(nil, mass, R.fill(R.create(n), 1))
	else
		mdiag = mass :: buffer
	end
	-- 1 for free components, 0 for fixed ones.
	local free = R.fill(R.create(n), 1)
	local fixed_list: { number } = o.fixed or {}
	local nfixed = #fixed_list
	for _, i in fixed_list do
		R.set(free, i, 0)
	end
	local tol, maxiter = o.tol or 1e-5, o.maxiter or 200
	local it: any = {
		t = t0,
		q = R.copy(nil, q0),
		v = R.mul(nil, free, v0),
		stats = { steps = 0, nfev = 0, njev = 0, iterations = 0 },
	}
	local dv_last = R.create(n)
	-- Work arrays, reused by every step.
	local xf, w = R.create(n), R.create(n)
	-- Combining K and D that share a sparse pattern: the values of
	-- M - h D - h^2 K on that pattern, with the diagonal's positions (cached per pattern).
	local diag_pos: { number }? = nil
	local diag_of: buffer? = nil
	-- The multigrid hierarchy (block path with options.amg), the pattern it was built on,
	-- and the steps since its last refill.
	local amg: any = nil
	local combined: buffer? = nil
	local amg_of: buffer? = nil
	local amg_age = 0

	local function apply(J: Jacobian, x: buffer, out: buffer): buffer
		if type(J) == "function" then
			return J(out, x)
		elseif (J :: any).block then
			return R.bsparse_matvec(out, J :: BlockSparse, x)
		end
		return R.sparse_matvec(out, J :: Sparse, x)
	end

	-- A matrix's index arrays (either format).
	local function index_arrays(A: any): (buffer, buffer)
		if A.block then
			return A.rowptr, A.colind
		end
		return A.colptr, A.rowind
	end

	-- Whether two sparse matrices have the same pattern (compared once per pair of index
	-- arrays, then remembered).
	local same_a: buffer?, same_b: buffer?, same_result = nil, nil, false
	local function same_pattern(A: any, B: any): boolean
		if (A.block ~= nil) ~= (B.block ~= nil) or A.block ~= B.block then
			return false
		end
		local ap, ai = index_arrays(A)
		local bp, bi = index_arrays(B)
		if ap == bp and ai == bi then
			return true
		end
		if same_a ~= ap or same_b ~= bp then
			same_a, same_b = ap, bp
			same_result = buffer.tostring(ap) == buffer.tostring(bp) and buffer.tostring(ai) == buffer.tostring(bi)
		end
		return same_result
	end

	-- Positions (1-based, in values) of the diagonal of a sparse pattern; every diagonal
	-- entry must be stored.
	local function diagonal_positions(P: any): { number }
		local ptr, ind = index_arrays(P)
		if diag_of == ptr and diag_pos then
			return diag_pos :: { number }
		end
		local pos = table.create(n, 0)
		local b = P.block or 1
		for c = 1, n // b do
			for p = buffer.readu32(ptr, 4 * (c - 1)), buffer.readu32(ptr, 4 * c) - 1 do
				if buffer.readu32(ind, 4 * p) + 1 == c then
					for r = 0, b - 1 do
						pos[(c - 1) * b + r + 1] = b * b * p + r * b + r + 1
					end
				end
			end
		end
		for i = 1, n do
			assert(pos[i] > 0, "ode.implicit_mechanics: the sparse Jacobians must store their diagonal")
		end
		diag_pos, diag_of = pos, ptr
		return pos
	end

	-- The rational scheme (options.rho_inf). Linearized at the current state y = (q, v),
	-- y' = L y + g with L = [[0, I], [M^-1 K, M^-1 D]] and g = (v, M^-1 F); one step is
	-- y += rho(hL) h g, so each mode is multiplied by R(z) = 1 + z rho(z). R is the third-
	-- order type (2, 2) rational function with |R(infinity)| = rho_inf: 0 gives Radau IIA's
	-- (L-stable), 1 Gauss's (fourth order, no damping), and every value in between is
	-- A-stable (|D(iy)|^2 - |N(iy)|^2 = q2^2 (1 - rho_inf^2) y^4). A slow mode is damped
	-- by O((h omega)^4) per step, where backward Euler damps it by O((h omega)^2): at
	-- h omega = 1, 98.8% of its amplitude survives each step (rho_inf = 0) against 70.7%.
	-- In partial fractions rho(z) = c / (z - sigma) + conj, so a step needs one complex
	-- solve with S = sigma^2 M - sigma h D - h^2 K (complex symmetric): COCG, preconditioned
	-- by multigrid or block Jacobi on Re S, which is positive definite because the poles
	-- are within 45 degrees of the real axis.
	-- the complex module of the same precision
	local CM: any = if buffer.len(R.create(1)) == 8 then c64 else c32
	local sigma_r, sigma_i, coef_r, coef_i = 0, 0, 0, 0
	if o.rho_inf ~= nil then
		local ri = o.rho_inf :: number
		assert(ri >= 0 and ri <= 1, "ode.implicit_mechanics: rho_inf must be in [0, 1]")
		-- R = (1 + p1 z + p2 z^2) / (1 + q1 z + q2 z^2), third order, R(infinity) = rho_inf
		local q1 = -(2 + ri) / (3 * (1 + ri))
		local q2 = -(1 + 3 * q1) / 6
		sigma_r = -q1 / (2 * q2)
		sigma_i = math.sqrt(4 * q2 - q1 * q1) / (2 * q2)
		-- rho(z) = (1 + (1/2 + q1) z) / (q2 (z - sigma)(z - conj sigma)):
		-- c = N(sigma) / (q2 (sigma - conj sigma)) = N(sigma) / (2 i q2 Im sigma)
		local nr, ni = 1 + (0.5 + q1) * sigma_r, (0.5 + q1) * sigma_i
		local d = 2 * q2 * sigma_i
		coef_r, coef_i = ni / d, -nr / d
	end
	it.rational = { sigma = { sigma_r, sigma_i }, c = { coef_r, coef_i } }

	-- Positions (1-based, in values) of the entries in fixed rows or columns of a block
	-- pattern, and whether each is on the diagonal (cached per pattern).
	local masked_of: buffer? = nil
	local masked_pos: { number }, masked_diag: { boolean } = {}, {}
	local function masked(P: any): ({ number }, { boolean })
		if masked_of == P.rowptr then
			return masked_pos, masked_diag
		end
		local isfixed = {}
		for _, c in fixed_list do
			isfixed[c] = true
		end
		local b = P.block
		local bb = b * b
		masked_pos, masked_diag = {}, {}
		for I = 0, P.rows // b - 1 do
			for p = buffer.readu32(P.rowptr, 4 * I), buffer.readu32(P.rowptr, 4 * I + 4) - 1 do
				local J = buffer.readu32(P.colind, 4 * p)
				for s = 0, b - 1 do
					for r = 0, b - 1 do
						local i, j = I * b + r + 1, J * b + s + 1
						if isfixed[i] or isfixed[j] then
							table.insert(masked_pos, bb * p + s * b + r + 1)
							table.insert(masked_diag, i == j)
						end
					end
				end
			end
		end
		masked_of = P.rowptr
		return masked_pos, masked_diag
	end

	-- The direct path: the system matrix copied into banded storage and factored (Cholesky
	-- for backward Euler's real symmetric positive definite matrix, LU for the rational
	-- step's complex symmetric one). Banded LU costs about n kl^2 multiply-adds where CG
	-- costs iterations times the nonzeros, so for chains and ropes (block tridiagonal: kl =
	-- 2 b - 1) it wins when CG needs more than a few iterations: on a 6-link tail with stiff
	-- rods, a rational step (force and Jacobian included) took 14 us against 35 by COCG.
	-- The half bandwidth of a block pattern (cached per pattern).
	local band_of: buffer? = nil
	local band_kl = 0
	local function bandwidth(P: any): number
		if band_of == P.rowptr then
			return band_kl
		end
		local b, kl = P.block, 0
		for I = 0, P.rows // b - 1 do
			for p = buffer.readu32(P.rowptr, 4 * I), buffer.readu32(P.rowptr, 4 * I + 4) - 1 do
				local J = buffer.readu32(P.colind, 4 * p)
				kl = math.max(kl, b * math.abs(I - J) + b - 1)
			end
		end
		band_of, band_kl = P.rowptr, kl
		return kl
	end
	local solver = o.solver or "auto"
	assert(solver == "auto" or solver == "direct" or solver == "cg", "ode.implicit_mechanics: solver must be \"auto\", \"direct\" or \"cg\"")
	local function use_direct(P: any): boolean
		if solver == "cg" or (solver == "auto" and o.amg ~= nil) then
			return false
		end
		local kl = bandwidth(P)
		return solver == "direct" or (kl <= 48 and n * kl * kl <= 2e5)
	end
	-- The matrix with block pattern P and values re (and im, for a complex one) in banded
	-- storage (as M.banded lays it out: entry (i, j) is element (j - 1) ld + kl + ku + i - j,
	-- 0-based), with fixed components' rows and columns replaced by the identity's. The
	-- positions are mapped once per pattern; each step is then one gather by direct buffer
	-- reads and writes (calling band_set per entry cost more than the factorization).
	local isfixed: { [number]: boolean } = {}
	for _, c in fixed_list do
		isfixed[c] = true
	end
	local width = buffer.len(R.create(1)) -- bytes per real number: 8 (f64) or 4 (f32)
	local map_of: buffer? = nil
	local map_src: buffer = buffer.create(0)
	local map_dst: buffer = buffer.create(0)
	local map_count, map_ld = 0, 0
	local function to_banded(M: any, P: any, re: buffer, im: buffer?): any
		local kl = bandwidth(P)
		local B = M.banded(n, kl, kl)
		if map_of ~= P.rowptr then
			local b = P.block
			local bb = b * b
			local src, dst = {}, {}
			for I = 0, P.rows // b - 1 do
				for p = buffer.readu32(P.rowptr, 4 * I), buffer.readu32(P.rowptr, 4 * I + 4) - 1 do
					local J = buffer.readu32(P.colind, 4 * p)
					for c = 0, b - 1 do
						for r = 0, b - 1 do
							local i, j = I * b + r, J * b + c
							if not (isfixed[i + 1] or isfixed[j + 1]) then
								table.insert(src, bb * p + c * b + r)
								table.insert(dst, j * B.ld + kl + kl + i - j)
							end
						end
					end
				end
			end
			map_count = #src
			map_src = buffer.create(4 * map_count)
			map_dst = buffer.create(4 * map_count)
			for k = 1, map_count do
				buffer.writeu32(map_src, 4 * (k - 1), src[k])
				buffer.writeu32(map_dst, 4 * (k - 1), dst[k])
			end
			map_of, map_ld = P.rowptr, B.ld
		end
		local d, src, dst = B.data, map_src, map_dst
		if im then
			-- complex: (re, im) pairs, 2 width bytes per entry
			local w2 = 2 * width
			for k = 0, map_count - 1 do
				local a, o = width * buffer.readu32(src, 4 * k), w2 * buffer.readu32(dst, 4 * k)
				if width == 8 then
					buffer.writef64(d, o, buffer.readf64(re, a))
					buffer.writef64(d, o + 8, buffer.readf64(im :: buffer, a))
				else
					buffer.writef32(d, o, buffer.readf32(re, a))
					buffer.writef32(d, o + 4, buffer.readf32(im :: buffer, a))
				end
			end
			for _, i in fixed_list do
				M.band_set(B, i, i, 1, 0)
			end
		else
			for k = 0, map_count - 1 do
				local a, o = width * buffer.readu32(src, 4 * k), width * buffer.readu32(dst, 4 * k)
				if width == 8 then
					buffer.writef64(d, o, buffer.readf64(re, a))
				else
					buffer.writef32(d, o, buffer.readf32(re, a))
				end
			end
			for _, i in fixed_list do
				M.band_set(B, i, i, 1)
			end
		end
		assert(B.ld == map_ld, "ode.implicit_mechanics: banded layout changed")
		return B
	end

	-- x^T y without conjugation, as (re, im).
	local function dotu(x: buffer, y: buffer): (number, number)
		local a, b = CM.dot(x, y)
		if type(a) == "vector" then
			return (a :: any).x, (a :: any).y
		end
		return a :: number, b :: number
	end
	local function cdiv(ar: number, ai: number, br: number, bi: number): (number, number)
		local d = br * br + bi * bi
		return (ar * br + ai * bi) / d, (ai * br - ar * bi) / d
	end

	-- COCG (conjugate gradients for complex symmetric A, with the bilinear form x^T y) for
	-- A x = b from x = 0, with a real symmetric positive definite preconditioner applied to
	-- the real and imaginary parts.
	local function cocg(A: (buffer) -> buffer, P: (buffer) -> buffer, b: buffer): (buffer, number)
		local CS = CM.space
		local x = CS.scale(CS.similar(b), 0, b)
		local r = CS.copy(CS.similar(b), b)
		local bnorm = CS.norm(b)
		if bnorm == 0 then
			return x, 0
		end
		local z = P(r)
		local p = CS.copy(CS.similar(z), z)
		local rr, ri = dotu(r, z)
		local iterations = 0
		while iterations < maxiter do
			iterations += 1
			local q = A(p)
			local pr, pi = dotu(p, q)
			if pr == 0 and pi == 0 then
				break
			end
			local ar, ai = cdiv(rr, ri, pr, pi)
			x = CS.caxpy(x, ar, ai, p, x)
			r = CS.caxpy(r, -ar, -ai, q, r)
			if CS.norm(r) <= tol * bnorm then
				break
			end
			z = P(r)
			local nr, ni = dotu(r, z)
			local br, bi = cdiv(nr, ni, rr, ri)
			p = CS.caxpy(p, br, bi, p, z)
			rr, ri = nr, ni
		end
		return x, iterations
	end

	local re_buf: buffer?, im_buf: buffer? = nil, nil
	local amg_r: any = nil
	local amg_r_of: buffer? = nil

	-- the rational step's complex solve by COCG (defined below)
	local rational_cocg: (self: any, K: any, reS: buffer, imS: buffer, rhs_re: buffer, rhs_im: buffer) -> buffer

	local function rational_step(self: any, h: number): buffer
		local t, q, v = self.t, self.q, self.v
		local F = force(t, q, v, R.create(n))
		self.stats.nfev += 1
		self.stats.njev += 1
		local K, D = (o.jacobian :: any)(t, q, v)
		assert(type(K) ~= "function" and K.block ~= nil and (D == nil or (type(D) ~= "function" and same_pattern(K, D))), "ode.implicit_mechanics: rho_inf needs block sparse K and D on one pattern")
		local sr, si = sigma_r, sigma_i
		local s2r, s2i = sr * sr - si * si, 2 * sr * si
		-- Re S = Re(sigma^2) M - Re(sigma) h D - h^2 K and Im S = Im(sigma^2) M - Im(sigma) h D
		local nv = R.length(K.values)
		if re_buf == nil or R.length(re_buf :: buffer) ~= nv then
			re_buf, im_buf = R.create(nv), R.create(nv)
		end
		local reS = R.scale(re_buf, -h * h, K.values)
		local imS = R.fill(im_buf :: buffer, 0)
		if D then
			reS = R.axpy(reS, -sr * h, D.values, reS)
			imS = R.axpy(imS, -si * h, D.values, imS)
		end
		local pos = diagonal_positions(K)
		for i = 1, n do
			local m = R.get(mdiag, i)
			R.set(reS, pos[i], R.get(reS, pos[i]) + s2r * m)
			R.set(imS, pos[i], R.get(imS, pos[i]) + s2i * m)
		end
		-- fixed components: identity rows and columns
		local mp, md = masked(K)
		for k, e in mp do
			R.set(reS, e, if md[k] then 1 else 0)
			R.set(imS, e, 0)
		end
		-- S a = -h^2 F + h (h D - sigma M) v
		local Mv = R.mul(nil, mdiag, v)
		local rhs_re = R.scale(nil, -h * h, F)
		if D then
			rhs_re = R.axpy(rhs_re, h * h, apply(D, v, R.create(n)), rhs_re)
		end
		rhs_re = R.axpy(rhs_re, -h * sr, Mv, rhs_re)
		local rhs_im = R.scale(nil, -h * si, Mv)
		rhs_re = R.mul(rhs_re, free, rhs_re)
		rhs_im = R.mul(rhs_im, free, rhs_im)
		local a: buffer
		if use_direct(K) then
			local LU = CM.band_lu(to_banded(CM, K, reS, imS))
			a = CM.band_lu_solve(nil, LU, CM.complexify(rhs_re, rhs_im))
			self.converged = not LU.singular
			self.path = "rational direct"
		else
			a = rational_cocg(self, K, reS, imS, rhs_re, rhs_im)
			self.path = "rational"
		end
		local ar, ai = CM.real(nil, a), CM.imag(nil, a)
		-- dq = 2 Re(c a); dv = 2 Re(c) v + (2 / h) Re(c sigma a)
		local cr, ci = coef_r, coef_i
		local gr, gi = cr * sr - ci * si, cr * si + ci * sr
		local dq = R.axpy(nil, -2 * ci, ai, R.scale(nil, 2 * cr, ar))
		local dv = R.axpy(nil, 2 * gr / h, ar, R.scale(nil, 2 * cr, v))
		dv = R.axpy(dv, -2 * gi / h, ai, dv)
		dq = R.mul(dq, free, dq)
		dv = R.mul(dv, free, dv)
		self.q = R.add(nil, q, dq)
		self.v = R.add(nil, v, dv)
		self.t = t + h
		self.stats.steps += 1
		return self.q
	end

	-- The rational step's complex solve by COCG, preconditioned on Re S.
	function rational_cocg(self: any, K: any, reS: buffer, imS: buffer, rhs_re: buffer, rhs_im: buffer): buffer
		local A: BlockSparse = { rows = n, cols = n, block = K.block, rowptr = K.rowptr, colind = K.colind, values = reS, symmetric = true }
		local Sc: any = { rows = n, cols = n, block = K.block, rowptr = K.rowptr, colind = K.colind, values = CM.complexify(reS, imS) }
		-- preconditioner on Re S
		local Preal: (buffer, buffer) -> buffer
		if o.amg then
			if amg_r == nil or amg_r_of ~= A.rowptr then
				local amg_options = table.clone(o.amg) :: any
				amg_options.fixed = fixed_list
				amg_r = R.bsparse_amg(A, amg_options)
				amg_r_of = A.rowptr
			else
				amg_r.update(A)
			end
			Preal = amg_r.apply
		else
			Preal = R.bsparse_jacobi(A)
		end
		local P = function(r: buffer): buffer
			local a = Preal(R.create(n), CM.real(nil, r))
			local b = Preal(R.create(n), CM.imag(nil, r))
			return CM.complexify(a, b)
		end
		local Aop = function(x: buffer): buffer
			return CM.bsparse_matvec(nil, Sc, x)
		end
		local a, iterations = cocg(Aop, P, CM.complexify(rhs_re, rhs_im))
		self.stats.iterations += iterations
		self.converged = iterations < maxiter
		return a
	end

	function it.step(self: any, h: number): buffer
		if o.rho_inf ~= nil then
			return rational_step(self, h)
		end
		local t, q, v = self.t, self.q, self.v
		local F = force(t, q, v, R.create(n))
		self.stats.nfev += 1
		self.stats.njev += 1
		-- A = M - h D - h^2 K as an operator, its diagonal, and K v.
		local applyA: (buffer, buffer) -> buffer
		local diag: buffer = mdiag
		local Kv: buffer
		local custom: ((buffer, buffer) -> buffer)? = nil
		-- with the direct solver: the pattern and A's values on it
		local direct_P: any, direct_values: buffer? = nil, nil
		if o.system then
			local A, d, kv = (o.system :: any)(t, q, v, h)
			applyA, Kv = A, kv
			self.path = "system"
			if type(d) == "function" then
				custom = d
			else
				diag = d
			end
		else
			local K, D = (o.jacobian :: any)(t, q, v)
			Kv = apply(K, v, R.create(n))
			if type(K) ~= "function" and (D == nil or (type(D) ~= "function" and same_pattern(K, D))) then
				-- One sparse matrix on the shared pattern.
				-- (into a buffer kept across steps: allocating it each step cost about as much as
				-- filling it)
				if combined == nil or R.length(combined :: buffer) ~= R.length(K.values) then
					combined = R.create(R.length(K.values))
				end
				local values = R.scale(combined, -h * h, K.values)
				if D then
					values = R.axpy(values, -h, D.values, values)
				end
				local pos = diagonal_positions(K)
				for i = 1, n do
					R.set(values, pos[i], R.get(values, pos[i]) + R.get(mdiag, i))
				end
				-- The diagonal, for the Jacobi preconditioner of the scalar path.
				if not K.block then
					diag = R.create(n)
					for i = 1, n do
						R.set(diag, i, R.get(values, pos[i]))
					end
				end
				-- M is diagonal, so A is symmetric when K and D are.
				local symmetric = K.symmetric and (D == nil or D.symmetric)
				if K.block and use_direct(K) then
					direct_P, direct_values = K, values
					self.path = "direct"
				elseif K.block then
					local A: BlockSparse = { rows = n, cols = n, block = K.block, rowptr = K.rowptr, colind = K.colind, values = values, symmetric = symmetric }
					self.path = "block"
					applyA = function(out: buffer, x: buffer): buffer
						return R.bsparse_matvec(out, A, x)
					end
					if o.amg then
						local refresh = o.amg.refresh or 1
						if amg == nil or amg_of ~= A.rowptr then
							local amg_options = table.clone(o.amg) :: any
							amg_options.fixed = fixed_list
							amg = R.bsparse_amg(A, amg_options)
							amg_of, amg_age = A.rowptr, 0
						else
							amg_age += 1
							if amg_age >= refresh then
								amg.update(A)
								amg_age = 0
							end
						end
						custom = amg.apply
					else
						custom = R.bsparse_jacobi(A)
					end
				else
					local A: Sparse = { rows = n, cols = n, colptr = K.colptr, rowind = K.rowind, values = values, symmetric = symmetric }
					self.path = "sparse"
					applyA = function(out: buffer, x: buffer): buffer
						return R.sparse_matvec(out, A, x)
					end
				end
			else
				self.path = "operators"
				applyA = function(out: buffer, x: buffer): buffer
					out = R.mul(out, mdiag, x)
					out = R.axpy(out, -h * h, apply(K, x, w), out)
					if D then
						out = R.axpy(out, -h, apply(D, x, w), out)
					end
					return out
				end
				diag = R.copy(nil, mdiag)
				local function diagonal(J: any): buffer
					return if J.block then R.bsparse_diag(J) else R.sparse_diag(J)
				end
				if type(K) ~= "function" then
					diag = R.axpy(diag, -h * h, diagonal(K), diag)
				end
				if D and type(D) ~= "function" then
					diag = R.axpy(diag, -h, diagonal(D), diag)
				end
			end
		end
		-- rhs = h (F + h K v), restricted to the free components.
		local rhs = R.axpy(nil, h, Kv, F)
		rhs = R.scale(rhs, h, rhs)
		rhs = R.mul(rhs, free, rhs)
		if direct_values then
			-- A is symmetric positive definite when K and D are negative semidefinite; if it
			-- is not (a model whose K is not), LU still solves it.
			local C = R.band_chol(to_banded(R, direct_P, direct_values :: buffer))
			local dv: buffer
			if C.definite then
				dv = R.band_chol_solve(nil, C, rhs)
				self.converged = true
			else
				local LU = R.band_lu(to_banded(R, direct_P, direct_values :: buffer))
				dv = R.band_lu_solve(nil, LU, rhs)
				self.converged = not LU.singular
			end
			dv_last = dv
			v = R.add(nil, v, dv)
			self.q = R.axpy(nil, h, v, q)
			self.v = v
			self.t = t + h
			self.stats.steps += 1
			return self.q
		end
		-- The system on the free components, and the identity on the fixed ones: the
		-- operator sees x with its fixed entries zeroed, and those entries of the result
		-- are x's own. (A copy and a few writes per call, rather than full passes.)
		local op = function(out: buffer, x: buffer): buffer
			local xin = x
			if nfixed > 0 then
				xf = R.copy(xf, x)
				for _, i in fixed_list do
					R.set(xf, i, 0)
				end
				xin = xf
			end
			out = applyA(out, xin)
			for _, i in fixed_list do
				R.set(out, i, R.get(x, i))
			end
			return out
		end
		-- The diagonal (Jacobi) preconditioner, or the model's own (applied to the free
		-- components, the identity on the fixed ones).
		local precond: (buffer, buffer) -> buffer
		if custom then
			local P = custom :: (buffer, buffer) -> buffer
			precond = function(out: buffer, r: buffer): buffer
				local rin = r
				if nfixed > 0 then
					xf = R.copy(xf, r)
					for _, i in fixed_list do
						R.set(xf, i, 0)
					end
					rin = xf
				end
				out = P(out, rin)
				for _, i in fixed_list do
					R.set(out, i, R.get(r, i))
				end
				return out
			end
		else
			local inv = R.create(n)
			for i = 1, n do
				local d = R.get(diag, i)
				R.set(inv, i, if R.get(free, i) == 0 or not (d > 0) then 1 else 1 / d)
			end
			precond = function(out: buffer, r: buffer): buffer
				return R.mul(out, inv, r)
			end
		end
		local cg_options: any = { x0 = R.mul(nil, free, dv_last), tol = tol, maxiter = maxiter, precond = precond }
		local dv, info = krylov.cg(S, op, rhs, cg_options)
		self.stats.iterations += info.iterations
		dv_last = dv
		v = R.add(nil, v, dv)
		self.q = R.axpy(nil, h, v, q)
		self.v = v
		self.t = t + h
		self.stats.steps += 1
		self.converged = info.converged
		return self.q
	end

	function it.advance(self: any, dt: number, h: number?): buffer
		local hs = h or dt
		local count = math.max(1, math.round(dt / hs))
		for _ = 1, count do
			self:step(dt / count)
		end
		return self.q
	end

	return it
end

return {
	implicit_mechanics = implicit_mechanics,
}
