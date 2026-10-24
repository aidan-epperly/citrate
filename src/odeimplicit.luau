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

	function it.step(self: any, h: number): buffer
		local t, q, v = self.t, self.q, self.v
		local F = force(t, q, v, R.create(n))
		self.stats.nfev += 1
		self.stats.njev += 1
		-- A = M - h D - h^2 K as an operator, its diagonal, and K v.
		local applyA: (buffer, buffer) -> buffer
		local diag: buffer = mdiag
		local Kv: buffer
		local custom: ((buffer, buffer) -> buffer)? = nil
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
				local values = R.scale(nil, -h * h, K.values)
				if D then
					values = R.axpy(values, -h, D.values, values)
				end
				local pos = diagonal_positions(K)
				diag = R.create(n)
				for i = 1, n do
					R.set(values, pos[i], R.get(values, pos[i]) + R.get(mdiag, i))
					R.set(diag, i, R.get(values, pos[i]))
				end
				-- M is diagonal, so A is symmetric when K and D are.
				local symmetric = K.symmetric and (D == nil or D.symmetric)
				if K.block then
					local A: BlockSparse = { rows = n, cols = n, block = K.block, rowptr = K.rowptr, colind = K.colind, values = values, symmetric = symmetric }
					self.path = "block"
					applyA = function(out: buffer, x: buffer): buffer
						return R.bsparse_matvec(out, A, x)
					end
					custom = R.bsparse_jacobi(A)
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
