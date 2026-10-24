--!strict
--!optimize 2

-- Boundary value problems by Chebyshev collocation, in the spirit of Chebfun's: m unknown
-- functions u_1 .. u_m on [a, b], where u_j satisfies an equation of order k_j,
--   u_j^(k_j) = f_j(x, z),   z = (u_1, u_1', .., u_1^(k_1 - 1), u_2, ..),
-- with k_1 + .. + k_m boundary conditions bc(z(a), z(b)) = 0 (separated, mixed or
-- periodic). First-order systems are the case k_j = 1; higher-order equations are solved
-- directly, without introducing their derivatives as unknowns.
--
-- The discretization is rectangular collocation (Driscoll and Hale 2016): the unknowns are
-- the values of each u_j at N Chebyshev points of the second kind; the equation for u_j is
-- imposed at the N - k_j Chebyshev points of the first kind, with u_j^(k_j) and z there
-- given by differentiation and barycentric resampling matrices; and the boundary
-- conditions fill the system to a square one, so no rows are replaced and any boundary
-- conditions work. Nonlinear problems are solved by damped Newton iteration, and N grows
-- (17, 33, 65, ...) until every u_j's Chebyshev coefficients decay below the tolerance;
-- each refinement starts from the last solution. The result is a Chebfun per u_j (see
-- cheb), which can be evaluated, differentiated, integrated and root-found. Everything is
-- in double precision with f64 arrays.

local types = require("./types")
local f64 = require("./f64")
local cheb = require("./cheb")

type Matrix = types.Matrix
type Chebfun = types.Chebfun
type Rows = { { number } }

-- f(x, z, out) -> the m highest derivatives f_j(x, z), for z an f64 array of the
-- K = k_1 + .. + k_m values (u_1, u_1', .., u_2, ..); bc(za, zb, out) -> the K residuals.
-- Optional jac(x, z) -> the m by K matrix df/dz and bc_jac(za, zb) -> (the K by K
-- matrices d bc / d za and d bc / d zb); finite differences without them. guess(x) gives
-- the initial guess as m values (or K, of which the values are used).
export type BvpOptions = {
	guess: ((x: number) -> buffer)?,
	tol: number?,
	max_points: number?,
	max_newton: number?,
	jac: ((x: number, z: buffer) -> Matrix)?,
	bc_jac: ((za: buffer, zb: buffer) -> (Matrix, Matrix))?,
}

local EPS = 2 ^ -52

-- The (N - k) by N matrix that resamples values at N Chebyshev points of the second kind
-- to the N - k points of the first kind (barycentric interpolation).
local function resample(N: number, k: number): Rows
	local x = cheb.chebpts(N, 2)
	local w = cheb.chebweights(N, 2)
	local t = cheb.chebpts(N - k, 1)
	local P: Rows = {}
	for r = 1, N - k do
		local tr = f64.get(t, r)
		local row = table.create(N, 0)
		local exact = 0
		for c = 1, N do
			if tr == f64.get(x, c) then
				exact = c
			end
		end
		if exact > 0 then
			row[exact] = 1
		else
			local s = 0
			for c = 1, N do
				row[c] = f64.get(w, c) / (tr - f64.get(x, c))
				s += row[c]
			end
			for c = 1, N do
				row[c] /= s
			end
		end
		P[r] = row
	end
	return P
end

-- A B for row tables (A: p by q, B: q by N).
local function product(A: Rows, B: Rows, N: number): Rows
	local C: Rows = {}
	for r = 1, #A do
		local row = table.create(N, 0)
		local Ar = A[r]
		for k = 1, #B do
			local p = Ar[k]
			if p ~= 0 then
				local Bk = B[k]
				for c = 1, N do
					row[c] += p * Bk[c]
				end
			end
		end
		C[r] = row
	end
	return C
end

-- A Jacobian by forward differences: d g / d z (rows by #z) for g(z) -> buffer of size
-- rows, given gz = g(z).
local function fd_matrix(g: (buffer) -> buffer, z: buffer, gz: buffer, rows: number): Matrix
	local n = f64.length(z)
	local J = f64.matrix(rows, n)
	local zp = f64.copy(nil, z)
	for j = 1, n do
		local zj = f64.get(z, j)
		local d = math.sqrt(EPS) * math.max(math.abs(zj), 1)
		f64.set(zp, j, zj + d)
		local gp = g(zp)
		for i = 1, rows do
			f64.matset(J, i, j, (f64.get(gp, i) - f64.get(gz, i)) / d)
		end
		f64.set(zp, j, zj)
	end
	return J
end

-- Solves the boundary value problem described above on domain = { a, b }. orders is the
-- list { k_1, .., k_m } of the equations' orders, or a number m for a first-order system
-- of m equations (y' = f(x, y) with bc(y(a), y(b)) = 0). options (all optional): guess
-- (default zero; nonlinear problems with several solutions converge to the one nearest the
-- guess), tol (default 1e-10: the relative size of the Chebyshev coefficients counted as
-- resolved, and the Newton tolerance), max_points (default 513), max_newton (iterations
-- per attempt, default 30), jac and bc_jac (see BvpOptions). Returns { y = { Chebfun per
-- unknown function }, status ("success" or "failed"), message, points (the final N),
-- newton (the total iterations), residual (the largest |u_j^(k_j) - f_j| at the
-- collocation points, relative to the size of f) }.
local function solve(f: (number, buffer, buffer) -> buffer, bc: (buffer, buffer, buffer) -> buffer, domain: { number }, orders: { number } | number, options: BvpOptions?): any
	local o: BvpOptions = options or {}
	local a, b = domain[1], domain[2]
	assert(b > a, "bvp: the domain must be an interval { a, b } with a < b")
	local ks: { number } = {}
	if type(orders) == "number" then
		for j = 1, orders do
			ks[j] = 1
		end
	else
		for j, k in orders do
			assert(k >= 1 and k % 1 == 0, "bvp: orders must be positive integers")
			ks[j] = k
		end
	end
	local m = #ks
	-- offset[j] + d + 1 is the position of u_j^(d) in z; K is the length of z.
	local offset: { number } = {}
	local K, kmax = 0, 0
	for j = 1, m do
		offset[j] = K
		K += ks[j]
		kmax = math.max(kmax, ks[j])
	end
	local tol = o.tol or 1e-10
	local max_points = o.max_points or 513
	local max_newton = o.max_newton or 30
	local guess = o.guess or function(_: number): buffer
		return f64.create(m)
	end

	-- The current values: Y[j] holds u_j at the N points.
	local N = 17
	while N <= 2 * kmax + 1 do
		N = 2 * N - 1
	end
	local Y: { buffer } = {}
	do
		local x = cheb.chebpts(N, 2, a, b)
		for j = 1, m do
			Y[j] = f64.create(N)
		end
		for c = 1, N do
			local g = guess(f64.get(x, c))
			local full = f64.length(g) == K and K ~= m
			for j = 1, m do
				f64.set(Y[j], c, f64.get(g, if full then offset[j] + 1 else j))
			end
		end
	end
	local total_newton = 0
	local jac_fn, bc_jac_fn = o.jac, o.bc_jac
	local failures = 0
	local status, message = "failed", "not resolved within max_points"
	local residual = math.huge

	while true do
		local x = cheb.chebpts(N, 2, a, b)
		-- Powers of the differentiation matrix: Dpow[d] (Dpow[0] the identity, implicit).
		local Dm = cheb.diffmat(N, a, b)
		local D1: Rows = {}
		for r = 1, N do
			local row = table.create(N, 0)
			for c = 1, N do
				row[c] = f64.matget(Dm, r, c)
			end
			D1[r] = row
		end
		local Dpow: { [number]: Rows } = { [1] = D1 }
		for d = 2, kmax do
			Dpow[d] = product(Dpow[d - 1], D1, N)
		end
		-- For each order k present: the collocation points t (first kind, N - k of them)
		-- and Q[k][d] = P_k D^d, the d-th derivative at t of the interpolant.
		local Q: { [number]: { [number]: Rows } } = {}
		local T: { [number]: buffer } = {}
		for _, k in ks do
			if Q[k] == nil then
				local P = resample(N, k)
				local Qk: { [number]: Rows } = { [0] = P }
				for d = 1, math.max(kmax - 1, k) do
					Qk[d] = product(P, Dpow[d], N)
				end
				Q[k] = Qk
				T[k] = cheb.chebpts(N - k, 1, a, b)
			end
		end
		-- Equation rows: block j starts after rowoff[j] rows.
		local rowoff: { number } = {}
		local rows = 0
		for j = 1, m do
			rowoff[j] = rows
			rows += N - ks[j]
		end
		local size = m * N
		assert(rows + K == size, "bvp: inconsistent system size")

		-- z at the collocation points of order k: Z[r] (an f64 array of K values).
		local function zs(Yv: { buffer }, k: number): { buffer }
			local Qk = Q[k]
			local n = N - k
			local out: { buffer } = {}
			for r = 1, n do
				out[r] = f64.create(K)
			end
			for i = 1, m do
				local yi = Yv[i]
				for d = 0, ks[i] - 1 do
					local Qd = Qk[d]
					for r = 1, n do
						local s = 0
						local row = Qd[r]
						for c = 1, N do
							s += row[c] * f64.get(yi, c)
						end
						f64.set(out[r], offset[i] + d + 1, s)
					end
				end
			end
			return out
		end
		-- z at an end (row 1 or N of each derivative matrix).
		local function z_end(Yv: { buffer }, e: number): buffer
			local z = f64.create(K)
			for i = 1, m do
				f64.set(z, offset[i] + 1, f64.get(Yv[i], e))
				for d = 1, ks[i] - 1 do
					local s = 0
					local row = Dpow[d][e]
					for c = 1, N do
						s += row[c] * f64.get(Yv[i], c)
					end
					f64.set(z, offset[i] + d + 1, s)
				end
			end
			return z
		end
		-- The residuals, with the z and f values at the collocation points per order, and
		-- the size of f.
		local function residuals(Yv: { buffer }): (buffer, { [number]: { buffer } }, { [number]: { buffer } }, number)
			local res = f64.create(size)
			local Zk: { [number]: { buffer } } = {}
			local Fk: { [number]: { buffer } } = {}
			local fscale = 0
			for _, k in ks do
				if Zk[k] == nil then
					local Z = zs(Yv, k)
					local F: { buffer } = {}
					local tk = T[k]
					for r = 1, N - k do
						F[r] = f(f64.get(tk, r), Z[r], f64.create(m))
						fscale = math.max(fscale, f64.norm(F[r]))
					end
					Zk[k], Fk[k] = Z, F
				end
			end
			for j = 1, m do
				local k = ks[j]
				local Qkk = Q[k][k]
				local F = Fk[k]
				local yj = Yv[j]
				for r = 1, N - k do
					local s = 0
					local row = Qkk[r]
					for c = 1, N do
						s += row[c] * f64.get(yj, c)
					end
					f64.set(res, rowoff[j] + r, s - f64.get(F[r], j))
				end
			end
			local g = bc(z_end(Yv, 1), z_end(Yv, N), f64.create(K))
			for s = 1, K do
				f64.set(res, rows + s, f64.get(g, s))
			end
			return res, Zk, Fk, fscale
		end

		-- Newton's method from Ystart: damped (Deuflhard's error-oriented strategy) or
		-- with full steps. Returns whether it converged, the iterate, its residuals and the
		-- size of f, and "singular" when the linear system was.
		local function newton(Ystart: { buffer }, damped: boolean): (boolean, { buffer }, buffer, number, string?)
			local Yc = Ystart
			local res, Zk, Fk, fscale = residuals(Yc)
			local prev_dnorm, prev_lambda = 0, 1
			local prev_dbar: buffer? = nil
			for _ = 1, max_newton do
				total_newton += 1
				-- The Newton matrix.
				local A = f64.matrix(size, size)
				local data = A.data
				local function add(r: number, c: number, v: number)
					local p = 8 * ((c - 1) * size + r - 1)
					buffer.writef64(data, p, buffer.readf64(data, p) + v)
				end
				-- The Jacobians df/dz at the collocation points of each order.
				local Jk: { [number]: { Matrix } } = {}
				for _, k in ks do
					if Jk[k] == nil then
						local list: { Matrix } = {}
						local tk = T[k]
						for r = 1, N - k do
							local tr = f64.get(tk, r)
							list[r] = if jac_fn
								then (jac_fn :: any)(tr, Zk[k][r])
								else fd_matrix(function(zz: buffer): buffer
									return f(tr, zz, f64.create(m))
								end, Zk[k][r], Fk[k][r], m)
						end
						Jk[k] = list
					end
				end
				for j = 1, m do
					local k = ks[j]
					local Qk = Q[k]
					local Qkk = Qk[k]
					for r = 1, N - k do
						local row = rowoff[j] + r
						local own = Qkk[r]
						for c = 1, N do
							add(row, (j - 1) * N + c, own[c])
						end
						local Jr = Jk[k][r]
						for i = 1, m do
							for d = 0, ks[i] - 1 do
								local w = f64.matget(Jr, j, offset[i] + d + 1)
								if w ~= 0 then
									local q = Qk[d][r]
									for c = 1, N do
										add(row, (i - 1) * N + c, -w * q[c])
									end
								end
							end
						end
					end
				end
				local za, zb = z_end(Yc, 1), z_end(Yc, N)
				local Ba: Matrix, Bb: Matrix
				if bc_jac_fn then
					Ba, Bb = (bc_jac_fn :: any)(za, zb)
				else
					local g0 = bc(za, zb, f64.create(K))
					Ba = fd_matrix(function(zz: buffer): buffer
						return bc(zz, zb, f64.create(K))
					end, za, g0, K)
					Bb = fd_matrix(function(zz: buffer): buffer
						return bc(za, zz, f64.create(K))
					end, zb, g0, K)
				end
				for s = 1, K do
					for i = 1, m do
						for d = 0, ks[i] - 1 do
							local col = offset[i] + d + 1
							local wa, wb = f64.matget(Ba, s, col), f64.matget(Bb, s, col)
							if d == 0 then
								add(rows + s, (i - 1) * N + 1, wa)
								add(rows + s, (i - 1) * N + N, wb)
							else
								local ra, rb = Dpow[d][1], Dpow[d][N]
								for c = 1, N do
									add(rows + s, (i - 1) * N + c, wa * ra[c] + wb * rb[c])
								end
							end
						end
					end
				end
				local LU = f64.lu(A)
				if LU.singular then
					return false, Yc, res, fscale, "singular"
				end
				local delta = f64.lu_solve(nil, LU, f64.scale(nil, -1, res))
				-- Damped Newton with Deuflhard's natural monotonicity test (as in Chebfun): a
				-- step lambda delta is accepted when the simplified Newton correction there,
				-- J^-1 F(y + lambda delta) with the same factorization, is smaller than delta
				-- by the factor 1 - lambda / 4. Unlike the residual, this measure is invariant
				-- under scaling the equations. The first lambda is predicted from the last
				-- step's contraction; a rejected one shrinks by Deuflhard's estimate (at
				-- least halving).
				local dnorm = f64.norm(delta)
				local lambda = 1
				if damped and prev_dbar ~= nil and dnorm > 0 then
					local gap = f64.norm(f64.sub(nil, prev_dbar :: buffer, delta))
					if gap > 0 then
						lambda = math.min(1, prev_dnorm * f64.norm(prev_dbar :: buffer) / (gap * dnorm) * prev_lambda)
					end
					lambda = math.max(lambda, 2 ^ -10)
				end
				local Ynew: { buffer } = {}
				local rnew, Znew, Fnew, fnew = res, Zk, Fk, fscale
				local dbar: buffer = delta
				while true do
					for j = 1, m do
						local yj = f64.copy(nil, Yc[j])
						for c = 1, N do
							f64.set(yj, c, f64.get(yj, c) + lambda * f64.get(delta, (j - 1) * N + c))
						end
						Ynew[j] = yj
					end
					rnew, Znew, Fnew, fnew = residuals(Ynew)
					if not damped then
						break
					end
					dbar = f64.lu_solve(nil, LU, f64.scale(nil, -1, rnew))
					local nbar = f64.norm(dbar)
					if dnorm == 0 or (nbar == nbar and nbar <= (1 - lambda / 4) * dnorm) or lambda <= 2 ^ -10 then
						break
					end
					-- |dbar - (1 - lambda) delta| measures the nonlinearity along the step.
					local wn = f64.norm(f64.axpy(nil, -(1 - lambda), delta, dbar))
					local estimate = if wn == wn and wn > 0 then lambda * lambda * dnorm / (2 * wn) else lambda / 2
					lambda = math.max(math.min(lambda / 2, estimate), lambda / 10, 2 ^ -10)
				end
				prev_dnorm, prev_lambda, prev_dbar = dnorm, lambda, dbar
				local ymax, dmax = 0, 0
				for j = 1, m do
					for c = 1, N do
						ymax = math.max(ymax, math.abs(f64.get(Ynew[j], c)))
						dmax = math.max(dmax, lambda * math.abs(f64.get(delta, (j - 1) * N + c)))
					end
				end
				Yc, res, Zk, Fk, fscale = Ynew, rnew, Znew, Fnew, fnew
				if dmax <= tol * math.max(ymax, 1) then
					return true, Yc, res, fscale, nil
				elseif not (dmax < math.huge) then
					break
				end
			end
			return false, Yc, res, fscale, nil
		end

		-- Damped Newton first, then full steps if it stalls (the damping is cautious). If
		-- neither converges, the grid may be too coarse to have a solution near the
		-- guess: refine it once and start again from the same values.
		local converged, Yn, res, fscale, why = newton(Y, true)
		if not converged and why == nil then
			converged, Yn, res, fscale, why = newton(Y, false)
		end
		residual = 0
		for i = 1, rows do
			residual = math.max(residual, math.abs(f64.get(res, i)))
		end
		residual /= math.max(fscale, 1)
		if why == "singular" then
			message = "the collocation system is singular (are the boundary conditions independent?)"
			break
		end
		local refine_from = Y
		if converged then
			Y = Yn
			failures = 0
		else
			-- Two failures in a row: the guess is the problem, not the grid.
			failures += 1
			if failures >= 2 or 2 * N - 1 > max_points then
				message = "Newton's method did not converge (try a better guess)"
				Y = Yn
				break
			end
		end
		-- Resolved when every function's coefficients have decayed below tol.
		if converged then
			local happy = true
			for j = 1, m do
				local c = cheb.vals2coeffs(Y[j])
				if cheb.standard_chop(c, tol) >= N then
					happy = false
				end
			end
			if happy then
				status, message = "success", "resolved"
				break
			end
			refine_from = Y
		end
		if 2 * N - 1 > max_points then
			message = "not resolved within max_points"
			break
		end
		-- Refine: the solution (or the starting values, if Newton failed) at twice as
		-- many points.
		local N2 = 2 * N - 1
		local x2 = cheb.chebpts(N2, 2, a, b)
		local w = cheb.chebweights(N, 2)
		local Y2: { buffer } = {}
		for j = 1, m do
			local v = f64.create(N2)
			for c = 1, N2 do
				f64.set(v, c, cheb.bary(f64.get(x2, c), x, refine_from[j], w))
			end
			Y2[j] = v
		end
		Y, N = Y2, N2
	end

	local out: { Chebfun } = {}
	for j = 1, m do
		local F = cheb.from_values(Y[j], { a, b })
		local keep = cheb.standard_chop(F.coeffs, tol)
		local c = f64.create(keep)
		buffer.copy(c, 0, F.coeffs, 0, 8 * keep)
		local G = cheb.from_coeffs(c, { a, b })
		G.happy = status == "success"
		out[j] = G
	end
	return { y = out, status = status, message = message, points = N, newton = total_newton, residual = residual }
end

return {
	solve = solve,
}
