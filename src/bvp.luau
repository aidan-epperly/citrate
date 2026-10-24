--!strict
--!optimize 2

-- Boundary value problems by Chebyshev collocation, in the spirit of Chebfun's: the first
-- order system y' = f(x, y) of m equations on [a, b] with m boundary conditions
-- bc(y(a), y(b)) = 0 (separated, mixed or periodic), solved for a polynomial in each
-- component. Higher-order equations are reduced to first order as usual (u'' = g(x, u,
-- u') becomes y = (u, u')).
--
-- The discretization is rectangular collocation (Driscoll and Hale 2016): the unknowns are
-- the values at N Chebyshev points of the second kind, the equations y' - f(x, y) are
-- imposed after resampling to the N - 1 Chebyshev points of the first kind, and the m
-- boundary conditions fill the system to a square one, so no rows are replaced and any
-- boundary conditions work. Nonlinear problems are solved by damped Newton iteration,
-- and N grows (17, 33, 65, ...) until every component's Chebyshev coefficients decay
-- below the tolerance; each refinement starts from the last solution. The result is a
-- Chebfun per component (see cheb), which can be evaluated, differentiated, integrated
-- and root-found. Everything is in double precision with f64 arrays.

local types = require("./types")
local f64 = require("./f64")
local cheb = require("./cheb")

type Matrix = types.Matrix
type Chebfun = types.Chebfun

-- f(x, y, out) -> f(x, y) for y an f64 array of the m components; bc(ya, yb, out) -> the
-- m residuals. Optional jac(x, y) -> the m by m matrix df/dy and bc_jac(ya, yb) -> (the
-- m by m matrices d bc / d ya and d bc / d yb); finite differences without them.
export type BvpOptions = {
	guess: ((x: number) -> buffer)?,
	tol: number?,
	max_points: number?,
	max_newton: number?,
	jac: ((x: number, y: buffer) -> Matrix)?,
	bc_jac: ((ya: buffer, yb: buffer) -> (Matrix, Matrix))?,
}

local EPS = 2 ^ -52

-- The (N - 1) by N matrix that resamples values at N Chebyshev points of the second kind
-- to the N - 1 points of the first kind (barycentric interpolation).
local function resample(N: number): { { number } }
	local x = cheb.chebpts(N, 2)
	local w = cheb.chebweights(N, 2)
	local t = cheb.chebpts(N - 1, 1)
	local P: { { number } } = {}
	for r = 1, N - 1 do
		local tr = f64.get(t, r)
		local row = table.create(N, 0)
		local exact = 0
		for k = 1, N do
			if tr == f64.get(x, k) then
				exact = k
			end
		end
		if exact > 0 then
			row[exact] = 1
		else
			local s = 0
			for k = 1, N do
				row[k] = f64.get(w, k) / (tr - f64.get(x, k))
				s += row[k]
			end
			for k = 1, N do
				row[k] /= s
			end
		end
		P[r] = row
	end
	return P
end

-- A Jacobian by forward differences: column j of d g / d y for g(y) -> buffer of size m.
local function fd_matrix(g: (buffer) -> buffer, y: buffer, gy: buffer, m: number): Matrix
	local J = f64.matrix(m, m)
	local yp = f64.copy(nil, y)
	for j = 1, m do
		local yj = f64.get(y, j)
		local d = math.sqrt(EPS) * math.max(math.abs(yj), 1)
		f64.set(yp, j, yj + d)
		local gp = g(yp)
		for i = 1, m do
			f64.matset(J, i, j, (f64.get(gp, i) - f64.get(gy, i)) / d)
		end
		f64.set(yp, j, yj)
	end
	return J
end

-- Solves the system of m equations y' = f(x, y) on domain = { a, b } with m boundary
-- conditions bc(y(a), y(b)) = 0. options (all optional): guess(x) -> the initial guess as an f64 array of m values (default zero;
-- nonlinear problems with several solutions converge to the one nearest the guess), tol
-- (default 1e-10: the relative size of the Chebyshev coefficients counted as resolved,
-- and the Newton tolerance), max_points (default 513), max_newton (iterations per
-- discretization, default 30), jac and bc_jac (see BvpOptions). Returns { y = { Chebfun per
-- component }, status ("success" or "failed"), message, points (the final N), newton
-- (the total iterations), residual (the largest |y' - f(x, y)| at the first-kind points,
-- relative to the size of f) }.
local function solve(f: (number, buffer, buffer) -> buffer, bc: (buffer, buffer, buffer) -> buffer, domain: { number }, m: number, options: BvpOptions?): any
	local o: BvpOptions = options or {}
	local a, b = domain[1], domain[2]
	assert(b > a, "bvp: the domain must be an interval { a, b } with a < b")
	local tol = o.tol or 1e-10
	local max_points = o.max_points or 513
	local max_newton = o.max_newton or 30
	local guess = o.guess or function(_: number): buffer
		return f64.create(m)
	end

	-- The current values: Y[j] holds component j at the N points.
	local N = 17
	local Y: { buffer } = {}
	do
		local x = cheb.chebpts(N, 2, a, b)
		for j = 1, m do
			Y[j] = f64.create(N)
		end
		for k = 1, N do
			local g = guess(f64.get(x, k))
			for j = 1, m do
				f64.set(Y[j], k, f64.get(g, j))
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
		local D = cheb.diffmat(N, a, b)
		local P = resample(N)
		local n = N - 1
		-- PD = P D, (N - 1) by N.
		local PD: { { number } } = {}
		for r = 1, n do
			local row = table.create(N, 0)
			local Pr = P[r]
			for k = 1, N do
				local p = Pr[k]
				if p ~= 0 then
					for c = 1, N do
						row[c] += p * f64.matget(D, k, c)
					end
				end
			end
			PD[r] = row
		end
		local size = m * N

		-- The point values of y at x_k, and f there.
		local function point(Yv: { buffer }, k: number): buffer
			local y = f64.create(m)
			for j = 1, m do
				f64.set(y, j, f64.get(Yv[j], k))
			end
			return y
		end
		local function residuals(Yv: { buffer }): (buffer, { buffer }, number)
			local F: { buffer } = {}
			local fscale = 0
			for k = 1, N do
				F[k] = f(f64.get(x, k), point(Yv, k), f64.create(m))
				fscale = math.max(fscale, f64.norm(F[k]))
			end
			local res = f64.create(size)
			for i = 1, m do
				local yi = Yv[i]
				for r = 1, n do
					local s = 0
					local PDr, Pr = PD[r], P[r]
					for k = 1, N do
						s += PDr[k] * f64.get(yi, k) - Pr[k] * f64.get(F[k], i)
					end
					f64.set(res, (i - 1) * n + r, s)
				end
			end
			local g = bc(point(Yv, 1), point(Yv, N), f64.create(m))
			for i = 1, m do
				f64.set(res, m * n + i, f64.get(g, i))
			end
			return res, F, fscale
		end

		-- Newton's method from Ystart: damped (Deuflhard's error-oriented strategy) or
		-- with full steps. Returns whether it converged, the iterate, its residuals and f
		-- values, and "singular" when the linear system was.
		local function newton(Ystart: { buffer }, damped: boolean): (boolean, { buffer }, buffer, { buffer }, number, string?)
			local Y = Ystart
			local res, F, fscale = residuals(Y)
			local prev_dnorm, prev_lambda = 0, 1
			local prev_dbar: buffer? = nil
			for _ = 1, max_newton do
				total_newton += 1
				-- The Newton matrix.
				local A = f64.matrix(size, size)
				for k = 1, N do
					local yk = point(Y, k)
					local xk = f64.get(x, k)
					local Jk = if jac_fn
						then (jac_fn :: any)(xk, yk)
						else fd_matrix(function(yy: buffer): buffer
							return f(xk, yy, f64.create(m))
						end, yk, F[k], m)
					for i = 1, m do
						for r = 1, n do
							local row = (i - 1) * n + r
							local p = P[r][k]
							f64.matset(A, row, (i - 1) * N + k, f64.matget(A, row, (i - 1) * N + k) + PD[r][k])
							if p ~= 0 then
								for j = 1, m do
									local col = (j - 1) * N + k
									f64.matset(A, row, col, f64.matget(A, row, col) - p * f64.matget(Jk, i, j))
								end
							end
						end
					end
				end
				local ya, yb = point(Y, 1), point(Y, N)
				local Ba: Matrix, Bb: Matrix
				if bc_jac_fn then
					Ba, Bb = (bc_jac_fn :: any)(ya, yb)
				else
					local g0 = bc(ya, yb, f64.create(m))
					Ba = fd_matrix(function(yy: buffer): buffer
						return bc(yy, yb, f64.create(m))
					end, ya, g0, m)
					Bb = fd_matrix(function(yy: buffer): buffer
						return bc(ya, yy, f64.create(m))
					end, yb, g0, m)
				end
				for i = 1, m do
					for j = 1, m do
						f64.matset(A, m * n + i, (j - 1) * N + 1, f64.matget(Ba, i, j))
						f64.matset(A, m * n + i, (j - 1) * N + N, f64.matget(A, m * n + i, (j - 1) * N + N) + f64.matget(Bb, i, j))
					end
				end
				local LU = f64.lu(A)
				if LU.singular then
					return false, Y, res, F, fscale, "singular"
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
				local rnew, Fnew, fnew = res, F, fscale
				local dbar: buffer = delta
				while true do
					for j = 1, m do
						local yj = f64.copy(nil, Y[j])
						for k = 1, N do
							f64.set(yj, k, f64.get(yj, k) + lambda * f64.get(delta, (j - 1) * N + k))
						end
						Ynew[j] = yj
					end
					rnew, Fnew, fnew = residuals(Ynew)
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
					for k = 1, N do
						ymax = math.max(ymax, math.abs(f64.get(Ynew[j], k)))
						dmax = math.max(dmax, lambda * math.abs(f64.get(delta, (j - 1) * N + k)))
					end
				end
				Y, res, F, fscale = Ynew, rnew, Fnew, fnew
				if dmax <= tol * math.max(ymax, 1) then
					return true, Y, res, F, fscale, nil
				elseif not (dmax < math.huge) then
					break
				end
			end
			return false, Y, res, F, fscale, nil
		end

		-- Damped Newton first, then full steps if it stalls (the damping is cautious). If
		-- neither converges, the grid may be too coarse to have a solution near the
		-- guess: refine it once and start again from the same values.
		local converged, Yn, res, _, fscale, why = newton(Y, true)
		if not converged and why == nil then
			converged, Yn, res, _, fscale, why = newton(Y, false)
		end
		residual = 0
		for i = 1, m * n do
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
		-- Resolved when every component's coefficients have decayed below tol.
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
			for k = 1, N2 do
				f64.set(v, k, cheb.bary(f64.get(x2, k), x, refine_from[j], w))
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
