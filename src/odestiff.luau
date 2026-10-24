--!strict
--!optimize 2

-- Stiff ODE (and index-1 DAE) solvers, used through ode.solve and ode.integrator with
-- options.method:
--   "radau"  Radau IIA of order 5 (Hairer and Wanner's RADAU5): fully implicit, L-stable
--            and stiffly accurate; the most robust choice, including tight tolerances,
--            very stiff problems and differential-algebraic equations.
-- They take M y' = f(t, y) with an optional constant mass matrix M (options.mass: a
-- matrix, or an array for a diagonal M; a singular M makes it a DAE, whose initial state
-- must be consistent). The Jacobian df/dy is options.jac (a function (t, y) -> matrix, or
-- a constant matrix), or else estimated by finite differences; options.jac_band = { kl,
-- ku } says it is banded (kl subdiagonals, ku superdiagonals), so the estimate needs only
-- kl + ku + 1 evaluations of f and the linear algebra is banded. States are f32 or f64
-- arrays (ode adapts vectors); the solves run in the state's precision.

local types = require("./types")
local core = require("./odecore")
local f32 = require("./f32")
local f64 = require("./f64")
local c32 = require("./c32")
local c64 = require("./c64")
local vec = require("./vec")

type Space<T> = types.Space<T>
type Matrix = types.Matrix
type Banded = types.Banded
type OdeFunction<T> = types.OdeFunction<T>
type Piece<T> = core.Piece<T>

local EPS = core.EPS
local weighted = core.weighted

-- The array modules of a real space: (real, complex).
local function modules(S: any): (any, any)
	if S == f64.space then
		return f64, c64
	elseif S == f32.space then
		return f32, c32
	end
	error("ode: the stiff solvers need f32 or f64 arrays (or vectors)")
end

-- Jacobians and linear systems

-- The Jacobian: evaluate(t, y, fy) gives J at (t, y) (dense Matrix or Banded); updatable
-- says whether that can change (false for a constant matrix).
type Jacobian = {
	evaluate: (number, buffer, buffer) -> any,
	updatable: boolean,
	banded: boolean,
	kl: number,
	ku: number,
}

local function make_jacobian(S: any, R: any, f: OdeFunction<buffer>, n: number, o: any, stats: any): Jacobian
	local band = o.jac_band
	local kl, ku = if band then band[1] else 0, if band then band[2] else 0
	local eps = math.sqrt(S.eps or EPS)
	if o.jac then
		if type(o.jac) == "function" then
			local user: any = o.jac
			return {
				evaluate = function(t: number, y: buffer, _: buffer): any
					stats.njev += 1
					return user(t, y)
				end,
				updatable = true,
				banded = band ~= nil,
				kl = kl,
				ku = ku,
			}
		end
		local Jc = o.jac
		return {
			evaluate = function(): any
				return Jc
			end,
			updatable = false,
			banded = band ~= nil,
			kl = kl,
			ku = ku,
		}
	end
	-- Finite differences: one evaluation per column, or per group of kl + ku + 1 columns
	-- for a band (columns that far apart touch disjoint rows).
	local function step_for(v: number): number
		local d = eps * math.max(math.abs(v), 1)
		return if v < 0 then -d else d
	end
	if band then
		return {
			evaluate = function(t: number, y: buffer, fy: buffer): any
				stats.njev += 1
				local B = R.banded(n, kl, ku)
				local yp = R.copy(nil, y)
				local fp = R.create(n)
				local w = kl + ku + 1
				for g = 1, math.min(w, n) do
					local steps = {}
					for j = g, n, w do
						local yj = R.get(y, j)
						R.set(yp, j, yj + step_for(yj))
						steps[j] = R.get(yp, j) - yj
					end
					fp = f(t, yp, fp)
					stats.nfev += 1
					for j = g, n, w do
						for i = math.max(1, j - ku), math.min(n, j + kl) do
							R.band_set(B, i, j, (R.get(fp, i) - R.get(fy, i)) / steps[j])
						end
						R.set(yp, j, R.get(y, j))
					end
				end
				return B
			end,
			updatable = true,
			banded = true,
			kl = kl,
			ku = ku,
		}
	end
	return {
		evaluate = function(t: number, y: buffer, fy: buffer): any
			stats.njev += 1
			local J = R.matrix(n, n)
			local yp = R.copy(nil, y)
			local fp = R.create(n)
			local bytes = buffer.len(y)
			for j = 1, n do
				local yj = R.get(y, j)
				R.set(yp, j, yj + step_for(yj))
				local d = R.get(yp, j) - yj
				fp = f(t, yp, fp)
				stats.nfev += 1
				local col = R.sub(nil, fp, fy)
				col = R.scale(col, 1 / d, col)
				buffer.copy(J.data, bytes * (j - 1), col)
				R.set(yp, j, yj)
			end
			return J
		end,
		updatable = true,
		banded = false,
		kl = 0,
		ku = 0,
	}
end

-- Mass matrix products: M x (identity when there is no mass matrix).
local function mass_apply(R: any, mass: any, x: buffer): buffer
	if mass == nil then
		return R.copy(nil, x)
	elseif type(mass) == "buffer" then
		return R.mul(nil, mass, x)
	end
	return R.matvec(nil, mass, x)
end

-- The mass matrix's diagonal entry i (for banded systems, whose mass must be diagonal).
local function mass_diag(R: any, mass: any, i: number): number
	if mass == nil then
		return 1
	elseif type(mass) == "buffer" then
		return R.get(mass, i)
	end
	return R.matget(mass, i, i)
end

-- A solver for (c M - J) x = b with real c, or nil if that matrix is singular.
local function real_system(R: any, J: any, mass: any, c: number, stats: any): ((buffer) -> buffer)?
	stats.nlu += 1
	if J.kl ~= nil and J.data ~= nil and J.n ~= nil then
		local B = R.band_copy(J)
		B.data = R.scale(B.data, -1, B.data)
		for i = 1, J.n do
			R.band_set(B, i, i, R.band_get(B, i, i) + c * mass_diag(R, mass, i))
		end
		local F = R.band_lu(B)
		if F.singular then
			return nil
		end
		return function(b: buffer): buffer
			return R.band_lu_solve(nil, F, b)
		end
	end
	local n = J.rows
	local A = R.matrix(n, n)
	A.data = R.scale(nil, -1, J.data)
	if mass == nil or type(mass) == "buffer" then
		for i = 1, n do
			R.matset(A, i, i, R.matget(A, i, i) + c * mass_diag(R, mass, i))
		end
	else
		A.data = R.axpy(A.data, c, mass.data, A.data)
	end
	local F = R.lu(A)
	if F.singular then
		return nil
	end
	return function(b: buffer): buffer
		return R.lu_solve(nil, F, b)
	end
end

-- A solver for ((cr + i ci) M - J) (xr + i xi) = br + i bi with real J and M, or nil.
local function complex_system(R: any, CM: any, J: any, mass: any, cr: number, ci: number, stats: any): ((buffer, buffer) -> (buffer, buffer))?
	stats.nlu += 1
	local solve: any
	if J.kl ~= nil and J.data ~= nil and J.n ~= nil then
		local Bre, Bim = R.band_copy(J), R.banded(J.n, J.kl, J.ku)
		Bre.data = R.scale(Bre.data, -1, Bre.data)
		for i = 1, J.n do
			local m = mass_diag(R, mass, i)
			R.band_set(Bre, i, i, R.band_get(Bre, i, i) + cr * m)
			R.band_set(Bim, i, i, ci * m)
		end
		local Bc = CM.banded(J.n, J.kl, J.ku)
		Bc.data = CM.complexify(Bre.data, Bim.data)
		local F = CM.band_lu(Bc)
		if F.singular then
			return nil
		end
		solve = function(b: buffer): buffer
			return CM.band_lu_solve(nil, F, b)
		end
	else
		local n = J.rows
		local Are, Aim = R.matrix(n, n), R.matrix(n, n)
		Are.data = R.scale(nil, -1, J.data)
		if mass == nil or type(mass) == "buffer" then
			for i = 1, n do
				local m = mass_diag(R, mass, i)
				R.matset(Are, i, i, R.matget(Are, i, i) + cr * m)
				R.matset(Aim, i, i, ci * m)
			end
		else
			Are.data = R.axpy(Are.data, cr, mass.data, Are.data)
			Aim.data = R.scale(nil, ci, mass.data)
		end
		local A = CM.matrix(n, n)
		A.data = CM.complexify(Are.data, Aim.data)
		local F = CM.lu(A)
		if F.singular then
			return nil
		end
		solve = function(b: buffer): buffer
			return CM.lu_solve(nil, F, b)
		end
	end
	return function(br: buffer, bi: buffer): (buffer, buffer)
		local x = solve(CM.complexify(br, bi))
		return CM.real(nil, x), CM.imag(nil, x)
	end
end

-- Radau IIA, order 5
--
-- Constants as in Hairer's RADAU5, via SciPy's radau.py (BSD-3-Clause; see the README's
-- note on third-party coefficients): the collocation nodes C, the error weights E, the
-- eigenvalues of A^-1 (one real, MU_REAL, and a complex pair MU_CR + i MU_CI) with the
-- transformations T and TI that decouple the 3n by 3n Newton system into one real and one
-- complex n by n system, and the coefficients P of the collocation polynomial.
local SQ6 = math.sqrt(6)
local RC = { (4 - SQ6) / 10, (4 + SQ6) / 10, 1 }
local RE = { (-13 - 7 * SQ6) / 3, (-13 + 7 * SQ6) / 3, -1 / 3 }
local MU_REAL = 3 + 3 ^ (2 / 3) - 3 ^ (1 / 3)
local MU_CR = 3 + 0.5 * (3 ^ (1 / 3) - 3 ^ (2 / 3))
local MU_CI = -0.5 * (3 ^ (5 / 6) + 3 ^ (7 / 6))
local RT = {
	{ 0.09443876248897524, -0.14125529502095421, 0.03002919410514742 },
	{ 0.25021312296533332, 0.20412935229379994, -0.38294211275726192 },
	{ 1, 1, 0 },
}
local RTI = {
	{ 4.17871859155190428, 0.32768282076106237, 0.52337644549944951 },
	{ -4.17871859155190428, -0.32768282076106237, 0.47662355450055044 },
	{ 0.50287263494578682, -2.57192694985560522, 0.59603920482822492 },
}
local RP = {
	{ 13 / 3 + 7 * SQ6 / 3, -23 / 3 - 22 * SQ6 / 3, 10 / 3 + 5 * SQ6 },
	{ 13 / 3 - 7 * SQ6 / 3, -23 / 3 + 22 * SQ6 / 3, 10 / 3 - 5 * SQ6 },
	{ 1 / 3, -8 / 3, 10 / 3 },
}
local RADAU_NEWTON = 6

-- sum_m coeffs[m] X[m] for three values X.
local function mix3(R: any, coeffs: { number }, X: { buffer }): buffer
	local o = R.scale(nil, coeffs[1], X[1])
	o = R.axpy(o, coeffs[2], X[2], o)
	return R.axpy(o, coeffs[3], X[3], o)
end

-- Gustafsson's predictive step-size factor (Hairer and Wanner II, IV.8).
local function predict_factor(h_abs: number, h_abs_old: number?, err: number, err_old: number?): number
	local multiplier = 1
	if err_old ~= nil and h_abs_old ~= nil and err ~= 0 then
		multiplier = h_abs / h_abs_old * (err_old / err) ^ 0.25
	end
	if err == 0 then
		return math.huge
	end
	return math.min(1, multiplier) * err ^ -0.25
end

local function radau(S: any, f: OdeFunction<buffer>, t0: number, y0: buffer, o: any): any
	local R, CM = modules(S)
	local n = R.length(y0)
	local rtol, atol = o.rtol or 1e-3, o.atol or 1e-6
	local maxstep = o.max_step or math.huge
	local mass = o.mass
	assert(not (o.jac_band and mass ~= nil and type(mass) ~= "buffer"), "ode: a banded Jacobian needs a diagonal mass (an array)")
	local stats = { nfev = 1, njev = 0, nlu = 0, steps = 0, rejected = 0 }
	local st: any = { t = t0, y = R.copy(nil, y0), piece = nil, stats = stats }
	local y = st.y
	local fy = f(t0, y, R.create(n))
	local jac = make_jacobian(S, R, f, n, o, stats)
	local J = jac.evaluate(t0, y, fy)
	local current_jac = true
	local LU_real: ((buffer) -> buffer)? = nil
	local LU_complex: ((buffer, buffer) -> (buffer, buffer))? = nil
	local h_abs: number? = if o.h then math.abs(o.h) else nil
	local h_abs_old: number? = nil
	local err_old: number? = nil
	local prev: Piece<buffer>? = nil
	local newton_tol = math.max(10 * (S.eps or EPS) / rtol, math.min(0.03, rtol ^ 0.5))

	local function nonfinite(x: buffer): boolean
		local v = R.norm(x)
		return not (v < math.huge)
	end

	-- The simplified Newton iteration for the stage increments Z (stage values y + Z_i).
	local function collocation(t: number, h: number, Z0: { buffer }): (boolean, number, { buffer }, number)
		local mr, mcr, mci = MU_REAL / h, MU_CR / h, MU_CI / h
		local W = { mix3(R, RTI[1], Z0), mix3(R, RTI[2], Z0), mix3(R, RTI[3], Z0) }
		local Z = Z0
		local dnorm_old: number? = nil
		local rate: number? = nil
		local F: { buffer } = {}
		for k = 1, RADAU_NEWTON do
			for i = 1, 3 do
				F[i] = f(t + RC[i] * h, R.add(nil, y, Z[i]), R.create(n))
				stats.nfev += 1
				if nonfinite(F[i]) then
					return false, k, Z, rate or 0
				end
			end
			local MW1 = mass_apply(R, mass, W[1])
			local MW2 = mass_apply(R, mass, W[2])
			local MW3 = mass_apply(R, mass, W[3])
			local fr = mix3(R, { RTI[1][1], RTI[1][2], RTI[1][3] }, F)
			fr = R.axpy(fr, -mr, MW1, fr)
			-- (mcr + i mci)(W2 + i W3) = (mcr W2 - mci W3) + i (mcr W3 + mci W2)
			local fcr = mix3(R, { RTI[2][1], RTI[2][2], RTI[2][3] }, F)
			fcr = R.axpy(fcr, -mcr, MW2, fcr)
			fcr = R.axpy(fcr, mci, MW3, fcr)
			local fci = mix3(R, { RTI[3][1], RTI[3][2], RTI[3][3] }, F)
			fci = R.axpy(fci, -mcr, MW3, fci)
			fci = R.axpy(fci, -mci, MW2, fci)
			local d1 = (LU_real :: (buffer) -> buffer)(fr)
			local d2, d3 = (LU_complex :: (buffer, buffer) -> (buffer, buffer))(fcr, fci)
			local e1 = weighted(S, d1, y, y, atol, rtol)
			local e2 = weighted(S, d2, y, y, atol, rtol)
			local e3 = weighted(S, d3, y, y, atol, rtol)
			local dnorm = math.sqrt((e1 * e1 + e2 * e2 + e3 * e3) / 3)
			if dnorm_old ~= nil then
				rate = dnorm / dnorm_old
			end
			if rate ~= nil and (rate >= 1 or rate ^ (RADAU_NEWTON - k + 1) / (1 - rate) * dnorm > newton_tol) then
				return false, k, Z, rate
			end
			W = { R.add(nil, W[1], d1), R.add(nil, W[2], d2), R.add(nil, W[3], d3) }
			Z = { mix3(R, RT[1], W), mix3(R, RT[2], W), mix3(R, RT[3], W) }
			if dnorm == 0 or (rate ~= nil and rate / (1 - rate) * dnorm < newton_tol) then
				return true, k, Z, rate or 0
			end
			dnorm_old = dnorm
		end
		return false, RADAU_NEWTON, Z, rate or 0
	end

	function st.step(self: any, t_bound: number, want: boolean): (boolean, string?)
		local t = self.t
		local direction = if t_bound >= t then 1 else -1
		if h_abs == nil then
			local hs, used = core.initial_step(S, f, t, y, fy, direction, math.abs(t_bound - t), atol, rtol, maxstep, 3)
			h_abs = hs
			stats.nfev += used
		end
		local min_step = 10 * EPS * math.max(math.abs(t), 1e-300)
		local hab = h_abs :: number
		local hab_old, eold = h_abs_old, err_old
		if hab > maxstep then
			hab, hab_old, eold = maxstep, nil, nil
		elseif hab < min_step then
			hab, hab_old, eold = min_step, nil, nil
		end
		local rejected = false
		local accepted = false
		local Z: { buffer } = {}
		local ynew: buffer = y
		local tnew = t
		local h = 0
		local n_iter, rate = 0, 0
		local err = 0
		local safety = 0.9
		while not accepted do
			if hab < min_step then
				return false, "the step size became too small"
			end
			h = hab * direction
			tnew = t + h
			if direction * (tnew - t_bound) > 0 or math.abs(t_bound - tnew) <= 4 * EPS * math.abs(t_bound) then
				tnew = t_bound
			end
			h = tnew - t
			hab = math.abs(h)
			local Z0: { buffer }
			if prev then
				local P = prev :: Piece<buffer>
				Z0 = {}
				for i = 1, 3 do
					Z0[i] = R.sub(nil, core.piece_eval(S, P, t + RC[i] * h), y)
				end
			else
				Z0 = { R.create(n), R.create(n), R.create(n) }
			end
			local converged = false
			while not converged do
				if LU_real == nil or LU_complex == nil then
					LU_real = real_system(R, J, mass, MU_REAL / h, stats)
					LU_complex = complex_system(R, CM, J, mass, MU_CR / h, MU_CI / h, stats)
					if LU_real == nil or LU_complex == nil then
						break
					end
				end
				converged, n_iter, Z, rate = collocation(t, h, Z0)
				if not converged then
					if current_jac or not jac.updatable then
						break
					end
					J = jac.evaluate(t, y, fy)
					current_jac = true
					LU_real, LU_complex = nil, nil
				end
			end
			if not converged then
				hab *= 0.5
				LU_real, LU_complex = nil, nil
				stats.rejected += 1
				continue
			end
			ynew = R.add(nil, y, Z[3])
			local ZE = mix3(R, RE, Z)
			ZE = R.scale(ZE, 1 / h, ZE)
			local MZE = mass_apply(R, mass, ZE)
			local errv = (LU_real :: (buffer) -> buffer)(R.add(nil, fy, MZE))
			err = weighted(S, errv, y, ynew, atol, rtol)
			safety = 0.9 * (2 * RADAU_NEWTON + 1) / (2 * RADAU_NEWTON + n_iter)
			if rejected and err > 1 then
				local fe = f(t, R.add(nil, y, errv), R.create(n))
				stats.nfev += 1
				errv = (LU_real :: (buffer) -> buffer)(R.add(nil, fe, MZE))
				err = weighted(S, errv, y, ynew, atol, rtol)
			end
			if err ~= err then
				return false, "the solution became NaN"
			end
			if err > 1 then
				local factor = predict_factor(hab, hab_old, err, eold)
				hab *= math.max(0.2, safety * factor)
				LU_real, LU_complex = nil, nil
				rejected = true
				stats.rejected += 1
			else
				accepted = true
			end
		end
		local recompute = jac.updatable and n_iter > 2 and rate > 1e-3
		local factor = math.min(10, safety * predict_factor(hab, hab_old, err, eold))
		if not recompute and factor < 1.2 then
			factor = 1
		else
			LU_real, LU_complex = nil, nil
		end
		local fnew = f(tnew, ynew, R.create(n))
		stats.nfev += 1
		if recompute then
			J = jac.evaluate(tnew, ynew, fnew)
			current_jac = true
		elseif jac.updatable then
			current_jac = false
		end
		h_abs_old = h_abs
		err_old = err
		h_abs = hab * factor
		-- The collocation polynomial y_old + sum_j Q_j theta^j, Q = Z P (stored divided by h,
		-- as "poly" pieces multiply by h).
		local q: { buffer } = {}
		for j = 1, 3 do
			local Qj = mix3(R, { RP[1][j], RP[2][j], RP[3][j] }, Z)
			q[j] = R.scale(Qj, 1 / h, Qj)
		end
		local piece: Piece<buffer> = { t0 = t, h = h, y0 = y, kind = "poly", q = q }
		prev = piece
		self.piece = if want then piece else nil
		y, fy = ynew, fnew
		self.t, self.y = tnew, ynew
		stats.steps += 1
		return true, nil
	end
	return st
end

local STEPPERS: { [string]: any } = { radau = radau }

-- Vector states: the stepper runs on f64 arrays of length 3, converting at the edges
-- (a user Jacobian or mass matrix is 3 by 3).
local function vector_stepper(make: any, f: OdeFunction<vector>, t0: number, y0: vector, o: any): any
	local function tobuf(v: vector, out: buffer?): buffer
		local b = out or buffer.create(24)
		buffer.writef64(b, 0, v.x)
		buffer.writef64(b, 8, v.y)
		buffer.writef64(b, 16, v.z)
		return b
	end
	local function tovec(b: buffer): vector
		return vector.create(buffer.readf64(b, 0), buffer.readf64(b, 8), buffer.readf64(b, 16))
	end
	local fb = function(t: number, y: buffer, out: buffer): buffer
		return tobuf(f(t, tovec(y), vector.zero), out)
	end
	local ob = table.clone(o)
	if type(o.jac) == "function" then
		local jac: any = o.jac
		ob.jac = function(t: number, y: buffer): any
			return jac(t, tovec(y))
		end
	end
	local inner = make(f64.space, fb, t0, tobuf(y0), ob)
	local st: any = { t = t0, y = y0, piece = nil, stats = inner.stats }
	function st.step(self: any, t_bound: number, want: boolean): (boolean, string?)
		local ok, why = inner:step(t_bound, want)
		self.t, self.y = inner.t, tovec(inner.y)
		local P = inner.piece
		self.piece = if want and P
			then {
				t0 = P.t0,
				h = P.h,
				y0 = tovec(P.y0),
				kind = "eval",
				q = {},
				eval = function(t: number): vector
					return tovec(core.piece_eval(f64.space, P, t))
				end,
			}
			else nil
		return ok, why
	end
	return st
end

-- A stepper for a stiff method (see ode.luau's driver).
local function stepper(S: any, f: any, t0: number, y0: any, o: any): any
	local make: any = STEPPERS[o.method]
	assert(make ~= nil, `ode: unknown method {o.method}`)
	if S == vec.space then
		return vector_stepper(make, f, t0, y0, o)
	end
	return (make :: any)(S, f, t0, y0, o)
end

return {
	stepper = stepper,
	methods = { radau = true } :: { [string]: boolean },
}
