--!strict
--!native
--!optimize 2

-- Stiff ODE (and index-1 DAE) solvers, used through ode.solve and ode.integrator with
-- options.method:
--   "radau"  Radau IIA of order 5 (Hairer and Wanner's RADAU5): fully implicit, L-stable
--            and stiffly accurate; the most robust choice, including tight tolerances,
--            very stiff problems, oscillatory stiff problems and DAEs. Real states (f32
--            or f64 arrays, and vectors) with dense or banded Jacobians.
--   "rodas5p"  Steinebach's Rosenbrock method of order 5: no Newton iterations, one
--            factorization and 8 solves per step, with J afresh each step; the most
--            efficient for small and medium systems at moderate to tight tolerances.
--            Real and complex states, dense or banded J. options.autonomous = true
--            skips the df/dt estimate (an evaluation of f per step) when f ignores t.
--   "bdf"    variable-order (1 to 5) numerical differentiation formulas (Shampine and
--            Reichelt's NDFs, as in MATLAB's ode15s and SciPy's BDF): one n by n solve
--            per Newton iteration, so the cheapest per step on large systems, and the one
--            with sparse and matrix-free (Newton-Krylov) linear algebra and complex
--            states. Orders 3 to 5 are not A-stable: for stiff oscillations (eigenvalues
--            near the imaginary axis) prefer radau.
-- All take M y' = f(t, y) with an optional constant mass matrix M (options.mass: a
-- matrix, a sparse matrix (bdf), or an array for a diagonal M, in the state's module; a
-- singular M makes it a DAE, whose initial state must be consistent).
--
-- radau and bdf are translations of SciPy's radau.py and bdf.py (BSD-3-Clause; see
-- licenses/SciPy-LICENSE.txt), and rodas5p uses OrdinaryDiffEq.jl's Rodas5P tableau (MIT;
-- see licenses/OrdinaryDiffEq.jl-LICENSE.md).
--
-- The Jacobian df/dy and the linear algebra:
--   jac           a function (t, y) -> J, or a constant J, where J is a matrix, a banded
--                 matrix or a sparse matrix (whose pattern must not change); without it
--                 J is estimated by finite differences
--   jac_band      { kl, ku }: J is banded, so the estimate takes kl + ku + 1 evaluations
--                 of f and the linear algebra is banded (M must then be diagonal)
--   jac_sparsity  a sparse matrix with J's pattern (bdf): the estimate groups columns
--                 that share no row (Curtis, Powell and Reid), one evaluation per group
--   linear        "direct" (LU, dense or banded; the default) or "krylov" (bdf): GMRES,
--                 preconditioned by ILU(0) for sparse J (the default for sparse J); with
--                 "krylov" and no jac, jac_band or jac_sparsity, Newton-Krylov is
--                 matrix-free: products J v by differences of f, no J at all
--   precond       for "krylov": a function (t, y, a, b) returning a preconditioner, an
--                 operator (out, r) -> out applying an approximate inverse of a M + b J
--                 at (t, y); it replaces ILU(0)
--   krylov        { restart, maxiter } for GMRES (defaults 30 and 300)
-- For complex states f must be complex-differentiable in y (the differences take real
-- steps). The solves run in the state's precision.

local types = require("./types")
local core = require("./odecore")
local krylov = require("./krylov")
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

local COMPLEX: { [any]: boolean } = { [c32] = true, [c64] = true }
-- The complex module of each real one (for Radau's complex-shifted systems).
local COMPLEXIFIED: { [any]: any } = { [f32] = c32, [f64] = c64 }

-- The array module whose space S is.
local function module_of(S: any): any
	for _, M in { f64, f32, c64, c32 } :: { any } do
		if S == M.space then
			return M
		end
	end
	error("ode: the stiff solvers need f32, f64, c32 or c64 arrays (or vectors)")
end

-- Element access as (re, im) pairs for every module: c64 takes pairs, c32 takes vector
-- scalars (x, y = re, im) and the real modules plain numbers.
local function eget(R: any, x: buffer, i: number): (number, number)
	if R == c32 then
		local v = R.get(x, i)
		return v.x, v.y
	elseif R == c64 then
		return R.get(x, i)
	end
	return R.get(x, i), 0
end

local function eset(R: any, x: buffer, i: number, re: number, im: number)
	if R == c32 then
		R.set(x, i, vector.create(re, im, 0))
	elseif R == c64 then
		R.set(x, i, re, im)
	else
		R.set(x, i, re)
	end
end

-- Entry (i, j) of a dense or banded matrix as (re, im), and setting it.
local function aget(R: any, A: any, i: number, j: number): (number, number)
	local banded = A.ld ~= nil
	if R == c32 then
		local v = if banded then R.band_get(A, i, j) else R.matget(A, i, j)
		return v.x, v.y
	elseif R == c64 then
		if banded then
			return R.band_get(A, i, j)
		end
		return R.matget(A, i, j)
	end
	return if banded then R.band_get(A, i, j) else R.matget(A, i, j), 0
end

local function aset(R: any, A: any, i: number, j: number, re: number, im: number)
	local banded = A.ld ~= nil
	local setter = if banded then R.band_set else R.matset
	if R == c32 then
		setter(A, i, j, vector.create(re, im, 0))
	elseif R == c64 then
		setter(A, i, j, re, im)
	else
		setter(A, i, j, re)
	end
end

-- A[i, i] += re + i im, for dense and banded matrices.
local function add_diag(R: any, A: any, i: number, re: number, im: number)
	local ar, ai = aget(R, A, i, i)
	aset(R, A, i, i, ar + re, ai + im)
end

-- What kind of Jacobian J is: "dense", "band", "sparse", or "free" (a linearization
-- point for matrix-free products).
local function kind_of(J: any): string
	if J.free then
		return "free"
	elseif J.colptr ~= nil then
		return "sparse"
	elseif J.ld ~= nil then
		return "band"
	end
	return "dense"
end

-- Mass matrices

-- M x (a copy of x without a mass matrix), into out (a new array if nil; not x).
local function mass_apply(R: any, mass: any, x: buffer, out: buffer?): buffer
	if mass == nil then
		return R.copy(out, x)
	elseif type(mass) == "buffer" then
		return R.mul(out, mass, x)
	elseif mass.colptr ~= nil then
		return R.sparse_matvec(out, mass, x)
	end
	return R.matvec(out, mass, x)
end

-- M[i, i] as (re, im).
local function mass_diag(R: any, mass: any, i: number): (number, number)
	if mass == nil then
		return 1, 0
	elseif type(mass) == "buffer" then
		return eget(R, mass, i)
	end
	return aget(R, mass, i, i)
end

-- y'(t0) from M y' = f0: M^-1 f0, or for a singular M its minimum-norm solution (zero in
-- the algebraic components of a diagonal M).
local function initial_slope(R: any, mass: any, f0: buffer): buffer
	if mass == nil then
		return f0
	elseif type(mass) == "buffer" then
		local n = R.length(f0)
		local yp = R.create(n)
		for i = 1, n do
			local mr, mi = eget(R, mass, i)
			local fr, fi = eget(R, f0, i)
			local d = mr * mr + mi * mi
			if d > 0 then
				eset(R, yp, i, (fr * mr + fi * mi) / d, (fi * mr - fr * mi) / d)
			end
		end
		return yp
	end
	local Md = if mass.colptr ~= nil then R.sparse_matrix(mass) else mass
	return R.matvec(nil, R.pinv(Md), f0)
end

-- f with M^-1 applied (as initial_slope), for the starting step size.
local function slope_function(R: any, f: OdeFunction<buffer>, mass: any): OdeFunction<buffer>
	if mass == nil then
		return f
	end
	return function(t: number, y: buffer, out: buffer): buffer
		return initial_slope(R, mass, f(t, y, out))
	end
end

-- Jacobians

-- evaluate(t, y, fy?) gives J at (t, y), where fy = f(t, y) if known; updatable is false
-- for a constant J.
type Jacobian = { evaluate: (number, buffer, buffer?) -> any, updatable: boolean }

local function make_jacobian(R: any, f: OdeFunction<buffer>, n: number, o: any, stats: any): Jacobian
	local band = o.jac_band
	local pattern = o.jac_sparsity
	local eps = math.sqrt(R.space.eps or EPS)
	if o.jac ~= nil then
		if type(o.jac) == "function" then
			local user: any = o.jac
			return {
				evaluate = function(t: number, y: buffer): any
					stats.njev += 1
					return user(t, y)
				end,
				updatable = true,
			}
		end
		local Jc = o.jac
		return {
			evaluate = function(): any
				return Jc
			end,
			updatable = false,
		}
	end
	-- Workspace for the finite differences, reused by every evaluation (the J returned is
	-- overwritten by the next one: each solver uses only its latest J).
	local fyw, ypw, fpw = R.create(n), R.create(n), R.create(n)
	local function value_at(t: number, y: buffer, fy: buffer?): buffer
		if fy then
			return fy
		end
		stats.nfev += 1
		fyw = f(t, y, fyw)
		return fyw
	end
	if (o.linear or "direct") == "krylov" and band == nil and pattern == nil then
		-- Matrix-free: J is the linearization point.
		return {
			evaluate = function(t: number, y: buffer, fy: buffer?): any
				return { free = true, t = t, y = R.copy(nil, y), fy = R.copy(nil, value_at(t, y, fy)) }
			end,
			updatable = true,
		}
	end
	-- Finite differences: perturb the real part of y_j by about sqrt(eps) |y_j|.
	local function perturb(yp: buffer, y: buffer, j: number): number
		local yr, yi = eget(R, y, j)
		local d = eps * math.max(math.sqrt(yr * yr + yi * yi), 1)
		eset(R, yp, j, yr + (if yr < 0 then -d else d), yi)
		return (eget(R, yp, j)) - yr
	end
	-- Column groups: columns in a group touch disjoint rows, so one evaluation of f
	-- estimates all of them. groups[g] lists the columns; rows(j) their row range.
	local groups: { { number } } = {}
	local colptr: buffer?, rowind: buffer? = nil, nil
	if band then
		local w = band[1] + band[2] + 1
		for g = 1, math.min(w, n) do
			local cols = {}
			for j = g, n, w do
				table.insert(cols, j)
			end
			groups[g] = cols
		end
	elseif pattern then
		assert(pattern.rows == n and pattern.cols == n, "ode: jac_sparsity has the wrong size")
		colptr, rowind = pattern.colptr, pattern.rowind
		local occupied: { { [number]: boolean } } = {}
		for j = 1, n do
			local s, e = buffer.readu32(pattern.colptr, 4 * (j - 1)), buffer.readu32(pattern.colptr, 4 * j) - 1
			local placed = false
			for g, occ in occupied do
				local free = true
				for p = s, e do
					if occ[buffer.readu32(pattern.rowind, 4 * p)] then
						free = false
						break
					end
				end
				if free then
					for p = s, e do
						occ[buffer.readu32(pattern.rowind, 4 * p)] = true
					end
					table.insert(groups[g], j)
					placed = true
					break
				end
			end
			if not placed then
				local occ = {}
				for p = s, e do
					occ[buffer.readu32(pattern.rowind, 4 * p)] = true
				end
				table.insert(occupied, occ)
				table.insert(groups, { j })
			end
		end
	end
	if band or pattern then
		local kl, ku = if band then band[1] else 0, if band then band[2] else 0
		local J: any
		if band then
			J = R.banded(n, kl, ku)
		else
			local pat = pattern :: any
			J = { rows = n, cols = n, colptr = pat.colptr, rowind = pat.rowind, values = R.create(buffer.len(pat.rowind) // 4) }
		end
		local steps: { [number]: number } = {}
		return {
			evaluate = function(t: number, y: buffer, fy0: buffer?): any
				stats.njev += 1
				local fy = value_at(t, y, fy0)
				local yp = R.copy(ypw, y)
				local fp = fpw
				for _, cols in groups do
					for _, j in cols do
						steps[j] = perturb(yp, y, j)
					end
					fp = f(t, yp, fp)
					fpw = fp
					stats.nfev += 1
					for _, j in cols do
						local d = steps[j]
						if band then
							for i = math.max(1, j - ku), math.min(n, j + kl) do
								local ar, ai = eget(R, fp, i)
								local br, bi = eget(R, fy, i)
								aset(R, J, i, j, (ar - br) / d, (ai - bi) / d)
							end
						else
							for p = buffer.readu32(colptr :: buffer, 4 * (j - 1)), buffer.readu32(colptr :: buffer, 4 * j) - 1 do
								local i = buffer.readu32(rowind :: buffer, 4 * p) + 1
								local ar, ai = eget(R, fp, i)
								local br, bi = eget(R, fy, i)
								eset(R, J.values, p + 1, (ar - br) / d, (ai - bi) / d)
							end
						end
						local yr, yi = eget(R, y, j)
						eset(R, yp, j, yr, yi)
					end
				end
				return J
			end,
			updatable = true,
		}
	end
	local J = R.matrix(n, n)
	local colw = R.create(n)
	return {
		evaluate = function(t: number, y: buffer, fy0: buffer?): any
			stats.njev += 1
			local fy = value_at(t, y, fy0)
			local yp = R.copy(ypw, y)
			local fp = fpw
			local bytes = buffer.len(y)
			for j = 1, n do
				local d = perturb(yp, y, j)
				fp = f(t, yp, fp)
				fpw = fp
				stats.nfev += 1
				local col = R.sub(colw, fp, fy)
				col = R.scale(col, 1 / d, col)
				buffer.copy(J.data, bytes * (j - 1), col)
				local yr, yi = eget(R, y, j)
				eset(R, yp, j, yr, yi)
			end
			return J
		end,
		updatable = true,
	}
end

-- Linear systems

-- A factory of solvers for (a M + b J) x = r with real a and b: linear(J, a, b, t, y)
-- returns solve(r, out?) -> x (into out if given, which may be r), or nil if the matrix is
-- singular; (t, y) is where the Newton iteration is, for a user preconditioner. ltol is
-- the relative residual for GMRES. With direct solves the matrix is assembled into one
-- workspace, so a new factorization replaces the one before (each solver uses only its
-- latest).
local function make_linear(R: any, f: OdeFunction<buffer>, n: number, mass: any, o: any, stats: any, ltol: number): (any, number, number, number, buffer) -> ((buffer, buffer?) -> buffer)?
	local linear = o.linear or (if o.jac_sparsity then "krylov" else "direct")
	local kopts = o.krylov or {}
	local S = R.space
	local eps = math.sqrt(S.eps or EPS)

	-- The diagonal of a M as a sparse matrix.
	local function sparse_mass(a: number): any
		if mass ~= nil and type(mass) ~= "buffer" then
			assert(mass.colptr ~= nil, "ode: with a sparse Jacobian the mass must be sparse or diagonal")
			return { rows = n, cols = n, colptr = mass.colptr, rowind = mass.rowind, values = R.scale(nil, a, mass.values) }
		end
		local cp, ri = buffer.create(4 * (n + 1)), buffer.create(4 * n)
		local vals = R.create(n)
		for i = 0, n do
			buffer.writeu32(cp, 4 * i, i)
		end
		for i = 1, n do
			buffer.writeu32(ri, 4 * (i - 1), i - 1)
			local mr, mi = mass_diag(R, mass, i)
			eset(R, vals, i, a * mr, a * mi)
		end
		return { rows = n, cols = n, colptr = cp, rowind = ri, values = vals }
	end

	-- a M + b J as a dense or banded matrix, in the workspace Aw (of J's shape).
	local Aw: any = nil
	local Mdense: any = nil
	local function assemble(J: any, a: number, b: number): any
		if Aw == nil or buffer.len(Aw.data) ~= buffer.len(J.data) or Aw.ld ~= J.ld then
			Aw = if J.ld ~= nil then R.band_copy(J) else R.matrix(n, n)
		end
		local A = Aw
		buffer.copy(A.data, 0, J.data)
		R.scale(A.data, b, A.data)
		if mass ~= nil and type(mass) ~= "buffer" and J.ld == nil then
			if Mdense == nil then
				Mdense = if mass.colptr ~= nil then R.sparse_matrix(mass) else mass
			end
			R.axpy(A.data, a, Mdense.data, A.data)
		else
			assert(mass == nil or type(mass) == "buffer", "ode: a banded Jacobian needs a diagonal mass (an array)")
			for i = 1, n do
				local mr, mi = mass_diag(R, mass, i)
				add_diag(R, A, i, a * mr, a * mi)
			end
		end
		return A
	end

	-- the current factorization, and solvers made once
	local Fcur: any = nil
	local function solve_dense(r: buffer, out: buffer?): buffer
		return R.lu_solve(out, Fcur, r)
	end
	local function solve_band(r: buffer, out: buffer?): buffer
		return R.band_lu_solve(out, Fcur, r)
	end
	return function(J: any, a: number, b: number, t: number, y: buffer): ((buffer, buffer?) -> buffer)?
		stats.nlu += 1
		local kind = kind_of(J)
		if linear == "direct" then
			assert(kind == "dense" or kind == "band", "ode: sparse Jacobians need options.linear = \"krylov\"")
			local A = assemble(J, a, b)
			if kind == "band" then
				Fcur = R.band_lu(A, Fcur)
				if Fcur.singular then
					return nil
				end
				return solve_band
			end
			Fcur = R.lu(A, Fcur)
			if Fcur.singular then
				return nil
			end
			return solve_dense
		end
		-- GMRES on the operator a M + b J.
		local op: (buffer, buffer) -> buffer
		local pre: any = nil
		if kind == "free" then
			local yl, fl = J.y, J.fy
			local ynorm = R.norm(yl)
			op = function(out: buffer, x: buffer): buffer
				local xn = R.norm(x)
				local r = mass_apply(R, mass, x)
				r = R.scale(r, a, r)
				if xn == 0 then
					return R.copy(out, r)
				end
				local sigma = eps * (1 + ynorm) / xn
				local fp = f(J.t, R.axpy(nil, sigma, x, yl), R.create(n))
				stats.nfev += 1
				local jv = R.sub(fp, fp, fl)
				r = R.axpy(r, b / sigma, jv, r)
				return R.copy(out, r)
			end
		elseif kind == "sparse" then
			local A = R.sparse_add({ rows = n, cols = n, colptr = J.colptr, rowind = J.rowind, values = R.scale(nil, b, J.values) }, sparse_mass(a))
			op = function(out: buffer, x: buffer): buffer
				return R.sparse_matvec(out, A, x)
			end
			if o.precond == nil then
				local F = R.sparse_ilu0_copy(A)
				if not F.singular then
					pre = function(out: buffer, r: buffer): buffer
						return R.sparse_ilu0_solve(out, F, r)
					end
				end
			end
		else
			local A = assemble(J, a, b)
			op = function(out: buffer, x: buffer): buffer
				return if kind == "band" then R.band_matvec(out, A, x) else R.matvec(out, A, x)
			end
		end
		local user: any = o.precond
		if user ~= nil then
			pre = (user :: any)(t, y, a, b)
		end
		local opts: any = { tol = ltol, restart = kopts.restart or 30, maxiter = kopts.maxiter or 300, precond = pre }
		return function(r: buffer, out: buffer?): buffer
			local x, info = krylov.gmres(S, op, r, opts)
			stats.nlinear = (stats.nlinear or 0) + info.iterations
			return if out then R.copy(out, x) else x
		end
	end
end

-- An upper bound on the spectral radius of J: its largest absolute row sum (nil for
-- matrix-free J). The automatic method choice compares h times it with the explicit
-- method's stability interval.
local function jac_norm(R: any, J: any, n: number): number?
	local kind = kind_of(J)
	if kind == "free" then
		return nil
	elseif kind == "dense" then
		return R.matnorm(J, "inf")
	end
	local rows = table.create(n, 0)
	if kind == "band" then
		for j = 1, n do
			for i = math.max(1, j - J.ku), math.min(n, j + J.kl) do
				local ar, ai = aget(R, J, i, j)
				rows[i] += math.sqrt(ar * ar + ai * ai)
			end
		end
	else
		for j = 1, n do
			for p = buffer.readu32(J.colptr, 4 * (j - 1)), buffer.readu32(J.colptr, 4 * j) - 1 do
				local i = buffer.readu32(J.rowind, 4 * p) + 1
				local ar, ai = eget(R, J.values, p + 1)
				rows[i] += math.sqrt(ar * ar + ai * ai)
			end
		end
	end
	local worst = 0
	for i = 1, n do
		worst = math.max(worst, rows[i])
	end
	return worst
end

-- A solver for ((cr + i ci) M - J) (xr + i xi) = br + i bi with real J and M, or nil.
local function complex_system(R: any, CM: any, n: number, mass: any, stats: any): (any, number, number) -> ((buffer, buffer, buffer, buffer) -> ())?
	-- Real and complex arrays share a layout (complex entries are (re, im) pairs of reals of
	-- width W), so the matrix and right-hand side are written directly. The complex matrix
	-- is one workspace, refilled for each factorization.
	local W = buffer.len(R.create(1))
	local rd: (buffer, number) -> number = if W == 8 then buffer.readf64 else buffer.readf32
	local wr: (buffer, number, number) -> () = if W == 8 then buffer.writef64 else buffer.writef32
	local C: any = nil
	local F: any = nil
	local cb = CM.create(n)
	local function solve(br: buffer, bi: buffer, outr: buffer, outi: buffer)
		for k = 0, n - 1 do
			wr(cb, 2 * W * k, rd(br, W * k))
			wr(cb, 2 * W * k + W, rd(bi, W * k))
		end
		if F.lu.ld ~= nil then
			CM.band_lu_solve(cb, F, cb)
		else
			CM.lu_solve(cb, F, cb)
		end
		CM.real(outr, cb)
		CM.imag(outi, cb)
	end
	-- adds (re, im) to complex entry k (0-based) of C
	local function add_at(k: number, re: number, im: number)
		local o = 2 * W * k
		wr(C.data, o, rd(C.data, o) + re)
		wr(C.data, o + W, rd(C.data, o + W) + im)
	end
	-- Factors cr M - J + i ci M; returns solve(br, bi, outr, outi), or nil if singular.
	return function(J: any, cr: number, ci: number): ((buffer, buffer, buffer, buffer) -> ())?
		stats.nlu += 1
		local banded = J.kl ~= nil and J.data ~= nil and J.n ~= nil
		if C == nil then
			C = if banded then CM.banded(n, J.kl, J.ku) else CM.matrix(n, n)
		end
		-- -J, entry for entry (the same layout), then the mass terms
		for k = 0, buffer.len(J.data) // W - 1 do
			wr(C.data, 2 * W * k, -rd(J.data, W * k))
			wr(C.data, 2 * W * k + W, 0)
		end
		if banded then
			-- entry (i, i) is element (i - 1) ld + kl + ku of the band storage
			for i = 1, n do
				local m = (mass_diag(R, mass, i))
				add_at((i - 1) * C.ld + C.kl + C.ku, cr * m, ci * m)
			end
			F = CM.band_lu(C, F)
		else
			if mass == nil or type(mass) == "buffer" then
				for i = 1, n do
					local m = (mass_diag(R, mass, i))
					add_at((i - 1) * n + i - 1, cr * m, ci * m)
				end
			else
				for k = 0, n * n - 1 do
					local m = rd(mass.data, W * k)
					add_at(k, cr * m, ci * m)
				end
			end
			F = CM.lu(C, F)
		end
		if F.singular then
			return nil
		end
		return solve
	end
end

-- Radau IIA, order 5
--
-- Constants as in Hairer's RADAU5, via SciPy's radau.py (BSD-3-Clause; see the third-party
-- code in docs/development.md): the collocation nodes C, the error weights E, the
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
-- RP's columns
local RPC = { { RP[1][1], RP[2][1], RP[3][1] }, { RP[1][2], RP[2][2], RP[3][2] }, { RP[1][3], RP[2][3], RP[3][3] } }

-- sum_m coeffs[m] X[m] for three values X, into out (not one of X).
local function mix3(R: any, coeffs: { number }, X: { buffer }, out: buffer?): buffer
	local o = R.scale(out, coeffs[1], X[1])
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
	local R = module_of(S)
	assert(not COMPLEX[R], "ode: radau needs real states (bdf takes complex ones)")
	assert((o.linear or "direct") == "direct" and o.jac_sparsity == nil, "ode: radau needs dense or banded linear algebra")
	local CM = COMPLEXIFIED[R]
	local n = R.length(y0)
	local rtol, atol = o.rtol or 1e-3, o.atol or 1e-6
	local maxstep = o.max_step or math.huge
	local mass = o.mass
	assert(not (o.jac_band and mass ~= nil and type(mass) ~= "buffer"), "ode: a banded Jacobian needs a diagonal mass (an array)")
	assert(mass == nil or type(mass) == "buffer" or mass.colptr == nil, "ode: radau needs a dense or diagonal mass")
	local stats = { nfev = 1, njev = 0, nlu = 0, steps = 0, rejected = 0 }
	local st: any = { t = t0, y = R.copy(nil, y0), piece = nil, stats = stats }
	local y = st.y
	local fy = f(t0, y, R.create(n))
	local jac = make_jacobian(R, f, n, o, stats)
	local linear = make_linear(R, f, n, mass, o, stats, 0)
	local yp0 = initial_slope(R, mass, fy)
	local J = jac.evaluate(t0, y, fy)
	local Jnorm = jac_norm(R, J, n)
	local current_jac = true
	local complex = complex_system(R, CM, n, mass, stats)
	local LU_real: ((buffer, buffer?) -> buffer)? = nil
	local LU_complex: ((buffer, buffer, buffer, buffer) -> ())? = nil
	-- Workspace for the Newton iteration and the step (a step allocates nothing beyond
	-- its dense output): W and Z (transformed and plain stage increments), F (stage
	-- slopes), the three right-hand sides and corrections, M W, a stage argument; the
	-- new state and slope (fnew alternates with fy), the error terms, the starting guess
	-- and the collocation polynomial kept for it (prev).
	local function arrays(k: number): { buffer }
		local t = {}
		for i = 1, k do
			t[i] = R.create(n)
		end
		return t
	end
	local Ww, Zw, Fw, Dw, MW, Z0w = arrays(3), arrays(3), arrays(3), arrays(3), arrays(3), arrays(3)
	local fr_w, fcr_w, fci_w, arg_w = R.create(n), R.create(n), R.create(n), R.create(n)
	local ynw, fnew_w, ze_w, mze_w, errv_w, tmp_w = R.create(n), R.create(n), R.create(n), R.create(n), R.create(n), R.create(n)
	local prev_y0, prev_q = R.create(n), arrays(3)
	local prev_w: Piece<buffer> = { t0 = 0, h = 0, y0 = prev_y0, kind = "poly", q = prev_q }
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
		local W = Ww
		for i = 1, 3 do
			W[i] = mix3(R, RTI[i], Z0, W[i])
		end
		local Z = Zw
		for i = 1, 3 do
			Z[i] = R.copy(Z[i], Z0[i])
		end
		local dnorm_old: number? = nil
		local rate: number? = nil
		local F = Fw
		for k = 1, RADAU_NEWTON do
			for i = 1, 3 do
				arg_w = R.add(arg_w, y, Z[i])
				F[i] = f(t + RC[i] * h, arg_w, F[i])
				stats.nfev += 1
				if nonfinite(F[i]) then
					return false, k, Z, rate or 0
				end
			end
			for i = 1, 3 do
				MW[i] = mass_apply(R, mass, W[i], MW[i])
			end
			local fr = mix3(R, RTI[1], F, fr_w)
			fr = R.axpy(fr, -mr, MW[1], fr)
			-- (mcr + i mci)(W2 + i W3) = (mcr W2 - mci W3) + i (mcr W3 + mci W2)
			local fcr = mix3(R, RTI[2], F, fcr_w)
			fcr = R.axpy(fcr, -mcr, MW[2], fcr)
			fcr = R.axpy(fcr, mci, MW[3], fcr)
			local fci = mix3(R, RTI[3], F, fci_w)
			fci = R.axpy(fci, -mcr, MW[3], fci)
			fci = R.axpy(fci, -mci, MW[2], fci)
			local d1 = (LU_real :: (buffer, buffer?) -> buffer)(fr, Dw[1])
			Dw[1] = d1
			;(LU_complex :: (buffer, buffer, buffer, buffer) -> ())(fcr, fci, Dw[2], Dw[3])
			local d2, d3 = Dw[2], Dw[3]
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
			W[1] = R.add(W[1], W[1], d1)
			W[2] = R.add(W[2], W[2], d2)
			W[3] = R.add(W[3], W[3], d3)
			for i = 1, 3 do
				Z[i] = mix3(R, RT[i], W, Z[i])
			end
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
			local hs, used = core.initial_step(S, slope_function(R, f, mass), t, y, yp0, direction, math.abs(t_bound - t), atol, rtol, maxstep, 3)
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
		local Z: { buffer } = Zw
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
			local Z0 = Z0w
			for i = 1, 3 do
				if prev then
					Z0[i] = core.piece_eval(S, prev :: Piece<buffer>, t + RC[i] * h, Z0[i])
					Z0[i] = R.sub(Z0[i], Z0[i], y)
				else
					buffer.fill(Z0[i], 0, 0)
				end
			end
			local converged = false
			while not converged do
				if LU_real == nil or LU_complex == nil then
					LU_real = linear(J, MU_REAL / h, -1, t, y)
					LU_complex = complex(J, MU_CR / h, MU_CI / h)
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
					Jnorm = jac_norm(R, J, n)
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
			ynw = R.add(ynw, y, Z[3])
			ynew = ynw
			ze_w = mix3(R, RE, Z, ze_w)
			ze_w = R.scale(ze_w, 1 / h, ze_w)
			mze_w = mass_apply(R, mass, ze_w, mze_w)
			tmp_w = R.add(tmp_w, fy, mze_w)
			errv_w = (LU_real :: (buffer, buffer?) -> buffer)(tmp_w, errv_w)
			err = weighted(S, errv_w, y, ynew, atol, rtol)
			safety = 0.9 * (2 * RADAU_NEWTON + 1) / (2 * RADAU_NEWTON + n_iter)
			if rejected and err > 1 then
				arg_w = R.add(arg_w, y, errv_w)
				local fe = f(t, arg_w, tmp_w)
				stats.nfev += 1
				tmp_w = R.add(fe, fe, mze_w)
				errv_w = (LU_real :: (buffer, buffer?) -> buffer)(tmp_w, errv_w)
				err = weighted(S, errv_w, y, ynew, atol, rtol)
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
		local fnew = f(tnew, ynew, fnew_w)
		stats.nfev += 1
		if recompute then
			J = jac.evaluate(tnew, ynew, fnew)
			Jnorm = jac_norm(R, J, n)
			current_jac = true
		elseif jac.updatable then
			current_jac = false
		end
		h_abs_old = h_abs
		err_old = err
		h_abs = hab * factor
		-- The collocation polynomial y_old + sum_j Q_j theta^j, Q = Z P (stored divided by h,
		-- as "poly" pieces multiply by h): kept for the next step's starting guess (in
		-- workspace), and new arrays when the dense output keeps it.
		local q: { buffer } = prev_q
		for j = 1, 3 do
			local Qj = mix3(R, RPC[j], Z, q[j])
			q[j] = R.scale(Qj, 1 / h, Qj)
		end
		local y0 = R.copy(prev_y0, y)
		local piece = prev_w
		piece.t0, piece.h, piece.y0, piece.q = t, h, y0, q
		prev = piece
		self.piece = if want then piece else nil
		-- the state in place; fnew's array becomes the next step's spare
		y = R.copy(y, ynew)
		fnew_w, fy = fy, fnew
		self.t, self.y = tnew, y
		self.stiffness = if Jnorm then hab * (Jnorm :: number) else nil
		stats.steps += 1
		return true, nil
	end
	return st
end

-- Numerical differentiation formulas, orders 1 to 5
--
-- The quasi-constant step NDFs of Shampine and Reichelt (the MATLAB ODE suite), in the
-- form of SciPy's bdf.py: the state is a table D of backward differences of the
-- interpolating polynomial at the current step size, rescaled when the step changes, and
-- the order follows the error estimates of the neighboring orders. The constants KAPPA
-- (Shampine and Reichelt's NDF coefficients) and the update formulas are as in SciPy's
-- bdf.py (BSD-3-Clause; see the third-party code in docs/development.md).
local BDF_MAX_ORDER = 5
local BDF_NEWTON = 4
local NDF_KAPPA: { [number]: number } = { [0] = 0, [1] = -0.1850, [2] = -1 / 9, [3] = -0.0823, [4] = -0.0415, [5] = 0 }
local NDF_GAMMA: { [number]: number } = { [0] = 0 }
local NDF_ALPHA: { [number]: number } = {}
local NDF_ERROR: { [number]: number } = {}
for k = 1, BDF_MAX_ORDER do
	NDF_GAMMA[k] = NDF_GAMMA[k - 1] + 1 / k
end
for k = 0, BDF_MAX_ORDER do
	NDF_ALPHA[k] = (1 - NDF_KAPPA[k]) * NDF_GAMMA[k]
	NDF_ERROR[k] = NDF_KAPPA[k] * NDF_GAMMA[k] + 1 / (k + 1)
end

-- The matrix (0 .. order square) that re-expresses the differences for a step size
-- changed by factor, into M (rows 0 .. order, kept and reused).
local function compute_R(order: number, factor: number, M: { [number]: { [number]: number } }): { [number]: { [number]: number } }
	for i = 0, order do
		local row = M[i]
		if row == nil then
			row = {}
			M[i] = row
		end
		for j = 0, order do
			row[j] = if i == 0 then 1 elseif j == 0 then 0 else (i - 1 - factor * j) / i
		end
	end
	for i = 1, order do
		for j = 0, order do
			M[i][j] *= M[i - 1][j]
		end
	end
	return M
end

-- compute_R's tables, and U = compute_R(order, 1) for each order (it does not change)
local R_WORK: { [number]: { [number]: number } } = {}
local U_OF: { [number]: { [number]: { [number]: number } } } = {}

-- Rescales the differences D[0 .. order] for a step size changed by factor, computing
-- into the arrays of spare and exchanging them with D's.
local function change_D(R: any, D: { [number]: buffer }, spare: { [number]: buffer }, order: number, factor: number)
	local Rm = compute_R(order, factor, R_WORK)
	local U = U_OF[order]
	if U == nil then
		U = compute_R(order, 1, {})
		U_OF[order] = U
	end
	for j = 0, order do
		local acc: buffer? = nil
		for k = 0, order do
			local w = 0
			for l = 0, order do
				w += Rm[k][l] * U[l][j]
			end
			acc = if acc then R.axpy(acc, w, D[k], acc) else R.scale(spare[j], w, D[k])
		end
		spare[j] = acc :: buffer
	end
	for j = 0, order do
		D[j], spare[j] = spare[j], D[j]
	end
end

local function bdf(S: any, f: OdeFunction<buffer>, t0: number, y0: buffer, o: any): any
	local R = module_of(S)
	local n = R.length(y0)
	local rtol, atol = o.rtol or 1e-3, o.atol or 1e-6
	local maxstep = o.max_step or math.huge
	local mass = o.mass
	local stats = { nfev = 1, njev = 0, nlu = 0, steps = 0, rejected = 0 }
	local st: any = { t = t0, y = R.copy(nil, y0), piece = nil, stats = stats }
	local fy = f(t0, st.y, R.create(n))
	local newton_tol = math.max(10 * (S.eps or EPS) / rtol, math.min(0.03, rtol ^ 0.5))
	local jac = make_jacobian(R, f, n, o, stats)
	local linear = make_linear(R, f, n, mass, o, stats, 0.05 * newton_tol)
	local free = (o.linear or "direct") == "krylov" and o.jac == nil and o.jac_band == nil and o.jac_sparsity == nil
	local J = jac.evaluate(t0, st.y, fy)
	local Jnorm = jac_norm(R, J, n)
	local LU: ((buffer, buffer?) -> buffer)? = nil
	local h_abs: number = if o.h then math.abs(o.h) else 0
	local D: { [number]: buffer } = {}
	-- Workspace (a step allocates nothing beyond its dense output): spare differences for
	-- change_D, the Newton iterate and correction, its slope, right-hand side and step,
	-- the prediction and psi.
	local Dspare: { [number]: buffer } = {}
	for k = 0, BDF_MAX_ORDER + 2 do
		Dspare[k] = R.create(n)
	end
	local yit, dit, fv_w, rhs_w, dy_w, md_w = R.create(n), R.create(n), R.create(n), R.create(n), R.create(n), R.create(n)
	local ypred_w, psi_w = R.create(n), R.create(n)
	local piece_w: Piece<buffer> = { t0 = 0, h = 0, y0 = R.create(n), kind = "ndf", q = {} }
	local order = 1
	local n_equal = 0
	local direction0: number? = nil

	local function nonfinite(x: buffer): boolean
		return not (R.norm(x) < math.huge)
	end

	-- Newton's method for the NDF equations M (d + psi) = c f(t_new, y_predict + d).
	local function newton(tnew: number, ypred: buffer, c: number, psi: buffer, solve: (buffer, buffer?) -> buffer): (boolean, number, buffer, buffer)
		buffer.fill(dit, 0, 0) -- (zero bytes are 0 in every representation)
		local d = dit
		local y = R.copy(yit, ypred)
		local dnorm_old: number? = nil
		for k = 1, BDF_NEWTON do
			local fv = f(tnew, y, fv_w)
			stats.nfev += 1
			if nonfinite(fv) then
				return false, k, y, d
			end
			local rhs = R.scale(rhs_w, c, fv)
			md_w = R.add(md_w, psi, d)
			rhs = R.sub(rhs, rhs, mass_apply(R, mass, md_w, dy_w))
			local dy = solve(rhs, dy_w)
			local dnorm = weighted(S, dy, ypred, ypred, atol, rtol)
			local rate: number? = if dnorm_old then dnorm / dnorm_old else nil
			if rate ~= nil and (rate >= 1 or rate ^ (BDF_NEWTON - k + 1) / (1 - rate) * dnorm > newton_tol) then
				return false, k, y, d
			end
			y = R.add(y, y, dy)
			d = R.add(d, d, dy)
			if dnorm == 0 or (rate ~= nil and rate / (1 - rate) * dnorm < newton_tol) then
				return true, k, y, d
			end
			dnorm_old = dnorm
		end
		return false, BDF_NEWTON, y, d
	end

	function st.step(self: any, t_bound: number, want: boolean): (boolean, string?)
		local t = self.t
		local direction = if t_bound >= t then 1 else -1
		if direction0 == nil then
			direction0 = direction
			local yp0 = initial_slope(R, mass, fy)
			if h_abs == 0 then
				local hs, used = core.initial_step(S, slope_function(R, f, mass), t, self.y, yp0, direction, math.abs(t_bound - t), atol, rtol, maxstep, 1)
				h_abs = hs
				stats.nfev += used
			end
			D[0] = R.copy(nil, self.y)
			D[1] = R.scale(nil, h_abs * direction, yp0)
			for k = 2, BDF_MAX_ORDER + 2 do
				D[k] = R.create(n)
			end
		end
		assert(direction == direction0, "ode: a bdf integration cannot change direction")
		local min_step = 10 * EPS * math.max(math.abs(t), 1e-300)
		local hab = h_abs
		if hab > maxstep then
			change_D(R, D, Dspare, order, maxstep / hab)
			hab = maxstep
			n_equal = 0
		elseif hab < min_step then
			change_D(R, D, Dspare, order, min_step / hab)
			hab = min_step
			n_equal = 0
		end
		local current_jac = not jac.updatable
		if free then
			-- Matrix-free products linearize afresh at each step (an evaluation of f).
			J = jac.evaluate(t, self.y)
			LU = nil
			current_jac = true
		end
		local accepted = false
		local tnew, h = t, 0
		local ynew, d = self.y, self.y
		local n_iter = 0
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
				change_D(R, D, Dspare, order, math.abs(tnew - t) / hab)
				n_equal = 0
				LU = nil
			end
			h = tnew - t
			hab = math.abs(h)
			local ypred = R.copy(ypred_w, D[0])
			for k = 1, order do
				ypred = R.add(ypred, ypred, D[k])
			end
			local psi = R.scale(psi_w, NDF_GAMMA[1] / NDF_ALPHA[order], D[1])
			for k = 2, order do
				psi = R.axpy(psi, NDF_GAMMA[k] / NDF_ALPHA[order], D[k], psi)
			end
			local c = h / NDF_ALPHA[order]
			local converged = false
			while not converged do
				if LU == nil then
					LU = linear(J, 1, -c, tnew, ypred)
					if LU == nil then
						break
					end
				end
				converged, n_iter, ynew, d = newton(tnew, ypred, c, psi, LU :: any)
				if not converged then
					if current_jac then
						break
					end
					J = jac.evaluate(tnew, ypred)
					Jnorm = jac_norm(R, J, n)
					LU = nil
					current_jac = true
				end
			end
			if not converged then
				hab *= 0.5
				change_D(R, D, Dspare, order, 0.5)
				n_equal = 0
				LU = nil
				stats.rejected += 1
				continue
			end
			safety = 0.9 * (2 * BDF_NEWTON + 1) / (2 * BDF_NEWTON + n_iter)
			err = NDF_ERROR[order] * weighted(S, d, ynew, ynew, atol, rtol)
			if err ~= err then
				return false, "the solution became NaN"
			elseif err > 1 then
				local factor = math.max(0.2, safety * err ^ (-1 / (order + 1)))
				hab *= factor
				change_D(R, D, Dspare, order, factor)
				n_equal = 0
				stats.rejected += 1
			else
				accepted = true
			end
		end
		n_equal += 1
		h_abs = hab
		-- The differences of the new interpolating polynomial (d is its order + 1st), in place.
		D[order + 2] = R.sub(D[order + 2], d, D[order + 1])
		D[order + 1] = R.copy(D[order + 1], d)
		for i = order, 0, -1 do
			D[i] = R.add(D[i], D[i], D[i + 1])
		end
		if want then
			-- y(s) = D_0 + sum_j D_j prod_{m <= j} (s - (t_new - (m - 1) h)) / (m h)
			-- (copies: D is updated in place)
			-- (one piece, refilled each step: whoever keeps it copies it)
			local P = piece_w
			local q = P.q
			for k = 0, order do
				q[k + 1] = R.copy(q[k + 1] or R.create(n), D[k])
			end
			for k = order + 2, #q do
				q[k] = nil
			end
			P.t0, P.h = t, h
			P.y0 = R.copy(P.y0, self.y)
			self.piece = P
		else
			self.piece = nil
		end
		self.t, self.y = tnew, R.copy(self.y, ynew)
		self.stiffness = if Jnorm then hab * (Jnorm :: number) else nil
		stats.steps += 1
		if n_equal < order + 1 then
			return true, nil
		end
		-- Order selection from the error estimates of orders order - 1, order, order + 1.
		local em = if order > 1 then NDF_ERROR[order - 1] * weighted(S, D[order], ynew, ynew, atol, rtol) else math.huge
		local ep = if order < BDF_MAX_ORDER then NDF_ERROR[order + 1] * weighted(S, D[order + 2], ynew, ynew, atol, rtol) else math.huge
		local factors = { em ^ (-1 / order), err ^ (-1 / (order + 1)), ep ^ (-1 / (order + 2)) }
		local best = 1
		for k = 2, 3 do
			if factors[k] > factors[best] then
				best = k
			end
		end
		order += best - 2
		local factor = math.min(10, safety * factors[best])
		h_abs *= factor
		change_D(R, D, Dspare, order, factor)
		n_equal = 0
		LU = nil
		return true, nil
	end
	return st
end

-- Rodas5P
--
-- Steinebach's Rodas5P (2023): a Rosenbrock-Wanner method of order 5 with 8 stages, an
-- embedded order 4 error estimate and a third-order dense output, L-stable and stiffly
-- accurate, so it suits index-1 DAEs. A Rosenbrock method makes no Newton iterations: each
-- step solves 8 linear systems with one matrix M / (h gamma) - J, so it needs J (and
-- df/dt) afresh at every step; it is the most efficient choice for small and medium
-- systems at moderate to tight tolerances. Real and complex states, dense or banded J.
-- The coefficients (in the form A, C, gamma, c, d, H of OrdinaryDiffEq.jl) are copied
-- from OrdinaryDiffEq.jl's rosenbrock_tableaus.jl (MIT; see the third-party code in
-- docs/development.md). Stage i solves
--   (M / (h gamma) - J) k_i = f(t + c_i h, y + sum_{j<i} A_ij k_j) + h d_i df/dt
--                             + M sum_{j<i} (C_ij / h) k_j,
-- and y_new = y + sum_j A_8j k_j + k_8 (the last stage), with error estimate k_8.
local RODAS_A = {
	{},
	{ 3.0 },
	{ 2.849394379747939, 0.45842242204463923 },
	{ -6.954028509809101, 2.489845061869568, -10.358996098473584 },
	{ 2.8029986275628964, 0.5072464736228206, -0.3988312541770524, -0.04721187230404641 },
	{ -7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946 },
	{ -7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1 },
	{ -7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1, 1 },
}
local RODAS_C = {
	{},
	{ -14.155112264123755 },
	{ -17.97296035885952, -2.859693295451294 },
	{ 147.12150275711716, -1.41221402718213, 71.68940251302358 },
	{ 165.43517024871676, -0.4592823456491126, 42.90938336958603, -5.961986721573306 },
	{ 24.854864614690072, -3.0009227002832186, 47.4931110020768, 5.5814197821558125, -0.6610691825249471 },
	{ 30.91273214028599, -3.1208243349937974, 77.79954646070892, 34.28646028294783, -19.097331116725623, -28.087943162872662 },
	{ 37.80277123390563, -3.2571969029072276, 112.26918849496327, 66.9347231244047, -40.06618937091002, -54.66780262877968, -9.48861652309627 },
}
local RODAS_GAMMA = 0.21193756319429014
local RODAS_CT = { 0, 0.6358126895828704, 0.4095798393397535, 0.9769306725060716, 0.4288403609558664, 1, 1, 1 }
local RODAS_D = { 0.21193756319429014, -0.42387512638858027, -0.3384627126235924, 1.8046452872882734, 2.325825639765069, 0, 0, 0 }
local RODAS_H = {
	{ 25.948786856663858, -2.5579724845846235, 10.433815404888879, -2.3679251022685204, 0.524948541321073, 1.1241088310450404, 0.4272876194431874, -0.17202221070155493 },
	{ -9.91568850695171, -0.9689944594115154, 3.0438037242978453, -24.495224566215796, 20.176138334709044, 15.98066361424651, -6.789040303419874, -6.710236069923372 },
	{ 11.419903575922262, 2.8879645146136994, 72.92137995996029, 80.12511834622643, -52.072871366152654, -59.78993625266729, -0.15582684282751913, 4.883087185713722 },
}

local function rodas5p(S: any, f: OdeFunction<buffer>, t0: number, y0: buffer, o: any): any
	local R = module_of(S)
	assert((o.linear or "direct") == "direct" and o.jac_sparsity == nil, "ode: rodas5p needs dense or banded linear algebra (bdf takes sparse and matrix-free)")
	local n = R.length(y0)
	local rtol, atol = o.rtol or 1e-3, o.atol or 1e-6
	local maxstep = o.max_step or math.huge
	local mass = o.mass
	assert(mass == nil or type(mass) == "buffer" or mass.colptr == nil, "ode: rodas5p needs a dense or diagonal mass")
	local stats = { nfev = 1, njev = 0, nlu = 0, steps = 0, rejected = 0 }
	local st: any = { t = t0, y = R.copy(nil, y0), piece = nil, stats = stats, stiffness = nil }
	local fy = f(t0, st.y, R.create(n))
	local jac = make_jacobian(R, f, n, o, stats)
	local linear = make_linear(R, f, n, mass, o, stats, 0)
	local autonomous = o.autonomous == true
	local h_abs: number = if o.h then math.abs(o.h) else 0
	local h_acc: number? = nil
	local err_acc: number? = nil
	local J: any = nil
	local Jnorm: number? = nil
	-- Workspace: the stages, their arguments and right-hand sides, the new state and slope
	-- (fnew alternates with fy), df/dt. A step allocates nothing beyond its dense output.
	local K: { buffer } = {}
	for i = 1, 8 do
		K[i] = R.create(n)
	end
	local u, fs_w, rhs, acc, macc = R.create(n), R.create(n), R.create(n), R.create(n), R.create(n)
	local ynw, fnew_w, fp_w, dT_w = R.create(n), R.create(n), R.create(n), R.create(n)
	local piece_w: Piece<buffer> = { t0 = 0, h = 0, y0 = R.create(n), kind = "rodas", q = { R.create(n), R.create(n), R.create(n), R.create(n) } }

	-- The 8 stages of a step of size h from (t, y) with the solver of M / (h gamma) - J:
	-- (y_new, stages, whether f stopped being finite).
	local function attempt(t: number, y: buffer, h: number, solve: (buffer, buffer?) -> buffer, dT: buffer?): (buffer, { buffer }, boolean)
		local fs = fy
		for i = 1, 8 do
			if i > 1 then
				u = R.copy(u, y)
				local Ai = RODAS_A[i]
				for j = 1, i - 1 do
					u = R.axpy(u, Ai[j], K[j], u)
				end
				fs_w = f(t + RODAS_CT[i] * h, u, fs_w)
				fs = fs_w
				stats.nfev += 1
				if not (R.norm(fs) < math.huge) then
					return y, K, true
				end
			end
			rhs = R.copy(rhs, fs)
			if dT and RODAS_D[i] ~= 0 then
				rhs = R.axpy(rhs, h * RODAS_D[i], dT, rhs)
			end
			if i > 1 then
				local Ci: { number } = RODAS_C[i] :: any
				acc = R.scale(acc, Ci[1] / h, K[1])
				for j = 2, i - 1 do
					acc = R.axpy(acc, Ci[j] / h, K[j], acc)
				end
				rhs = R.add(rhs, rhs, mass_apply(R, mass, acc, macc))
			end
			K[i] = solve(rhs, K[i])
		end
		ynw = R.copy(ynw, y)
		local A8 = RODAS_A[8]
		for j = 1, 7 do
			ynw = R.axpy(ynw, A8[j], K[j], ynw)
		end
		ynw = R.add(ynw, ynw, K[8])
		return ynw, K, false
	end

	-- J at (t, y) when it can change, and df/dt there unless f is autonomous.
	local function linearize(t: number, y: buffer): buffer?
		if J == nil or jac.updatable then
			J = jac.evaluate(t, y, fy)
			Jnorm = jac_norm(R, J, n)
		end
		if autonomous then
			return nil
		end
		local dt = math.sqrt(S.eps or EPS) * math.max(math.abs(t), 1)
		fp_w = f(t + dt, y, fp_w)
		stats.nfev += 1
		dT_w = R.sub(dT_w, fp_w, fy)
		dT_w = R.scale(dT_w, 1 / dt, dT_w)
		return dT_w
	end

	-- The new state into st.y (updated in place) and its slope into fy.
	local function accept_state(self: any, tnew: number, ynew: buffer, fnew: buffer)
		self.t, self.y = tnew, R.copy(self.y, ynew)
		fnew_w, fy = fy, fnew
	end

	-- One step of exactly h, without error control (integrator:step): a fixed cost of one
	-- Jacobian, one factorization and 8 evaluations and solves, with no iterations, which
	-- suits frame-locked simulation. It is still stable on stiff problems, as the method
	-- is L-stable.
	function st.fixed_step(self: any, h: number): buffer
		local t, y = self.t, self.y
		local dT = linearize(t, y)
		local solve = linear(J, 1 / (h * RODAS_GAMMA), -1, t, y)
		assert(solve ~= nil, "ode: rodas5p: the step's linear system is singular")
		local ynew, _, bad = attempt(t, y, h, solve :: (buffer, buffer?) -> buffer, dT)
		assert(not bad, "ode: rodas5p: f is not finite during the step")
		local fnew = f(t + h, ynew, fnew_w)
		stats.nfev += 1
		stats.steps += 1
		accept_state(self, t + h, ynew, fnew)
		self.piece = nil
		self.stiffness = if Jnorm then math.abs(h) * (Jnorm :: number) else nil
		return self.y
	end

	function st.step(self: any, t_bound: number, want: boolean): (boolean, string?)
		local t, y = self.t, self.y
		local direction = if t_bound >= t then 1 else -1
		if h_abs == 0 then
			local hs, used = core.initial_step(S, slope_function(R, f, mass), t, y, initial_slope(R, mass, fy), direction, math.abs(t_bound - t), atol, rtol, maxstep, 4)
			h_abs = hs
			stats.nfev += used
		end
		-- J and df/dt at (t, y), once per step (kept across rejections).
		local dT = linearize(t, y)
		local min_step = 10 * EPS * math.max(math.abs(t), 1e-300)
		local hab = math.min(h_abs, maxstep)
		local requested = hab
		local rejected = false
		while true do
			if hab < min_step then
				return false, "the step size became too small"
			end
			local h = hab * direction
			local tnew = t + h
			if direction * (tnew - t_bound) > 0 or math.abs(t_bound - tnew) <= 4 * EPS * math.abs(t_bound) then
				tnew = t_bound
			end
			h = tnew - t
			hab = math.abs(h)
			local solve = linear(J, 1 / (h * RODAS_GAMMA), -1, t, y)
			if solve == nil then
				hab *= 0.5
				stats.rejected += 1
				continue
			end
			local ynew, Ks, bad = attempt(t, y, h, solve :: (buffer, buffer?) -> buffer, dT)
			local err = if bad then math.huge else weighted(S, Ks[8], y, ynew, atol, rtol)
			if err ~= err then
				return false, "the solution became NaN"
			end
			-- Hairer's step-size control with Gustafsson's predictor (as in RODAS).
			local fac = math.clamp(err ^ 0.2 / 0.9, 1 / 6, 5)
			if err <= 1 then
				if h_acc ~= nil and err_acc ~= nil and err > 0 then
					local facgus = math.clamp((h_acc :: number) / hab * (err * err / (err_acc :: number)) ^ 0.2 / 0.9, 1 / 6, 5)
					fac = math.max(fac, facgus)
				end
				h_acc, err_acc = hab, math.max(err, 1e-2)
				local hnext = hab / fac
				if rejected then
					hnext = math.min(hnext, hab)
				end
				-- A step cut short at t_bound keeps the step size it was cut from.
				if tnew == t_bound and hab < requested then
					hnext = math.max(hnext, requested * math.min(1, 1 / fac))
				end
				h_abs = math.min(hnext, maxstep)
				local fnew = f(tnew, ynew, fnew_w)
				stats.nfev += 1
				if want then
					-- y(theta) = (1 - theta) y0 + theta (y1 + (1 - theta)(k1 + theta (k2 + theta k3)))
					-- with k_j = sum_i H_ji K_i. (Copies: the workspace is reused next step.)
					-- (one piece, refilled each step: whoever keeps it copies it)
					local P = piece_w
					P.t0, P.h = t, h
					P.y0 = R.copy(P.y0, y)
					P.q[1] = R.copy(P.q[1], ynew)
					for r = 1, 3 do
						local hk = R.scale(P.q[r + 1], RODAS_H[r][1], Ks[1])
						for i = 2, 8 do
							hk = R.axpy(hk, RODAS_H[r][i], Ks[i], hk)
						end
						P.q[r + 1] = hk
					end
					self.piece = P
				else
					self.piece = nil
				end
				accept_state(self, tnew, ynew, fnew)
				self.stiffness = if Jnorm then hab * (Jnorm :: number) else nil
				stats.steps += 1
				return true, nil
			end
			hab /= if bad then 4 else math.max(fac, 1.2)
			rejected = true
			stats.rejected += 1
		end
	end
	return st
end

local STEPPERS: { [string]: any } = { radau = radau, bdf = bdf, rodas5p = rodas5p }

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
	local vpiece: any = { t0 = 0, h = 0, y0 = y0, kind = "poly", q = {} }
	function st.step(self: any, t_bound: number, want: boolean): (boolean, string?)
		local ok, why = inner:step(t_bound, want)
		self.t, self.y, self.stiffness = inner.t, tovec(inner.y), inner.stiffness
		local P = inner.piece
		local piece: any = nil
		if want and P then
			piece = vpiece
			local q = piece.q
			table.clear(q)
			for i, v in P.q do
				q[i] = tovec(v)
			end
			piece.t0, piece.h, piece.y0, piece.kind = P.t0, P.h, tovec(P.y0), P.kind
		end
		self.piece = piece
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

-- The Jacobian df/dy of f at (t, y) by finite differences, as the stiff solvers estimate
-- it: a matrix, or with options.jac_band = { kl, ku } a banded matrix, or with
-- options.jac_sparsity a sparse matrix of that pattern (one evaluation of f per group of
-- columns that share no row). y is an f32, f64, c32 or c64 array.
local function jacobian(S: any, f: OdeFunction<buffer>, t: number, y: buffer, options: any?): any
	local R = module_of(S)
	local o = table.clone(options or {})
	o.jac, o.linear = nil, "direct"
	local stats = { nfev = 0, njev = 0 }
	return make_jacobian(R, f, R.length(y), o, stats).evaluate(t, y)
end

-- The array module whose space S is, or nil for other spaces.
local function array_module(S: any): any
	for _, M in { f64, f32, c64, c32 } :: { any } do
		if S == M.space then
			return M
		end
	end
	return nil
end

return {
	array_module = array_module,
	stepper = stepper,
	jacobian = jacobian,
	methods = { radau = true, bdf = true, rodas5p = true } :: { [string]: boolean },
}
