--!strict
--!native
--!optimize 2

-- Lightning solvers (Gopal and Trefethen, "Solving Laplace problems with corner
-- singularities via rational functions", 2019): Laplace's (or Poisson's, with a constant
-- source) equation on a polygon, with Dirichlet or Neumann data on each side.
--
-- The solution is a sum of harmonic functions, the real and imaginary parts of
--   a polynomial in z (orthogonalized on the sample points by the Vandermonde with Arnoldi
--   process of Brubeck, Nakatsukasa and Trefethen), and
--   simple poles d / (z - p) clustered exponentially close to each vertex, outside the
--   polygon, at distances d_j = L exp(-sigma (sqrt(N) - sqrt(j))), j = 1 .. N,
-- fitted to the boundary data by least squares at points that cluster the same way. Each
-- term satisfies the equation exactly, so only the boundary is sampled. Near a corner, or
-- where the boundary condition changes, the solution has a singularity (r^(1/2) where a
-- Dirichlet side meets a Neumann side in a straight line); the clustered poles resolve it,
-- and the error decreases like exp(-C sqrt(N)) where polynomial or finite element
-- approximations converge only algebraically. A change of condition in the middle of a
-- side is given as a vertex there.

local f64 = require("./f64")
local c64 = require("./c64")
local rational = require("./rational")

local types = require("./types")

type Condition = types.LightningCondition
type LightningOptions = types.LightningOptions
type LightningSolution = types.LightningSolution
type LightningCompressed = types.LightningCompressed

-- The fit's basis at a point z = (x, y): fills vals[k] (Re and Im parts interleaved by
-- column: column 2k - 1 is Re of basis function k, 2k its Im) and, if wanted, the complex
-- derivatives.
type Basis = {
	cx: number,
	cy: number,
	scale: number,
	degree: number,
	H: { { number } }, -- Arnoldi coefficients: H[j][k] complex as { re, im } interleaved
	poles: { number }, -- interleaved px, py
	dist: { number }, -- each pole's distance (its numerator)
}

-- Evaluates the complex basis functions and derivatives at z: re[k], im[k], dre[k], dim[k]
-- for k = 1 .. count (the polynomial terms first, then the poles).
local function evaluate(B: Basis, x: number, y: number, re: { number }, im: { number }, dre: { number }, dim: { number })
	local sr, si = (x - B.cx) / B.scale, (y - B.cy) / B.scale
	local n = B.degree
	-- q_0 = 1; q_k = (s q_{k-1} - sum_j H[j][k] q_j) / H[k + 1][k] (complex), and the same
	-- recurrence differentiated (ds/dz = 1 / scale).
	re[1], im[1], dre[1], dim[1] = 1, 0, 0, 0
	for k = 1, n do
		local col = B.H[k]
		local ar = sr * re[k] - si * im[k]
		local ai = sr * im[k] + si * re[k]
		local br = re[k] / B.scale + sr * dre[k] - si * dim[k]
		local bi = im[k] / B.scale + sr * dim[k] + si * dre[k]
		for j = 1, k do
			local hr, hi = col[2 * j - 1], col[2 * j]
			ar -= hr * re[j] - hi * im[j]
			ai -= hr * im[j] + hi * re[j]
			br -= hr * dre[j] - hi * dim[j]
			bi -= hr * dim[j] + hi * dre[j]
		end
		local h = col[2 * k + 1] -- real and positive
		re[k + 1], im[k + 1], dre[k + 1], dim[k + 1] = ar / h, ai / h, br / h, bi / h
	end
	-- d / (z - p), derivative -d / (z - p)^2
	local base = n + 1
	for j = 1, #B.dist do
		local dx, dy = x - B.poles[2 * j - 1], y - B.poles[2 * j]
		local m2 = dx * dx + dy * dy
		local d = B.dist[j]
		local vr, vi = d * dx / m2, -d * dy / m2 -- d / (dx + i dy)
		re[base + j], im[base + j] = vr, vi
		-- -(v^2) / d
		dre[base + j] = -(vr * vr - vi * vi) / d
		dim[base + j] = -(2 * vr * vi) / d
	end
end

-- The Arnoldi coefficients of the polynomial part on the sample points (complex numbers
-- as interleaved pairs; MGS applied twice).
local function arnoldi(sx: { number }, sy: { number }, cx: number, cy: number, scale: number, n: number): { { number } }
	local M = #sx
	local qr = { table.create(M, 1) }
	local qi = { table.create(M, 0) }
	local H = {}
	for k = 1, n do
		local vr, vi = table.create(M, 0), table.create(M, 0)
		local pr, pi = qr[k], qi[k]
		for t = 1, M do
			local s_r, s_i = (sx[t] - cx) / scale, (sy[t] - cy) / scale
			vr[t] = s_r * pr[t] - s_i * pi[t]
			vi[t] = s_r * pi[t] + s_i * pr[t]
		end
		local col = table.create(2 * k + 1, 0)
		for _ = 1, 2 do
			for j = 1, k do
				-- h = q_j^H v / M
				local hr, hi = 0, 0
				local ar, ai = qr[j], qi[j]
				for t = 1, M do
					hr += ar[t] * vr[t] + ai[t] * vi[t]
					hi += ar[t] * vi[t] - ai[t] * vr[t]
				end
				hr, hi = hr / M, hi / M
				for t = 1, M do
					vr[t] -= hr * ar[t] - hi * ai[t]
					vi[t] -= hr * ai[t] + hi * ar[t]
				end
				col[2 * j - 1] += hr
				col[2 * j] += hi
			end
		end
		local nrm = 0
		for t = 1, M do
			nrm += vr[t] * vr[t] + vi[t] * vi[t]
		end
		nrm = math.sqrt(nrm / M)
		assert(nrm > 0, "lightning: too few sample points for the polynomial degree")
		col[2 * k + 1] = nrm
		for t = 1, M do
			vr[t] /= nrm
			vi[t] /= nrm
		end
		qr[k + 1], qi[k + 1] = vr, vi
		H[k] = col
	end
	return H
end

-- Solves Laplace's equation (Poisson's, with options.source) on the polygon with the given
-- vertices ({ x, y } pairs, counterclockwise); conditions[i] holds on the side from vertex
-- i to vertex i + 1 (the last side closing the polygon).
local function laplace(vertices: { { number } }, conditions: { Condition }, options: LightningOptions?): LightningSolution
	local o: LightningOptions = options or {}
	local m = #vertices
	assert(m >= 3 and #conditions == m, "lightning.laplace: one condition per side of the polygon")
	local N = o.poles or 20
	local sigma = o.sigma or 4
	local degree = o.degree or math.max(N, 4)
	local f = o.source or 0
	local X, Y = table.create(m, 0), table.create(m, 0)
	local area = 0
	for i = 1, m do
		X[i], Y[i] = vertices[i][1], vertices[i][2]
	end
	for i = 1, m do
		local j = i % m + 1
		area += X[i] * Y[j] - X[j] * Y[i]
	end
	assert(area > 0, "lightning.laplace: the vertices must go counterclockwise")
	local cx, cy, scale = 0, 0, 0
	for i = 1, m do
		cx += X[i] / m
		cy += Y[i] / m
	end
	for i = 1, m do
		scale = math.max(scale, math.sqrt((X[i] - cx) ^ 2 + (Y[i] - cy) ^ 2))
	end
	local function side_length(i: number): number
		local j = i % m + 1
		return math.sqrt((X[j] - X[i]) ^ 2 + (Y[j] - Y[i]) ^ 2)
	end

	-- Poles: at each vertex, along the exterior bisector (the outward normal where the
	-- polygon is straight).
	local poles, dist = {}, {}
	local cluster = table.create(m) :: { { number } } -- each vertex's distances
	for k = 1, m do
		local prev, nxt = (k - 2) % m + 1, k % m + 1
		local lp, ln = side_length(prev), side_length(k)
		local ux, uy = (X[prev] - X[k]) / lp, (Y[prev] - Y[k]) / lp
		local vx, vy = (X[nxt] - X[k]) / ln, (Y[nxt] - Y[k]) / ln
		-- outward normal of the side k -> k + 1 (counterclockwise: (vy, -vx))
		local ex, ey = -(ux + vx), -(uy + vy)
		local cross = vx * uy - vy * ux -- > 0 for a convex corner
		local len = math.sqrt(ex * ex + ey * ey)
		if len < 1e-12 then
			ex, ey = vy, -vx
		else
			ex, ey = ex / len, ey / len
			if cross < 0 then
				-- reentrant corner: the exterior is on the other side
				ex, ey = -ex, -ey
			end
		end
		local L = math.min(lp, ln)
		local ds = table.create(N, 0)
		for j = 1, N do
			local d = L * math.exp(-sigma * (math.sqrt(N) - math.sqrt(j)))
			ds[j] = d
			table.insert(poles, X[k] + d * ex)
			table.insert(poles, Y[k] + d * ey)
			table.insert(dist, d)
		end
		cluster[k] = ds
	end

	-- Sample points on each side: uniform, plus clustered toward both ends like the poles.
	local function samples(density: number): ({ number }, { number }, { number }, { number }, { number })
		local px, py, nx, ny, side = {}, {}, {}, {}, {}
		for i = 1, m do
			local j = i % m + 1
			local L = side_length(i)
			local tx, ty = (X[j] - X[i]) / L, (Y[j] - Y[i]) / L
			local ts = {}
			local uniform = math.max(20, 2 * degree) * density
			for t = 1, uniform do
				table.insert(ts, L * (t - 0.5) / uniform)
			end
			for _, frac in { 1 / 3, 2 / 3, 1 } do
				for _, d in cluster[i] do
					local s = d * frac / density
					if s < L / 2 then
						table.insert(ts, s)
					end
				end
				for _, d in cluster[j] do
					local s = d * frac / density
					if s < L / 2 then
						table.insert(ts, L - s)
					end
				end
			end
			for _, s in ts do
				table.insert(px, X[i] + s * tx)
				table.insert(py, Y[i] + s * ty)
				table.insert(nx, ty)
				table.insert(ny, -tx)
				table.insert(side, i)
			end
		end
		return px, py, nx, ny, side
	end

	local px, py, nx, ny, side = samples(1)
	local B: Basis = {
		cx = cx,
		cy = cy,
		scale = scale,
		degree = degree,
		H = arnoldi(px, py, cx, cy, scale, degree),
		poles = poles,
		dist = dist,
	}
	local count = degree + 1 + #dist
	-- The real unknowns: Re and Im of every basis function except Im of the constant.
	local unknowns = 2 * count - 1
	local re, im = table.create(count, 0), table.create(count, 0)
	local dre, dim = table.create(count, 0), table.create(count, 0)

	-- The particular solution f |z - c|^2 / 4 and its outward normal derivative.
	local function particular(x: number, y: number): number
		return f * ((x - cx) ^ 2 + (y - cy) ^ 2) / 4
	end
	local function particular_dn(x: number, y: number, ux: number, uy: number): number
		return f / 2 * ((x - cx) * ux + (y - cy) * uy)
	end

	-- The rows of the least-squares system at the points given: the condition applied to
	-- each basis function, and the data (less the particular solution's part).
	local function rows(qx: { number }, qy: { number }, qnx: { number }, qny: { number }, qside: { number }, A: any?, rhs: buffer?)
		for t = 1, #qx do
			local x, y = qx[t], qy[t]
			local cond = conditions[qside[t]]
			evaluate(B, x, y, re, im, dre, dim)
			local dirichlet = cond.kind == "dirichlet"
			assert(dirichlet or cond.kind == "neumann", "lightning.laplace: kind must be \"dirichlet\" or \"neumann\"")
			local datum = cond.value(x, y)
			if dirichlet then
				datum -= particular(x, y)
			else
				datum -= particular_dn(x, y, qnx[t], qny[t])
			end
			if rhs then
				f64.set(rhs, t, datum)
			end
			if A then
				local col = 1
				for k = 1, count do
					local vr, vi
					if dirichlet then
						vr, vi = re[k], im[k]
					else
						-- d/dn Re(phi) = Re(phi' n), d/dn Im(phi) = Im(phi' n)
						vr = dre[k] * qnx[t] - dim[k] * qny[t]
						vi = dre[k] * qny[t] + dim[k] * qnx[t]
					end
					f64.matset(A, t, col, vr)
					col += 1
					if k > 1 then
						f64.matset(A, t, col, vi)
						col += 1
					end
				end
			end
		end
	end

	local A = f64.matrix(#px, unknowns)
	local rhs = f64.create(#px)
	rows(px, py, nx, ny, side, A, rhs)
	-- Row weights, in the distance r to the nearest vertex: sqrt(r) on Dirichlet rows and
	-- r sqrt(r) on Neumann rows. Weighting keeps the points crowded near the vertices from
	-- dominating the fit; the extra r puts derivatives on the scale of values (near an
	-- r^(1/2) singularity a derivative is r^(-1/2)). The power 1/2 matches the interior's
	-- sensitivity to boundary values near a change from Dirichlet to Neumann (the harmonic
	-- measure's density grows like r^(-1/2) there), so errors the weighting tolerates do
	-- not move the solution: with weights r and r^2 (Gopal and Trefethen's r on Dirichlet
	-- problems) the hanging membrane's sag stalled near 1e-4 while the residual fell to
	-- 1e-12; with these it converges to 10 digits. Unweighted, even a single singularity
	-- converged only algebraically.
	local function weight(x: number, y: number, i: number): number
		local r = math.huge
		for k = 1, m do
			r = math.min(r, math.sqrt((x - X[k]) ^ 2 + (y - Y[k]) ^ 2))
		end
		local s = math.sqrt(r)
		return if conditions[i].kind == "dirichlet" then s else s * r
	end
	for t = 1, #px do
		local w = weight(px[t], py[t], side[t])
		f64.set(rhs, t, w * f64.get(rhs, t))
		for k = 1, unknowns do
			f64.matset(A, t, k, w * f64.matget(A, t, k))
		end
	end
	local coef = f64.solve(nil, A, rhs)
	local c = table.create(unknowns, 0)
	for k = 1, unknowns do
		c[k] = f64.get(coef, k)
	end

	local function harmonic(x: number, y: number): (number, number, number)
		evaluate(B, x, y, re, im, dre, dim)
		local u, gx, gy = 0, 0, 0
		local col = 1
		for k = 1, count do
			-- u += a Re(phi) + b Im(phi); grad Re(phi) = (Re phi', -Im phi'),
			-- grad Im(phi) = (Im phi', Re phi')
			local a = c[col]
			u += a * re[k]
			gx += a * dre[k]
			gy -= a * dim[k]
			col += 1
			if k > 1 then
				local b = c[col]
				u += b * im[k]
				gx += b * dim[k]
				gy += b * dre[k]
				col += 1
			end
		end
		return u, gx, gy
	end

	-- The analytic function f with u = Re f (+ the particular solution): coefficient a of
	-- Re(phi) and b of Im(phi) make (a - i b) phi.
	local function analytic(x: number, y: number): (number, number)
		evaluate(B, x, y, re, im, dre, dim)
		local fr, fi = 0, 0
		local col = 1
		for k = 1, count do
			local a = c[col]
			col += 1
			local b = 0
			if k > 1 then
				b = c[col]
				col += 1
			end
			fr += a * re[k] + b * im[k]
			fi += a * im[k] - b * re[k]
		end
		return fr, fi
	end

	-- The residual on twice as many boundary points, in the weighted norm of the fit.
	local fx, fy, fnx, fny, fside = samples(2)
	local frhs = f64.create(#fx)
	rows(fx, fy, fnx, fny, fside, nil, frhs)
	local worst, size = 0, 0
	for t = 1, #fx do
		local u, gx, gy = harmonic(fx[t], fy[t])
		local value = if conditions[fside[t]].kind == "dirichlet" then u else gx * fnx[t] + gy * fny[t]
		local datum = f64.get(frhs, t)
		local w = weight(fx[t], fy[t], fside[t])
		worst = math.max(worst, w * math.abs(value - datum))
		size = math.max(size, w * math.abs(datum))
	end

	local residual = if size > 0 then worst / size else worst

	-- AAA on f's values at the boundary points: by the maximum modulus principle the error
	-- inside is no larger than on the boundary, and AAA places its poles only where f needs
	-- them, where the fit put a fixed cluster at every vertex.
	local function compress(tol: number?): LightningCompressed
		local n = #fx
		local zr, zi, vr, vi = table.create(n, 0), table.create(n, 0), table.create(n, 0), table.create(n, 0)
		local mean = 0
		for t = 1, n do
			zr[t], zi[t] = fx[t], fy[t]
			vr[t], vi[t] = analytic(fx[t], fy[t])
			mean += vi[t] / n
		end
		-- Only Re f is fitted, so Im f (the harmonic conjugate) carries an arbitrary
		-- constant, and the fit's can be large (13065 against |Re f| < 1 on the hanging
		-- membrane), which would set AAA's relative tolerance: drop it, and measure the
		-- tolerance against Re f.
		local remax, fmax = 0, 0
		for t = 1, n do
			vi[t] -= mean
			remax = math.max(remax, math.abs(vr[t]))
			fmax = math.max(fmax, math.sqrt(vr[t] * vr[t] + vi[t] * vi[t]))
		end
		local want = tol or math.max(residual, 1e-13)
		local R = rational.aaa(c64.from(zr, zi), c64.from(vr, vi), { tol = want * remax / math.max(fmax, 1e-300), mmax = 150 })
		return {
			rational = R,
			terms = c64.length(R.support),
			error = R.error * fmax / math.max(remax, 1e-300),
			eval = function(x: number, y: number): number
				local ur = R.eval(x, y)
				return ur + particular(x, y)
			end,
		}
	end

	return {
		eval = function(x: number, y: number): number
			local u = harmonic(x, y)
			return u + particular(x, y)
		end,
		gradient = function(x: number, y: number): (number, number)
			local _, gx, gy = harmonic(x, y)
			return gx + f / 2 * (x - cx), gy + f / 2 * (y - cy)
		end,
		analytic = analytic,
		compress = compress,
		error = residual,
		unknowns = unknowns,
	}
end

return {
	laplace = laplace,
}
