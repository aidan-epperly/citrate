--!strict

-- Stiff solvers: reference solutions of standard stiff problems, Jacobians (finite
-- differences, user functions, constant, banded), mass matrices and DAEs, dense output,
-- events, integrators and every representation.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local ode, f64, f32, vec = lib.ode, lib.f64, lib.f32, lib.vec

local STIFF = { "radau", "bdf" }

-- Per-method accuracy and work expectations. The work bounds are regressions against
-- SciPy's Radau and BDF on the same problems (their counts are in the comments).
local EXPECT: { [string]: any } = {
	-- SciPy Radau: Robertson 78 steps; Van der Pol 311, 1551 and 8504 steps with 2905,
	-- 12630 and 64621 evaluations.
	radau = {
		robertson = 1e-7,
		robertson_steps = 85,
		vdp_error = 100,
		vdp_steps = { 340, 1650, 9000 },
		vdp_nfev = { 3300, 13500, 72000 },
		vdp_rejected = { 40, 45, 60 },
		dense = 1e-7,
	},
	-- SciPy BDF: Robertson 144 steps; Van der Pol 559, 1820 and 5875 steps with 1786,
	-- 5511 and 18761 evaluations.
	bdf = {
		robertson = 1e-6,
		robertson_steps = 150,
		vdp_error = 100,
		vdp_steps = { 580, 1900, 6100 },
		vdp_nfev = { 2100, 6100, 19500 },
		vdp_rejected = { 90, 35, 20 },
		dense = 1e-6,
	},
}

-- Robertson's chemical kinetics (Hairer and Wanner II, IV.1): rate constants spanning
-- eleven orders of magnitude. Reference at t = 40 from Hairer's test set.
local function robertson(t: number, y: buffer, out: buffer): buffer
	local a, b, c = f64.get(y, 1), f64.get(y, 2), f64.get(y, 3)
	f64.set(out, 1, -0.04 * a + 1e4 * b * c)
	f64.set(out, 2, 0.04 * a - 1e4 * b * c - 3e7 * b * b)
	f64.set(out, 3, 3e7 * b * b)
	return out
end
local ROBERTSON_40 = { 0.7158270687193262, 9.185534764557e-6, 0.2841637457458 }

-- Van der Pol with eps = 1e-6 (mu = 1000 in Hairer's scaling); reference at t = 2 from
-- the IVP test set (Mazzia and Magherini).
local function vdp(_: number, y: buffer, out: buffer): buffer
	local a, b = f64.get(y, 1), f64.get(y, 2)
	f64.set(out, 1, b)
	f64.set(out, 2, ((1 - a * a) * b - a) / 1e-6)
	return out
end
local VDP_2 = { 1.706167732170483, -0.8928097010248125 }

-- Prothero-Robinson: y_i' = lambda_i (y_i - g_i(t)) + g_i'(t) has the solution y = g for
-- y(0) = g(0), with transients decaying at rates lambda (here up to 1e6).
local LAMBDA = { -1, -1e4, -1e6 }
local function g(t: number): { number }
	return { math.sin(t), math.cos(t), math.exp(-t) }
end
local function dg(t: number): { number }
	return { math.cos(t), -math.sin(t), -math.exp(-t) }
end
local function prothero(M: any): (number, buffer, buffer) -> buffer
	return function(t: number, y: buffer, out: buffer): buffer
		local gt, dgt = g(t), dg(t)
		for i = 1, 3 do
			M.set(out, i, LAMBDA[i] * (M.get(y, i) - gt[i]) + dgt[i])
		end
		return out
	end
end

-- The worst relative error of y against g(t).
local function prothero_error(M: any, t: number, y: buffer): number
	local gt = g(t)
	local worst = 0
	for i = 1, 3 do
		worst = math.max(worst, math.abs(M.get(y, i) - gt[i]) / math.max(1e-3, math.abs(gt[i])))
	end
	return worst
end

-- The 1D Brusselator (Hairer and Wanner II, IV.1) on n interior points, interleaved as
-- (u_1, v_1, u_2, v_2, ...), so its Jacobian has two sub- and superdiagonals.
local function brusselator(n: number): (number, buffer, buffer) -> buffer
	local c = (n + 1) ^ 2 / 50
	return function(_: number, y: buffer, out: buffer): buffer
		for i = 1, n do
			local u, v = f64.get(y, 2 * i - 1), f64.get(y, 2 * i)
			local ul = if i > 1 then f64.get(y, 2 * i - 3) else 1
			local ur = if i < n then f64.get(y, 2 * i + 1) else 1
			local vl = if i > 1 then f64.get(y, 2 * i - 2) else 3
			local vr = if i < n then f64.get(y, 2 * i + 2) else 3
			f64.set(out, 2 * i - 1, 1 + u * u * v - 4 * u + c * (ul - 2 * u + ur))
			f64.set(out, 2 * i, 3 * u - u * u * v + c * (vl - 2 * v + vr))
		end
		return out
	end
end
local function brusselator_start(n: number): buffer
	local y = f64.create(2 * n)
	for i = 1, n do
		local x = i / (n + 1)
		f64.set(y, 2 * i - 1, 1 + math.sin(2 * math.pi * x))
		f64.set(y, 2 * i, 3)
	end
	return y
end
-- Its exact Jacobian, as a band.
local function brusselator_jac(n: number): (number, buffer) -> any
	local c = (n + 1) ^ 2 / 50
	return function(_: number, y: buffer): any
		local B = f64.banded(2 * n, 2, 2)
		for i = 1, n do
			local u, v = f64.get(y, 2 * i - 1), f64.get(y, 2 * i)
			local r, s = 2 * i - 1, 2 * i
			f64.band_set(B, r, r, 2 * u * v - 4 - 2 * c)
			f64.band_set(B, r, s, u * u)
			f64.band_set(B, s, r, 3 - 2 * u * v)
			f64.band_set(B, s, s, -u * u - 2 * c)
			if i > 1 then
				f64.band_set(B, r, r - 2, c)
				f64.band_set(B, s, s - 2, c)
			end
			if i < n then
				f64.band_set(B, r, r + 2, c)
				f64.band_set(B, s, s + 2, c)
			end
		end
		return B
	end
end

local function last(sol: any): buffer
	return sol.y[#sol.y]
end

test("stiff reference problems", function()
	for _, method in STIFF do
		local sol = ode.solve(f64.space, robertson, { 0, 40 }, f64.from({ 1, 0, 0 }), { method = method, rtol = 1e-6, atol = 1e-10 })
		check(sol.status == "success", `{method} robertson: {sol.message}`)
		local y = last(sol)
		local ex = EXPECT[method]
		close(f64.get(y, 1), ROBERTSON_40[1], ex.robertson, `{method} robertson y1`)
		close(f64.get(y, 2) / ROBERTSON_40[2], 1, 10 * ex.robertson, `{method} robertson y2`)
		close(f64.get(y, 3), ROBERTSON_40[3], ex.robertson, `{method} robertson y3`)
		-- Mass conservation: y1 + y2 + y3 = 1 holds for any linear one-step method.
		close(f64.get(y, 1) + f64.get(y, 2) + f64.get(y, 3), 1, 1e-13, `{method} robertson mass`)
		check(sol.stats.steps <= ex.robertson_steps, `{method} robertson: {sol.stats.steps} steps`)

		for _, tol in { 1e-4, 1e-7, 1e-10 } do
			sol = ode.solve(f64.space, vdp, { 0, 2 }, f64.from({ 2, 0 }), { method = method, rtol = tol, atol = tol })
			y = last(sol)
			check(sol.status == "success", `{method} vdp {tol}: {sol.message}`)
			local err = math.max(math.abs(f64.get(y, 1) - VDP_2[1]), math.abs(f64.get(y, 2) - VDP_2[2]))
			check(err <= ex.vdp_error * tol, `{method} vdp {tol}: error {err}`)
			-- For Radau the evaluations catch a Jacobian that is never refreshed and a
			-- missing starting guess from the last step's polynomial; the rejections at 1e-4
			-- catch a missing Gustafsson predictor or safety factor.
			local k = if tol == 1e-4 then 1 elseif tol == 1e-7 then 2 else 3
			check(sol.stats.steps <= ex.vdp_steps[k], `{method} vdp {tol}: {sol.stats.steps} steps`)
			check(sol.stats.nfev <= ex.vdp_nfev[k], `{method} vdp {tol}: {sol.stats.nfev} evaluations`)
			check(sol.stats.rejected <= ex.vdp_rejected[k], `{method} vdp {tol}: {sol.stats.rejected} rejections`)
		end
	end
end)

test("stiff Jacobians: finite differences, functions, constants and bands", function()
	for _, method in STIFF do
		-- Prothero-Robinson with the exact (constant) Jacobian, as a function and a matrix.
		local J = f64.matrix(3, 3)
		for i = 1, 3 do
			f64.matset(J, i, i, LAMBDA[i])
		end
		local f = prothero(f64)
		local base: { [string]: any } = {}
		for _, variant in { "fd", "function", "constant" } do
			local o: any = { method = method, rtol = 1e-8, atol = 1e-10 }
			if variant == "function" then
				o.jac = function(): any
					return J
				end
			elseif variant == "constant" then
				o.jac = J
			end
			local sol = ode.solve(f64.space, f, { 0, 5 }, f64.from(g(0)), o)
			check(sol.status == "success", `{method} {variant}: {sol.message}`)
			check(prothero_error(f64, 5, last(sol)) <= 1e-6, `{method} {variant}: error {prothero_error(f64, 5, last(sol))}`)
			base[variant] = sol
		end
		-- A constant Jacobian is never re-evaluated; finite differences cost f evaluations.
		check(base.constant.stats.njev == 0, `{method}: constant Jacobian evaluated {base.constant.stats.njev} times`)
		check(base["function"].stats.njev >= 1, `{method}: Jacobian function never called`)
		check(base.fd.stats.nfev > base["function"].stats.nfev, `{method}: finite differences cost no evaluations`)

		-- The Brusselator: banded finite differences (5 evaluations per Jacobian instead of
		-- 2n), an exact banded Jacobian, and dense finite differences agree.
		local n = 40
		local bf = brusselator(n)
		local tol = { rtol = 1e-6, atol = 1e-8 }
		local dense = ode.solve(f64.space, bf, { 0, 10 }, brusselator_start(n), { method = method, rtol = tol.rtol, atol = tol.atol })
		local banded = ode.solve(f64.space, bf, { 0, 10 }, brusselator_start(n), { method = method, rtol = tol.rtol, atol = tol.atol, jac_band = { 2, 2 } })
		local exact = ode.solve(f64.space, bf, { 0, 10 }, brusselator_start(n), { method = method, rtol = tol.rtol, atol = tol.atol, jac = brusselator_jac(n), jac_band = { 2, 2 } })
		for name, sol in ({ dense = dense, banded = banded, exact = exact } :: { [string]: any }) do
			check(sol.status == "success", `{method} brusselator {name}: {sol.message}`)
		end
		local worst = 0
		local worst_exact = 0
		for i = 1, 2 * n do
			local d = f64.get(last(dense), i)
			worst = math.max(worst, math.abs(f64.get(last(banded), i) - d) / math.max(1, math.abs(d)))
			worst_exact = math.max(worst_exact, math.abs(f64.get(last(exact), i) - d) / math.max(1, math.abs(d)))
		end
		check(worst <= 1e-4, `{method} brusselator banded vs dense: {worst}`)
		check(worst_exact <= 1e-4, `{method} brusselator exact vs dense: {worst_exact}`)
		-- Each banded estimate takes kl + ku + 1 = 5 evaluations instead of 2n = 80.
		check((banded.stats.njev :: number) >= 1, `{method} brusselator banded: njev {banded.stats.njev}`)
		check(banded.stats.nfev < dense.stats.nfev, `{method} brusselator: banded nfev {banded.stats.nfev}, dense {dense.stats.nfev}`)
		check(exact.stats.nfev < banded.stats.nfev, `{method} brusselator exact: nfev {exact.stats.nfev}`)
	end
end)

test("mass matrices and DAEs", function()
	for _, method in STIFF do
		-- M y' = M (Prothero-Robinson) with a dense, nondiagonal M has the same solution.
		local M = f64.matrix(3, 3)
		local entries = { { 2, 1, 0 }, { 1, 3, 1 }, { 0, 1, 4 } }
		for i = 1, 3 do
			for j = 1, 3 do
				f64.matset(M, i, j, entries[i][j])
			end
		end
		local f = prothero(f64)
		local fm = function(t: number, y: buffer, out: buffer): buffer
			return f64.matvec(out, M, f(t, y, f64.create(3)))
		end
		local sol = ode.solve(f64.space, fm, { 0, 5 }, f64.from(g(0)), { method = method, rtol = 1e-8, atol = 1e-10, mass = M })
		check(sol.status == "success", `{method} dense mass: {sol.message}`)
		check(prothero_error(f64, 5, last(sol)) <= 1e-6, `{method} dense mass: error {prothero_error(f64, 5, last(sol))}`)
		-- A diagonal mass as an array: (d_i y_i)' scaled equations.
		local d = f64.from({ 0.5, 2, 10 })
		local fd = function(t: number, y: buffer, out: buffer): buffer
			return f64.mul(out, d, f(t, y, f64.create(3)))
		end
		sol = ode.solve(f64.space, fd, { 0, 5 }, f64.from(g(0)), { method = method, rtol = 1e-8, atol = 1e-10, mass = d })
		check(prothero_error(f64, 5, last(sol)) <= 1e-6, `{method} diagonal mass: error {prothero_error(f64, 5, last(sol))}`)
		-- Scaling the equations by M changes nothing: the same steps as without M (the
		-- starting step too, which must come from M^-1 f).
		local unscaled = ode.solve(f64.space, f, { 0, 5 }, f64.from(g(0)), { method = method, rtol = 1e-8, atol = 1e-10 })
		check(math.abs(sol.stats.steps - unscaled.stats.steps) <= 2, `{method} diagonal mass: {sol.stats.steps} steps, {unscaled.stats.steps} without M`)
		close(sol.t[2], unscaled.t[2], 1e-6, `{method} diagonal mass: first step`)

		-- Robertson as an index-1 DAE: the third equation replaced by y1 + y2 + y3 = 1.
		local dae = function(t: number, y: buffer, out: buffer): buffer
			robertson(t, y, out)
			f64.set(out, 3, f64.get(y, 1) + f64.get(y, 2) + f64.get(y, 3) - 1)
			return out
		end
		for _, mass in { f64.from({ 1, 1, 0 }), (function(): any
			local Md = f64.matrix(3, 3)
			f64.matset(Md, 1, 1, 1)
			f64.matset(Md, 2, 2, 1)
			return Md
		end)() } do
			sol = ode.solve(f64.space, dae, { 0, 40 }, f64.from({ 1, 0, 0 }), { method = method, rtol = 1e-6, atol = 1e-10, mass = mass })
			check(sol.status == "success", `{method} robertson DAE: {sol.message}`)
			local y = last(sol)
			close(f64.get(y, 1), ROBERTSON_40[1], 1e-6, `{method} robertson DAE y1`)
			close(f64.get(y, 2) / ROBERTSON_40[2], 1, 1e-5, `{method} robertson DAE y2`)
			close(f64.get(y, 3), ROBERTSON_40[3], 1e-6, `{method} robertson DAE y3`)
			-- The algebraic constraint holds at every step and in the dense output.
			local drift = 0
			for k = 1, #sol.y do
				local yk = sol.y[k]
				drift = math.max(drift, math.abs(f64.get(yk, 1) + f64.get(yk, 2) + f64.get(yk, 3) - 1))
			end
			local ym = sol.at(20.5)
			drift = math.max(drift, math.abs(f64.get(ym, 1) + f64.get(ym, 2) + f64.get(ym, 3) - 1))
			check(drift <= 1e-12, `{method} robertson DAE constraint: {drift}`)
		end

		-- The Brusselator with a diagonal mass and a band.
		local n = 20
		local bf = brusselator(n)
		local scale = f64.create(2 * n)
		for i = 1, 2 * n do
			f64.set(scale, i, 1 + i / n)
		end
		local bscaled = function(t: number, y: buffer, out: buffer): buffer
			return f64.mul(out, scale, bf(t, y, f64.create(2 * n)))
		end
		local plain = ode.solve(f64.space, bf, { 0, 5 }, brusselator_start(n), { method = method, rtol = 1e-7, atol = 1e-9, jac_band = { 2, 2 } })
		local massed = ode.solve(f64.space, bscaled, { 0, 5 }, brusselator_start(n), { method = method, rtol = 1e-7, atol = 1e-9, jac_band = { 2, 2 }, mass = scale })
		local worst = 0
		for i = 1, 2 * n do
			worst = math.max(worst, math.abs(f64.get(last(plain), i) - f64.get(last(massed), i)))
		end
		check(worst <= 1e-5, `{method} banded mass: {worst}`)
	end
end)

test("stiff dense output, t_eval, events and backward integration", function()
	for _, method in STIFF do
		local f = prothero(f64)
		local t_eval = { 0, 0.25, 1, 2.5, 4 }
		local sol = ode.solve(f64.space, f, { 0, 4 }, f64.from(g(0)), { method = method, rtol = 1e-9, atol = 1e-12, t_eval = t_eval })
		local worst = 0
		for k = 0, 80 do
			local t = 4 * k / 80
			worst = math.max(worst, prothero_error(f64, t, sol.at(t)))
		end
		check(worst <= EXPECT[method].dense, `{method} dense output: {worst}`)
		for k, t in t_eval do
			check(prothero_error(f64, t, (sol.y_eval :: { buffer })[k]) <= EXPECT[method].dense, `{method} t_eval {t}`)
		end
		-- Backward from t = 4 to 0.
		local back = ode.solve(f64.space, function(t: number, y: buffer, out: buffer): buffer
			-- Backward in time the stiff modes grow, so integrate the non-stiff one only.
			f64.set(out, 1, math.cos(t))
			f64.set(out, 2, -f64.get(y, 1))
			return out
		end, { 4, 0 }, f64.from({ math.sin(4), math.cos(4) }), { method = method, rtol = 1e-9, atol = 1e-12 })
		close(f64.get(last(back), 1), 0, 1e-7, `{method} backward y1`)
		close(f64.get(last(back), 2), 1, 1e-7, `{method} backward y2`)
		check(back.t[#back.t] == 0, `{method} backward ends at {back.t[#back.t]}`)

		-- Events: the zero of sin t at pi, and a terminal one at cos t = 0.
		local ev = ode.solve(f64.space, f, { 0, 5 }, f64.from(g(0)), {
			method = method,
			rtol = 1e-9,
			atol = 1e-12,
			events = {
				{ g = function(_: number, y: buffer): number
					return f64.get(y, 1) - 0.5
				end },
				{ g = function(_: number, y: buffer): number
					return f64.get(y, 2) + 0.5
				end, terminal = true },
			},
		})
		check(ev.status == "event", `{method} terminal event: {ev.message}`)
		close(ev.t[#ev.t], 2 * math.pi / 3, 1e-7, `{method} terminal event time`)
		check(#ev.events == 2 and ev.events[1].index == 1, `{method}: {#ev.events} events`)
		close(ev.events[1].t, math.pi / 6, 1e-7, `{method} event time`)
		close(f64.get(ev.events[1].y, 1), 0.5, 1e-7, `{method} event state`)
	end
end)

test("stiff integrators and representations", function()
	for _, method in STIFF do
		-- Frame stepping through Van der Pol lands on each frame time exactly.
		local it = ode.integrator(f64.space, vdp, 0, f64.from({ 2, 0 }), { method = method, rtol = 1e-8, atol = 1e-8 })
		for frame = 1, 40 do
			it:advance(0.05)
			check(it.t == 0.05 * frame or math.abs(it.t - 0.05 * frame) <= 1e-15, `{method} frame {frame}: t = {it.t}`)
		end
		close(f64.get(it.y, 1), VDP_2[1], 1e-5, `{method} integrator vdp y1`)
		check(not pcall(function()
			it:step(0.1)
		end), `{method}: step(h) should need an explicit method`)

		-- f32 arrays: the solves run in single precision.
		local sol32 = ode.solve(f32.space, prothero(f32), { 0, 5 }, f32.from(g(0)), { method = method, rtol = 1e-5, atol = 1e-6 })
		check(sol32.status == "success", `{method} f32: {sol32.message}`)
		check(prothero_error(f32, 5, last(sol32)) <= 1e-4, `{method} f32: error {prothero_error(f32, 5, last(sol32))}`)

		-- Vectors: converted at the edges; the Jacobian sees vectors. Vectors store single
		-- precision, which bounds the attainable accuracy (f is evaluated at rounded states).
		local fv = function(t: number, y: vector, _: vector): vector
			local gt, dgt = g(t), dg(t)
			return vector.create(LAMBDA[1] * (y.x - gt[1]) + dgt[1], LAMBDA[2] * (y.y - gt[2]) + dgt[2], LAMBDA[3] * (y.z - gt[3]) + dgt[3])
		end
		local seen = false
		local jv = function(_: number, y: vector): any
			seen = typeof(y) == "vector"
			local J = f64.matrix(3, 3)
			for i = 1, 3 do
				f64.matset(J, i, i, LAMBDA[i])
			end
			return J
		end
		local g0 = g(0)
		local solv = ode.solve(vec.space, fv, { 0, 5 }, vector.create(g0[1], g0[2], g0[3]), { method = method, rtol = 1e-4, atol = 1e-6, jac = jv, t_eval = { 2.5 } })
		check(solv.status == "success", `{method} vector: {solv.message}`)
		local yv = last(solv) :: any
		local g5 = g(5)
		check(typeof(yv) == "vector", `{method} vector: the result is a {typeof(yv)}`)
		check(math.abs(yv.x - g5[1]) <= 1e-5 and math.abs(yv.y - g5[2]) <= 1e-5 and math.abs(yv.z - g5[3]) <= 1e-5, `{method} vector: {yv}`)
		check(seen, `{method} vector: the Jacobian did not get a vector`)
		local ym = solv.at(1.3) :: any
		local g13 = g(1.3)
		check(typeof(ym) == "vector" and math.abs(ym.x - g13[1]) <= 1e-4, `{method} vector dense output: {ym}`)
		check(typeof((solv.y_eval :: any)[1]) == "vector", `{method} vector t_eval`)
	end
	-- Complex arrays are not supported by the Radau solver.
	check(not pcall(function()
		ode.solve(lib.c64.space, function(_: number, y: buffer, out: buffer): buffer
			return lib.c64.copy(out, y)
		end, { 0, 1 }, lib.c64.from({ 1 }), { method = "radau" })
	end), "radau: complex arrays should be rejected")
end)

test("stiff failures are reported", function()
	for _, method in STIFF do
		-- y' = y^2 blows up at t = 1.
		local sol = ode.solve(f64.space, function(_: number, y: buffer, out: buffer): buffer
			return f64.mul(out, y, y)
		end, { 0, 2 }, f64.from({ 1 }), { method = method })
		check(sol.status == "failed", `{method} blow-up: {sol.status}`)
		check(math.abs(sol.t[#sol.t] - 1) < 1e-2, `{method} blow-up stopped at {sol.t[#sol.t]}`)
		-- A step limit.
		sol = ode.solve(f64.space, vdp, { 0, 2 }, f64.from({ 2, 0 }), { method = method, max_steps = 10 })
		check(sol.status == "failed" and #sol.t == 11, `{method} max_steps: {sol.status}, {#sol.t}`)
	end
end)

-- The 2D Brusselator on an m by m grid (u, v interleaved per point, points by columns),
-- with its sparsity pattern and exact Jacobian.
local function brusselator2(m: number): ((number, buffer, buffer) -> buffer, any, (number, buffer) -> any)
	local c = (m + 1) ^ 2 / 50
	local function u_at(i: number, j: number): number
		return 2 * ((j - 1) * m + i) - 1
	end
	local function f(_: number, y: buffer, out: buffer): buffer
		for j = 1, m do
			for i = 1, m do
				local r = u_at(i, j)
				local u, v = f64.get(y, r), f64.get(y, r + 1)
				local lu, lv = -4 * u, -4 * v
				for _, nb in { { i - 1, j }, { i + 1, j }, { i, j - 1 }, { i, j + 1 } } do
					local a, b = nb[1], nb[2]
					if a >= 1 and a <= m and b >= 1 and b <= m then
						lu += f64.get(y, u_at(a, b))
						lv += f64.get(y, u_at(a, b) + 1)
					else
						lu += 1
						lv += 3
					end
				end
				f64.set(out, r, 1 + u * u * v - 4 * u + c * lu)
				f64.set(out, r + 1, 3 * u - u * u * v + c * lv)
			end
		end
		return out
	end
	local function entries(y: buffer?): ({ number }, { number }, { number })
		local ii, jj, vv = {}, {}, {}
		local function add(i: number, j: number, v: number)
			table.insert(ii, i)
			table.insert(jj, j)
			table.insert(vv, v)
		end
		for j = 1, m do
			for i = 1, m do
				local r = u_at(i, j)
				local u = if y then f64.get(y, r) else 1
				local v = if y then f64.get(y, r + 1) else 1
				add(r, r, 2 * u * v - 4 - 4 * c)
				add(r, r + 1, u * u)
				add(r + 1, r, 3 - 2 * u * v)
				add(r + 1, r + 1, -u * u - 4 * c)
				for _, nb in { { i - 1, j }, { i + 1, j }, { i, j - 1 }, { i, j + 1 } } do
					local a, b = nb[1], nb[2]
					if a >= 1 and a <= m and b >= 1 and b <= m then
						add(r, u_at(a, b), c)
						add(r + 1, u_at(a, b) + 1, c)
					end
				end
			end
		end
		return ii, jj, vv
	end
	local n = 2 * m * m
	local ii, jj, vv = entries(nil)
	local pattern = f64.sparse(n, n, ii, jj, f64.from(vv))
	local jac = function(_: number, y: buffer): any
		local i2, j2, v2 = entries(y)
		return f64.sparse(n, n, i2, j2, f64.from(v2))
	end
	return f, pattern, jac
end

local function brusselator2_start(m: number): buffer
	local y = f64.create(2 * m * m)
	for j = 1, m do
		for i = 1, m do
			local r = 2 * ((j - 1) * m + i) - 1
			f64.set(y, r, 1 + math.sin(2 * math.pi * i / (m + 1)) * math.sin(math.pi * j / (m + 1)))
			f64.set(y, r + 1, 3)
		end
	end
	return y
end

local function max_diff(M: any, a: buffer, b: buffer): number
	local worst = 0
	for i = 1, M.length(a) do
		local ar, ai = M.get(a, i)
		local br, bi = M.get(b, i)
		worst = math.max(worst, math.abs(ar - br) + math.abs((ai or 0) - (bi or 0)))
	end
	return worst
end

test("bdf: sparse, matrix-free and preconditioned Newton-Krylov", function()
	local m = 6
	local f, pattern, jac = brusselator2(m)
	local tspan = { 0, 2 }
	local base = { method = "bdf", rtol = 1e-6, atol = 1e-8 }
	local function run(extra: { [string]: any }): any
		local o: { [string]: any } = table.clone(base)
		for k, v in extra do
			o[k] = v
		end
		local sol = ode.solve(f64.space, f, tspan, brusselator2_start(m), o :: any)
		check(sol.status == "success", `bdf brusselator2: {sol.message}`)
		return sol
	end
	local dense = run({})
	local sparse_fd = run({ jac_sparsity = pattern })
	local sparse_exact = run({ jac = jac, linear = "krylov" })
	local free = run({ linear = "krylov" })
	local precond_calls = 0
	-- A preconditioner built by the user: ILU(0) of a I + b J from the exact Jacobian.
	local n = 2 * m * m
	local ii, jj, ones = {}, {}, {}
	for i = 1, n do
		ii[i], jj[i], ones[i] = i, i, 1
	end
	local onesb = f64.from(ones)
	local prec = run({
		linear = "krylov",
		precond = function(t: number, yy: buffer, a: number, b: number): any
			precond_calls += 1
			local J = jac(t, yy)
			J.values = f64.scale(J.values, b, J.values)
			local I = f64.sparse(n, n, ii, jj, f64.scale(nil, a, onesb))
			local F = f64.sparse_ilu0(f64.sparse_add(J, I))
			return function(out: buffer, r: buffer): buffer
				return f64.sparse_ilu0_solve(out, F, r)
			end
		end,
	})
	local y = last(dense)
	for name, sol in ({ sparse_fd = sparse_fd, sparse_exact = sparse_exact, free = free, prec = prec } :: { [string]: any }) do
		local d = max_diff(f64, last(sol), y)
		check(d <= 1e-5, `bdf brusselator2 {name} vs dense: {d}`)
	end
	-- Grouped differences: a 5-point stencil with two species needs about a dozen
	-- evaluations per Jacobian, not n = 72.
	local per_jac = (sparse_fd.stats.nfev - sparse_exact.stats.nfev) / math.max(sparse_fd.stats.njev :: number, 1)
	check(per_jac < 20, `bdf sparse differences: {per_jac} evaluations per Jacobian`)
	check(sparse_fd.stats.nfev < dense.stats.nfev, `bdf sparse nfev {sparse_fd.stats.nfev}, dense {dense.stats.nfev}`)
	-- The preconditioner is used and saves products.
	check(precond_calls >= 1, "bdf: the preconditioner was never built")
	check(prec.stats.nfev < free.stats.nfev, `bdf preconditioned nfev {prec.stats.nfev}, unpreconditioned {free.stats.nfev}`)
	-- Matrix-free never forms a Jacobian.
	check(free.stats.njev == 0, `bdf matrix-free njev {free.stats.njev}`)

	-- The heat equation on a 10 by 10 grid (eigenvalues down to about -950): ILU(0) of
	-- the sparse system saves GMRES iterations over unpreconditioned matrix-free products
	-- (527 against 942).
	local k = 10
	local hn = k * k
	local hc = (k + 1) ^ 2
	local function heat(_: number, yy: buffer, out: buffer): buffer
		for j = 1, k do
			for i = 1, k do
				local r = (j - 1) * k + i
				local acc = -4 * f64.get(yy, r)
				if i > 1 then
					acc += f64.get(yy, r - 1)
				end
				if i < k then
					acc += f64.get(yy, r + 1)
				end
				if j > 1 then
					acc += f64.get(yy, r - k)
				end
				if j < k then
					acc += f64.get(yy, r + k)
				end
				f64.set(out, r, hc * acc)
			end
		end
		return out
	end
	local hi, hj, hv = {}, {}, {}
	for r = 1, hn do
		for _, q in { r - k, r - 1, r, r + 1, r + k } do
			local same_row = math.abs(q - r) ~= 1 or (q - 1) // k == (r - 1) // k
			if q >= 1 and q <= hn and same_row then
				table.insert(hi, r)
				table.insert(hj, q)
				table.insert(hv, 1)
			end
		end
	end
	local hpattern = f64.sparse(hn, hn, hi, hj, f64.from(hv))
	local h0 = f64.create(hn)
	for r = 1, hn do
		f64.set(h0, r, math.sin(r))
	end
	local hsparse = ode.solve(f64.space, heat, { 0, 0.2 }, h0, { method = "bdf", jac_sparsity = hpattern, rtol = 1e-5, atol = 1e-8 })
	local hfree = ode.solve(f64.space, heat, { 0, 0.2 }, h0, { method = "bdf", linear = "krylov", rtol = 1e-5, atol = 1e-8 })
	check(max_diff(f64, last(hsparse), last(hfree)) <= 1e-5, `bdf heat sparse vs matrix-free: {max_diff(f64, last(hsparse), last(hfree))}`)
	check((hsparse.stats.nlinear :: number) < 0.7 * (hfree.stats.nlinear :: number), `bdf heat ILU: {hsparse.stats.nlinear} GMRES iterations, matrix-free {hfree.stats.nlinear}`)

	-- A sparse mass matrix with the sparse Jacobian.
	local mv = {}
	for i = 1, n do
		mv[i] = 1 + i / n
	end
	local Ms = f64.sparse(n, n, ii, jj, f64.from(mv))
	local fm = function(t: number, yy: buffer, out: buffer): buffer
		return f64.sparse_matvec(out, Ms, f(t, yy, f64.create(n)))
	end
	local massed = ode.solve(f64.space, fm, tspan, brusselator2_start(m), { method = "bdf", rtol = 1e-6, atol = 1e-8, jac_sparsity = pattern, mass = Ms })
	check(massed.status == "success", `bdf sparse mass: {massed.message}`)
	check(max_diff(f64, last(massed), y) <= 1e-5, `bdf sparse mass vs dense: {max_diff(f64, last(massed), y)}`)
end)

test("bdf: complex states", function()
	-- y_k' = mu_k y_k: y = exp(mu_k t), with stiff decay and oscillation.
	local mu = { { -1, 2 }, { -1e4, 5 }, { -1e6, 1 } }
	local function linear(M: any): (number, buffer, buffer) -> buffer
		return function(_: number, y: buffer, out: buffer): buffer
			for k = 1, 3 do
				if M == lib.c32 then
					local z = M.get(y, k)
					M.set(out, k, vector.create(mu[k][1] * z.x - mu[k][2] * z.y, mu[k][1] * z.y + mu[k][2] * z.x, 0))
				else
					local yr, yi = M.get(y, k)
					M.set(out, k, mu[k][1] * yr - mu[k][2] * yi, mu[k][1] * yi + mu[k][2] * yr)
				end
			end
			return out
		end
	end
	local function exact(t: number, k: number): (number, number)
		local r = math.exp(mu[k][1] * t)
		return r * math.cos(mu[k][2] * t), r * math.sin(mu[k][2] * t)
	end
	for _, case in ({ { lib.c64, 1e-8, 1e-10, 1e-5 }, { lib.c32, 1e-4, 1e-6, 2e-3 } } :: { { any } }) do
		local M, rtol, atol, tol = case[1], case[2], case[3], case[4]
		for _, linear_kind in { "direct", "krylov" } do
			local sol = ode.solve(M.space, linear(M), { 0, 1 }, M.from({ 1, 1, 1 }, { 0, 0, 0 }), { method = "bdf", rtol = rtol, atol = atol, linear = linear_kind })
			check(sol.status == "success", `bdf complex {linear_kind}: {sol.message}`)
			local function at(x: buffer): (number, number)
				if M == lib.c32 then
					local v = M.get(x, 1) :: any
					return v.x, v.y
				end
				return M.get(x, 1)
			end
			local yr, yi = at(last(sol))
			local er, ei = exact(1, 1)
			check(math.abs(yr - er) + math.abs(yi - ei) <= tol, `bdf complex {linear_kind}: {yr} + {yi} i, expected {er} + {ei} i`)
			local mr, mi = at(sol.at(0.5))
			local xr, xi = exact(0.5, 1)
			check(math.abs(mr - xr) + math.abs(mi - xi) <= tol, `bdf complex dense output {linear_kind}`)
		end
	end
	-- An integrator cannot reverse.
	local it = ode.integrator(f64.space, prothero(f64), 0, f64.from(g(0)), { method = "bdf" })
	it:advance(0.5)
	check(not pcall(function()
		it:advance(-0.25)
	end), "bdf integrator: reversing should be an error")
end)

test("finite-difference Jacobians", function()
	-- f_i = sum_{j = i - kl}^{i + ku} w_ij y_j^2 + y_i^3 / 10, with w_ij = (i + 2 j) / 10:
	-- banded with kl = 1, ku = 2 and complex-differentiable, so every module can check it.
	local n, kl, ku = 7, 1, 2
	local function w(i: number, j: number): number
		return (i + 2 * j) / 10
	end
	for _, M in { f64, f32, lib.c64, lib.c32 } :: { any } do
		local cplx = M == lib.c64 or M == lib.c32
		local function get(x: any, i: number): (number, number)
			if M == lib.c32 then
				local v = M.get(x, i)
				return v.x, v.y
			elseif M == lib.c64 then
				return M.get(x, i)
			end
			return M.get(x, i), 0
		end
		local function set(x: any, i: number, re: number, im: number)
			if M == lib.c32 then
				M.set(x, i, vector.create(re, im, 0))
			elseif M == lib.c64 then
				M.set(x, i, re, im)
			else
				M.set(x, i, re)
			end
		end
		local function cmul(ar: number, ai: number, br: number, bi: number): (number, number)
			return ar * br - ai * bi, ar * bi + ai * br
		end
		local f = function(_: number, y: buffer, out: buffer): buffer
			for i = 1, n do
				local sr, si = 0, 0
				for j = math.max(1, i - kl), math.min(n, i + ku) do
					local yr, yi = get(y, j)
					local qr, qi = cmul(yr, yi, yr, yi)
					sr += w(i, j) * qr
					si += w(i, j) * qi
				end
				local yr, yi = get(y, i)
				local qr, qi = cmul(yr, yi, yr, yi)
				local cr, ci = cmul(qr, qi, yr, yi)
				set(out, i, sr + cr / 10, si + ci / 10)
			end
			return out
		end
		local y = M.create(n)
		for j = 1, n do
			set(y, j, 0.5 + 0.1 * j, if cplx then 0.3 - 0.05 * j else 0)
		end
		-- The exact entry (i, j).
		local function exact(i: number, j: number): (number, number)
			if j < i - kl or j > i + ku then
				return 0, 0
			end
			local yr, yi = get(y, j)
			local er, ei = 2 * w(i, j) * yr, 2 * w(i, j) * yi
			if i == j then
				local qr, qi = cmul(yr, yi, yr, yi)
				er += 0.3 * qr
				ei += 0.3 * qi
			end
			return er, ei
		end
		local tol = if M == f32 or M == lib.c32 then 2e-3 else 1e-6
		local name = if M == f64 then "f64" elseif M == f32 then "f32" elseif M == lib.c64 then "c64" else "c32"
		local function compare(kind: string, entry: (number, number) -> (number, number))
			local worst = 0
			for i = 1, n do
				for j = 1, n do
					local ar, ai = entry(i, j)
					local er, ei = exact(i, j)
					worst = math.max(worst, math.abs(ar - er) + math.abs(ai - ei))
				end
			end
			check(worst <= tol, `jacobian {name} {kind}: error {worst}`)
		end
		local Jd = ode.jacobian(M.space, f, 0, y)
		compare("dense", function(i: number, j: number): (number, number)
			if M == lib.c32 then
				local v = M.matget(Jd, i, j)
				return v.x, v.y
			elseif M == lib.c64 then
				return M.matget(Jd, i, j)
			end
			return M.matget(Jd, i, j), 0
		end)
		local Jb = ode.jacobian(M.space, f, 0, y, { jac_band = { kl, ku } })
		compare("banded", function(i: number, j: number): (number, number)
			if M == lib.c32 then
				local v = M.band_get(Jb, i, j)
				return v.x, v.y
			elseif M == lib.c64 then
				return M.band_get(Jb, i, j)
			end
			return M.band_get(Jb, i, j), 0
		end)
		local ii, jj = {}, {}
		for i = 1, n do
			for j = math.max(1, i - kl), math.min(n, i + ku) do
				table.insert(ii, i)
				table.insert(jj, j)
			end
		end
		local pattern = M.sparse(n, n, ii, jj, M.create(#ii))
		local calls = 0
		local counted = function(t: number, yy: buffer, out: buffer): buffer
			calls += 1
			return f(t, yy, out)
		end
		local Js = ode.jacobian(M.space, counted, 0, y, { jac_sparsity = pattern })
		compare("sparse", function(i: number, j: number): (number, number)
			if M == lib.c32 then
				local v = M.sparse_get(Js, i, j)
				return v.x, v.y
			elseif M == lib.c64 then
				return M.sparse_get(Js, i, j)
			end
			return M.sparse_get(Js, i, j), 0
		end)
		-- Columns of a band of width kl + ku + 1 = 4 fall into 4 groups (plus f(y) itself).
		check(calls == 5, `jacobian {name} sparse: {calls} evaluations`)
	end
end)

return nil
