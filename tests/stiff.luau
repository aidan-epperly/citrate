--!strict

-- Stiff solvers: reference solutions of standard stiff problems, Jacobians (finite
-- differences, user functions, constant, banded), mass matrices and DAEs, dense output,
-- events, integrators and every representation.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local ode, f64, f32, vec = lib.ode, lib.f64, lib.f32, lib.vec

local STIFF = { "radau" }

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
		close(f64.get(y, 1), ROBERTSON_40[1], 1e-7, `{method} robertson y1`)
		close(f64.get(y, 2) / ROBERTSON_40[2], 1, 1e-6, `{method} robertson y2`)
		close(f64.get(y, 3), ROBERTSON_40[3], 1e-7, `{method} robertson y3`)
		-- Mass conservation: y1 + y2 + y3 = 1 holds for any linear one-step method.
		close(f64.get(y, 1) + f64.get(y, 2) + f64.get(y, 3), 1, 1e-13, `{method} robertson mass`)
		-- A regression bound on the work (SciPy's Radau takes 78 steps).
		check(sol.stats.steps <= 85, `{method} robertson: {sol.stats.steps} steps`)

		for _, tol in { 1e-4, 1e-7, 1e-10 } do
			sol = ode.solve(f64.space, vdp, { 0, 2 }, f64.from({ 2, 0 }), { method = method, rtol = tol, atol = tol })
			y = last(sol)
			check(sol.status == "success", `{method} vdp {tol}: {sol.message}`)
			local err = math.max(math.abs(f64.get(y, 1) - VDP_2[1]), math.abs(f64.get(y, 2) - VDP_2[2]))
			check(err <= 100 * tol, `{method} vdp {tol}: error {err}`)
			-- Regression bounds on the work (SciPy's Radau: 311, 1551 and 8504 steps; 2905,
			-- 12630 and 64621 evaluations). The evaluations catch a Jacobian that is never
			-- refreshed and a missing starting guess from the last step's polynomial; the
			-- rejections at 1e-4 catch a missing Gustafsson predictor or safety factor.
			local k = if tol == 1e-4 then 1 elseif tol == 1e-7 then 2 else 3
			check(sol.stats.steps <= ({ 340, 1650, 9000 })[k], `{method} vdp {tol}: {sol.stats.steps} steps`)
			check(sol.stats.nfev <= ({ 3300, 13500, 72000 })[k], `{method} vdp {tol}: {sol.stats.nfev} evaluations`)
			check(sol.stats.rejected <= ({ 40, 45, 60 })[k], `{method} vdp {tol}: {sol.stats.rejected} rejections`)
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
		check(worst <= 1e-7, `{method} dense output: {worst}`)
		for k, t in t_eval do
			check(prothero_error(f64, t, (sol.y_eval :: { buffer })[k]) <= 1e-7, `{method} t_eval {t}`)
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
		check(math.abs(sol.t[#sol.t] - 1) < 1e-3, `{method} blow-up stopped at {sol.t[#sol.t]}`)
		-- A step limit.
		sol = ode.solve(f64.space, vdp, { 0, 2 }, f64.from({ 2, 0 }), { method = method, max_steps = 10 })
		check(sol.status == "failed" and #sol.t == 11, `{method} max_steps: {sol.status}, {#sol.t}`)
	end
end)

return nil
