--!strict

-- Second-order equations: Runge-Kutta-Nystrom orders, accuracy, dense output and
-- efficiency; reduction to first order with explicit and stiff methods; events, output
-- times and vectors.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local ode, f64, vec = lib.ode, lib.f64, lib.vec

local RKN = { "dprkn6", "dprkn8", "dprkn12" }
local STAGES: { [string]: number } = { dprkn6 = 6, dprkn8 = 9, dprkn12 = 17 }

-- y'' = -y, y(0) = 0, y'(0) = 1: y = sin t, v = cos t.
local function oscillator(_: number, y: buffer, _: buffer, out: buffer): buffer
	return f64.scale(out, -1, y)
end

-- Kepler with eccentricity 0.5 (period 2 pi).
local E = 0.5
local function kepler(_: number, q: buffer, _: buffer, out: buffer): buffer
	local x, y = f64.get(q, 1), f64.get(q, 2)
	local r3 = (x * x + y * y) ^ 1.5
	f64.set(out, 1, -x / r3)
	f64.set(out, 2, -y / r3)
	return out
end
local function kepler_start(): (buffer, buffer)
	return f64.from({ 1 - E, 0 }), f64.from({ 0, math.sqrt((1 + E) / (1 - E)) })
end

local function last(sol: any): (buffer, buffer)
	return sol.y[#sol.y], sol.v[#sol.v]
end

-- The forced oscillator y'' = -y + sin 2t (non-autonomous, so stage times matter),
-- y(0) = 0, y'(0) = 1: y = 5/3 sin t - 1/3 sin 2t, v = 5/3 cos t - 2/3 cos 2t.
local function forced(t: number, y: buffer, _: buffer, out: buffer): buffer
	out = f64.scale(out, -1, y)
	f64.set(out, 1, f64.get(out, 1) + math.sin(2 * t))
	return out
end

test("Runge-Kutta-Nystrom orders", function()
	-- Constant steps (tolerances too loose to reject; max_step fixes h) on the forced
	-- oscillator.
	local expected: { [string]: { number } } = { dprkn6 = { 6, 10 }, dprkn8 = { 8, 8 }, dprkn12 = { 12, 8 } }
	for _, method in RKN do
		local p, n = expected[method][1], expected[method][2]
		local errs = {}
		for k, steps in { n, 2 * n } do
			local hs = 10 / steps
			local sol = ode.solve_second_order(f64.space, forced, { 0, 10 }, f64.from({ 0 }), f64.from({ 1 }), { method = method, h = hs, max_step = hs, rtol = 1e6, atol = 1e6 })
			local y, v = last(sol)
			local ey = 5 / 3 * math.sin(10) - math.sin(20) / 3
			local ev = 5 / 3 * math.cos(10) - 2 / 3 * math.cos(20)
			errs[k] = math.abs(f64.get(y, 1) - ey) + math.abs(f64.get(v, 1) - ev)
		end
		local observed = math.log(errs[1] / errs[2], 2)
		check(observed >= p - 1 and observed <= p + 1, `{method}: order {observed} (expected {p})`)
	end
end)

test("Runge-Kutta-Nystrom accuracy, dense output and work", function()
	local q0, v0 = kepler_start()
	local ref = ode.solve_second_order(f64.space, kepler, { 0, 4 * math.pi }, q0, v0, { method = "dprkn12", rtol = 1e-15, atol = 1e-15 })
	local function distance(a: buffer, b: buffer): number
		return math.abs(f64.get(a, 1) - f64.get(b, 1)) + math.abs(f64.get(a, 2) - f64.get(b, 2))
	end
	for _, method in RKN do
		for _, tol in { 1e-6, 1e-10 } do
			local calls = 0
			local counted = function(t: number, q: buffer, v: buffer, out: buffer): buffer
				calls += 1
				return kepler(t, q, v, out)
			end
			local sol = ode.solve_second_order(f64.space, counted, { 0, 4 * math.pi }, q0, v0, { method = method, rtol = tol, atol = tol })
			check(sol.status == "success", `{method} {tol}: {sol.message}`)
			check(calls == sol.stats.nfev, `{method}: {calls} calls of f, {sol.stats.nfev} counted`)
			-- The error at the steps tracks the tolerance; the dense output (positions and
			-- velocities) is as accurate as the steps.
			local at_steps, dense, dense_v = 0, 0, 0
			for k, t in sol.t do
				local yr = ref.at(t)
				at_steps = math.max(at_steps, distance(sol.y[k], yr))
			end
			for i = 0, 200 do
				local t = 4 * math.pi * (i + 0.37) / 201
				local y, v = sol.at(t)
				local yr, vr = ref.at(t)
				dense = math.max(dense, distance(y, yr))
				dense_v = math.max(dense_v, distance(v, vr))
			end
			check(at_steps <= 1000 * tol, `{method} {tol}: error {at_steps}`)
			check(dense <= 2 * at_steps + 1e-13, `{method} {tol}: dense error {dense} against {at_steps} at the steps`)
			check(dense_v <= 20 * at_steps + 1e-12, `{method} {tol}: dense velocity error {dense_v}`)
			-- Evaluations: the first stage of each step is the last step's final
			-- acceleration (and for the FSAL pairs, the last stage is the next first).
			local s = STAGES[method]
			local per = if method == "dprkn12" then s else s - 1
			local expected_nfev = per * sol.stats.steps + (s - 1) * sol.stats.rejected + 2
			check(sol.stats.nfev <= expected_nfev, `{method}: {sol.stats.nfev} evaluations for {sol.stats.steps} steps`)
		end
	end
	-- Fewer evaluations than DOP853 on the first-order system, and more accurate.
	for _, case in { { "dprkn8", 1e-10 }, { "dprkn12", 1e-13 } } do
		local method, tol = case[1] :: string, case[2] :: number
		local a = ode.solve_second_order(f64.space, kepler, { 0, 4 * math.pi }, q0, v0, { method = method, rtol = tol, atol = tol })
		local b = ode.solve_second_order(f64.space, kepler, { 0, 4 * math.pi }, q0, v0, { method = "dop853", rtol = tol, atol = tol })
		local ya, yb, yr = a.y[#a.y], b.y[#b.y], ref.y[#ref.y]
		check(a.stats.nfev < b.stats.nfev, `{method}: {a.stats.nfev} evaluations, dop853 {b.stats.nfev}`)
		check(distance(ya, yr) < distance(yb, yr), `{method}: error {distance(ya, yr)}, dop853 {distance(yb, yr)}`)
	end
end)

test("second-order equations by reduction", function()
	-- Damped: y'' = -y - y' / 5 (f depends on v), y(0) = 1, y'(0) = 0.
	local w = math.sqrt(1 - 0.01)
	local function damped_exact(t: number): (number, number)
		local e = math.exp(-0.1 * t)
		local y = e * (math.cos(w * t) + 0.1 / w * math.sin(w * t))
		local v = e * (-w * math.sin(w * t) + 0.1 * math.cos(w * t)) - 0.1 * y
		return y, v
	end
	local damped = function(_: number, y: buffer, v: buffer, out: buffer): buffer
		out = f64.scale(out, -1, y)
		return f64.axpy(out, -0.2, v, out)
	end
	for _, method in { "dp5", "dop853", "radau", "bdf", "rodas5p", "auto" } do
		local sol = ode.solve_second_order(f64.space, damped, { 0, 5 }, f64.from({ 1 }), f64.from({ 0 }), { method = method, rtol = 1e-8, atol = 1e-10, t_eval = { 1, 2.5 } })
		check(sol.status == "success", `{method} damped: {sol.message}`)
		local y, v = last(sol)
		local ey, ev = damped_exact(5)
		check(math.abs(f64.get(y, 1) - ey) <= 1e-6 and math.abs(f64.get(v, 1) - ev) <= 1e-6, `{method} damped: y {f64.get(y, 1)} v {f64.get(v, 1)}, expected {ey} {ev}`)
		local ye, ve = (sol.y_eval :: any)[2], (sol.v_eval :: any)[2]
		local xy, xv = damped_exact(2.5)
		check(math.abs(f64.get(ye, 1) - xy) <= 1e-6 and math.abs(f64.get(ve, 1) - xv) <= 1e-6, `{method} damped t_eval`)
		local ym, vm = sol.at(3.3)
		local zy, zv = damped_exact(3.3)
		check(math.abs(f64.get(ym, 1) - zy) <= 1e-6 and math.abs(f64.get(vm, 1) - zv) <= 1e-6, `{method} damped dense output`)
	end

	-- Stiff: y'' = -1000 y - 1001 y' (rates -1 and -1000), y(0) = 1, y'(0) = 0:
	-- y = (1000 e^-t - e^-1000t) / 999. The stiff methods take few steps.
	local stiff_spring = function(_: number, y: buffer, v: buffer, out: buffer): buffer
		out = f64.scale(out, -1000, y)
		return f64.axpy(out, -1001, v, out)
	end
	local explicit = ode.solve_second_order(f64.space, stiff_spring, { 0, 10 }, f64.from({ 1 }), f64.from({ 0 }), { method = "dp5", rtol = 1e-6, atol = 1e-9 })
	for _, method in { "radau", "bdf", "rodas5p" } do
		local sol = ode.solve_second_order(f64.space, stiff_spring, { 0, 10 }, f64.from({ 1 }), f64.from({ 0 }), { method = method, rtol = 1e-6, atol = 1e-9 })
		local y = last(sol)
		close(f64.get(y, 1), 1000 * math.exp(-10) / 999, 1e-6, `{method} stiff spring`)
		check(sol.stats.steps * 10 < explicit.stats.steps, `{method} stiff spring: {sol.stats.steps} steps, dp5 {explicit.stats.steps}`)
	end

	-- Unsupported combinations are refused.
	check(not pcall(function()
		ode.solve_second_order(f64.space, damped, { 0, 1 }, f64.from({ 1 }), f64.from({ 0 }), { method = "radau", jac = function(): any
			return nil
		end })
	end), "solve_second_order: jac should be refused")
	check(not pcall(function()
		ode.solve_second_order(vec.space, function(_: number, y: vector): vector
			return -y
		end, { 0, 1 }, vector.create(1, 0, 0), vector.create(0, 0, 0), { method = "radau" })
	end), "solve_second_order: stiff methods with vectors should be refused")
end)

test("second-order events, output times and vectors", function()
	-- The oscillator's y first decreases through zero at pi, and v first increases
	-- through zero at 3 pi / 2 (terminal).
	for _, method in { "dprkn6", "dprkn8", "dprkn12", "dp5" } do
		local sol = ode.solve_second_order(f64.space, oscillator, { 0, 10 }, f64.from({ 0 }), f64.from({ 1 }), {
			method = method,
			rtol = 1e-10,
			atol = 1e-12,
			t_eval = { 0, 1, 2 },
			events = {
				{ g = function(_: number, y: buffer, _: buffer): number
					return f64.get(y, 1)
				end, direction = -1 },
				{ g = function(_: number, _: buffer, v: buffer): number
					return f64.get(v, 1)
				end, direction = 1, terminal = true },
			},
		} :: any)
		check(sol.status == "event", `{method} events: {sol.message}`)
		close(sol.t[#sol.t], 3 * math.pi / 2, 1e-8, `{method} terminal event time`)
		check(#sol.events == 2 and sol.events[1].index == 1, `{method}: {#sol.events} events`)
		close(sol.events[1].t, math.pi, 1e-8, `{method} event time`)
		close(f64.get(sol.events[1].v, 1), -1, 1e-8, `{method} event velocity`)
		close(f64.get((sol.v_eval :: any)[2], 1), math.cos(1), 1e-8, `{method} v_eval`)
	end
	-- Vectors stay vectors: Kepler in the plane.
	local accel = function(_: number, q: vector, _: vector): vector
		return -q / vector.magnitude(q) ^ 3
	end
	for _, method in { "dprkn8", "dop853" } do
		local sol = ode.solve_second_order(vec.space, accel, { 0, 2 * math.pi }, vector.create(1 - E, 0, 0), vector.create(0, math.sqrt((1 + E) / (1 - E)), 0), { method = method, rtol = 1e-6, atol = 1e-6 })
		local y, v = sol.y[#sol.y] :: any, sol.v[#sol.v] :: any
		check(type(y) == "vector" and type(v) == "vector", `{method} vectors: {typeof(y)}`)
		check(math.abs(y.x - (1 - E)) <= 1e-4 and math.abs(y.y) <= 1e-4, `{method} vector orbit: {y}`)
		local ym, vm = sol.at(1)
		check(type(ym) == "vector" and type(vm) == "vector", `{method} vector dense output`)
	end
end)

return nil
