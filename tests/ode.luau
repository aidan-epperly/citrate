--!strict

-- Explicit Runge-Kutta methods: order of accuracy, adaptive step control, dense output,
-- events, output times, backward integration, integrators, and every representation.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local ode, f64, c64, vec = lib.ode, lib.f64, lib.c64, lib.vec

-- y' = y cos t, y(0) = 1: y = exp(sin t). Nonlinear in t, so it tests every stage.
local function growth(t: number, y: buffer, out: buffer): buffer
	return f64.scale(out, math.cos(t), y)
end
local function exact(t: number): number
	return math.exp(math.sin(t))
end

local function final(sol: lib.OdeSolution<buffer>): number
	return f64.get(sol.y[#sol.y], 1)
end

test("order of accuracy of every method", function()
	-- Fixed steps h and h/2 over [0, 2]: the error ratio is about 2^order.
	local orders: { [string]: number } = { euler = 1, midpoint = 2, heun = 2, rk4 = 4, bs3 = 3, dp5 = 5, dop853 = 8 }
	local steps: { [string]: number } = { euler = 0.01, midpoint = 0.02, heun = 0.02, rk4 = 0.05, bs3 = 0.05, dp5 = 0.1, dop853 = 0.25 }
	for method, p in orders do
		local errs: { number } = {}
		for k, hs in { steps[method], steps[method] / 2 } do
			local it = ode.integrator(f64.space, growth, 0, f64.from({ 1 }), { method = method })
			local n = math.round(2 / hs)
			for _ = 1, n do
				it:step(hs)
			end
			errs[k] = math.abs(f64.get(it.y, 1) - exact(2))
		end
		local observed = math.log(errs[1] / errs[2], 2)
		check(math.abs(observed - p) <= 0.3, `{method}: order {observed} (expected {p})`)
	end
end)

test("adaptive methods meet their tolerance", function()
	for _, method in { "bs3", "dp5", "dop853" } do
		local last = math.huge
		local ok = true
		for _, tol in { 1e-4, 1e-7, 1e-10 } do
			local sol = ode.solve(f64.space, growth, { 0, 10 }, f64.from({ 1 }), { method = method, rtol = tol, atol = tol })
			local err = math.abs(final(sol) - exact(10)) / exact(10)
			ok = ok and sol.status == "success" and err <= 100 * tol and err < last
			last = err
		end
		check(ok, `{method}: global error tracks the tolerance`)
	end
	-- Higher order takes fewer steps at a tight tolerance.
	local counts = {}
	for _, method in { "bs3", "dp5", "dop853" } do
		counts[method] = ode.solve(f64.space, growth, { 0, 10 }, f64.from({ 1 }), { method = method, rtol = 1e-10, atol = 1e-12 }).stats.steps
	end
	check(counts.dop853 < counts.dp5 and counts.dp5 < counts.bs3, `steps: bs3 {counts.bs3}, dp5 {counts.dp5}, dop853 {counts.dop853}`)
	-- Regression bounds on the step counts (rtol = atol = 1e-8; seeded, deterministic):
	-- the PI controller's exponent and safety factor show up here, not in the accuracy.
	local bounds: { [string]: number } = { bs3 = 1150, dp5 = 85, dop853 = 26 }
	for method, bound in bounds do
		local steps = ode.solve(f64.space, growth, { 0, 10 }, f64.from({ 1 }), { method = method, rtol = 1e-8, atol = 1e-8 }).stats.steps
		check(steps <= bound, `{method}: {steps} steps at rtol 1e-8 (at most {bound})`)
	end
end)

test("dense output, t_eval and backward integration", function()
	for _, method in { "bs3", "dp5", "dop853" } do
		local tol = 1e-8
		local sol = ode.solve(f64.space, growth, { 0, 10 }, f64.from({ 1 }), { method = method, rtol = tol, atol = tol })
		local worst = 0
		for k = 0, 200 do
			local t = 10 * k / 200
			worst = math.max(worst, math.abs(f64.get(sol.at(t), 1) - exact(t)) / exact(t))
		end
		check(worst <= 100 * tol, `{method}: dense output between steps ({worst})`)
		local agree = true
		for i, t in sol.t do
			agree = agree and math.abs(f64.get(sol.at(t), 1) - f64.get(sol.y[i], 1)) <= 1e-12 * exact(t)
		end
		check(agree, `{method}: dense output passes through the steps`)
	end
	local times = { 0, 0.5, 1.7, 3, 9.99 }
	local sol = ode.solve(f64.space, growth, { 0, 10 }, f64.from({ 1 }), { t_eval = times, rtol = 1e-9, atol = 1e-12, dense = false })
	local ok = sol.y_eval ~= nil
	for i, t in times do
		ok = ok and math.abs(f64.get((sol.y_eval :: { buffer })[i], 1) - exact(t)) <= 1e-7 * exact(t)
	end
	check(ok, "t_eval without keeping the dense output")
	check(not pcall(function()
		sol.at(1)
	end), "at needs dense output")
	-- Backward from t = 10 to 0.
	local back = ode.solve(f64.space, growth, { 10, 0 }, f64.from({ exact(10) }), { rtol = 1e-10, atol = 1e-12 })
	check(math.abs(final(back) - 1) <= 1e-8 and back.t[#back.t] == 0, "backward integration")
	close(f64.get(back.at(4.2), 1), exact(4.2), 1e-7, "backward: dense output")
	-- Fixed-step methods interpolate with cubic Hermite polynomials.
	local fixed = ode.solve(f64.space, growth, { 0, 2 }, f64.from({ 1 }), { method = "rk4", h = 0.01 })
	close(f64.get(fixed.at(1.234), 1), exact(1.234), 1e-7, "rk4: Hermite dense output")
	check(math.abs(fixed.t[#fixed.t] - 2) <= 1e-12 and #fixed.t == 201, "rk4: fixed steps up to the end")
end)

test("events", function()
	-- A ball dropped from height 5 (y'' = -9.8): it lands at t = sqrt(10 / 9.8).
	local fall = function(_: number, y: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(y, 2))
		f64.set(out, 2, -9.8)
		return out
	end
	local ground: lib.OdeEvent<buffer> = {
		g = function(_: number, y: buffer): number
			return f64.get(y, 1)
		end,
		terminal = true,
		direction = -1,
	}
	local sol = ode.solve(f64.space, fall, { 0, 10 }, f64.from({ 5, 0 }), { events = { ground } })
	local land = math.sqrt(10 / 9.8)
	check(sol.status == "event" and #sol.events == 1, "a terminal event stops the integration")
	close(sol.events[1].t, land, 1e-12, "landing time")
	close(sol.t[#sol.t], land, 1e-12, "the solution ends at the event")
	close(f64.get(sol.events[1].y, 2), -9.8 * land, 1e-9, "velocity at landing")
	-- Non-terminal events record every crossing; direction filters them.
	local wave = function(t: number, _: buffer, out: buffer): buffer
		f64.set(out, 1, math.cos(t))
		return out
	end
	local zero: lib.OdeEvent<buffer> = {
		g = function(_: number, y: buffer): number
			return f64.get(y, 1)
		end,
	}
	local up: lib.OdeEvent<buffer> = {
		g = function(_: number, y: buffer): number
			return f64.get(y, 1)
		end,
		direction = 1,
	}
	local s2 = ode.solve(f64.space, wave, { 0, 10 }, f64.from({ 0.5 }), { events = { zero, up }, rtol = 1e-10, atol = 1e-12 })
	-- sin t + 0.5 = 0 at t = 7 pi / 6 + 2 pi k and 11 pi / 6 + 2 pi k; up-crossings at 11 pi / 6.
	local all, ups = {}, {}
	for _, e in s2.events do
		table.insert(if e.index == 1 then all else ups, e.t)
	end
	check(#all == 3 and #ups == 1, `all crossings {#all}, upward {#ups}`)
	close(all[1], 7 * math.pi / 6, 1e-9, "first crossing")
	close(ups[1], 11 * math.pi / 6, 1e-9, "the upward crossing")
	check(s2.status == "success", "non-terminal events do not stop it")
end)

test("conservation on the Kepler problem", function()
	-- An eccentric orbit (e = 0.6): energy and angular momentum stay put, and after one
	-- period (2 pi) the body is back where it started.
	local e = 0.6
	local kepler = function(_: number, y: buffer, out: buffer): buffer
		local x, yy = f64.get(y, 1), f64.get(y, 2)
		local r3 = (x * x + yy * yy) ^ 1.5
		f64.set(out, 1, f64.get(y, 3))
		f64.set(out, 2, f64.get(y, 4))
		f64.set(out, 3, -x / r3)
		f64.set(out, 4, -yy / r3)
		return out
	end
	local y0 = f64.from({ 1 - e, 0, 0, math.sqrt((1 + e) / (1 - e)) })
	local function invariants(y: buffer): (number, number)
		local x, yy, u, v = f64.get(y, 1), f64.get(y, 2), f64.get(y, 3), f64.get(y, 4)
		return 0.5 * (u * u + v * v) - 1 / math.sqrt(x * x + yy * yy), x * v - yy * u
	end
	local E0, L0 = invariants(y0)
	for _, method in { "dp5", "dop853" } do
		local sol = ode.solve(f64.space, kepler, { 0, 2 * math.pi }, y0, { method = method, rtol = 1e-10, atol = 1e-12 })
		local yend = sol.y[#sol.y]
		local E1, L1 = invariants(yend)
		check(math.abs(E1 - E0) <= 1e-8 and math.abs(L1 - L0) <= 1e-8, `{method}: energy and angular momentum`)
		check(f64.norm(f64.sub(nil, yend, y0)) <= 1e-7, `{method}: back after one period`)
	end
end)

test("every representation", function()
	-- The harmonic oscillator x'' = -x on vectors, f64 buffers and complex numbers.
	local osc = function(_: number, y: vector, _out: vector): vector
		return vector.create(y.y, -y.x, 0)
	end
	local sv = ode.solve(vec.space, osc, { 0, 2 }, vector.create(1, 0, 0), { rtol = 1e-6, atol = 1e-9 })
	local yv = sv.y[#sv.y]
	check(typeof(yv) == "vector" and typeof(sv.at(1)) == "vector", "vectors stay vectors")
	check(math.abs(yv.x - math.cos(2)) <= 1e-5 and math.abs(yv.y + math.sin(2)) <= 1e-5, "vector oscillator")
	local oscb = function(_: number, y: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(y, 2))
		f64.set(out, 2, -f64.get(y, 1))
		f64.set(out, 3, 0)
		return out
	end
	local sb = ode.solve(f64.space, oscb, { 0, 2 }, f64.from({ 1, 0, 0 }), { rtol = 1e-6, atol = 1e-9 })
	local yb = sb.y[#sb.y]
	-- (Vector arithmetic rounds to single precision, so the step counts may differ slightly.)
	check(math.abs(f64.get(yb, 1) - yv.x) <= 1e-5 and math.abs(#sb.t - #sv.t) <= 2, `vectors and buffers agree (steps {#sv.t - 1} and {#sb.t - 1})`)
	-- Complex: y' = i y, y = e^(i t).
	local rot = function(_: number, y: buffer, out: buffer): buffer
		return c64.cscale(out, 0, 1, y)
	end
	local sc = ode.solve(c64.space, rot, { 0, math.pi }, c64.from({ 1 }, { 0 }), { rtol = 1e-10, atol = 1e-12 })
	local re, im = c64.get(sc.y[#sc.y], 1)
	check(math.abs(re + 1) <= 1e-8 and math.abs(im) <= 1e-8, "complex: e^(i pi) = -1")
	-- Componentwise error control: a component of size 1e-6 next to one of 1e6 still gets
	-- relative accuracy (a whole-vector norm would let it go).
	local decay = function(_: number, y: buffer, out: buffer): buffer
		return f64.scale(out, -1, y)
	end
	local sd = ode.solve(f64.space, decay, { 0, 3 }, f64.from({ 1e6, 1e-6 }), { rtol = 1e-8, atol = 1e-30 })
	close(f64.get(sd.y[#sd.y], 2), 1e-6 * math.exp(-3), 1e-6, "small component, relative accuracy")
	-- A space without errnorm falls back to whole-vector norms.
	local plain: lib.Space<buffer> = table.clone(f64.space) :: any
	;(plain :: any).errnorm = nil
	local sp = ode.solve(plain, decay, { 0, 3 }, f64.from({ 1, 2 }), { rtol = 1e-8, atol = 1e-10 })
	close(f64.get(sp.y[#sp.y], 1), math.exp(-3), 1e-7, "fallback norm without errnorm")
end)

test("integrators", function()
	-- advance(dt) in frames matches solve.
	local it = ode.integrator(f64.space, growth, 0, f64.from({ 1 }), { rtol = 1e-9, atol = 1e-12 })
	for _ = 1, 100 do
		it:advance(0.1)
	end
	close(it.t, 10, 1e-12, "advance lands exactly on the frame times")
	close(f64.get(it.y, 1), exact(10), 1e-7, "adaptive advance")
	check(it.stats.steps > 0 and it.stats.nfev > it.stats.steps, "statistics")
	-- Fixed steps: step(h) agrees with solve's fixed steps exactly.
	local rk = ode.integrator(f64.space, growth, 0, f64.from({ 1 }), { method = "rk4" })
	for _ = 1, 20 do
		rk:step(0.1)
	end
	local sol = ode.solve(f64.space, growth, { 0, 2 }, f64.from({ 1 }), { method = "rk4", h = 0.1 })
	check(math.abs(f64.get(rk.y, 1) - final(sol)) <= 1e-13, "rk4 integrator matches solve")
	local fixed = ode.integrator(f64.space, growth, 0, f64.from({ 1 }), { method = "rk4", h = 0.01 })
	fixed:advance(1)
	close(f64.get(fixed.y, 1), exact(1), 1e-8, "fixed-step advance uses substeps of h")
	-- Vectors, stepping a frame at a time.
	local spring = ode.integrator(vec.space, function(_: number, y: vector, _out: vector): vector
		return vector.create(y.y, -4 * y.x, 0)
	end, 0, vector.create(1, 0, 0))
	for _ = 1, 60 do
		spring:advance(1 / 60)
	end
	check(typeof(spring.y) == "vector" and math.abs(spring.y.x - math.cos(2)) <= 1e-2, "vector integrator")
end)

test("failures are reported", function()
	check(not pcall(function()
		ode.solve(f64.space, growth, { 0, 1 }, f64.from({ 1 }), { method = "rk7" })
	end), "unknown method")
	check(not pcall(function()
		ode.solve(f64.space, growth, { 0, 1 }, f64.from({ 1 }), { method = "rk4" })
	end), "fixed-step methods need h")
	-- y' = y^2 blows up at t = 1.
	local blow = function(_: number, y: buffer, out: buffer): buffer
		return f64.scale(out, f64.get(y, 1), y)
	end
	local sol = ode.solve(f64.space, blow, { 0, 2 }, f64.from({ 1 }))
	check(sol.status == "failed" and sol.t[#sol.t] < 1 and sol.t[#sol.t] > 0.99, `a singularity stops it: {sol.message}`)
	local capped = ode.solve(f64.space, growth, { 0, 100 }, f64.from({ 1 }), { max_steps = 5 })
	check(capped.status == "failed" and #capped.t == 6, "max_steps")
	local nan = ode.solve(f64.space, function(t: number, _: buffer, out: buffer): buffer
		f64.set(out, 1, if t > 0.5 then 0 / 0 else 1)
		return out
	end, { 0, 1 }, f64.from({ 0 }))
	check(nan.status == "failed", `NaN is reported: {nan.message}`)
end)

return nil
