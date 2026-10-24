--!strict

-- Boundary value problems by Chebyshev collocation: linear, nonlinear (both Bratu
-- branches, Blasius), boundary layers, periodic conditions, user Jacobians and failures.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local ode, f64, cheb = lib.ode, lib.f64, lib.cheb

-- The largest |F(x) - g(x)| at count + 1 equally spaced points of [a, b].
local function max_error(F: lib.Chebfun, g: (number) -> number, count: number?): number
	local c = count or 100
	local worst = 0
	for i = 0, c do
		local x = F.a + (F.b - F.a) * i / c
		worst = math.max(worst, math.abs(cheb.eval(F, x) - g(x)))
	end
	return worst
end

-- u'' = g(x, u, u') as the system y = (u, u').
local function second_order(g: (number, number, number) -> number): (number, buffer, buffer) -> buffer
	return function(x: number, y: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(y, 2))
		f64.set(out, 2, g(x, f64.get(y, 1), f64.get(y, 2)))
		return out
	end
end

-- u(a) = ua, u(b) = ub.
local function dirichlet(ua: number, ub: number): (buffer, buffer, buffer) -> buffer
	return function(ya: buffer, yb: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(ya, 1) - ua)
		f64.set(out, 2, f64.get(yb, 1) - ub)
		return out
	end
end

test("linear boundary value problems", function()
	-- u'' = -u, u(0) = 0, u(pi/2) = 1: u = sin x, u' = cos x.
	local sol = ode.bvp(second_order(function(_: number, u: number, _: number): number
		return -u
	end), dirichlet(0, 1), { 0, math.pi / 2 }, 2)
	check(sol.status == "success", `sin: {sol.message}`)
	check(max_error(sol.y[1], math.sin) <= 1e-10, `sin: error {max_error(sol.y[1], math.sin)}`)
	check(max_error(sol.y[2], math.cos) <= 1e-10, `sin: derivative error {max_error(sol.y[2], math.cos)}`)
	check(sol.residual <= 1e-10, `sin: residual {sol.residual}`)
	-- The result is a Chebfun: its derivative matches the second component.
	local du = cheb.diff(sol.y[1])
	check(max_error(du, math.cos) <= 1e-8, "sin: diff of the solution")

	-- Periodic conditions on [0, 2 pi] for u'' - u = cos x: u = -cos(x) / 2.
	local periodic = ode.bvp(second_order(function(x: number, u: number, _: number): number
		return u + math.cos(x)
	end), function(ya: buffer, yb: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(ya, 1) - f64.get(yb, 1))
		f64.set(out, 2, f64.get(ya, 2) - f64.get(yb, 2))
		return out
	end, { 0, 2 * math.pi }, 2)
	check(periodic.status == "success", `periodic: {periodic.message}`)
	check(max_error(periodic.y[1], function(x: number): number
		return -math.cos(x) / 2
	end) <= 1e-10, "periodic: error")

	-- A Robin condition at one end and a mixed one: u'' = 0 with u'(0) = 1 and
	-- u(1) + u'(1) = 4: u = x + 2.
	local robin = ode.bvp(second_order(function(): number
		return 0
	end), function(ya: buffer, yb: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(ya, 2) - 1)
		f64.set(out, 2, f64.get(yb, 1) + f64.get(yb, 2) - 4)
		return out
	end, { 0, 1 }, 2)
	check(max_error(robin.y[1], function(x: number): number
		return x + 2
	end) <= 1e-12, "robin: error")

	-- Boundary layers: eps u'' + u' = 0, u(0) = 0, u(1) = 1. Thinner layers need more
	-- points, found adaptively.
	local points: { number } = {}
	for k, eps in { 0.1, 0.01 } do
		local layer = ode.bvp(second_order(function(_: number, _: number, du: number): number
			return -du / eps
		end), dirichlet(0, 1), { 0, 1 }, 2)
		local exact = function(x: number): number
			return (1 - math.exp(-x / eps)) / (1 - math.exp(-1 / eps))
		end
		check(layer.status == "success", `layer {eps}: {layer.message}`)
		check(max_error(layer.y[1], exact, 400) <= 1e-9, `layer {eps}: error {max_error(layer.y[1], exact, 400)}`)
		points[k] = layer.points
	end
	check(points[2] > points[1], `layers: {points[1]} and {points[2]} points`)
end)

test("nonlinear boundary value problems", function()
	-- Bratu: u'' + e^u = 0, u(0) = u(1) = 0 has two solutions
	-- u = -2 log(cosh((x - 1/2) theta / 2) / cosh(theta / 4)) with theta = sqrt(2)
	-- cosh(theta / 4); the guess picks the branch.
	local function theta_root(t0: number): number
		local t = t0
		for _ = 1, 50 do
			local g = t - math.sqrt(2) * math.cosh(t / 4)
			local dg = 1 - math.sqrt(2) / 4 * math.sinh(t / 4)
			t -= g / dg
		end
		return t
	end
	local bratu = second_order(function(_: number, u: number, _: number): number
		return -math.exp(u)
	end)
	-- (The guess of height 30 overshoots: damped Newton stalls on the coarsest grid and
	-- full steps finish the job.)
	for _, branch in { { theta_root(1), 0 }, { theta_root(11), 16 }, { theta_root(11), 30 } } do
		local theta, height = branch[1], branch[2]
		local exact = function(x: number): number
			return -2 * math.log(math.cosh((x - 0.5) * theta / 2) / math.cosh(theta / 4))
		end
		local sol = ode.bvp(bratu, dirichlet(0, 0), { 0, 1 }, 2, {
			guess = function(x: number): buffer
				return f64.from({ height * x * (1 - x), height * (1 - 2 * x) })
			end,
		})
		check(sol.status == "success", `bratu theta {theta}: {sol.message}`)
		check(max_error(sol.y[1], exact) <= 1e-11, `bratu theta {theta}: error {max_error(sol.y[1], exact)}`)
	end

	-- Blasius: f''' + f f'' / 2 = 0, f(0) = f'(0) = 0, f'(10) = 1 (a truncated infinite
	-- domain): f''(0) = 0.33205733621519630 (the classical value).
	local blasius = function(_: number, y: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(y, 2))
		f64.set(out, 2, f64.get(y, 3))
		f64.set(out, 3, -f64.get(y, 1) * f64.get(y, 3) / 2)
		return out
	end
	local bc = function(ya: buffer, yb: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(ya, 1))
		f64.set(out, 2, f64.get(ya, 2))
		f64.set(out, 3, f64.get(yb, 2) - 1)
		return out
	end
	local sol = ode.bvp(blasius, bc, { 0, 10 }, 3, {
		guess = function(x: number): buffer
			return f64.from({ x * x / 20, x / 10, 1 / 10 })
		end,
	})
	check(sol.status == "success", `blasius: {sol.message}`)
	close(cheb.eval(sol.y[3], 0), 0.33205733621519630, 1e-8, "blasius f''(0)")

	-- Troesch's problem u'' = mu sinh(mu u), u(0) = 0, u(1) = 1, a classic test of
	-- shooting and Newton methods: u'(0) = 0.0457504614 for mu = 5.
	local troesch = ode.bvp(second_order(function(_: number, u: number, _: number): number
		return 5 * math.sinh(5 * u)
	end), dirichlet(0, 1), { 0, 1 }, 2, {
		guess = function(x: number): buffer
			return f64.from({ x, 1 })
		end,
	})
	check(troesch.status == "success", `troesch: {troesch.message}`)
	close(cheb.eval(troesch.y[2], 0), 0.045750461403, 1e-9, "troesch u'(0)")
end)

test("damped Newton", function()
	-- Problems where Newton needs damping (Carrier's: full steps diverge) and where a
	-- damping test on the residual stalls (u'' = u^3 from a zero guess, whose
	-- residual grows on the way to the solution). Each solution is checked by
	-- substituting the Chebfun back into the equation.
	local function check_residual(name: string, sol: any, g: (number, number, number) -> number, a: number, b: number)
		check(sol.status == "success", `{name}: {sol.message}`)
		local u, du = sol.y[1], cheb.diff(sol.y[1])
		local ddu = cheb.diff(du)
		local worst = 0
		for i = 0, 50 do
			local x = a + (b - a) * i / 50
			worst = math.max(worst, math.abs(cheb.eval(ddu, x) - g(x, cheb.eval(u, x), cheb.eval(du, x))))
		end
		local scale = math.max(cheb.norm(ddu), 1)
		check(worst <= 1e-5 * scale, `{name}: residual {worst} (scale {scale})`)
	end
	local carrier = function(x: number, u: number, _: number): number
		return -(2 * (1 - x * x) * u + u * u - 1) / 0.03
	end
	check_residual("carrier", ode.bvp(second_order(carrier), dirichlet(0, 0), { -1, 1 }, 2), carrier, -1, 1)
	local cubic = function(_: number, u: number, _: number): number
		return u ^ 3
	end
	check_residual("cubic", ode.bvp(second_order(cubic), dirichlet(5, 5), { 0, 1 }, 2), cubic, 0, 1)
	local expu = function(_: number, u: number, _: number): number
		return math.exp(u)
	end
	check_residual("exp", ode.bvp(second_order(expu), dirichlet(10, 10), { 0, 1 }, 2), expu, 0, 1)
end)

test("boundary value Jacobians and failures", function()
	-- Bratu's lower branch with exact Jacobians agrees with finite differences.
	local bratu = second_order(function(_: number, u: number, _: number): number
		return -math.exp(u)
	end)
	local fd = ode.bvp(bratu, dirichlet(0, 0), { 0, 1 }, 2)
	local calls = 0
	local exact = ode.bvp(bratu, dirichlet(0, 0), { 0, 1 }, 2, {
		jac = function(_: number, y: buffer): lib.Matrix
			calls += 1
			local J = f64.matrix(2, 2)
			f64.matset(J, 1, 2, 1)
			f64.matset(J, 2, 1, -math.exp(f64.get(y, 1)))
			return J
		end,
		bc_jac = function(): (lib.Matrix, lib.Matrix)
			local Ba, Bb = f64.matrix(2, 2), f64.matrix(2, 2)
			f64.matset(Ba, 1, 1, 1)
			f64.matset(Bb, 2, 1, 1)
			return Ba, Bb
		end,
	})
	check(calls > 0, "bvp: the Jacobian was never called")
	check(max_error(exact.y[1], function(x: number): number
		return cheb.eval(fd.y[1], x)
	end) <= 1e-12, "bvp: exact and finite-difference Jacobians disagree")

	-- Unresolvable within max_points: reported, not hidden.
	local layer = ode.bvp(second_order(function(_: number, _: number, du: number): number
		return -du / 1e-3
	end), dirichlet(0, 1), { 0, 1 }, 2, { max_points = 65 })
	check(layer.status == "failed" and not layer.y[1].happy, `max_points: {layer.status}`)
	-- Dependent boundary conditions (both on u(0)) make the system singular.
	local bad = ode.bvp(bratu, function(ya: buffer, _: buffer, out: buffer): buffer
		f64.set(out, 1, f64.get(ya, 1))
		f64.set(out, 2, 2 * f64.get(ya, 1))
		return out
	end, { 0, 1 }, 2)
	check(bad.status == "failed", `singular conditions: {bad.status}, {bad.message}`)
end)

return nil
