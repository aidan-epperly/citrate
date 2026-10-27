--!strict
-- The optimize module's classical methods against their Chebyshev and AAA alternatives
-- (luau -O2 [--codegen] bench/optimize.luau), on the scalar functions below and the
-- Moré-Garbow-Hillstrom problems (tests/mgh.luau); bench/optimize.py runs SciPy on the
-- same problems. Counts are function evaluations (and gradient or Jacobian evaluations);
-- "solved" means reaching the known minimum (or below: some problems have several).
--
-- The cost model, stated before measuring:
-- 1. Scalar roots: Brent needs 5-15 evaluations for one root; an interpolant needs N
--    (N ~ 20-200 for smooth f, growing as 1/width near steep fronts and poles for
--    Chebyshev) but finds every root. AAA should stay small near poles and fronts.
-- 2. Scalar minima: Brent is cheapest but local; the interpolants are global.
-- 3. Line searches with exact gradients (a gradient costing about one f): Wolfe takes
--    about 1.1-1.5 (f, g) pairs an iteration, an interpolating search k = 9 values and one
--    gradient, so it costs about 4 times more unless it saves iterations; it may find
--    better minima on nonconvex problems, seeing further along the line.
-- 4. With finite-difference gradients (n + 1 evaluations each) Wolfe costs about
--    1.3 (n + 1) an iteration and the interpolating search k + n + 1: about even at
--    n ~ 27, the interpolating search ahead beyond (if iterations are alike).
-- 5. Levenberg-Marquardt's damping: Moré's Newton iteration needs 2-3 trials of lambda,
--    AAA or Chebyshev about 20-40 (their samples); each trial is O(n^2) after the QR
--    factorization (an O(m n^2) one per Jacobian), and the steps (hence iterations) should
--    match. (Measured: the steps match, but AAA and Chebyshev need 30 and 130 times the
--    trials, which does cost time.)
-- 7. Jacobian reuse: consecutive Jacobians differ by a nearly low-rank change (rank 1 within
--    10% for 92% of the steps), so a Broyden update or plain reuse should cut the Jacobians
--    by about half, at a few more evaluations (worth it when a Jacobian costs n of them).
-- 6. Newton's line search: the unit step is usually accepted, so the alternatives only
--    matter where backtracking is needed, where they cost k evaluations against 1-3.
local lib = require("../src")
local f64, opt = lib.f64, lib.optimize
local mgh = require("../tests/mgh")

local function tab(b: buffer): { number }
	local t = {}
	for k = 1, buffer.len(b) // 8 do
		t[k] = buffer.readf64(b, 8 * (k - 1))
	end
	return t
end

-- seconds per call (repeated to 0.05 s)
local function time(f: () -> ()): number
	local reps, t0 = 0, os.clock()
	repeat
		f()
		reps += 1
	until os.clock() - t0 > 0.05
	return (os.clock() - t0) / reps
end
local function us(s: number): string
	return if s < 1e-3 then string.format("%.0f us", s * 1e6) else string.format("%.2f ms", s * 1e3)
end

print("## Scalar roots (evaluations, time)\n")
local ROOTS = {
	{ "cos x - x on [0, 1]", function(x: number): number return math.cos(x) - x end, 0, 1 },
	{ "x^3 - 2x - 5 on [2, 3]", function(x: number): number return x ^ 3 - 2 * x - 5 end, 2, 3 },
	{ "pole nearby: 1/(x - 1.01) + 1.5 on [-1, 0.9]", function(x: number): number return 1 / (x - 1.01) + 1.5 end, -1, 0.9 },
	{ "kink: |x - 0.3| - 0.1 on [0, 0.25]", function(x: number): number return math.abs(x - 0.3) - 0.1 end, 0, 0.25 },
	{ "front: tanh(50 (x - 0.2)) on [-1, 1]", function(x: number): number return math.tanh(50 * (x - 0.2)) end, -1, 1 },
	{ "exp(x) - 1000 on [0, 10]", function(x: number): number return math.exp(x) - 1000 end, 0, 10 },
}
print("| function | brent | chebyshev | aaa |")
print("| --- | --- | --- | --- |")
for _, p in ROOTS do
	local f, a, b = p[2] :: (number) -> number, p[3] :: number, p[4] :: number
	local cells = { p[1] :: string }
	for _, m in { "brent", "chebyshev", "aaa" } do
		local x, info = opt.root_scalar(f, a, b, { method = m })
		local t = time(function()
			opt.root_scalar(f, a, b, { method = m })
		end)
		table.insert(cells, string.format("%d, %s%s", info.evaluations, us(t), if x and math.abs(f(x)) < 1e-9 then "" else " (failed)"))
	end
	print("| " .. table.concat(cells, " | ") .. " |")
end

print("\n## Scalar minima (evaluations, x)\n")
local MINS = {
	{ "(x - 0.3)^2 on [-1, 1]", function(x: number): number return (x - 0.3) ^ 2 end, -1, 1 },
	{ "x^4 - x^2 + 0.3 x on [-1.5, 1.5] (two minima)", function(x: number): number return x ^ 4 - x ^ 2 + 0.3 * x end, -1.5, 1.5 },
	{ "x^4 - x^2 - 0.3 x on [-1.5, 1.5] (global on the right)", function(x: number): number return x ^ 4 - x ^ 2 - 0.3 * x end, -1.5, 1.5 },
	{ "sin(5 x) + 0.5 x on [0, 3]", function(x: number): number return math.sin(5 * x) + 0.5 * x end, 0, 3 },
}
print("| function | brent | chebyshev | aaa |")
print("| --- | --- | --- | --- |")
for _, p in MINS do
	local f, a, b = p[2] :: (number) -> number, p[3] :: number, p[4] :: number
	local cells = { p[1] :: string }
	for _, m in { "brent", "chebyshev", "aaa" } do
		local x, fx, info = opt.minimize_scalar(f, a, b, { method = m })
		table.insert(cells, string.format("%d, x = %.6f (f %.5f)", info.evaluations, x, fx))
	end
	print("| " .. table.concat(cells, " | ") .. " |")
end

-- the MGH problems as functions of f64 arrays
type Adapted = { r: any, jac: any, f: any, grad: any }
local function adapt(p: any): Adapted
	local function r(x: buffer, out: buffer?): buffer
		local v = p.r(tab(x))
		local o = out or buffer.create(8 * p.m)
		for i = 1, p.m do
			buffer.writef64(o, 8 * (i - 1), v[i])
		end
		return o
	end
	local function jac(x: buffer, J: any): any
		local v = p.jac(tab(x))
		for i = 1, p.m do
			for j = 1, p.n do
				buffer.writef64(J.data, 8 * ((j - 1) * p.m + i - 1), v[i][j])
			end
		end
		return J
	end
	local function f(x: buffer): number
		local v = p.r(tab(x))
		local s = 0
		for i = 1, p.m do
			s += v[i] ^ 2
		end
		return 0.5 * s
	end
	local function grad(x: buffer, out: buffer): buffer
		local xt = tab(x)
		local v, J = p.r(xt), p.jac(xt)
		for j = 1, p.n do
			local s = 0
			for i = 1, p.m do
				s += J[i][j] * v[i]
			end
			buffer.writef64(out, 8 * (j - 1), s)
		end
		return out
	end
	return { r = r, jac = jac, f = f, grad = grad }
end
local function solved(v: number, fstar: number): boolean
	return v <= fstar + 1e-5 * math.max(1e-3, fstar) + 1e-9
end

print("\n## Least squares: Levenberg-Marquardt's damping (evaluations / Jacobians / lambda trials)\n")
local LAMS = { "newton", "aaa", "chebyshev" }
print("| problem | " .. table.concat(LAMS, " | ") .. " |")
print("| --- | --- | --- | --- |")
local lam_total: { [string]: { number } } = {}
for _, lm in LAMS do
	lam_total[lm] = { 0, 0, 0, 0, 0 }
end
for _, p in mgh.problems do
	local A = adapt(p)
	local cells = { p.name }
	for _, lm in LAMS do
		local _, info = opt.least_squares(f64.space, A.r, f64.from(p.x0), { jac = A.jac, lambda = lm })
		local ok = solved(info.cost, p.fstar)
		local t = time(function()
			opt.least_squares(f64.space, A.r, f64.from(p.x0), { jac = A.jac, lambda = lm })
		end)
		local tot = lam_total[lm]
		tot[1] += info.evaluations
		tot[2] += info.jacobians
		tot[3] += info.lambda_trials
		tot[4] += if ok then 1 else 0
		tot[5] += t
		table.insert(cells, string.format("%d / %d / %d%s", info.evaluations, info.jacobians, info.lambda_trials, if ok then "" else " (not solved)"))
	end
	print("| " .. table.concat(cells, " | ") .. " |")
end
local cells = { "**total** (solved, time)" }
for _, lm in LAMS do
	local t = lam_total[lm]
	table.insert(cells, string.format("%d / %d / %d (%d of %d, %s)", t[1], t[2], t[3], t[4], #mgh.problems, us(t[5])))
end
print("| " .. table.concat(cells, " | ") .. " |")

print("\n## Least squares: Jacobians every iteration, reused while they predict well, or Broyden-updated (evaluations / Jacobians)\n")
print("| problem | every, exact | lazy, exact | broyden, exact | every, differences | lazy, differences | broyden, differences |")
print("| --- | --- | --- | --- | --- | --- | --- |")
local mode_total: { [string]: { number } } = {}
for _, p in mgh.problems do
	local A = adapt(p)
	local row = { p.name }
	for _, exact in { true, false } do
		for _, mode in { "every", "lazy", "broyden" } do
			local _, info = opt.least_squares(f64.space, A.r, f64.from(p.x0), { jac = if exact then A.jac else nil, jacobian = mode })
			local ok = solved(info.cost, p.fstar)
			local key = mode .. (if exact then " exact" else " differences")
			local tot = mode_total[key] or { 0, 0, 0 }
			mode_total[key] = tot
			tot[1] += info.evaluations
			tot[2] += info.jacobians
			tot[3] += if ok then 1 else 0
			table.insert(row, string.format("%d / %d%s", info.evaluations, info.jacobians, if ok then "" else " (not solved)"))
		end
	end
	print("| " .. table.concat(row, " | ") .. " |")
end
local totals = { "**total** (solved)" }
for _, exact in { true, false } do
	for _, mode in { "every", "lazy", "broyden" } do
		local t = mode_total[mode .. (if exact then " exact" else " differences")]
		table.insert(totals, string.format("%d / %d (%d of %d)", t[1], t[2], t[3], #mgh.problems))
	end
end
print("| " .. table.concat(totals, " | ") .. " |")

print("\n## Minimization with exact gradients: line searches (f evaluations / gradients)\n")
local LS = { "wolfe", "chebyshev", "aaa", "hybrid" }
print("| problem | " .. table.concat(LS, " | ") .. " | nelder-mead |")
print("| --- | --- | --- | --- | --- | --- |")
local ls_total: { [string]: { number } } = {}
for _, ls in LS do
	ls_total[ls] = { 0, 0, 0, 0 }
end
ls_total["nelder-mead"] = { 0, 0, 0, 0 }
for _, p in mgh.problems do
	local A = adapt(p)
	local row = { p.name }
	for _, ls in { "wolfe", "chebyshev", "aaa", "hybrid", "nelder-mead" } do
		local o: any = if ls == "nelder-mead" then { method = "nelder-mead", maxiter = 20000 } else { grad = A.grad, line_search = ls }
		local _, fx, info = opt.minimize(f64.space, A.f, f64.from(p.x0), o)
		local ok = solved(fx, p.fstar)
		local t = time(function()
			opt.minimize(f64.space, A.f, f64.from(p.x0), o)
		end)
		local tot = ls_total[ls]
		tot[1] += info.evaluations
		tot[2] += info.gradients
		tot[3] += if ok then 1 else 0
		tot[4] += t
		table.insert(row, string.format("%d / %d%s", info.evaluations, info.gradients, if ok then "" else string.format(" (f = %.2g)", fx)))
	end
	print("| " .. table.concat(row, " | ") .. " |")
end
local cells2 = { "**total** (solved, time)" }
for _, ls in { "wolfe", "chebyshev", "aaa", "hybrid", "nelder-mead" } do
	local t = ls_total[ls]
	table.insert(cells2, string.format("%d / %d (%d of %d, %s)", t[1], t[2], t[3], #mgh.problems, us(t[4])))
end
print("| " .. table.concat(cells2, " | ") .. " |")

print("\n## Minimization with finite-difference gradients, by size (f evaluations, iterations)\n")
print("| problem | wolfe | chebyshev | hybrid |")
print("| --- | --- | --- | --- |")
for _, gen in { { mgh.extended_rosenbrock, { 10, 30, 100 } }, { mgh.discrete_bv, { 10, 30, 100 } } } do
	for _, n in gen[2] :: { number } do
		local p = (gen[1] :: any)(n)
		local A = adapt(p)
		local cells3 = { p.name }
		for _, ls in { "wolfe", "chebyshev", "hybrid" } do
			local _, fx, info = opt.minimize(f64.space, A.f, f64.from(p.x0), { line_search = ls })
			table.insert(cells3, string.format("%d, %d%s", info.evaluations, info.iterations, if solved(fx, 1e-9) then "" else string.format(" (f = %.2g)", fx)))
		end
		print("| " .. table.concat(cells3, " | ") .. " |")
	end
end

print("\n## Systems: Newton's line search (evaluations / Jacobians)\n")
print("| problem | backtracking | chebyshev | aaa |")
print("| --- | --- | --- | --- |")
for _, p in mgh.problems do
	if p.root then
		local A = adapt(p)
		local cells4 = { p.name }
		for _, ls in { "backtracking", "chebyshev", "aaa" } do
			local _, info = opt.root(f64.space, A.r, f64.from(p.x0), { jac = A.jac, line_search = ls })
			table.insert(cells4, string.format("%d / %d%s", info.evaluations, info.jacobians, if info.converged then "" else " (not solved)"))
		end
		print("| " .. table.concat(cells4, " | ") .. " |")
	end
end
