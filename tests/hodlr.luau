--!strict

-- HODLR kernel matrices against dense ones: products, solves and log-determinants, a
-- refactorization with another lambda, an ill-conditioned (Gaussian) system, and errors.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local f64, hodlr, random = lib.f64, lib.hodlr, lib.random

local function setup(n: number, phi: (number) -> number): ({ { number } }, any)
	local rng = random.new(n)
	local pts = {}
	for k = 1, n do
		pts[k] = { rng:uniform(), rng:uniform() }
	end
	local A = f64.matrix(n, n)
	for j = 1, n do
		for i = 1, n do
			local dx, dy = pts[i][1] - pts[j][1], pts[i][2] - pts[j][2]
			f64.matset(A, i, j, phi(math.sqrt(dx * dx + dy * dy)))
		end
	end
	return pts, A
end

local function shifted(A: any, lambda: number): any
	local B = f64.matcopy(nil, A)
	for i = 1, A.rows do
		f64.matset(B, i, i, f64.matget(B, i, i) + lambda)
	end
	return B
end

test("hodlr: products, solves and log-determinants against dense", function()
	local n = 600
	local phi = function(r: number): number
		return math.exp(-r / 0.05) -- Matern 1/2: well conditioned, ranks grow with n
	end
	local pts, A = setup(n, phi)
	local H = hodlr.new(pts, phi, { tol = 1e-12, leaf = 48 })
	check(#H.ranks >= 3, "several levels")
	local b = f64.create(n)
	for i = 1, n do
		f64.set(b, i, math.sin(7 * i) + 0.3)
	end
	local y = f64.matvec(nil, A, b)
	check(f64.norm(f64.sub(nil, H:matvec(b), y)) / f64.norm(y) < 1e-10, "matvec")
	local top = f64.get(f64.eigh_copy(A).values, n)
	close(H:norm(), top, 1e-2 * top, "norm estimate (the largest eigenvalue)")
	for _, lambda in { 1e-3, 0.5 } do
		H:factor(lambda)
		local B = shifted(A, lambda)
		local x = H:solve(b)
		local res = f64.norm(f64.sub(nil, f64.matvec(nil, B, x), b)) / f64.norm(b)
		check(res < 1e-9, string.format("solve, lambda %g: residual %.1e", lambda, res))
		local F = f64.chol_copy(B)
		local ld = 0
		for i = 1, n do
			ld += 2 * math.log(f64.matget(F.chol, i, i))
		end
		close(H:logdet(), ld, 1e-9 * math.abs(ld), string.format("logdet, lambda %g", lambda))
	end
end)

test("hodlr: an ill-conditioned Gaussian system fits as well as dense", function()
	local n = 500
	local phi = function(r: number): number
		local s = r / 0.3
		return math.exp(-s * s)
	end
	local pts, A = setup(n, phi)
	local tol = 1e-10
	local H = hodlr.new(pts, phi, { tol = tol })
	local lambda = 2 * tol * H:norm()
	H:factor(lambda)
	-- smooth data: the fits agree where the dense solve is accurate (at the points' values)
	local y = f64.create(n)
	for k = 1, n do
		f64.set(y, k, math.sin(5 * pts[k][1]) * math.cos(3 * pts[k][2]))
	end
	local w = H:solve(y)
	local F = f64.chol_copy(shifted(A, lambda))
	check(F.definite, "dense factors")
	local wd = f64.chol_solve(nil, F, y)
	local fit, fitd = f64.matvec(nil, A, w), f64.matvec(nil, A, wd)
	local e, ed = 0, 0
	for k = 1, n do
		e = math.max(e, math.abs(f64.get(fit, k) - f64.get(y, k)))
		ed = math.max(ed, math.abs(f64.get(fitd, k) - f64.get(y, k)))
	end
	check(e < 10 * ed + 1e-8, string.format("misfit %.1e against dense %.1e", e, ed))
	check(not pcall(function()
		H:factor(1e-14)
	end), "too small a lambda is refused (not positive definite)")
end)

test("hodlr: one point, a leaf, and errors", function()
	local H = hodlr.new({ { 0, 0 } }, function(r: number): number
		return math.exp(-r)
	end)
	H:factor(1)
	close(f64.get(H:solve(f64.from({ 4 })), 1), 2, 1e-15, "one point: (1 + 1) x = 4")
	close(H:logdet(), math.log(2), 1e-15, "one point: logdet")
	check(not pcall(function()
		hodlr.new({}, math.exp)
	end), "no points")
	local H2 = hodlr.new({ { 0, 0 }, { 1, 0 } }, function(r: number): number
		return math.exp(-r)
	end)
	check(not pcall(function()
		H2:solve(f64.from({ 1, 2 }))
	end), "solve before factor")
	check(not pcall(function()
		H2:factor(-1)
	end), "negative lambda")
end)

return nil
