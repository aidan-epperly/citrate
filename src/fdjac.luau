--!strict
--!native
--!optimize 2

-- Finite-difference Jacobians, shared by the stiff ODE solvers (odestiff) and optimize:
-- element access for every array module, the step for each component, and the dense
-- (possibly rectangular) Jacobian.

local c32 = require("./c32")
local c64 = require("./c64")

local EPS = 2 ^ -52

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

-- Perturbs the real part of component j of yp (a copy of y) by about sqrt(eps) max(|y_j|,
-- 1), and returns the step actually taken (after rounding).
local function perturb(R: any, yp: buffer, y: buffer, j: number): number
	local eps = math.sqrt(R.space.eps or EPS)
	local yr, yi = eget(R, y, j)
	local d = eps * math.max(math.sqrt(yr * yr + yi * yi), 1)
	eset(R, yp, j, yr + (if yr < 0 then -d else d), yi)
	return (eget(R, yp, j)) - yr
end

-- The m by n Jacobian of F at x (n = length(x)) by forward differences, into J (an m by n
-- matrix of module R): column j is (F(x + d e_j) - F(x)) / d. fx = F(x); F(x, out) writes
-- F into out (length m). yp, fp and col are workspace arrays of lengths n, m and m.
-- Returns J and the evaluations used (n).
local function dense(R: any, F: (buffer, buffer) -> buffer, x: buffer, fx: buffer, J: any, yp: buffer, fp: buffer, col: buffer): (any, number)
	local n = R.length(x)
	local bytes = buffer.len(fx)
	yp = R.copy(yp, x)
	for j = 1, n do
		local d = perturb(R, yp, x, j)
		fp = F(yp, fp)
		col = R.sub(col, fp, fx)
		col = R.scale(col, 1 / d, col)
		buffer.copy(J.data, bytes * (j - 1), col)
		local yr, yi = eget(R, x, j)
		eset(R, yp, j, yr, yi)
	end
	return J, n
end

return {
	eget = eget,
	eset = eset,
	perturb = perturb,
	dense = dense,
}
