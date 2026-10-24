--!strict
--!optimize 2

-- `vector` as a state space: real vectors of dimension at most 3, with native
-- single-precision arithmetic. Vectors are immutable, so every `out` slot is ignored.

local types = require("./types")

local function similar(x: vector): vector
	return x
end

local function copy(_: vector?, x: vector): vector
	return x
end

local function add(_: vector?, x: vector, y: vector): vector
	return x + y
end

local function sub(_: vector?, x: vector, y: vector): vector
	return x - y
end

local function scale(_: vector?, a: number, x: vector): vector
	return x * a
end

local function axpy(_: vector?, a: number, x: vector, y: vector): vector
	return x * a + y
end

local function dot(x: vector, y: vector): number
	return vector.dot(x, y)
end

local function norm(x: vector): number
	return vector.magnitude(x)
end

-- The weighted RMS norm of an ODE error estimate over the three components.
local function errnorm(e: vector, y: vector, ynew: vector, atol: number, rtol: number): number
	local ex = e.x / (atol + rtol * math.max(math.abs(y.x), math.abs(ynew.x)))
	local ey = e.y / (atol + rtol * math.max(math.abs(y.y), math.abs(ynew.y)))
	local ez = e.z / (atol + rtol * math.max(math.abs(y.z), math.abs(ynew.z)))
	return math.sqrt((ex * ex + ey * ey + ez * ez) / 3)
end

local space: types.Space<vector> = {
	eps = 2 ^ -24,
	real = true,
	errnorm = errnorm,
	similar = similar,
	copy = copy,
	add = add,
	sub = sub,
	scale = scale,
	axpy = axpy,
	dot = dot,
	norm = norm,
}

return {
	space = space,

	similar = similar,
	copy = copy,
	add = add,
	sub = sub,
	scale = scale,
	axpy = axpy,
	dot = dot,
	norm = norm,
}
