--!strict
--!optimize 2

-- `vector` as a state space: real vectors of dimension at most 3, with native
-- single-precision arithmetic. Vectors are immutable, so every `out` slot is ignored.

local types = require("./types")

-- A zero vector (vectors are values, so there is nothing to allocate).
local function similar(x: vector): vector
	return x
end

-- x (vectors are values; the out argument is ignored).
local function copy(_: vector?, x: vector): vector
	return x
end

-- x + y.
local function add(_: vector?, x: vector, y: vector): vector
	return x + y
end

-- x - y.
local function sub(_: vector?, x: vector, y: vector): vector
	return x - y
end

-- a x.
local function scale(_: vector?, a: number, x: vector): vector
	return x * a
end

-- a x + y.
local function axpy(_: vector?, a: number, x: vector, y: vector): vector
	return x * a + y
end

-- The dot product.
local function dot(x: vector, y: vector): number
	return vector.dot(x, y)
end

-- The Euclidean norm.
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

-- vector as a Space (see types.Space), for the solvers that take one.
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
