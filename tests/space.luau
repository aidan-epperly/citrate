--!strict

-- One generic algorithm run on every representation: it must compute in the
-- representation it was given and return a value of the same kind.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close

-- x - 2 y + 3 z, written only against the Space interface.
local function combination<T>(S: lib.Space<T>, x: T, y: T, z: T): T
	local acc = S.copy(S.similar(x), x)
	acc = S.axpy(acc, -2, y, acc)
	acc = S.axpy(acc, 3, z, acc)
	return acc
end

-- Expected norm of x - 2y + 3z for x = (1, 2, 3), y = (0, 1, 0), z = (1, 0, 1).
local expected = math.sqrt(4 * 4 + 0 * 0 + 6 * 6)

test("generic algorithm on vector", function()
	local r = combination(lib.vec.space, vector.create(1, 2, 3), vector.create(0, 1, 0), vector.create(1, 0, 1))
	check(typeof(r) == "vector", "stays a vector")
	close(lib.vec.norm(r), expected, h.eps32)
end)

local function real_case(name: string, R: typeof(lib.f64), eps: number)
	test(`generic algorithm on {name} buffers`, function()
		local x, y, z = R.from({ 1, 2, 3 }), R.from({ 0, 1, 0 }), R.from({ 1, 0, 1 })
		local r = combination(R.space, x, y, z)
		check(typeof(r) == "buffer" and R.length(r) == 3, "stays a buffer")
		check(R.get(x, 1) == 1, "inputs untouched")
		close(R.norm(r), expected, 4 * eps)
	end)
end

-- The same real vectors placed in the imaginary parts.
local function complex_case(name: string, space: lib.Space<buffer>, from: ({ number }, { number }?) -> buffer, eps: number)
	test(`generic algorithm on {name} buffers`, function()
		local zero = { 0, 0, 0 }
		local r = combination(space, from(zero, { 1, 2, 3 }), from(zero, { 0, 1, 0 }), from(zero, { 1, 0, 1 }))
		close(space.norm(r), expected, 4 * eps)
	end)
end

real_case("f32", lib.f32, h.eps32)
real_case("f64", lib.f64, h.eps64)
complex_case("c32", lib.c32.space, lib.c32.from, h.eps32)
complex_case("c64", lib.c64.space, lib.c64.from, h.eps64)

return nil
