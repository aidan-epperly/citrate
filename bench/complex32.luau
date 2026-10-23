--!strict

-- Single-precision complex scalar operations, three ways:
--   current  complex32 as shipped (double-precision formulas, rounded once)
--   inline   local copies of those formulas, as a reference for the call overhead
--   vector   true single precision: native vector arithmetic only
-- The original version of complex32 called complex64 instead; that cost about 10 ns per
-- call, was replaced by the inline formulas, and the vector variant lost on speed,
-- accuracy and overflow.
-- Run from the repository root: luau -O2 [--codegen] bench/complex32.luau

local complex32 = require("../src/complex32")

local function mul_inline(a: vector, b: vector): vector
	local ar, ai, br, bi = a.x, a.y, b.x, b.y
	return vector.create(ar * br - ai * bi, ar * bi + ai * br, 0)
end

local function mul_vector(a: vector, b: vector): vector
	return vector.create(a.x, a.x, 0) * b + vector.create(-a.y, a.y, 0) * vector.create(b.y, b.x, 0)
end

-- Single-precision inputs cannot overflow br^2 + bi^2 in double precision.
local function div_inline(a: vector, b: vector): vector
	local ar, ai, br, bi = a.x, a.y, b.x, b.y
	local d = br * br + bi * bi
	return vector.create((ar * br + ai * bi) / d, (ai * br - ar * bi) / d, 0)
end

local function div_vector(a: vector, b: vector): vector
	return mul_vector(a, vector.create(b.x, -b.y, 0)) / vector.dot(b, b)
end

local function abs_inline(z: vector): number
	local r, i = z.x, z.y
	return math.sqrt(r * r + i * i)
end

local function abs_vector(z: vector): number
	return vector.magnitude(z)
end

local N = 2_000_000
local M = 1024
math.randomseed(7)
local xs = table.create(M) :: { vector }
for k = 1, M do
	xs[k] = vector.create(4 * math.random() - 2, 4 * math.random() - 2, 0)
end

local function time_binary(f: (vector, vector) -> vector): number
	local acc = vector.zero
	local t = os.clock()
	for k = 0, N - 1 do
		acc += f(xs[k % M + 1], xs[(k * 7) % M + 1])
	end
	local dt = os.clock() - t
	return dt / N * 1e9
end

local function time_unary(f: (vector) -> number): number
	local acc = 0
	local t = os.clock()
	for k = 0, N - 1 do
		acc += f(xs[k % M + 1])
	end
	return (os.clock() - t) / N * 1e9
end

local baseline = time_binary(function(a, _)
	return a
end)

local unary_baseline = time_unary(function(a)
	return a.x
end)

print(string.format("ns per call, loop overhead subtracted (%.1f ns binary, %.1f ns unary)", baseline, unary_baseline))
print(string.format("%-6s %9s %9s %9s", "", "current", "inline", "vector"))
print(string.format("%-6s %9.1f %9.1f %9.1f", "mul",
	time_binary(complex32.mul) - baseline, time_binary(mul_inline) - baseline, time_binary(mul_vector) - baseline))
print(string.format("%-6s %9.1f %9.1f %9.1f", "div",
	time_binary(complex32.div) - baseline, time_binary(div_inline) - baseline, time_binary(div_vector) - baseline))
print(string.format("%-6s %9.1f %9.1f %9.1f", "abs",
	time_unary(complex32.abs) - unary_baseline, time_unary(abs_inline) - unary_baseline,
	time_unary(abs_vector) - unary_baseline))

-- Accuracy against a double-precision reference, in units of single-precision epsilon
-- relative to the modulus of the exact result.
local EPS32 = 2 ^ -23
local function err(z: vector, rr: number, ri: number): number
	return math.max(math.abs(z.x - rr), math.abs(z.y - ri)) / math.sqrt(rr * rr + ri * ri) / EPS32
end

local worst = { cm = 0, vm = 0, cd = 0, vd = 0 }
for k = 1, 200_000 do
	local a, b = xs[k % M + 1], vector.create(4 * math.random() - 2, 4 * math.random() - 2, 0)
	local rr, ri = a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x
	worst.cm = math.max(worst.cm, err(complex32.mul(a, b), rr, ri))
	worst.vm = math.max(worst.vm, err(mul_vector(a, b), rr, ri))
	local d = b.x * b.x + b.y * b.y
	local qr, qi = (a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d
	worst.cd = math.max(worst.cd, err(complex32.div(a, b), qr, qi))
	worst.vd = math.max(worst.vd, err(div_vector(a, b), qr, qi))
end
print(string.format("worst error / eps32   mul: current %.2f vector %.2f   div: current %.2f vector %.2f",
	worst.cm, worst.vm, worst.cd, worst.vd))

local big = vector.create(3e19, 4e19, 0)
print("abs(3e19 + 4e19 i):  current", complex32.abs(big), " vector", abs_vector(big))
print("big / big:           current", complex32.div(big, big), " vector", div_vector(big, big))
