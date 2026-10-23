--!strict

-- 1-based vs 0-based element access on f64 buffers and column-major matrices.
-- "called" goes through a module table like user code would (no inlining);
-- "inlined" uses local functions the compiler can inline at -O2.
-- Run from the repository root: luau -O2 [--codegen] bench/indexing.luau

type Matrix = { rows: number, cols: number, data: buffer }

local function get1(x: buffer, i: number): number
	return buffer.readf64(x, 8 * (i - 1))
end

local function get0(x: buffer, i: number): number
	return buffer.readf64(x, 8 * i)
end

local function set1(x: buffer, i: number, v: number)
	buffer.writef64(x, 8 * (i - 1), v)
end

local function set0(x: buffer, i: number, v: number)
	buffer.writef64(x, 8 * i, v)
end

local function matget1(A: Matrix, i: number, j: number): number
	return buffer.readf64(A.data, 8 * ((j - 1) * A.rows + i - 1))
end

local function matget0(A: Matrix, i: number, j: number): number
	return buffer.readf64(A.data, 8 * (j * A.rows + i))
end

local mod = { get1 = get1, get0 = get0, set1 = set1, set0 = set0, matget1 = matget1, matget0 = matget0 }

local N = 1_000_000
local x = buffer.create(8 * N)
local side = 1000
local A: Matrix = { rows = side, cols = side, data = buffer.create(8 * side * side) }

local cases: { { name: string, run: () -> () } } = {
	{ name = "get, called, 1-based", run = function()
		local s = 0
		for i = 1, N do
			s += mod.get1(x, i)
		end
	end },
	{ name = "get, called, 0-based", run = function()
		local s = 0
		for i = 0, N - 1 do
			s += mod.get0(x, i)
		end
	end },
	{ name = "get, inlined, 1-based", run = function()
		local s = 0
		for i = 1, N do
			s += get1(x, i)
		end
	end },
	{ name = "get, inlined, 0-based", run = function()
		local s = 0
		for i = 0, N - 1 do
			s += get0(x, i)
		end
	end },
	{ name = "set, called, 1-based", run = function()
		for i = 1, N do
			mod.set1(x, i, i)
		end
	end },
	{ name = "set, called, 0-based", run = function()
		for i = 0, N - 1 do
			mod.set0(x, i, i)
		end
	end },
	{ name = "matget, called, 1-based", run = function()
		local s = 0
		for j = 1, side do
			for i = 1, side do
				s += mod.matget1(A, i, j)
			end
		end
	end },
	{ name = "matget, called, 0-based", run = function()
		local s = 0
		for j = 0, side - 1 do
			for i = 0, side - 1 do
				s += mod.matget0(A, i, j)
			end
		end
	end },
	{ name = "matget, inlined, 1-based", run = function()
		local s = 0
		for j = 1, side do
			for i = 1, side do
				s += matget1(A, i, j)
			end
		end
	end },
	{ name = "matget, inlined, 0-based", run = function()
		local s = 0
		for j = 0, side - 1 do
			for i = 0, side - 1 do
				s += matget0(A, i, j)
			end
		end
	end },
}

-- Best of several trials, interleaved so drift affects both bases alike.
local TRIALS = 15
local best: { number } = table.create(#cases, math.huge)
for _ = 1, TRIALS do
	for k, case in cases do
		local t = os.clock()
		case.run()
		best[k] = math.min(best[k], (os.clock() - t) / N * 1e9)
	end
end
for k, case in cases do
	print(string.format("%-26s %6.2f ns/element", case.name, best[k]))
end
