--!strict
-- The table in docs/performance.md: run it native (luau -O2 --codegen bench/table.luau, as
-- Roblox servers and Studio run citrate) and interpreted (luau -O2 bench/table.luau, as
-- players' devices run it).
local lib = require("../src")
local f64, c64, krylov = lib.f64, lib.c64, lib.krylov

-- best of several runs of f, each at least 0.2 s
local function time(f: () -> ()): number
	local best = math.huge
	for _ = 1, 3 do
		local reps, t0 = 0, os.clock()
		repeat
			f()
			reps += 1
		until os.clock() - t0 > 0.2
		best = math.min(best, (os.clock() - t0) / reps)
	end
	return best
end
local seed = 7
local function rnd(): number
	seed = (seed * 1103515245 + 12345) % 2147483648
	return seed / 2147483648 - 0.5
end
local function random(M: any, rows: number, cols: number, complex: boolean?): any
	local A = M.matrix(rows, cols)
	for k = 0, rows * cols * (if complex then 2 else 1) - 1 do
		buffer.writef64(A.data, 8 * k, rnd())
	end
	return A
end
local function row(name: string, seconds: number)
	print(string.format("| %s | %.3g ms |", name, seconds * 1e3))
end

local A, B = random(f64, 256, 256), random(f64, 256, 256)
row("`matmul`, f64, n = 256", time(function()
	f64.matmul(nil, A, B)
end))
local Ac, Bc = random(c64, 256, 256, true), random(c64, 256, 256, true)
row("`matmul`, c64, n = 256", time(function()
	c64.matmul(nil, Ac, Bc)
end))
local L = random(f64, 120, 120)
local b = f64.create(120)
for i = 1, 120 do
	f64.set(b, i, rnd())
end
row("LU solve, n = 120", time(function()
	(f64.solve :: any)(f64.create(120), L, b)
end))
local Hs = random(f64, 120, 120)
local Hsym = f64.matrix(120, 120)
Hsym.data = f64.add(nil, Hs.data, f64.transpose(nil, Hs).data)
row("`eigh`, n = 120 (with vectors)", time(function()
	f64.eigh_copy(Hsym)
end))
row("`eig`, n = 120 (with vectors)", time(function()
	f64.eig_copy(L)
end))
row("`svd`, n = 120 (with vectors)", time(function()
	f64.svd_copy(L)
end))
local x = c64.create(4096)
for k = 0, 2 * 4096 - 1 do
	buffer.writef64(x, 8 * k, rnd())
end
row("`fft`, 4096 points", time(function()
	c64.fft(x)
end))
local X = random(c64, 128, 128, true)
row("`fft2`, 128 by 128", time(function()
	c64.fft2(X)
end))
-- CG on the 5-point Laplacian of a 200 by 200 grid, matrix-free: time per iteration
local n = 200
local N = n * n
local function lap(out: buffer, u: buffer): buffer
	for j = 0, n - 1 do
		for i = 0, n - 1 do
			local k = j * n + i
			local c = 4 * buffer.readf64(u, 8 * k)
			if i > 0 then c -= buffer.readf64(u, 8 * (k - 1)) end
			if i < n - 1 then c -= buffer.readf64(u, 8 * (k + 1)) end
			if j > 0 then c -= buffer.readf64(u, 8 * (k - n)) end
			if j < n - 1 then c -= buffer.readf64(u, 8 * (k + n)) end
			buffer.writef64(out, 8 * k, c)
		end
	end
	return out
end
local rhs = f64.fill(f64.create(N), 1)
local iters = 0
local per = time(function()
	local _, info = krylov.cg(f64.space, lap, rhs, { maxiter = 20, tol = 1e-30 })
	iters = info.iterations
end)
row("CG iteration, 200x200 Laplacian", per / iters)
