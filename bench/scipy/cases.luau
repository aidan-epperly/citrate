--!strict
-- The citrate side of the SciPy comparison (bench/scipy/run.py runs it):
--   luau -O2 [--codegen] bench/scipy/cases.luau -a <case> time    -- prints seconds per call
--   luau -O2 [--codegen] bench/scipy/cases.luau -a <case> mem <r> -- the inputs, r calls
-- (r = 0 builds the inputs only: the baseline for memory). The inputs are the same as
-- cases.py's: the same generator, sizes and problems.
local lib = require("../../src")
local f64, c64, krylov, ode, curve, rational, graph, spatial = lib.f64, lib.c64, lib.krylov, lib.ode, lib.curve, lib.rational, lib.graph, lib.spatial

local case, mode, reps = ...
assert(type(case) == "string" and type(mode) == "string", "usage: cases.luau -a <case> time|mem [reps]")

-- Park-Miller, exact in doubles; values in [-0.5, 0.5)
local seed = 12345
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647 - 0.5
end
local function random_matrix(n: number): any
	local A = f64.matrix(n, n)
	for k = 0, n * n - 1 do -- column-major, as numpy's order="F"
		buffer.writef64(A.data, 8 * k, rnd())
	end
	return A
end
local function random_vector(n: number): buffer
	local b = f64.create(n)
	for k = 0, n - 1 do
		buffer.writef64(b, 8 * k, rnd())
	end
	return b
end

-- the 5-point Laplacian of an m by m grid, written straight into compressed columns (no
-- temporary tables, so building it leaves no garbage to mask the calls' memory)
local function laplacian(m: number): any
	local N = m * m
	local nnz = 5 * N - 4 * m
	local colptr, rowind, values = buffer.create(4 * (N + 1)), buffer.create(4 * nnz), f64.create(nnz)
	local p = 0
	for k = 0, N - 1 do
		buffer.writeu32(colptr, 4 * k, p)
		local i, j = k % m, k // m
		local function add(r: number, v: number)
			buffer.writeu32(rowind, 4 * p, r)
			buffer.writef64(values, 8 * p, v)
			p += 1
		end
		if j > 0 then add(k - m, -1) end
		if i > 0 then add(k - 1, -1) end
		add(k, 4)
		if i < m - 1 then add(k + 1, -1) end
		if j < m - 1 then add(k + m, -1) end
	end
	buffer.writeu32(colptr, 4 * N, p)
	assert(p == nnz)
	return { rows = N, cols = N, colptr = colptr, rowind = rowind, values = values, symmetric = true }
end

local setups: { [string]: () -> () -> () } = {}

setups.matmul_256 = function()
	local A, B = random_matrix(256), random_matrix(256)
	return function() f64.matmul(nil, A, B) end
end
for _, n in { 8, 120, 500 } do
	setups["solve_" .. n] = function()
		local A, b = random_matrix(n), random_vector(n)
		return function() (f64.solve :: any)(nil, A, b) end
	end
end
setups.eigh_200 = function()
	local A = random_matrix(200)
	local H = f64.matrix(200, 200) -- A + A^T, in place
	for j = 0, 199 do
		for i = 0, 199 do
			buffer.writef64(H.data, 8 * (j * 200 + i), buffer.readf64(A.data, 8 * (j * 200 + i)) + buffer.readf64(A.data, 8 * (i * 200 + j)))
		end
	end
	return function() f64.eigh_copy(H) end
end
setups.eig_200 = function()
	local A = random_matrix(200)
	return function() f64.eig_copy(A) end
end
setups.svd_200 = function()
	local A = random_matrix(200)
	return function() f64.svd_copy(A) end
end
for _, n in { 4096, 65536 } do
	setups["fft_" .. n] = function()
		local x = c64.create(n)
		for k = 0, 2 * n - 1 do
			buffer.writef64(x, 8 * k, rnd())
		end
		local y = c64.create(n)
		-- out of place, as SciPy's: copy, then transform in place
		return function() c64.fft(c64.copy(y, x)) end
	end
end
setups.banded_chol_100000 = function()
	-- symmetric band storage (the diagonal and kd subdiagonals), as SciPy's lower form
	local n, kd = 100000, 10
	local B = f64.symbanded(n, kd)
	for j = 1, n do
		for i = j, math.min(n, j + kd) do
			f64.symband_set(B, i, j, if i == j then 2 * kd + 1 else rnd())
		end
	end
	local b = random_vector(n)
	return function() f64.band_chol_solve(nil, f64.band_chol_copy(B), b) end
end
setups.sparse_matvec_500 = function()
	local S = laplacian(500)
	local x = random_vector(500 * 500)
	local y = f64.create(500 * 500)
	return function() f64.sparse_matvec(y, S, x) end
end
setups.cg_200 = function()
	local S = laplacian(200)
	local b = random_vector(200 * 200)
	local function A(out: buffer, x: buffer): buffer
		return f64.sparse_matvec(out, S, x)
	end
	return function()
		local _, info = krylov.cg(f64.space, A, b, { maxiter = 100, tol = 1e-30 })
		assert(info.iterations == 100)
	end
end

local function lorenz(_t: number, y: buffer, out: buffer): buffer
	local x1, x2, x3 = buffer.readf64(y, 0), buffer.readf64(y, 8), buffer.readf64(y, 16)
	buffer.writef64(out, 0, 10 * (x2 - x1))
	buffer.writef64(out, 8, x1 * (28 - x3) - x2)
	buffer.writef64(out, 16, x1 * x2 - 8 / 3 * x3)
	return out
end
local function vdp(_t: number, y: buffer, out: buffer): buffer
	local x1, x2 = buffer.readf64(y, 0), buffer.readf64(y, 8)
	buffer.writef64(out, 0, x2)
	buffer.writef64(out, 8, 1000 * ((1 - x1 * x1) * x2) - x1)
	return out
end
local STATS = ""
for _, m in { { "dp5", "lorenz" }, { "dop853", "lorenz" }, { "radau", "vdp" }, { "bdf", "vdp" } } do
	setups["ode_" .. m[1] .. "_" .. m[2]] = function()
		local f = if m[2] == "lorenz" then lorenz else vdp
		local y0 = if m[2] == "lorenz" then f64.from({ 1, 1, 1 }) else f64.from({ 2, 0 })
		local tspan = if m[2] == "lorenz" then { 0, 100 } else { 0, 3000 }
		local tol = if m[2] == "lorenz" then { 1e-8, 1e-10 } else { 1e-6, 1e-6 }
		return function()
			local sol = ode.solve(f64.space, f, tspan, y0, { method = m[1], rtol = tol[1], atol = tol[2] })
			assert(sol.status == "success", sol.message)
			STATS = string.format("steps %d nfev %d", sol.stats.steps, sol.stats.nfev)
		end
	end
end
local function spline_points(): ({ buffer }, { number })
	local n = 1000
	local pts, xs = {}, {}
	for i = 1, n do
		local x = (i - 1) / (n - 1) * 10
		xs[i] = x
		pts[i] = f64.from({ math.sin(x) + 0.1 * rnd() })
	end
	return pts, xs
end
setups.spline_build_1000 = function()
	local pts, xs = spline_points()
	return function() curve.cubic_spline(pts, { space = f64.space, breaks = xs, ends = "not-a-knot" }) end
end
setups.spline_eval_100000 = function()
	local pts, xs = spline_points()
	local C = curve.cubic_spline(pts, { space = f64.space, breaks = xs, ends = "not-a-knot" })
	local out = f64.create(1)
	return function()
		for k = 0, 99999 do
			curve.eval(C, k / 99999 * 10, out)
		end
	end
end
setups.spline_eval_one = function()
	local pts, xs = spline_points()
	local C = curve.cubic_spline(pts, { space = f64.space, breaks = xs, ends = "not-a-knot" })
	local out = f64.create(1)
	return function() curve.eval(C, 3.7, out) end
end
setups.aaa_1000 = function()
	local n = 1000
	local Z, F = c64.create(n), c64.create(n)
	for k = 0, n - 1 do
		local x = -1 + 2 * k / (n - 1)
		c64.set(Z, k + 1, x, 0)
		c64.set(F, k + 1, math.exp(x) / (1.05 - x * x) + math.tanh(5 * x), 0)
	end
	return function() rational.aaa(Z, F) end
end

-- an m by m grid graph with its vertices numbered at random (a Fisher-Yates shuffle by
-- the same generator) and random edge lengths in [0.5, 1.5)
local function shuffled_grid(m: number): (any, buffer)
	local N = m * m
	local label = table.create(N, 0)
	for k = 1, N do
		label[k] = k
	end
	for i = N - 1, 1, -1 do
		seed = (seed * 16807) % 2147483647
		local j = seed % (i + 1)
		label[i + 1], label[j + 1] = label[j + 1], label[i + 1]
	end
	local ii, jj, ww = {}, {}, {}
	for y = 0, m - 1 do
		for x = 0, m - 1 do
			local k = y * m + x
			if x < m - 1 then
				table.insert(ii, label[k + 1])
				table.insert(jj, label[k + 2])
				table.insert(ww, 1 + rnd())
			end
			if y < m - 1 then
				table.insert(ii, label[k + 1])
				table.insert(jj, label[k + m + 1])
				table.insert(ww, 1 + rnd())
			end
		end
	end
	local G, w = graph.from_edges(N, ii, jj, { weights = ww })
	return G, w :: buffer
end
setups.rcm_grid_200 = function()
	local G = shuffled_grid(200)
	local perm = graph.rcm(G)
	STATS = string.format("bandwidth %d -> %d", graph.bandwidth(G), graph.bandwidth(G, perm))
	return function() graph.rcm(G) end
end
setups.dijkstra_grid_200 = function()
	local G, w = shuffled_grid(200)
	return function() graph.dijkstra(G, w, 1) end
end

-- n points uniform in a cube sized for about 12 neighbors within 1 each
local function cube_points(n: number): { vector }
	local L = (n * (4 / 3) * math.pi / 12) ^ (1 / 3)
	local P = table.create(n) :: { vector }
	for i = 1, n do
		local x = (rnd() + 0.5) * L
		local y = (rnd() + 0.5) * L
		local z = (rnd() + 0.5) * L
		P[i] = vector.create(x, y, z)
	end
	return P
end
setups.kdtree_build_10000 = function()
	local P = cube_points(10000)
	return function() spatial.kdtree(P) end
end
setups.kdtree_knn8_10000 = function()
	local P = cube_points(10000)
	local T = spatial.kdtree(P)
	local idx, dist = {}, {}
	return function()
		for i = 1, #P do
			T:knn(P[i], 8, idx, dist)
		end
	end
end
setups.kdtree_ball_10000 = function()
	local P = cube_points(10000)
	local T = spatial.kdtree(P)
	local out = {}
	return function()
		for i = 1, #P do
			T:radius(P[i], 1, out)
		end
	end
end
setups.grid_ball_10000 = function()
	local P = cube_points(10000)
	local G = spatial.grid(P, 1)
	local out = {}
	return function()
		G:rebuild(P)
		for i = 1, #P do
			G:radius(P[i], 1, out)
		end
	end
end

setups.none = function()
	return function() end
end

local make = setups[case]
assert(make, "unknown case " .. case)
local op = make()
if mode == "list" then
	return
elseif mode == "time" then
	op() -- warm up (and caches, as SciPy's)
	local best = math.huge
	for _ = 1, 5 do
		local n, t0 = 0, os.clock()
		repeat
			op()
			n += 1
		until os.clock() - t0 > 0.2
		best = math.min(best, (os.clock() - t0) / n)
	end
	print(string.format("%.9g %s", best, STATS))
else
	for _ = 1, tonumber(reps) or 1 do
		op()
	end
end
