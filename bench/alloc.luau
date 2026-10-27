--!strict
-- Bytes allocated per call (luau -O2 [--codegen] bench/alloc.luau): what a call leaves for
-- the garbage collector, beyond its inputs. A 64 MB ballast keeps the collector paused, so
-- the heap's growth over a call is exactly what the call allocated (Luau's heap count has
-- 1 KB resolution: small calls are repeated and averaged). Each function is called once
-- before measuring, so one-time caches (FFT plans, arclength tables) are not counted.
local lib = require("../src")
local f64, c64, krylov, ode, curve, cheb, xpbd, graph, optimize, spatial = lib.f64, lib.c64, lib.krylov, lib.ode, lib.curve, lib.cheb, lib.xpbd, lib.graph, lib.optimize, lib.spatial

local ballast = buffer.create(64 * 2 ^ 20)
-- (collectgarbage is in the CLI's environment but not in the type checker's)
local collectgarbage: (string) -> number = (getfenv() :: any).collectgarbage
local function per_call(f: () -> (), reps: number): number
	f()
	collectgarbage("collect")
	local c0 = collectgarbage("count")
	for _ = 1, reps do
		f()
	end
	return (collectgarbage("count") - c0) * 1024 / reps
end

local seed = 12345
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647 - 0.5
end
local function vec(n: number): buffer
	local b = f64.create(n)
	for k = 0, n - 1 do
		buffer.writef64(b, 8 * k, rnd())
	end
	return b
end
local function mat(n: number, spd: boolean?): any
	local A = f64.matrix(n, n)
	A.data = vec(n * n)
	if spd then
		for i = 1, n do
			buffer.writef64(A.data, 8 * ((i - 1) * n + i - 1), n)
		end
		A = f64.matmul(nil, A, f64.transpose(nil, A))
	end
	return A
end

local rows: { { any } } = {}
local function row(name: string, bytes: number, note: string?)
	table.insert(rows, { name, bytes, note or "" })
end

-- dense: n = 100, an n-vector is 800 bytes and an n by n matrix 80,000
local n = 100
local A, S, b, out = mat(n), mat(n, true), vec(n), f64.create(n)
local LU, CH, QR = f64.lu_copy(A), f64.chol_copy(S), f64.qr_copy(A)
local AA = f64.matrix(n, n)
row("matmul (out given)", per_call(function() f64.matmul(AA, A, A) end, 10))
row("lu_solve (out given)", per_call(function() f64.lu_solve(out, LU, b) end, 200))
row("chol_solve (out given)", per_call(function() f64.chol_solve(out, CH, b) end, 200))
row("qr_solve (out given)", per_call(function() f64.qr_solve(out, QR, b) end, 200))
row("solve (out given)", per_call(function() (f64.solve :: any)(out, A, b) end, 20), "factors a copy of A")
row("lu_copy", per_call(function() f64.lu_copy(A) end, 20), "the copy: 80,000")
row("lu_copy (reuse given)", per_call(function() LU = f64.lu_copy(A, LU) end, 20))
row("chol_copy", per_call(function() f64.chol_copy(S) end, 20), "the copy: 80,000")
row("chol_copy (reuse given)", per_call(function() CH = f64.chol_copy(S, CH) end, 20))
row("eigh_copy", per_call(function() f64.eigh_copy(S) end, 3), "the copy: 80,000")
row("svd_copy", per_call(function() f64.svd_copy(A) end, 3), "the copy: 80,000")
row("eig_copy", per_call(function() f64.eig_copy(A) end, 3), "the copy: 80,000")
row("expm", per_call(function() f64.expm(A) end, 2), "the result: 80,000")

-- structured: n = 10,000
local N = 10000
local bN, oN = vec(N), f64.create(N)
local T = f64.tridiagonal(vec(N - 1), f64.fill(f64.create(N), 4), vec(N - 1))
local TL = f64.tridiag_lu_copy(T)
row("tridiag_lu_copy", per_call(function() f64.tridiag_lu_copy(T) end, 10), "copies 3 bands + pivots: ~320,000")
row("tridiag_lu_solve (out given)", per_call(function() f64.tridiag_lu_solve(oN, TL, bN) end, 10))
local kd = 5
local B = f64.banded(N, kd, kd)
for j = 1, N do
	for i = math.max(1, j - kd), math.min(N, j + kd) do
		f64.band_set(B, i, j, if i == j then 4 * kd else -1 / (1 + math.abs(i - j)))
	end
end
local BC, BL = f64.band_chol_copy(B), f64.band_lu_copy(B)
row("band_matvec (out given)", per_call(function() f64.band_matvec(oN, B, bN) end, 10))
row("band_chol_copy", per_call(function() f64.band_chol_copy(B) end, 10), `the copy: {8 * B.ld * N} (Cholesky reads {8 * (kd + 1) * N})`)
row("band_chol_solve (out given)", per_call(function() f64.band_chol_solve(oN, BC, bN) end, 10))
row("band_chol_copy (reuse given)", per_call(function() BC = f64.band_chol_copy(B, BC) end, 10))
local SB = f64.symbanded(N, kd)
for j = 1, N do
	for i = j, math.min(N, j + kd) do
		f64.symband_set(SB, i, j, f64.band_get(B, i, j))
	end
end
local SC = f64.band_chol_copy(SB)
row("band_chol_copy, symmetric band storage", per_call(function() f64.band_chol_copy(SB) end, 10), `the copy: {8 * (kd + 1) * N}`)
row("band_chol_copy, symmetric, reuse given", per_call(function() SC = f64.band_chol_copy(SB, SC) end, 10))
row("band_lu_copy", per_call(function() f64.band_lu_copy(B) end, 10), `the copy: {8 * B.ld * N}`)
row("band_lu_copy (reuse given)", per_call(function() BL = f64.band_lu_copy(B, BL) end, 10))
row("band_lu_solve (out given)", per_call(function() f64.band_lu_solve(oN, BL, bN) end, 10))
row("banded storage per entry of the band", 8 * B.ld / (2 * kd + 1), "bytes; 8 would be the band alone")
row("symmetric band storage per entry of its lower band", 8 * (kd + 1) / (kd + 1), "bytes")

-- sparse and Krylov: the 5-point Laplacian of a 100 by 100 grid (10,000 unknowns)
local m = 100
local ii, jj, vv = {}, {}, {}
for k = 1, m * m do
	local i, j = (k - 1) % m, (k - 1) // m
	local function add(c: number, v: number)
		table.insert(ii, k)
		table.insert(jj, c)
		table.insert(vv, v)
	end
	add(k, 4)
	if i > 0 then add(k - 1, -1) end
	if i < m - 1 then add(k + 1, -1) end
	if j > 0 then add(k - m, -1) end
	if j < m - 1 then add(k + m, -1) end
end
local L = f64.sparse(m * m, m * m, ii, jj, f64.from(vv))
L.symmetric = true
row("sparse_matvec (out given)", per_call(function() f64.sparse_matvec(oN, L, bN) end, 10))
local function op(o: buffer, x: buffer): buffer
	return f64.sparse_matvec(o, L, x)
end
local cg10 = per_call(function() krylov.cg(f64.space, op, bN, { maxiter = 10, tol = 1e-30 }) end, 3)
local cg60 = per_call(function() krylov.cg(f64.space, op, bN, { maxiter = 60, tol = 1e-30 }) end, 3)
row("cg, per call", cg10, "an n-vector is 80,000")
row("cg, per iteration", (cg60 - cg10) / 50)
local g10 = per_call(function() krylov.gmres(f64.space, op, bN, { maxiter = 10, tol = 1e-30 }) end, 3)
local g60 = per_call(function() krylov.gmres(f64.space, op, bN, { maxiter = 60, tol = 1e-30 }) end, 3)
row("gmres, per iteration", (g60 - g10) / 50, "Krylov basis vectors are kept: 80,000 each")

-- FFT: 4096 points, in place
local x = c64.create(4096)
row("fft (in place)", per_call(function() c64.fft(x) end, 20))
row("rfft (out given)", per_call(function() f64.rfft(c64.create(2049), vec(4096)) end, 20), "inputs made in the loop: 32,768 + 32,784")

-- ODEs: per step, from runs of two lengths
local function lorenz(_t: number, y: buffer, o: buffer): buffer
	local x1, x2, x3 = buffer.readf64(y, 0), buffer.readf64(y, 8), buffer.readf64(y, 16)
	buffer.writef64(o, 0, 10 * (x2 - x1))
	buffer.writef64(o, 8, x1 * (28 - x3) - x2)
	buffer.writef64(o, 16, x1 * x2 - 8 / 3 * x3)
	return o
end
local y0 = f64.from({ 1, 1, 1 })
for _, method in { "dp5", "dop853", "rodas5p", "radau", "bdf" } do
	local steps = {}
	local function run(T1: number): () -> ()
		return function()
			local sol = ode.solve(f64.space, lorenz, { 0, T1 }, y0, { method = method, rtol = 1e-8, atol = 1e-10 })
			steps[T1] = sol.stats.steps
		end
	end
	local a1, a2 = per_call(run(2), 1), per_call(run(4), 1)
	row(`ode.solve {method}, per step`, (a2 - a1) / (steps[4] - steps[2]), "the state is 24 bytes")
	local function run_nd(T1: number): () -> ()
		return function()
			local sol = ode.solve(f64.space, lorenz, { 0, T1 }, y0, { method = method, rtol = 1e-8, atol = 1e-10, dense = false })
			steps[T1] = sol.stats.steps
		end
	end
	local d1, d2 = per_call(run_nd(2), 1), per_call(run_nd(4), 1)
	row(`ode.solve {method} dense = false, per step`, (d2 - d1) / (steps[4] - steps[2]))
	local it = ode.integrator(f64.space, lorenz, 0, y0, { method = method, rtol = 1e-8, atol = 1e-10 })
	row(`integrator {method}, per advance(1/60)`, per_call(function() it:advance(1 / 60) end, 60))
end

-- implicit mechanics: a 6-node chain (18 unknowns), block tridiagonal K, direct solver
do
	local NB = 6
	local bi, bj = {}, {}
	for i = 1, NB do
		table.insert(bi, i)
		table.insert(bj, i)
		if i > 1 then
			table.insert(bi, i)
			table.insert(bj, i - 1)
			table.insert(bi, i - 1)
			table.insert(bj, i)
		end
	end
	local vals = f64.create(9 * #bi)
	for k = 1, #bi do
		for a = 0, 2 do
			f64.set(vals, 9 * (k - 1) + 4 * a + 1, if bi[k] == bj[k] then -2e4 else 1e4)
		end
	end
	local K = f64.bsparse(3 * NB, 3 * NB, 3, bi, bj, vals)
	K.symmetric = true
	local D = table.clone(K)
	D.values = f64.scale(nil, 1e-2, K.values)
	local q0 = vec(3 * NB)
	for _, rho in { false, 0.5 } do
		local it = ode.implicit_mechanics(f64.space, function(_t, q, _v, o)
			return f64.bsparse_matvec(o, K, q)
		end, 0, q0, f64.create(3 * NB), { jacobian = function() return K, D end, rho_inf = if rho then rho else nil })
		row(`implicit_mechanics step ({if rho then "rational" else "backward Euler"}, 18 unknowns)`, per_call(function() it:step(1 / 60) end, 100), "the state is 2 x 144 bytes")
	end
end

-- per-frame calls
local P: { vector } = {}
for k = 0, 9 do
	P[k + 1] = vector.create(10 * math.cos(k), k, 10 * math.sin(k))
end
local C = curve.cubic_spline(P, { closed = true })
local Fr = curve.framing(C)
local Lc = curve.length(C)
local s = 0
row("curve.eval", per_call(function() s += 0.37; curve.eval(C, s % 10) end, 1000))
row("curve.tangent", per_call(function() s += 0.37; curve.tangent(C, s % 10) end, 1000))
row("curve.curvature", per_call(function() s += 0.37; curve.curvature(C, s % 10) end, 1000))
row("curve.at_length", per_call(function() s += 0.37; curve.at_length(C, s % Lc) end, 1000))
row("framing:at", per_call(function() s += 0.37; Fr:at(s) end, 1000))
local tq = 0
row("curve.closest_near", per_call(function() s += 0.37; tq = curve.closest_near(C, vector.create(9, 3, 1), tq) end, 1000))
local Fc = cheb.fun(math.exp)
row("cheb.eval", per_call(function() cheb.eval(Fc, 0.3) end, 1000))

-- simulation steps
local pos = {}
for k = 1, 20 do
	pos[k] = vector.create(k, 0, 0)
end
local W = xpbd.new(pos, { gravity = vector.create(0, -9.8, 0) })
W:pin(1)
for k = 2, 20 do
	W:distance(k - 1, k, 0)
end
row("xpbd step (20 particles, 4 substeps)", per_call(function() W:step(1 / 60, 4) end, 100))

-- graphs: a 50 by 50 grid (2500 vertices, 4900 edges)
do
	local gm = 50
	local gi, gj, ww = {}, {}, {}
	for gy = 0, gm - 1 do
		for gx = 0, gm - 1 do
			local k = gy * gm + gx + 1
			if gx < gm - 1 then
				table.insert(gi, k)
				table.insert(gj, k + 1)
				table.insert(ww, 1.5 + rnd())
			end
			if gy < gm - 1 then
				table.insert(gi, k)
				table.insert(gj, k + gm)
				table.insert(ww, 1.5 + rnd())
			end
		end
	end
	local G, w = graph.from_edges(gm * gm, gi, gj, { weights = ww })
	row("graph.rcm (2500 vertices)", per_call(function() graph.rcm(G) end, 10), "25 bytes a vertex: the permutation and its workspace")
	row("graph.bfs", per_call(function() graph.bfs(G, 1) end, 10), "12 bytes a vertex: the order and levels returned, and the queue")
	row("graph.dijkstra", per_call(function() graph.dijkstra(G, w, 1) end, 10), "20 bytes a vertex: distances and predecessors returned, and the heap")
end

-- spatial queries: 2000 points with about 12 neighbors within 1 each
do
	local side = (2000 * (4 / 3) * math.pi / 12) ^ (1 / 3)
	local pts = table.create(2000) :: { vector }
	for i = 1, 2000 do
		pts[i] = vector.create((rnd() + 0.5) * side, (rnd() + 0.5) * side, (rnd() + 0.5) * side)
	end
	local G = spatial.grid(pts, 1)
	local tree = spatial.kdtree(pts)
	local found, dist = {}, {}
	local q = 0
	row("spatial grid:rebuild (2000 points)", per_call(function() G:rebuild(pts) end, 10))
	row("grid:radius, into out", per_call(function() q = q % 2000 + 1; G:radius(pts[q], 1, found) end, 1000))
	row("kdtree:rebuild (2000 points)", per_call(function() tree:rebuild(pts) end, 10), "the build's closure")
	row("kdtree:refit", per_call(function() tree:refit(pts) end, 10))
	row("kdtree:radius, into out", per_call(function() q = q % 2000 + 1; tree:radius(pts[q], 1, found) end, 1000))
	row("kdtree:knn, k = 8, into out", per_call(function() q = q % 2000 + 1; tree:knn(pts[q], 8, found, dist) end, 1000))
	row("kdtree:nearest", per_call(function() q = q % 2000 + 1; tree:nearest(pts[q]) end, 1000))
end

-- optimization: Rosenbrock's least squares (2 residuals, 2 unknowns) from (-1.2, 1)
do
	local function r(xr: buffer, out: buffer?): buffer
		local o = out or f64.create(2)
		local a, b = f64.get(xr, 1), f64.get(xr, 2)
		f64.set(o, 1, 10 * (b - a * a))
		f64.set(o, 2, 1 - a)
		return o
	end
	local function jac(xj: buffer, J: any): any
		-- column-major, 2 by 2
		buffer.writef64(J.data, 0, -20 * f64.get(xj, 1))
		buffer.writef64(J.data, 8, -1)
		buffer.writef64(J.data, 16, 10)
		buffer.writef64(J.data, 24, 0)
		return J
	end
	local x0 = f64.from({ -1.2, 1 })
	local _, info = optimize.least_squares(f64.space, r, x0, { jac = jac })
	row("optimize.least_squares (Rosenbrock)", per_call(function() optimize.least_squares(f64.space, r, x0, { jac = jac }) end, 10), "its workspace, per call")
	row("optimize.least_squares, reuse", per_call(function() optimize.least_squares(f64.space, r, x0, { jac = jac, jacobian = "lazy", reuse = info, maxiter = 4 }) end, 10), "4 iterations from the last call's Jacobian")
end

print("| call | bytes per call | note |")
print("| --- | --- | --- |")
for _, r in rows do
	print(string.format("| %s | %.0f | %s |", r[1], r[2], r[3]))
end
print(#tostring(ballast) > 0 and "" or "")
