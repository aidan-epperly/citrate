--!strict

-- Krylov eigensolvers: thick-restart Lanczos (eigsh) and Krylov-Schur (eigs).

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local krylov, f64, f32, c64, vec = lib.krylov, lib.f64, lib.f32, lib.c64, lib.vec

local function random_values(n: number): { number }
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = 2 * math.random() - 1
	end
	return t
end

-- The 1D Laplacian tridiag(-1, 2, -1) of size n as an operator on module M's arrays;
-- eigenvalues 2 - 2 cos(k pi / (n + 1)).
local function laplacian(M: any, n: number): lib.Operator<buffer>
	return function(out: buffer, x: buffer): buffer
		local o = out or M.create(n)
		for i = 1, n do
			local v = 2 * M.get(x, i)
			if i > 1 then
				v -= M.get(x, i - 1)
			end
			if i < n then
				v -= M.get(x, i + 1)
			end
			M.set(o, i, v)
		end
		return o
	end
end

test("eigsh: thick-restart Lanczos", function()
	math.randomseed(601)
	local n = 150
	local A = laplacian(f64, n)
	local v0 = f64.from(random_values(n))
	local function exact(k: number): number
		return 2 - 2 * math.cos(k * math.pi / (n + 1))
	end
	local values, vectors, info = krylov.eigsh(f64.space, A, v0, 4, { which = "LA" })
	check(info.converged and info.nconv >= 4, `LA converged ({info.restarts} restarts)`)
	local ok = true
	for q = 1, 4 do
		ok = ok and math.abs(f64.get(values, q) - exact(n + 1 - q)) <= 1e-10
		local r = f64.sub(nil, A(f64.create(n), vectors[q]), f64.scale(nil, f64.get(values, q), vectors[q]))
		ok = ok and f64.norm(r) <= 1e-10 and math.abs(f64.norm(vectors[q]) - 1) <= 1e-12
	end
	check(ok, "LA: the largest eigenvalues, in order, with small residuals")
	local orth = 0
	for a = 1, 4 do
		for b = a + 1, 4 do
			orth = math.max(orth, math.abs(f64.dot(vectors[a], vectors[b])))
		end
	end
	check(orth <= 1e-10, "eigenvectors are orthogonal")
	local small, _, sinfo = krylov.eigsh(f64.space, A, v0, 2, { which = "SA", maxiter = 3000 })
	check(sinfo.converged, `SA converged ({sinfo.restarts} restarts)`)
	close(f64.get(small, 1), exact(1), 1e-9, "SA: the smallest eigenvalue")
	-- LM on an indefinite operator: 2 - Laplacian has eigenvalues in (-2, 2) symmetric.
	local B: lib.Operator<buffer> = function(out: buffer, x: buffer): buffer
		local o = A(out, x)
		return f64.axpy(o, -1, o, f64.scale(nil, 2, x))
	end
	local lm = krylov.eigsh(f64.space, B, v0, 2, { which = "LM" })
	check(math.abs(math.abs(f64.get(lm, 1)) - (2 - exact(1))) <= 1e-9, "LM on an indefinite operator")

	-- A complex Hermitian operator against the dense eigh.
	local m = 40
	local re, im = random_values(m * m), random_values(m * m)
	local G = c64.matrix(m, m)
	G.data = c64.from(re, im)
	local H = c64.adjoint(nil, G)
	H.data = c64.add(nil, H.data, G.data)
	local Hop: lib.Operator<buffer> = function(out: buffer, x: buffer): buffer
		return c64.matvec(out, H, x)
	end
	local hv, hx, hinfo = krylov.eigsh(c64.space, Hop, c64.from(random_values(m), random_values(m)), 3, { which = "LA" })
	local dense = f64.totable(c64.eigh_copy(H, false).values)
	check(hinfo.converged, "complex Hermitian converged")
	local agree = true
	for q = 1, 3 do
		agree = agree and math.abs(f64.get(hv, q) - dense[m + 1 - q]) <= 1e-9 * math.abs(dense[m])
		local r = c64.sub(nil, Hop(c64.create(m), hx[q]), c64.scale(nil, f64.get(hv, q), hx[q]))
		agree = agree and c64.norm(r) <= 1e-9 * math.abs(dense[m])
	end
	check(agree, "complex Hermitian: matches the dense eigh")

	-- Vectors (3 dimensions): the Krylov space is exhausted, so the answer is exact.
	local S3: lib.Operator<vector> = function(_: vector, x: vector): vector
		return vector.create(4 * x.x + x.y, x.x + 3 * x.y + x.z, x.y + 2 * x.z)
	end
	local v3, x3, i3 = krylov.eigsh(vec.space, S3, vector.create(1, 0.3, -0.2), 2, { which = "LA" })
	local d3 = f64.totable(f64.eigh_copy(f64.matfrom({ { 4, 1, 0 }, { 1, 3, 1 }, { 0, 1, 2 } }), false).values)
	check(i3.converged and math.abs(f64.get(v3, 1) - d3[3]) <= 1e-5 and type(x3[1]) == "vector", "vectors: exact after the space is exhausted")
	-- Single precision: the default tolerance follows the space.
	local fv, _, finfo = krylov.eigsh(f32.space, laplacian(f32, 60), f32.from(random_values(60)), 2, { which = "LA" })
	check(finfo.converged and math.abs(f64.get(fv, 1) - (2 - 2 * math.cos(60 * math.pi / 61))) <= 1e-4, "f32 space converges")
	check(not pcall(function()
		krylov.eigsh(f64.space, A, f64.create(n), 2)
	end), "a zero start vector is an error")
	check(not pcall(function()
		krylov.eigsh(f64.space, A, v0, 5, { ncv = 5 })
	end), "k must be less than ncv")
end)

test("eigs: Krylov-Schur", function()
	math.randomseed(602)
	-- A normal real matrix with known complex eigenvalues: Q diag(blocks) Q^T, each 2 by 2
	-- block [[a, -b], [b, a]] giving a +- b i, plus some real ones.
	local n = 60
	local Q = f64.qr_q(nil, f64.qr(f64.matfrom((function()
		local rows = {}
		for i = 1, n do
			rows[i] = random_values(n)
		end
		return rows
	end)())))
	local D = f64.matrix(n, n)
	local expected: { { number } } = {}
	local i = 1
	while i <= n do
		if i < n and i % 3 ~= 0 then
			local a, b = 2 * math.random() - 1, math.random()
			if i == 1 then
				a, b = 1.5, 2.5 -- the largest in magnitude
			end
			f64.matset(D, i, i, a)
			f64.matset(D, i + 1, i + 1, a)
			f64.matset(D, i, i + 1, -b)
			f64.matset(D, i + 1, i, b)
			table.insert(expected, { a, b })
			table.insert(expected, { a, -b })
			i += 2
		else
			local a = 2 * math.random() - 1
			f64.matset(D, i, i, a)
			table.insert(expected, { a, 0 })
			i += 1
		end
	end
	local A = f64.matmul(nil, f64.matmul(nil, Q, D), f64.transpose(nil, Q))
	local Aop: lib.Operator<buffer> = function(out: buffer, x: buffer): buffer
		return f64.matvec(out, A, x)
	end
	local values, vectors, info = krylov.eigs(f64.space, Aop, f64.from(random_values(n)), 2)
	check(info.converged, `converged ({info.restarts} restarts)`)
	local re, im = c64.totable(values)
	close(math.sqrt(re[1] ^ 2 + im[1] ^ 2), math.sqrt(1.5 ^ 2 + 2.5 ^ 2), 1e-10, "LM: the largest magnitude")
	check(math.abs(re[1] - 1.5) <= 1e-10 and math.abs(math.abs(im[1]) - 2.5) <= 1e-10, "LM: 1.5 +- 2.5 i")
	check(math.abs(re[2] - 1.5) <= 1e-10 and math.abs(im[2] + im[1]) <= 1e-10, "LM: then its conjugate")
	-- Real space: the vectors are (re, im) pairs; check A x = lambda x.
	local ok = true
	for q = 1, 2 do
		local x = vectors[q] :: any
		local Axr, Axi = Aop(f64.create(n), x.re), Aop(f64.create(n), x.im)
		local rr = f64.sub(nil, Axr, f64.sub(nil, f64.scale(nil, re[q], x.re), f64.scale(nil, im[q], x.im)))
		local ri = f64.sub(nil, Axi, f64.add(nil, f64.scale(nil, re[q], x.im), f64.scale(nil, im[q], x.re)))
		local nx = math.sqrt(f64.norm(x.re) ^ 2 + f64.norm(x.im) ^ 2)
		ok = ok and math.sqrt(f64.norm(rr) ^ 2 + f64.norm(ri) ^ 2) <= 1e-9 and math.abs(nx - 1) <= 1e-10
	end
	check(ok, "real space: A x = lambda x for (re, im) pairs")
	-- Other orderings.
	local lr = c64.totable(krylov.eigs(f64.space, Aop, f64.from(random_values(n)), 1, { which = "LR" }))
	local best = -math.huge
	for _, e in expected do
		best = math.max(best, e[1])
	end
	close(lr[1], best, 1e-9, "LR: the largest real part")
	local _, si = c64.totable(krylov.eigs(f64.space, Aop, f64.from(random_values(n)), 1, { which = "SI" }))
	local lowest = math.huge
	for _, e in expected do
		lowest = math.min(lowest, e[2])
	end
	close(si[1], lowest, 1e-9, "SI: the smallest imaginary part")

	-- A complex space: a random complex operator against the dense eig.
	local m = 30
	local Cm = c64.matrix(m, m)
	Cm.data = c64.from(random_values(m * m), random_values(m * m))
	local Cop: lib.Operator<buffer> = function(out: buffer, x: buffer): buffer
		return c64.matvec(out, Cm, x)
	end
	local cv, cx, cinfo = krylov.eigs(c64.space, Cop, c64.from(random_values(m), random_values(m)), 3, { ncv = 15 })
	local dr, di = c64.totable(c64.eig_copy(Cm, false).values)
	local mags = {}
	for k = 1, m do
		mags[k] = math.sqrt(dr[k] ^ 2 + di[k] ^ 2)
	end
	table.sort(mags, function(a: number, b: number): boolean
		return a > b
	end)
	local cr, ci = c64.totable(cv)
	local cok = cinfo.converged
	for q = 1, 3 do
		cok = cok and math.abs(math.sqrt(cr[q] ^ 2 + ci[q] ^ 2) - mags[q]) <= 1e-9
		local r = c64.sub(nil, Cop(c64.create(m), cx[q]), c64.cscale(nil, cr[q], ci[q], cx[q]))
		cok = cok and c64.norm(r) <= 1e-9
	end
	check(cok, "complex space: the three largest in magnitude, with eigenvectors")
end)

return nil
