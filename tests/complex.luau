--!strict

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local C64, C32 = lib.c64, lib.c32

-- The two complex modules differ only in how scalars are passed: (re, im) for c64,
-- a vector for c32. These adapters let one test body cover both.
type Adapter = {
	name: string,
	M: any,
	eps: number,
	get: (x: buffer, i: number) -> (number, number),
	matset: (A: lib.Matrix, i: number, j: number, re: number, im: number) -> (),
	dot: (x: buffer, y: buffer) -> (number, number),
	dotc: (x: buffer, y: buffer) -> (number, number),
	det: (A: lib.Matrix) -> (number, number),
	caxpy: (out: buffer?, re: number, im: number, x: buffer, y: buffer) -> buffer,
}

local function unpack_vector(z: vector): (number, number)
	return z.x, z.y
end

local adapters: { Adapter } = {
	{
		name = "c64",
		M = C64,
		eps = h.eps64,
		get = C64.get,
		matset = C64.matset,
		dot = C64.dot,
		dotc = C64.dotc,
		det = C64.det,
		caxpy = C64.caxpy,
	},
	{
		name = "c32",
		M = C32,
		eps = h.eps32,
		get = function(x, i)
			return unpack_vector(C32.get(x, i))
		end,
		matset = function(A, i, j, re, im)
			C32.matset(A, i, j, vector.create(re, im, 0))
		end,
		dot = function(x, y)
			return unpack_vector(C32.dot(x, y))
		end,
		dotc = function(x, y)
			return unpack_vector(C32.dotc(x, y))
		end,
		det = function(A)
			return unpack_vector(C32.det(A))
		end,
		caxpy = function(out, re, im, x, y)
			return C32.caxpy(out, vector.create(re, im, 0), x, y)
		end,
	},
}

for _, a in adapters do
	local M, eps, name = a.M, a.eps, a.name
	local tol = 8 * eps

	test(`{name} construction and access`, function()
		local x = M.from({ 1, 2 }, { 3, 4 })
		check(M.length(x) == 2)
		local r, i = a.get(x, 2)
		check(r == 2 and i == 4, "get")
		local re, im = M.totable(x)
		check(re[1] == 1 and im[1] == 3, "totable")
		re, im = M.totable(M.from({ 5 }))
		check(re[1] == 5 and im[1] == 0, "imaginary part defaults to 0")
		local parts = M.complexify(M.real(nil, x), M.imag(nil, x))
		r, i = a.get(parts, 1)
		check(r == 1 and i == 3, "real/imag/complexify round trip")
		local m = M.abs(nil, M.from({ 3 }, { 4 }))
		check(buffer.len(m) == buffer.len(M.real(nil, M.from({ 3 }))) and M.norm(M.from({ 3 }, { 4 })) == 5, "abs")
	end)

	test(`{name} kernels`, function()
		local x, y = M.from({ 1, 2 }, { 3, 4 }), M.from({ 5, 6 }, { 7, 8 })
		local r, i = a.get(M.mul(nil, x, y), 1)
		check(r == -16 and i == 22, "Hadamard product (1+3i)(5+7i)")
		r, i = a.get(M.conj(nil, x), 2)
		check(r == 2 and i == -4, "conj")
		r, i = a.get(M.axpy(nil, 2, x, y), 1)
		check(r == 7 and i == 13, "real axpy")
		r, i = a.get(a.caxpy(nil, 0, 1, x, y), 1)
		check(r == 2 and i == 8, "complex axpy: i(1+3i) + (5+7i)")
		r, i = a.dot(x, y)
		check(r == -16 + (12 - 32) and i == 22 + (16 + 24), "dot without conjugation")
		r, i = a.dotc(x, x)
		check(r == 30 and i == 0, "dotc(x, x) = |x|^2")
		close(M.norm(x), math.sqrt(30), tol, "norm")
		close(M.amax(x), math.sqrt(20), tol, "amax is the largest modulus")
	end)

	test(`{name} matrices`, function()
		local A = M.matfrom({ { 1, 2 }, { 3, 4 } }, { { 1, 0 }, { 0, -1 } })
		local T = M.transpose(nil, A)
		local H = M.adjoint(nil, A)
		local tr, ti = M.mattotable(T)
		local hr, hi = M.mattotable(H)
		check(tr[1][2] == 3 and ti[2][2] == -1 and hi[2][2] == 1 and hr[1][1] == 1 and hi[1][1] == -1, "transpose/adjoint")
		local r, i = a.get(M.matvec(nil, A, M.from({ 1, 1 })), 2)
		check(r == 7 and i == -1, "matvec")
		local P = M.matmul(nil, A, M.identity(2))
		local pr, pi = M.mattotable(P)
		check(pr[2][2] == 4 and pi[2][2] == -1 and pi[1][1] == 1, "matmul by identity")
		-- det [[1+i, 2], [3, 4-i]] = (1+i)(4-i) - 6 = -1 + 3i
		r, i = a.det(A)
		close(r, -1, tol, "det re")
		close(i, 3, tol, "det im")
	end)

	test(`{name} solve`, function()
		math.randomseed(2)
		for _, n in { 1, 6, 30 } do
			local A = M.matrix(n, n)
			for p = 1, n do
				for q = 1, n do
					a.matset(A, p, q, 2 * math.random() - 1, 2 * math.random() - 1)
				end
			end
			local re, im = {}, {}
			for p = 1, n do
				re[p], im[p] = 2 * math.random() - 1, 2 * math.random() - 1
			end
			local b = M.from(re, im)
			local x = M.solve(nil, A, b)
			local res = M.sub(nil, M.matvec(nil, A, x), b)
			local bound = 50 * n * eps * M.norm(A.data) * M.norm(x)
			check(M.norm(res) <= bound, `residual n={n}: {M.norm(res)} > {bound}`)
		end
		local S = M.matfrom({ { 1, 2 }, { 2, 4 } }, { { 1, 2 }, { 2, 4 } })
		check(M.lu_copy(S).singular, "singular matrix detected")
	end)
end

return nil
