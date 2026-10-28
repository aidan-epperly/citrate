--!strict

-- reuse for eigh, eig and svd (and their _copy forms) on all four types: reused results
-- equal fresh ones, keep their arrays, survive a change of size, and leave A unchanged.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

type Case = { M: any, C: any, complex: boolean, eps: number, name: string }
local cases: { Case } = {
	{ M = lib.f64, C = lib.c64, complex = false, eps = h.eps64, name = "f64" },
	{ M = lib.f32, C = lib.c32, complex = false, eps = h.eps32, name = "f32" },
	{ M = lib.c64, C = lib.c64, complex = true, eps = h.eps64, name = "c64" },
	{ M = lib.c32, C = lib.c32, complex = true, eps = h.eps32, name = "c32" },
}

local seed = 17
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647 - 0.5
end

local function random_matrix(case: Case, m: number, n: number, hermitian: boolean): any
	local M = case.M
	local A = M.matrix(m, n)
	for j = 1, n do
		for i = 1, m do
			if case.name == "c32" then
				M.matset(A, i, j, vector.create(rnd(), rnd(), 0))
			elseif case.complex then
				M.matset(A, i, j, rnd(), rnd())
			else
				M.matset(A, i, j, rnd())
			end
		end
	end
	if hermitian then
		local H = M.add(nil, A.data, (if case.complex then M.adjoint(nil, A) else M.transpose(nil, A)).data)
		return { rows = m, cols = n, data = H }
	end
	return A
end

-- the largest difference between two arrays of the same type, by their norm
local function diff(M: any, a: buffer, b: buffer): number
	return M.norm(M.sub(nil, a, b)) / math.max(M.norm(a), 1e-300)
end

test("reuse: eigh, eig and svd refill earlier results", function()
	for _, case in cases do
		local M = case.M
		local R = if case.complex then M else (if case.name == "f64" then lib.f64 else lib.f32)
		local tol = 100 * case.eps
		for _, n in { 7, 12, 7 } do
			-- eigh
			local S = random_matrix(case, n, n, true)
			local before = M.copy(nil, S.data)
			local fresh = M.eigh_copy(S)
			local E0 = M.eigh_copy(random_matrix(case, n, n, true))
			local E = M.eigh_copy(S, true, E0)
			check(E.values == E0.values and E.vectors == E0.vectors, case.name .. ": eigh arrays kept")
			check(R.norm(R.sub(nil, E.values, fresh.values)) <= tol * R.norm(fresh.values), case.name .. ": eigh values")
			check(diff(M, S.data, before) == 0, case.name .. ": A unchanged")
			-- values only
			local Ev = M.eigh_copy(S, false, M.eigh_copy(S, false))
			check(R.norm(R.sub(nil, Ev.values, fresh.values)) <= tol * R.norm(fresh.values), case.name .. ": eigh values only")
			-- eig
			local A = random_matrix(case, n, n, false)
			local G0 = M.eig_copy(random_matrix(case, n, n, false))
			local G = M.eig_copy(A, true, true, G0)
			local Gf = M.eig_copy(A)
			check(G.values == G0.values and G.vectors == G0.vectors, case.name .. ": eig arrays kept")
			local C = case.C
			check(diff(C, G.values, Gf.values) <= tol, case.name .. ": eig values")
			check(diff(C, G.vectors.data, Gf.vectors.data) <= 1000 * case.eps, case.name .. ": eig vectors")
			-- svd, tall and wide
			for _, shape in { { n + 3, n }, { n, n + 2 } } do
				local B = random_matrix(case, shape[1], shape[2], false)
				local F0 = M.svd_copy(random_matrix(case, shape[1], shape[2], false))
				local F = M.svd_copy(B, true, F0)
				local Ff = M.svd_copy(B)
				check(F.values == F0.values, case.name .. ": svd values kept")
				check(R.norm(R.sub(nil, F.values, Ff.values)) <= tol * R.norm(Ff.values), case.name .. string.format(": svd %d by %d", shape[1], shape[2]))
				check(diff(M, F.u.data, Ff.u.data) <= 1000 * case.eps and diff(M, F.v.data, Ff.v.data) <= 1000 * case.eps, case.name .. ": svd vectors")
			end
		end
	end
end)

return nil
