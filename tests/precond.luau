--!strict

-- Sparse preconditioners: their defining properties, and that they help the solvers.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check
local krylov = lib.krylov

type Case = { name: string, M: any, complex: boolean, eps: number }

local cases: { Case } = {
	{ name = "f32", M = lib.f32, complex = false, eps = h.eps32 },
	{ name = "f64", M = lib.f64, complex = false, eps = h.eps64 },
	{ name = "c32", M = lib.c32, complex = true, eps = h.eps32 },
	{ name = "c64", M = lib.c64, complex = true, eps = h.eps64 },
}

local function random_values(n: number): { number }
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = 2 * math.random() - 1
	end
	return t
end

local function values(case: Case, re: { number }, im: { number }?): buffer
	if case.complex then
		return case.M.from(re, im or table.create(#re, 0))
	end
	return case.M.from(re)
end

local function random_vector(case: Case, n: number): buffer
	return values(case, random_values(n), if case.complex then random_values(n) else nil)
end

-- The 5-point Laplacian on a k by k grid plus `shift` on the diagonal and a convection
-- term `conv` (nonsymmetric if nonzero); complex cases may add `ishift` i to the diagonal.
local function laplacian(case: Case, k: number, shift: number, conv: number, ishift: number?): lib.Sparse
	local ii, jj, re, im = {}, {}, {}, {}
	local function add(i: number, j: number, v: number, w: number)
		table.insert(ii, i)
		table.insert(jj, j)
		table.insert(re, v)
		table.insert(im, w)
	end
	for gy = 0, k - 1 do
		for gx = 0, k - 1 do
			local p = gy * k + gx + 1
			add(p, p, 4 + shift, ishift or 0)
			if gx > 0 then
				add(p, p - 1, -1 - conv, 0)
			end
			if gx < k - 1 then
				add(p, p + 1, -1 + conv, 0)
			end
			if gy > 0 then
				add(p, p - k, -1, 0)
			end
			if gy < k - 1 then
				add(p, p + k, -1, 0)
			end
		end
	end
	return case.M.sparse(k * k, k * k, ii, jj, values(case, re, im))
end

-- The dense L (unit lower, strictly below the diagonal of F) and U (on and above).
local function split_lu(case: Case, F: lib.Matrix): (lib.Matrix, lib.Matrix)
	local re, im = case.M.mattotable(F)
	local n = F.rows
	local lr, li, ur, ui = {}, {}, {}, {}
	for i = 1, n do
		lr[i], li[i], ur[i], ui[i] = table.create(n, 0), table.create(n, 0), table.create(n, 0), table.create(n, 0)
		for j = 1, n do
			if i > j then
				lr[i][j], li[i][j] = re[i][j], if im then im[i][j] else 0
			else
				ur[i][j], ui[i][j] = re[i][j], if im then im[i][j] else 0
			end
		end
		lr[i][i] = 1
	end
	if case.complex then
		return case.M.matfrom(lr, li), case.M.matfrom(ur, ui)
	end
	return case.M.matfrom(lr), case.M.matfrom(ur)
end

-- The largest |B_ij - A_ij| over the stored positions of S (which has A's pattern).
local function pattern_error(case: Case, S: lib.Sparse, A: lib.Matrix, B: lib.Matrix, lower_only: boolean): number
	local ar, ai = case.M.mattotable(A)
	local br, bi = case.M.mattotable(B)
	local worst = 0
	for j = 0, S.cols - 1 do
		for p = buffer.readu32(S.colptr, 4 * j), buffer.readu32(S.colptr, 4 * j + 4) - 1 do
			local i = buffer.readu32(S.rowind, 4 * p)
			if not lower_only or i >= j then
				local d = math.abs(br[i + 1][j + 1] - ar[i + 1][j + 1])
				if ai and bi then
					d += math.abs(bi[i + 1][j + 1] - ai[i + 1][j + 1])
				end
				worst = math.max(worst, d)
			end
		end
	end
	return worst
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps
	local tol = 1e3 * eps

	test(`{name} SSOR applies M^-1`, function()
		math.randomseed(111)
		local S = laplacian(case, 5, 0.3, 0.2, if case.complex then 0.4 else nil)
		local A = M.sparse_matrix(S)
		local n = A.rows
		local omega = 1.3
		-- M = (D + w L) D^-1 (D + w U) / (w (2 - w)), built densely.
		local re, im = M.mattotable(A)
		local function part(keep: (number, number) -> boolean, scale: number): lib.Matrix
			local pr, pi = {}, {}
			for i = 1, n do
				pr[i], pi[i] = table.create(n, 0), table.create(n, 0)
				for j = 1, n do
					if keep(i, j) then
						local f = if i == j then 1 else scale
						pr[i][j], pi[i][j] = f * re[i][j], f * (if im then im[i][j] else 0)
					end
				end
			end
			return if case.complex then M.matfrom(pr, pi) else M.matfrom(pr)
		end
		local DL = part(function(i, j)
			return i >= j
		end, omega)
		local DU = part(function(i, j)
			return i <= j
		end, omega)
		local D = part(function(i, j)
			return i == j
		end, 1)
		local r = random_vector(case, n)
		local z = M.sparse_ssor(S, omega)(M.create(n), r)
		-- M z = (D + wL) D^-1 (D + wU) z / (w (2 - w)) should be r.
		local t = M.matvec(nil, DU, z)
		t = M.solve(nil, D, t)
		t = M.matvec(nil, DL, t)
		t = M.scale(nil, 1 / (omega * (2 - omega)), t)
		check(M.norm(M.sub(nil, t, r)) <= tol * M.norm(r), "M (M^-1 r) = r")
	end)

	test(`{name} ILU(0) and IC(0) factorizations`, function()
		math.randomseed(112)
		-- On the pattern of A, ILU(0) reproduces A exactly: (L U)_ij = a_ij.
		local S = laplacian(case, 6, 0.5, 0.3, if case.complex then 0.4 else nil)
		local A = M.sparse_matrix(S)
		local before = M.copy(nil, S.values)
		local F = M.sparse_ilu0_copy(S)
		check(M.norm(M.sub(nil, S.values, before)) == 0, "sparse_ilu0_copy leaves S alone")
		check(not F.singular, "ILU(0) succeeded")
		local L, U = split_lu(case, M.sparse_matrix(F.lu))
		local err = pattern_error(case, S, A, M.matmul(nil, L, U), false)
		check(err <= tol * 8, `ILU(0): (L U)_ij = a_ij on the pattern ({err})`)

		-- On the lower pattern of a Hermitian positive definite A, IC(0) gives (L L^H)_ij = a_ij.
		local P = laplacian(case, 6, 0, 0)
		local Pa = M.sparse_matrix(P)
		local G = M.sparse_ic0_copy(P)
		check(G.definite, "IC(0) succeeded on the Laplacian")
		local lr, li = M.mattotable(M.sparse_matrix(G.ic))
		for i = 1, #lr do
			for j = i + 1, #lr do
				lr[i][j] = 0
				if li then
					li[i][j] = 0
				end
			end
		end
		local Ld = if li then M.matfrom(lr, li) else M.matfrom(lr)
		local LLh = M.matmul(nil, Ld, if case.complex then M.adjoint(nil, Ld) else M.transpose(nil, Ld))
		local cerr = pattern_error(case, P, Pa, LLh, true)
		check(cerr <= tol * 8, `IC(0): (L L^H)_ij = a_ij on the lower pattern ({cerr})`)

		-- With no fill-in to drop (tridiagonal), both are exact.
		local n = 12
		local ii, jj, vv = {}, {}, {}
		for i = 1, n do
			table.insert(ii, i)
			table.insert(jj, i)
			table.insert(vv, 4)
			if i < n then
				for _, pair in { { i, i + 1 }, { i + 1, i } } do
					table.insert(ii, pair[1])
					table.insert(jj, pair[2])
					table.insert(vv, -1)
				end
			end
		end
		local T = M.sparse(n, n, ii, jj, values(case, vv))
		local b = random_vector(case, n)
		local want = M.solve(nil, M.sparse_matrix(T), b)
		check(M.norm(M.sub(nil, M.sparse_ilu0_solve(nil, M.sparse_ilu0_copy(T), b), want)) <= tol * M.norm(want), "tridiagonal: ILU(0) is exact")
		check(M.norm(M.sub(nil, M.sparse_ic0_solve(nil, M.sparse_ic0_copy(T), b), want)) <= tol * M.norm(want), "tridiagonal: IC(0) is exact")

		local Ind = laplacian(case, 3, -4.5, 0)
		check(not M.sparse_ic0_copy(Ind).definite, "IC(0) reports breakdown on an indefinite matrix")
		local NoDiag = M.sparse(2, 2, { 1, 2 }, { 2, 1 }, values(case, { 1, 1 }))
		check(M.sparse_ilu0_copy(NoDiag).singular, "ILU(0) needs the diagonal")
	end)

	test(`{name} preconditioners reduce iterations`, function()
		math.randomseed(113)
		local rtol = if eps > 1e-10 then 1e-4 else 1e-8
		-- Symmetric positive definite: CG with IC(0) and SSOR, MINRES with IC(0).
		local S = laplacian(case, 16, 0, 0)
		local A: lib.Operator<buffer> = function(out, x)
			return M.sparse_matvec(out, S, x)
		end
		local b = random_vector(case, S.rows)
		local IC = M.sparse_ic0_copy(S)
		local ic: lib.Operator<buffer> = function(out, r)
			return M.sparse_ic0_solve(out, IC, r)
		end
		local _, plain = krylov.cg(M.space, A, b, { tol = rtol })
		local _, withic = krylov.cg(M.space, A, b, { tol = rtol, precond = ic })
		local _, withssor = krylov.cg(M.space, A, b, { tol = rtol, precond = M.sparse_ssor(S) })
		check(plain.converged and withic.converged and withssor.converged, "CG converged with each")
		check(withic.iterations < plain.iterations and withssor.iterations < plain.iterations,
			`CG iterations: plain {plain.iterations}, IC(0) {withic.iterations}, SSOR {withssor.iterations}`)
		local _, mplain = krylov.minres(M.space, A, b, { tol = rtol })
		local _, mic = krylov.minres(M.space, A, b, { tol = rtol, precond = ic })
		check(mic.converged and mic.iterations < mplain.iterations, `MINRES iterations: plain {mplain.iterations}, IC(0) {mic.iterations}`)

		-- Symmetric indefinite: MINRES with Jacobi (the diagonal is still positive).
		local Ind = laplacian(case, 10, -3.05, 0)
		local Ai: lib.Operator<buffer> = function(out, x)
			return M.sparse_matvec(out, Ind, x)
		end
		local bi = random_vector(case, Ind.rows)
		local _, ij = krylov.minres(M.space, Ai, bi, { tol = rtol, precond = M.sparse_jacobi(Ind) })
		check(ij.converged, `MINRES with Jacobi on an indefinite matrix ({ij.residual} after {ij.iterations})`)

		-- Nonsymmetric: GMRES and BiCGSTAB with ILU(0).
		local N = laplacian(case, 16, 0.2, 0.4, if case.complex then 0.5 else nil)
		local An: lib.Operator<buffer> = function(out, x)
			return M.sparse_matvec(out, N, x)
		end
		local bn = random_vector(case, N.rows)
		local LU = M.sparse_ilu0_copy(N)
		local ilu: lib.Operator<buffer> = function(out, r)
			return M.sparse_ilu0_solve(out, LU, r)
		end
		local _, g0 = krylov.gmres(M.space, An, bn, { tol = rtol })
		local _, g1 = krylov.gmres(M.space, An, bn, { tol = rtol, precond = ilu })
		check(g1.converged and g1.iterations < g0.iterations, `GMRES iterations: plain {g0.iterations}, ILU(0) {g1.iterations}`)
		local _, s0 = krylov.bicgstab(M.space, An, bn, { tol = rtol })
		local _, s1 = krylov.bicgstab(M.space, An, bn, { tol = rtol, precond = ilu })
		check(s1.converged and s1.iterations < s0.iterations, `BiCGSTAB iterations: plain {s0.iterations}, ILU(0) {s1.iterations}`)
	end)
end

return nil
