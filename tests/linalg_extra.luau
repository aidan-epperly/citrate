--!strict

-- Matrix utilities, norms and condition numbers, rank and subspaces, and the matrix
-- exponential, square root and logarithm.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close

type Case = { name: string, M: any, R: any, C: any, complex: boolean, eps: number }

local cases: { Case } = {
	{ name = "f32", M = lib.f32, R = lib.f32, C = lib.c32, complex = false, eps = h.eps32 },
	{ name = "f64", M = lib.f64, R = lib.f64, C = lib.c64, complex = false, eps = h.eps64 },
	{ name = "c32", M = lib.c32, R = lib.f32, C = lib.c32, complex = true, eps = h.eps32 },
	{ name = "c64", M = lib.c64, R = lib.f64, C = lib.c64, complex = true, eps = h.eps64 },
}

local function random_values(n: number): { number }
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = 2 * math.random() - 1
	end
	return t
end

local function random_vector(case: Case, n: number): buffer
	return if case.complex then case.M.from(random_values(n), random_values(n)) else case.M.from(random_values(n))
end

local function random_matrix(case: Case, m: number, n: number): lib.Matrix
	local A = case.M.matrix(m, n)
	A.data = random_vector(case, m * n)
	return A
end

local function from_rows(case: Case, rows: { { number } }): lib.Matrix
	if case.complex then
		local zeros = {}
		for i, row in rows do
			zeros[i] = table.create(#row, 0)
		end
		return case.M.matfrom(rows, zeros)
	end
	return case.M.matfrom(rows)
end

local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

local function diff(M: any, A: lib.Matrix, B: lib.Matrix): number
	return M.norm(M.sub(nil, A.data, B.data))
end

-- A complex scalar result as (re, im), whatever the module returns.
local function pair(case: Case, ...: any): (number, number)
	local a, b = ...
	if not case.complex then
		return a, 0
	elseif type(a) == "vector" then
		return a.x, a.y
	end
	return a, b
end

for _, case in cases do
	local M, R, name, eps = case.M, case.R, case.name, case.eps
	local tol = 100 * eps

	test(`{name} matrix utilities`, function()
		math.randomseed(401)
		local A = random_matrix(case, 4, 4)
		local tr, ti = pair(case, M.trace(A))
		local d = M.diag(nil, A)
		local sr, si = pair(case, M.sum(d))
		check(math.abs(tr - sr) <= tol and math.abs(ti - si) <= tol, "trace is the sum of the diagonal")
		local D = M.diagm(d)
		check(M.norm(M.sub(nil, M.diag(nil, D), d)) == 0, "diag(diagm(x)) = x")
		local off = M.matcopy(nil, D)
		check(diff(M, D, M.matmul(nil, M.identity(4), off)) == 0, "diagm is diagonal")

		-- kron: the mixed product property (A x B)(C x D) = AC x BD.
		local B, C, Dm = random_matrix(case, 2, 3), random_matrix(case, 4, 2), random_matrix(case, 3, 2)
		local AB = M.kron(A, B)
		check(AB.rows == 8 and AB.cols == 12, "kron shape")
		local lhs = M.matmul(nil, AB, M.kron(C, Dm))
		local rhs = M.kron(M.matmul(nil, A, C), M.matmul(nil, B, Dm))
		check(diff(M, lhs, rhs) <= tol * M.norm(rhs.data), "kron: mixed product property")

		-- outer and ger.
		local x, y = random_vector(case, 4), random_vector(case, 3)
		local X = M.matrix(4, 1)
		X.data = M.copy(nil, x)
		local Y = M.matrix(3, 1)
		Y.data = M.copy(nil, y)
		local O = M.outer(nil, x, y)
		check(diff(M, O, M.matmul(nil, X, adjoint(case, Y))) <= tol, "outer(x, y) = x y^H")
		local G = random_matrix(case, 4, 3)
		local G0 = M.matcopy(nil, G)
		check(M.ger(G, 2.5, x, y) == G, "ger is in place")
		local want = M.matrix(4, 3)
		want.data = M.axpy(nil, 2.5, O.data, G0.data)
		check(diff(M, G, want) <= tol, "ger: A + a x y^H")

		-- Submatrices and stacking.
		local W = random_matrix(case, 5, 6)
		local left, right = M.submatrix(nil, W, 1, 5, 1, 2), M.submatrix(nil, W, 1, 5, 3, 6)
		check(left.cols == 2 and right.cols == 4 and diff(M, M.hstack({ left, right }), W) == 0, "hstack of column blocks")
		local top, bottom = M.submatrix(nil, W, 1, 2, 1, 6), M.submatrix(nil, W, 3, 5, 1, 6)
		check(diff(M, M.vstack({ top, bottom }), W) == 0, "vstack of row blocks")
		local Z = M.matrix(5, 6)
		M.setsubmatrix(Z, 1, 1, M.submatrix(nil, W, 1, 3, 1, 4))
		M.setsubmatrix(Z, 4, 1, M.submatrix(nil, W, 4, 5, 1, 4))
		M.setsubmatrix(Z, 1, 5, M.submatrix(nil, W, 1, 5, 5, 6))
		check(diff(M, Z, W) == 0, "setsubmatrix reassembles the blocks")
		check(not pcall(function()
			M.submatrix(nil, W, 0, 2, 1, 2)
		end) and not pcall(function()
			M.setsubmatrix(Z, 5, 5, W)
		end), "out of range is an error")

		-- matvec_adjoint and gemm.
		local v = random_vector(case, 5)
		check(M.norm(M.sub(nil, M.matvec_adjoint(nil, W, v), M.matvec(nil, adjoint(case, W), v))) <= tol, "matvec_adjoint = A^H x")
		local P, Q = random_matrix(case, 3, 4), random_matrix(case, 4, 2)
		local Cm = random_matrix(case, 3, 2)
		local function op(Mx: lib.Matrix, o: string): lib.Matrix
			if o == "n" then
				return Mx
			elseif o == "t" then
				return M.transpose(nil, Mx)
			end
			return adjoint(case, Mx)
		end
		local all = true
		for _, oa in { "n", "t", "c" } do
			for _, ob in { "n", "t", "c" } do
				local Pa = if oa == "n" then P else op(P, oa)
				local Qb = if ob == "n" then Q else op(Q, ob)
				-- Pass the matrices so that op(.) brings them back to P and Q.
				local got = M.gemm(M.matcopy(nil, Cm), 1.5, Pa, Qb, -0.5, oa, ob)
				local expect = M.matrix(3, 2)
				expect.data = M.axpy(nil, 1.5, M.matmul(nil, op(Pa, oa), op(Qb, ob)).data, M.scale(nil, -0.5, Cm.data))
				all = all and diff(M, got, expect) <= tol
			end
		end
		check(all, "gemm with every combination of ops")
		check(diff(M, M.gemm(nil, 1, P, Q), M.matmul(nil, P, Q)) <= tol, "gemm with beta = 0 makes C")

		-- slogdet: no overflow.
		local S = from_rows(case, { { 0, 2, 0 }, { 0, 0, -3 }, { 4, 0, 0 } })
		local sr2, si2, sl = M.slogdet(S)
		local pr, pi = pair(case, sr2, si2)
		local logabs = if case.complex then (if type(sr2) == "vector" then si2 else sl) else si2
		close(logabs, math.log(24), tol, "slogdet: log |det|")
		local dr, di = pair(case, M.det(S))
		check(math.abs(pr * 24 - dr) <= tol * 24 and math.abs(pi * 24 - di) <= tol * 24, "slogdet: sign times |det| is det")
		local big = M.matrix(200, 200)
		for i = 1, 200 do
			if case.complex then
				if name == "c32" then
					M.matset(big, i, i, vector.create(0, 30, 0))
				else
					M.matset(big, i, i, 0, 30)
				end
			else
				M.matset(big, i, i, -30)
			end
		end
		local b1, b2, b3 = M.slogdet(big)
		local bl = if case.complex then (if type(b1) == "vector" then b2 else b3) else b2
		close(bl, 200 * math.log(30), 1e-12, "slogdet of a 200 x 200 matrix whose det overflows")
		local br, bi = pair(case, b1, if case.complex and type(b1) ~= "vector" then b2 else nil)
		-- (-30)^200 > 0; (30 i)^200 = 30^200 i^200 = 30^200.
		check(math.abs(br - 1) <= 1e-5 and math.abs(bi) <= 1e-5, "slogdet: the sign")
	end)

	test(`{name} norms, condition numbers and subspaces`, function()
		math.randomseed(402)
		local A = from_rows(case, { { 1, -2, 3 }, { -4, 5, -6 } })
		close(M.matnorm(A, 1), 9, tol, "1-norm: largest column sum")
		close(M.matnorm(A, "inf"), 15, tol, "inf-norm: largest row sum")
		close(M.matnorm(A), math.sqrt(91), tol, "Frobenius norm (default)")
		local s = R.totable(M.svd_copy(A, false).values)
		close(M.matnorm(A, 2), s[1], tol, "2-norm: largest singular value")
		check(not pcall(function()
			M.matnorm(A, 3)
		end), "unknown norm is an error")

		local Dg = from_rows(case, { { 1, 0, 0 }, { 0, 1e-2, 0 }, { 0, 0, 1e-3 } })
		close(M.cond(Dg), 1e3, 1e3 * eps, "cond: s_max / s_min")
		close(M.cond(Dg, 1), 1e3, 1e3 * eps, "cond in the 1-norm")
		check(M.cond(from_rows(case, { { 1, 2 }, { 2, 4 } })) >= 1 / (10 * eps), "cond of a singular matrix is huge")

		-- rcond: an estimate of 1 / cond_1, never below the truth, within a factor of 3.
		local ok = true
		for _, n in { 1, 2, 5, 12, 30 } do
			local B = random_matrix(case, n, n)
			local truth = 1 / M.cond(B, 1)
			local est = M.rcond(B)
			ok = ok and est >= truth * (1 - 1e3 * eps) and est <= 3 * truth
		end
		check(ok, "rcond within [truth, 3 truth]")
		check(M.rcond(from_rows(case, { { 1, 2 }, { 2, 4 } })) == 0, "rcond of a singular matrix is 0")

		-- lu_solve_adjoint.
		local B = random_matrix(case, 6, 6)
		local b = random_vector(case, 6)
		local x = M.lu_solve_adjoint(nil, M.lu_copy(B), b)
		check(M.norm(M.sub(nil, M.matvec(nil, adjoint(case, B), x), b)) <= tol * 10, "lu_solve_adjoint solves A^H x = b")

		-- Rank and subspaces.
		local L = M.matmul(nil, random_matrix(case, 7, 3), random_matrix(case, 3, 5))
		check(M.rank(L) == 3 and M.rank(random_matrix(case, 4, 6)) == 4 and M.rank(M.matrix(3, 3)) == 0, "rank")
		local Ob = M.orth(L)
		check(Ob.rows == 7 and Ob.cols == 3, "orth: rows by rank")
		check(diff(M, M.matmul(nil, adjoint(case, Ob), Ob), M.identity(3)) <= tol, "orth: orthonormal")
		local proj = M.matmul(nil, Ob, M.matmul(nil, adjoint(case, Ob), L))
		check(diff(M, proj, L) <= tol * M.norm(L.data), "orth spans the range")
		local N = M.null_space(L)
		check(N.rows == 5 and N.cols == 2, "null_space: cols by cols - rank")
		check(diff(M, M.matmul(nil, adjoint(case, N), N), M.identity(2)) <= tol, "null_space: orthonormal")
		check(M.norm(M.matmul(nil, L, N).data) <= tol * M.norm(L.data), "A N = 0")
		local Wd = random_matrix(case, 3, 6)
		local Nw = M.null_space(Wd)
		check(Nw.cols == 3 and M.norm(M.matmul(nil, Wd, Nw).data) <= tol * M.norm(Wd.data), "null_space of a wide matrix")
		check(diff(M, M.null_space(M.matrix(2, 4)), M.identity(4)) == 0, "null_space of zero is everything")
		check(M.null_space(random_matrix(case, 5, 3)).cols == 0, "full column rank: trivial null space")
	end)

	test(`{name} matrix exponential`, function()
		math.randomseed(403)
		local n = 5
		check(diff(M, M.expm(M.matrix(n, n)), M.identity(n)) == 0, "expm(0) = I")
		local Dg = from_rows(case, { { 1, 0 }, { 0, -2 } })
		local E = M.expm(Dg)
		local er = M.mattotable(E)
		close(er[1][1], math.exp(1), tol, "expm of a diagonal matrix")
		close(er[2][2], math.exp(-2), tol)
		local Nil = M.expm(from_rows(case, { { 0, 1 }, { 0, 0 } }))
		check(diff(M, Nil, from_rows(case, { { 1, 1 }, { 0, 1 } })) <= tol, "nilpotent: I + N exactly")
		local th = 2.3
		local Rot = M.expm(from_rows(case, { { 0, -th }, { th, 0 } }))
		check(diff(M, Rot, from_rows(case, { { math.cos(th), -math.sin(th) }, { math.sin(th), math.cos(th) } })) <= tol, "rotation generator")
		-- Every degree and the scaling path: exp(A) exp(-A) = I and exp(2A) = exp(A)^2.
		for _, scale in { 0.005, 0.1, 0.5, 1.5, 4, 40 } do
			local A = random_matrix(case, n, n)
			A.data = M.scale(nil, scale / M.matnorm(A, 1), A.data)
			local X = M.expm(A)
			local negA = M.matrix(n, n)
			negA.data = M.scale(nil, -1, A.data)
			local inv = M.matmul(nil, X, M.expm(negA))
			local growth = M.matnorm(X, 1) * M.matnorm(M.expm(negA), 1)
			check(diff(M, inv, M.identity(n)) <= 30 * tol * growth, `|A|_1 = {scale}: exp(A) exp(-A) = I`)
			local twoA = M.matrix(n, n)
			twoA.data = M.scale(nil, 2, A.data)
			check(diff(M, M.expm(twoA), M.matmul(nil, X, X)) <= 30 * tol * M.norm(X.data) ^ 2, `|A|_1 = {scale}: exp(2A) = exp(A)^2`)
		end
		-- Against the eigendecomposition V exp(L) V^-1 of a diagonalizable matrix.
		local A = random_matrix(case, 4, 4)
		local F = M.eig_copy(A)
		local C = case.C
		local V = F.vectors :: lib.Matrix
		local re, im = C.totable(F.values)
		local er2, ei2 = {}, {}
		for i = 1, 4 do
			er2[i], ei2[i] = table.create(4, 0), table.create(4, 0)
			er2[i][i] = math.exp(re[i]) * math.cos(im[i])
			ei2[i][i] = math.exp(re[i]) * math.sin(im[i])
		end
		local viaEig = C.matmul(nil, C.matmul(nil, V, C.matfrom(er2, ei2)), C.inv_copy(V))
		local X = M.expm(A)
		local Xc = if case.complex then X else (function()
			local Z = C.matrix(4, 4)
			Z.data = C.complexify(X.data)
			return Z
		end)()
		check(C.norm(C.sub(nil, Xc.data, viaEig.data)) <= 1e3 * eps * C.norm(viaEig.data), "agrees with V exp(L) V^-1")
		local nanA = random_matrix(case, 2, 2)
		nanA.data = M.scale(nil, 0 / 0, nanA.data)
		check(h.all_nan(M.expm(nanA).data, eps == h.eps32), "NaN gives NaN")
	end)

	test(`{name} matrix square root and logarithm`, function()
		math.randomseed(404)
		local n = 5
		-- A nonsymmetric matrix with eigenvalues in the right half plane.
		local A = random_matrix(case, n, n)
		A.data = M.scale(nil, 0.3, A.data)
		A.data = M.axpy(nil, 2, M.identity(n).data, A.data)
		local X = M.sqrtm(A)
		check(X.rows == n and getmetatable(X :: any) == nil, "sqrtm returns a matrix of the same module")
		check(diff(M, M.matmul(nil, X, X), A) <= tol * M.norm(A.data), "sqrtm(A)^2 = A")
		local Lg = M.logm(A)
		check(diff(M, M.expm(Lg), A) <= 10 * tol * M.norm(A.data), "expm(logm(A)) = A")
		local B = random_matrix(case, n, n)
		B.data = M.scale(nil, 0.5 / M.matnorm(B, 2), B.data)
		check(diff(M, M.logm(M.expm(B)), B) <= 10 * tol, "logm(expm(B)) = B for small B")
		check(M.norm(M.logm(M.identity(n)).data) <= tol, "logm(I) = 0")
		-- Symmetric positive definite: the square root is symmetric positive definite.
		local G = random_matrix(case, n, n)
		local Spd = M.matmul(nil, adjoint(case, G), G)
		Spd.data = M.axpy(nil, 1, M.identity(n).data, Spd.data)
		local Sq = M.sqrtm(Spd)
		check(diff(M, Sq, adjoint(case, Sq)) <= tol * M.norm(Sq.data), "SPD: the square root is Hermitian")
		local values = R.totable(M.eigh_copy(Sq, false).values)
		check(values[1] > 0, "SPD: the square root is positive definite")
		-- Large norm: many square roots in logm.
		local Big = M.matrix(n, n)
		Big.data = M.scale(nil, 1e6, A.data)
		check(diff(M, M.expm(M.logm(Big)), Big) <= 1e3 * tol * M.norm(Big.data), "logm of a large matrix")

		local Neg = from_rows(case, { { -4, 0 }, { 0, 9 } })
		if case.complex then
			local Xn = M.sqrtm(Neg)
			check(diff(M, M.matmul(nil, Xn, Xn), Neg) <= tol * 10, "complex: square root of a negative eigenvalue")
			local Ln = M.logm(Neg)
			check(diff(M, M.expm(Ln), Neg) <= tol * 100, "complex: logarithm of a negative eigenvalue")
		else
			check(not pcall(function()
				M.sqrtm(Neg)
			end) and not pcall(function()
				M.logm(Neg)
			end), "real: negative eigenvalues are an error")
		end
		check(not pcall(function()
			M.logm(from_rows(case, { { 1, 2 }, { 2, 4 } }))
		end), "logm of a singular matrix is an error")
	end)
end

return nil
