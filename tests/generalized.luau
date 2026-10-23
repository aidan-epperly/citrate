--!strict

-- Symmetric indefinite LDL^T, generalized eigenproblems (Hermitian-definite and QZ), and
-- Sylvester and Lyapunov equations.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

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

-- A random Hermitian matrix (indefinite in general).
local function hermitian(case: Case, n: number): lib.Matrix
	local B = random_matrix(case, n, n)
	local A = adjoint(case, B)
	A.data = case.M.add(nil, A.data, B.data)
	return A
end

-- As a complex matrix of the same precision.
local function as_complex(case: Case, A: lib.Matrix): lib.Matrix
	if case.complex then
		return A
	end
	local Z = case.C.matrix(A.rows, A.cols)
	Z.data = case.C.complexify(A.data)
	return Z
end

for _, case in cases do
	local M, R, C, name, eps = case.M, case.R, case.C, case.name, case.eps
	local tol = 100 * eps

	test(`{name} Bunch-Kaufman LDL^T`, function()
		math.randomseed(501)
		for _, n in { 1, 2, 5, 12, 30 } do
			local A = hermitian(case, n)
			local A0 = M.matcopy(nil, A)
			local F = M.ldl(A)
			check(F.ldl == A and not F.singular, `n={n}: in place, nonsingular`)
			local b = random_vector(case, n)
			local x = M.ldl_solve(nil, F, b)
			local r = M.norm(M.sub(nil, M.matvec(nil, A0, x), b))
			check(r <= 30 * n * eps * M.norm(A0.data) * M.norm(x), `n={n}: residual {r}`)
			check(M.norm(M.sub(nil, M.solve(nil, F, b, "ldl"), x)) == 0, `n={n}: form "ldl"`)
			-- Inertia agrees with the eigenvalues.
			local pos, neg, zero = M.ldl_inertia(F)
			local values = R.totable(M.eigh_copy(A0, false).values)
			local epos, eneg = 0, 0
			for _, v in values do
				if v > 0 then
					epos += 1
				elseif v < 0 then
					eneg += 1
				end
			end
			check(pos == epos and neg == eneg and zero == 0, `n={n}: inertia ({pos}, {neg}, {zero})`)
		end
		-- Zero diagonals force 2 by 2 pivots.
		for _, rows in { { { 0, 1 }, { 1, 0 } }, { { 0, 1, 2 }, { 1, 0, 3 }, { 2, 3, 0 } } } do
			local A = from_rows(case, rows)
			local F = M.ldl_copy(A)
			local twobytwo = false
			for _, p in F.piv do
				twobytwo = twobytwo or p < 0
			end
			local b = random_vector(case, #rows)
			local x = M.ldl_solve(nil, F, b)
			check(twobytwo and M.norm(M.sub(nil, M.matvec(nil, A, x), b)) <= tol, `zero diagonal ({#rows}): a 2 by 2 pivot, and the solve works`)
		end
		-- Only the lower triangle is read.
		local A = hermitian(case, 6)
		local re, im = M.mattotable(A)
		for i = 1, 6 do
			for j = i + 1, 6 do
				re[i][j] = 0 / 0
				if im then
					im[i][j] = 0 / 0
				end
			end
		end
		local messy = if im then M.matfrom(re, im) else M.matfrom(re)
		local b = random_vector(case, 6)
		check(M.norm(M.sub(nil, M.ldl_solve(nil, M.ldl(messy), b), M.ldl_solve(nil, M.ldl_copy(A), b))) == 0, "reads only the lower triangle")
		-- Matrix right-hand sides, copies and singular matrices.
		local B = random_matrix(case, 6, 3)
		local X = M.solve(nil, M.ldl_copy(A), B, "ldl")
		check(diff(M, M.matmul(nil, A, X), B) <= 30 * tol * M.norm(X.data), "matrix right-hand side")
		local A0 = M.matcopy(nil, A)
		M.ldl_copy(A)
		check(diff(M, A, A0) == 0, "ldl_copy leaves A alone")
		check(M.ldl_copy(M.matrix(3, 3)).singular, "the zero matrix is singular")
	end)

	test(`{name} Hermitian-definite generalized eigenproblem`, function()
		math.randomseed(502)
		local n = 6
		local A = hermitian(case, n)
		local G = random_matrix(case, n, n)
		local B = M.matmul(nil, adjoint(case, G), G)
		B.data = M.axpy(nil, 1, M.identity(n).data, B.data)
		local A0, B0 = M.matcopy(nil, A), M.matcopy(nil, B)
		local F = M.eigh_gen(A, B)
		check(F.vectors == A, "eigenvectors overwrite A")
		local X = F.vectors :: lib.Matrix
		local values = R.totable(F.values)
		local ascending = true
		for i = 2, n do
			ascending = ascending and values[i] >= values[i - 1]
		end
		check(ascending, "ascending")
		-- A X = B X L and X^H B X = I.
		local rows = {}
		for i = 1, n do
			rows[i] = table.create(n, 0)
			rows[i][i] = values[i]
		end
		local L = from_rows(case, rows)
		local res = diff(M, M.matmul(nil, A0, X), M.matmul(nil, M.matmul(nil, B0, X), L))
		check(res <= 30 * tol * M.norm(A0.data) * M.norm(X.data), `A X = B X L ({res})`)
		check(diff(M, M.matmul(nil, M.matmul(nil, adjoint(case, X), B0), X), M.identity(n)) <= 30 * tol, "X^H B X = I")
		local only = R.totable(M.eigh_gen_copy(A0, B0, false).values)
		local agree = true
		for i = 1, n do
			agree = agree and math.abs(only[i] - values[i]) <= 30 * tol * (1 + math.abs(values[i]))
		end
		check(agree, "values without vectors agree")
		check(not pcall(function()
			M.eigh_gen_copy(A0, from_rows(case, { { 1, 0 }, { 0, -1 } }))
		end), "B must be positive definite")
	end)

	test(`{name} QZ generalized eigenproblem`, function()
		math.randomseed(503)
		-- |beta A v - alpha B v| relative to |beta| |A| + |alpha| |B|, for every pair.
		local function residual(A: lib.Matrix, B: lib.Matrix, F: lib.GenEig): number
			local Ac, Bc = as_complex(case, A), as_complex(case, B)
			local V = F.vectors :: lib.Matrix
			local n = A.rows
			local are, aim = C.totable(F.alpha)
			local bre, bim = C.totable(F.beta)
			local worst = 0
			for k = 1, n do
				local v = C.create(n)
				buffer.copy(v, 0, V.data, buffer.len(v) * (k - 1), buffer.len(v))
				local Av, Bv = C.matvec(nil, Ac, v), C.matvec(nil, Bc, v)
				local alpha = if name == "c32" or name == "f32" then vector.create(are[k], aim[k], 0) else nil
				local beta = if alpha then vector.create(bre[k], bim[k], 0) else nil
				local lhs = if alpha
					then C.sub(nil, C.cscale(nil, beta, Av), C.cscale(nil, alpha, Bv))
					else C.sub(nil, C.cscale(nil, bre[k], bim[k], Av), C.cscale(nil, are[k], aim[k], Bv))
				local scale = math.sqrt(bre[k] ^ 2 + bim[k] ^ 2) * C.norm(Ac.data) + math.sqrt(are[k] ^ 2 + aim[k] ^ 2) * C.norm(Bc.data)
				worst = math.max(worst, C.norm(lhs) / scale)
				worst = math.max(worst, math.abs(C.norm(v) - 1))
			end
			return worst
		end
		for _, n in { 1, 3, 8, 20 } do
			local A, B = random_matrix(case, n, n), random_matrix(case, n, n)
			local F = M.eig_gen_copy(A, B)
			local r = residual(A, B, F)
			check(r <= 30 * n * eps, `n={n}: residual {r}`)
		end
		-- B = I gives the standard eigenvalues.
		local A = random_matrix(case, 5, 5)
		local F = M.eig_gen_copy(A, M.identity(5), false)
		local E = M.eig_copy(A, false)
		local fr, fi = C.totable(F.values)
		local er, ei = C.totable(E.values)
		local matched = true
		for k = 1, 5 do
			local best = math.huge
			for j = 1, 5 do
				best = math.min(best, math.abs(fr[k] - er[j]) + math.abs(fi[k] - ei[j]))
			end
			matched = matched and best <= 1e3 * eps
		end
		check(matched, "B = I: the eigenvalues of A")
		-- A singular B gives infinite eigenvalues, also with the zero inside the matrix
		-- (chased to the top of the active block).
		for _, zeros in { { 3 }, { 2 }, { 1, 3 } } do
			local rows = { { 1, 0.5, 0, 0 }, { 0, 1, 0.25, 0 }, { 0, 0, 1, 0.5 }, { 0, 0, 0, 1 } }
			for _, z in zeros do
				for j = 1, 4 do
					rows[z][j] = 0
				end
			end
			local Bs = from_rows(case, rows)
			local As = random_matrix(case, 4, 4)
			local Fs = M.eig_gen_copy(As, Bs)
			local br, bi = C.totable(Fs.beta)
			local vr = C.totable(Fs.values)
			local infinite = 0
			for k = 1, 4 do
				if br[k] == 0 and bi[k] == 0 then
					infinite += 1
					check(vr[k] == math.huge, "infinite eigenvalue reported as inf")
				end
			end
			check(infinite == #zeros, `singular B ({#zeros} zero rows): {infinite} infinite eigenvalues`)
			check(residual(As, Bs, Fs) <= 3e3 * eps, "singular B: residual")
		end
		local nanA = random_matrix(case, 3, 3)
		nanA.data = M.scale(nil, 0 / 0, nanA.data)
		check(h.all_nan(M.eig_gen_copy(nanA, M.identity(3)).values, eps == h.eps32), "NaN gives NaN")

		-- The generalized Schur form itself (complex modules): A = Q S Z^H, B = Q P Z^H
		-- with S, P upper triangular and Q, Z unitary, also with a singular B.
		if case.complex then
			for _, singular in { false, true } do
				local n = 7
				local A0, B0 = random_matrix(case, n, n), random_matrix(case, n, n)
				if singular then
					local re, im = M.mattotable(B0)
					for j = 1, n do
						re[3][j], im[3][j] = 0, 0
						re[j][5], im[j][5] = 0, 0
					end
					B0 = M.matfrom(re, im)
				end
				local S, P = M.matcopy(nil, A0), M.matcopy(nil, B0)
				local Q, Z = M.qz(S, P, true)
				local triangular = true
				local sr, si = M.mattotable(S)
				local pr, pi = M.mattotable(P)
				for i = 1, n do
					for j = 1, i - 1 do
						triangular = triangular and sr[i][j] == 0 and si[i][j] == 0 and pr[i][j] == 0 and pi[i][j] == 0
					end
				end
				local label = if singular then "singular B" else "random"
				check(triangular, `qz ({label}): S and P are upper triangular`)
				local Zh = M.adjoint(nil, Z)
				local ra = diff(M, M.matmul(nil, M.matmul(nil, Q, S), Zh), A0) / M.norm(A0.data)
				local rb = diff(M, M.matmul(nil, M.matmul(nil, Q, P), Zh), B0) / M.norm(B0.data)
				check(ra <= 30 * n * eps and rb <= 30 * n * eps, `qz ({label}): A = Q S Z^H ({ra}), B = Q P Z^H ({rb})`)
				local uq = diff(M, M.matmul(nil, M.adjoint(nil, Q), Q), M.identity(n))
				local uz = diff(M, M.matmul(nil, Zh, Z), M.identity(n))
				check(uq <= 30 * n * eps and uz <= 30 * n * eps, `qz ({label}): Q and Z are unitary`)
			end
		end
	end)

	test(`{name} Sylvester and Lyapunov equations`, function()
		math.randomseed(504)
		local m, n = 5, 3
		local A = random_matrix(case, m, m)
		A.data = M.axpy(nil, 3, M.identity(m).data, A.data)
		local B = random_matrix(case, n, n)
		B.data = M.axpy(nil, 3, M.identity(n).data, B.data)
		local Cm = random_matrix(case, m, n)
		local X = M.sylvester(A, B, Cm)
		check(X.rows == m and X.cols == n, "shape")
		local res = M.norm(M.sub(nil, M.add(nil, M.matmul(nil, A, X).data, M.matmul(nil, X, B).data), Cm.data))
		check(res <= 30 * tol * M.norm(Cm.data), `A X + X B = C ({res})`)
		-- Lyapunov: a stable A and a positive definite Q give a negative definite X
		-- (A X + X A^H = Q has X = -integral of e^(At) Q e^(A^H t)).
		local S = random_matrix(case, 4, 4)
		S.data = M.axpy(nil, -4, M.identity(4).data, S.data)
		local G = random_matrix(case, 4, 4)
		local Q = M.matmul(nil, G, adjoint(case, G))
		Q.data = M.axpy(nil, 1, M.identity(4).data, Q.data)
		local Xl = M.lyapunov(S, Q)
		local lres = M.norm(M.sub(nil, M.add(nil, M.matmul(nil, S, Xl).data, M.matmul(nil, Xl, adjoint(case, S)).data), Q.data))
		check(lres <= 30 * tol * M.norm(Q.data), `A X + X A^H = Q ({lres})`)
		check(diff(M, Xl, adjoint(case, Xl)) <= 30 * tol * M.norm(Xl.data), "X is Hermitian")
		check(R.totable(M.eigh_copy(Xl, false).values)[4] < 0, "X is negative definite")
		check(not pcall(function()
			M.sylvester(from_rows(case, { { 1 } }), from_rows(case, { { -1 } }), from_rows(case, { { 1 } }))
		end), "A and -B sharing an eigenvalue is an error")
	end)
end

return nil
