--!strict

-- Randomly pivoted Cholesky and QR.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

type Case = { name: string, M: any, R: any, complex: boolean, eps: number }

local cases: { Case } = {
	{ name = "f32", M = lib.f32, R = lib.f32, complex = false, eps = h.eps32 },
	{ name = "f64", M = lib.f64, R = lib.f64, complex = false, eps = h.eps64 },
	{ name = "c32", M = lib.c32, R = lib.f32, complex = true, eps = h.eps32 },
	{ name = "c64", M = lib.c64, R = lib.f64, complex = true, eps = h.eps64 },
}

local function random_values(n: number): { number }
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = 2 * math.random() - 1
	end
	return t
end

local function random_matrix(case: Case, m: number, n: number): lib.Matrix
	local A = case.M.matrix(m, n)
	A.data = if case.complex
		then case.M.from(random_values(m * n), random_values(m * n))
		else case.M.from(random_values(m * n))
	return A
end

local function adjoint(case: Case, A: lib.Matrix): lib.Matrix
	return if case.complex then case.M.adjoint(nil, A) else case.M.transpose(nil, A)
end

local function diff(M: any, A: lib.Matrix, B: lib.Matrix): number
	return M.norm(M.sub(nil, A.data, B.data))
end

-- Q diag(values) Q^H for a random unitary Q.
local function with_spectrum(case: Case, n: number, values: { number }): lib.Matrix
	local M = case.M
	local Q = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
	local rows, zeros = {}, {}
	for i = 1, n do
		rows[i], zeros[i] = table.create(n, 0), table.create(n, 0)
		rows[i][i] = values[i] or 0
	end
	local D = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
	return M.matmul(nil, M.matmul(nil, Q, D), adjoint(case, Q))
end

local function trace_real(M: any, A: lib.Matrix): number
	local t = M.trace(A)
	return if type(t) == "vector" then (t :: any).x else t
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps
	local tol = 100 * eps

	test(`{name} randomly pivoted Cholesky`, function()
		math.randomseed(901)
		local n = 40
		-- Exactly rank 5: five steps recover A, and the tolerance stops it there.
		local G = random_matrix(case, n, 5)
		local A = M.matmul(nil, G, adjoint(case, G))
		local A0 = M.matcopy(nil, A)
		local F = M.rpcholesky(A, 10, { tol = 1e3 * eps, seed = 1 })
		check(diff(M, A, A0) == 0, "A is not modified")
		check(F.rank == 5 and F.factor.cols == 5 and #F.pivots == 5, `rank 5 found (got {F.rank})`)
		local FF = M.matmul(nil, F.factor, adjoint(case, F.factor))
		check(diff(M, FF, A) <= tol * M.norm(A.data), "rank 5: F F^H = A")
		local distinct = {}
		local ok = true
		for _, s in F.pivots do
			ok = ok and not distinct[s]
			distinct[s] = true
		end
		check(ok, "pivots are distinct")
		-- The rows of F at the pivots form a lower triangular matrix (it is a Cholesky
		-- factorization of A(S, S)).
		local re, im = M.mattotable(F.factor)
		local lower = 0
		for j, s in F.pivots do
			for l = j + 1, F.rank do
				lower = math.max(lower, math.abs(re[s][l]), if im then math.abs(im[s][l]) else 0)
			end
		end
		check(lower <= tol, `pivot rows are lower triangular ({lower})`)

		-- A decaying spectrum 2^-i: the trace error falls with k and stays near the best.
		local values = {}
		for i = 1, n do
			values[i] = 2 ^ -(i - 1)
		end
		local B = with_spectrum(case, n, values)
		local traceB = trace_real(M, B)
		local last = math.huge
		local fine = true
		for _, k in { 5, 10, 20 } do
			local P = M.rpcholesky(B, k, { seed = 2 })
			-- The residual is the trace of B - F F^H, computed directly.
			local E = M.matrix(n, n)
			E.data = M.sub(nil, B.data, M.matmul(nil, P.factor, adjoint(case, P.factor)).data)
			fine = fine and math.abs(trace_real(M, E) - P.residual) <= 1e3 * eps * traceB
			-- Best rank-k trace error: sum of the eigenvalues past k, about 2^-(k-1).
			local best = 2 ^ -(k - 1)
			fine = fine and P.residual <= 20 * best + 1e3 * eps * traceB and P.residual <= last
			last = P.residual
		end
		check(fine, "decaying spectrum: the residual is the trace error, near the best, and falls with k")
		local T = M.rpcholesky(B, n, { tol = 1e-3 })
		check(T.residual <= 1e-3 * traceB and T.rank < n, `tol stops it early (rank {T.rank})`)

		-- Matrix-free: the same seed gives the same pivots and factor.
		local diag = {}
		local dre = M.mattotable(B)
		for i = 1, n do
			diag[i] = dre[i][i]
		end
		local requested = 0
		local C = M.rpcholesky_columns(n, diag, function(j: number): buffer
			requested += 1
			local col = M.create(n)
			buffer.copy(col, 0, B.data, buffer.len(col) * (j - 1), buffer.len(col))
			return col
		end, 8, { seed = 3 })
		local D = M.rpcholesky(B, 8, { seed = 3 })
		local same = #C.pivots == #D.pivots and diff(M, C.factor, D.factor) == 0
		for i, s in C.pivots do
			same = same and D.pivots[i] == s
		end
		check(same, "rpcholesky_columns matches rpcholesky")
		check(requested == 8, `only the chosen columns are requested ({requested})`)
		local E1, E2 = M.rpcholesky(B, 6, { seed = 4 }), M.rpcholesky(B, 6, { seed = 4 })
		check(diff(M, E1.factor, E2.factor) == 0, "a seed makes it reproducible")
		check(M.rpcholesky(M.matrix(5, 5), 3).rank == 0, "the zero matrix: rank 0")
		-- Coherent: diag(2^-i) unrotated. Sampling by the residual diagonal picks the big
		-- entries; uniform sampling would mostly pick tiny ones.
		local rows, zeros = {}, {}
		for i = 1, n do
			rows[i], zeros[i] = table.create(n, 0), table.create(n, 0)
			rows[i][i] = 2 ^ -(i - 1)
		end
		local Dg = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
		local worst = 0
		for seed = 10, 19 do
			worst = math.max(worst, M.rpcholesky(Dg, 5, { seed = seed }).residual)
		end
		check(worst <= 0.5, `coherent matrix: residual after 5 steps at most {worst} (uniform sampling: about 1.9)`)
	end)

	test(`{name} randomly pivoted QR`, function()
		math.randomseed(902)
		local m, n = 30, 20
		-- Exactly rank 4.
		local A = M.matmul(nil, random_matrix(case, m, 4), random_matrix(case, 4, n))
		local A0 = M.matcopy(nil, A)
		local F = M.rpqr(A, 8, { tol = 1e3 * eps, seed = 1 })
		check(diff(M, A, A0) == 0, "A is not modified")
		check(F.rank == 4 and F.q.cols == 4 and F.r.rows == 4 and F.r.cols == n, `rank 4 found (got {F.rank})`)
		check(diff(M, M.matmul(nil, F.q, F.r), A) <= tol * M.norm(A.data), "rank 4: Q R = A")
		check(diff(M, M.matmul(nil, adjoint(case, F.q), F.q), M.identity(4)) <= tol, "Q is orthonormal")
		check(diff(M, F.r, M.matmul(nil, adjoint(case, F.q), A)) <= tol * M.norm(A.data), "R = Q^H A")
		-- The chosen columns lie in the span of Q.
		local Sel = M.matrix(m, #F.pivots)
		for l, s in F.pivots do
			buffer.copy(Sel.data, buffer.len(Sel.data) // #F.pivots * (l - 1), A.data, buffer.len(A.data) // n * (s - 1), buffer.len(A.data) // n)
		end
		local proj = M.matmul(nil, F.q, M.matmul(nil, adjoint(case, F.q), Sel))
		check(diff(M, proj, Sel) <= tol * M.norm(Sel.data), "Q spans the chosen columns")

		-- Decaying singular values 2^-i: the residual is |A - Q R|_F^2 and near the best.
		local values = {}
		for i = 1, n do
			values[i] = 2 ^ -(i - 1)
		end
		local U = M.qr_q(nil, M.qr(random_matrix(case, m, n)))
		local V = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
		local rows, zeros = {}, {}
		for i = 1, n do
			rows[i], zeros[i] = table.create(n, 0), table.create(n, 0)
			rows[i][i] = values[i]
		end
		local Dg = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
		local B = M.matmul(nil, M.matmul(nil, U, Dg), adjoint(case, V))
		local total = M.norm(B.data) ^ 2
		local fine = true
		local last = math.huge
		for _, k in { 3, 6, 12 } do
			local P = M.rpqr(B, k, { seed = 5 })
			-- Late columns are nearly dependent on earlier ones, so this needs the second
			-- Gram-Schmidt pass.
			fine = fine and diff(M, M.matmul(nil, adjoint(case, P.q), P.q), M.identity(P.rank)) <= tol
			local err = diff(M, M.matmul(nil, P.q, P.r), B) ^ 2
			fine = fine and math.abs(err - P.residual) <= 1e3 * eps * total
			local best = 0
			for i = k + 1, n do
				best += values[i] ^ 2
			end
			fine = fine and P.residual <= 50 * best + 1e3 * eps * total and P.residual <= last
			last = P.residual
		end
		check(fine, "decaying spectrum: Q orthonormal, the residual is the error, near the best, and falls with k")
		local T = M.rpqr(B, n, { tol = 1e-4 })
		check(T.residual <= 1e-4 * total and T.rank < n, `tol stops it early (rank {T.rank})`)
		check(M.rpqr(M.matrix(4, 3), 2).rank == 0, "the zero matrix: rank 0")
		-- Coherent: columns scaled by 2^-i. Sampling by residual norms picks the big ones.
		local Cm = M.matrix(m, n)
		local cre, cim = M.mattotable(random_matrix(case, m, n))
		for i = 1, m do
			for j = 1, n do
				cre[i][j] *= 2 ^ -(j - 1)
				if cim then
					cim[i][j] *= 2 ^ -(j - 1)
				end
			end
		end
		Cm = if cim then M.matfrom(cre, cim) else M.matfrom(cre)
		local cworst = 0
		for seed = 10, 19 do
			cworst = math.max(cworst, M.rpqr(Cm, 5, { seed = seed }).residual / M.norm(Cm.data) ^ 2)
		end
		check(cworst <= 0.05, `coherent matrix: relative residual after 5 steps at most {cworst}`)
	end)
end

return nil
