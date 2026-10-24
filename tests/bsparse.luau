--!strict

-- The unrolled sparse products (column lengths covering every remainder of the unrolling,
-- the symmetric and hermitian marks) and block sparse matrices (generated kernels for
-- blocks of 2 to 4, the generic loops for others) against dense references.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

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

local function values(case: Case, n: number): buffer
	if case.complex then
		return case.M.from(random_values(n), random_values(n))
	end
	return case.M.from(random_values(n))
end

local function near(M: any, x: buffer, y: buffer, tol: number): boolean
	return M.norm(M.sub(nil, x, y)) <= tol * math.max(M.norm(y), 1e-300)
end

-- A rows by cols sparse matrix whose column j holds (j - 1) % 10 entries (capped by rows).
local function random_sparse(case: Case, rows: number, cols: number): any
	local ii, jj = {}, {}
	for j = 1, cols do
		local want = math.min(rows, (j - 1) % 10)
		local used = {}
		local got = 0
		while got < want do
			local i = math.random(rows)
			if not used[i] then
				used[i] = true
				got += 1
				table.insert(ii, i)
				table.insert(jj, j)
			end
		end
	end
	return case.M.sparse(rows, cols, ii, jj, values(case, #ii))
end

local function dense_equal(M: any, A: any, B: any, tol: number): boolean
	return A.rows == B.rows and A.cols == B.cols and near(M, A.data, B.data, tol)
end

for _, case in cases do
	local M, name = case.M, case.name
	local tol = 1e3 * case.eps

	test(`{name} unrolled sparse products and symmetry marks`, function()
		for _, size in { { 1, 1 }, { 7, 5 }, { 13, 23 }, { 30, 30 } } do
			local m, n = size[1], size[2]
			local S = random_sparse(case, m, n)
			local A = M.sparse_matrix(S)
			local x, y = values(case, n), values(case, m)
			check(near(M, M.sparse_matvec(nil, S, x), M.matvec(nil, A, x), tol), `matvec {m}x{n}`)
			local AH = M.sparse_matrix(M.sparse_adjoint(S))
			check(near(M, M.sparse_matvec_adjoint(nil, S, y), M.matvec(nil, AH, y), tol), `adjoint {m}x{n}`)
			-- into an existing output with stale contents
			local o = values(case, m)
			check(near(M, M.sparse_matvec(o, S, x), M.matvec(nil, A, x), tol), `matvec into out {m}x{n}`)
			if m == n then
				-- S + S^T marked symmetric, and (complex) S + S^H marked hermitian
				local Ssym = M.sparse_add(S, M.sparse_transpose(S))
				Ssym.symmetric = true
				check(near(M, M.sparse_matvec(nil, Ssym, x), M.matvec(nil, M.sparse_matrix(Ssym), x), tol), `symmetric matvec {m}`)
				if case.complex then
					local Sh = M.sparse_add(S, M.sparse_adjoint(S))
					Sh.hermitian = true
					check(near(M, M.sparse_matvec(nil, Sh, x), M.matvec(nil, M.sparse_matrix(Sh), x), tol), `hermitian matvec {m}`)
					check(M.sparse_adjoint(Sh).hermitian == true and M.sparse_transpose(Sh).hermitian == true, "marks carry over")
				end
			end
		end
	end)

	test(`{name} block sparse matrices`, function()
		for _, b in { 1, 2, 3, 4, 5 } do
			local nb, mb = 7, 5
			local S = random_sparse(case, b * nb, b * mb)
			local A = M.sparse_matrix(S)
			local B = M.bsparse_from_sparse(S, b)
			check(B.block == b and M.bsparse_nnz(B) % (b * b) == 0 and M.bsparse_nnz(B) >= M.sparse_nnz(S), `b={b} blocks hold every entry`)
			check(dense_equal(M, M.sparse_matrix(M.bsparse_sparse(B)), A, 0), `b={b} round trip`)
			local x, y = values(case, b * mb), values(case, b * nb)
			check(near(M, M.bsparse_matvec(nil, B, x), M.matvec(nil, A, x), tol), `b={b} matvec`)
			local AH = M.sparse_matrix(M.sparse_adjoint(S))
			check(near(M, M.bsparse_matvec_adjoint(nil, B, y), M.matvec(nil, AH, y), tol), `b={b} adjoint`)
			-- entries, blocks and the diagonal
			local ok = true
			for i = 1, b * nb do
				for j = 1, b * mb do
					local g, w = M.bsparse_get(B, i, j), M.sparse_get(S, i, j)
					if type(g) == "vector" then
						ok = ok and (g :: any).x == (w :: any).x and (g :: any).y == (w :: any).y
					elseif case.complex then
						local gr, gi = M.bsparse_get(B, i, j)
						local wr, wi = M.sparse_get(S, i, j)
						ok = ok and gr == wr and gi == wi
					else
						ok = ok and g == w
					end
				end
			end
			check(ok, `b={b} bsparse_get`)
			local Sq = random_sparse(case, b * nb, b * nb)
			local Bq = M.bsparse_from_sparse(Sq, b)
			check(near(M, M.bsparse_diag(Bq), M.sparse_diag(Sq), 0), `b={b} bsparse_diag`)
			local p = M.bsparse_block(B, 1, 1)
			check(p == nil or (p >= 1 and p <= M.bsparse_nnz(B) // (b * b)), `b={b} bsparse_block`)
			check(M.bsparse_block(B, nb, mb + 1) == nil, `b={b} block outside the pattern`)

			-- Triplets with a repeated position, and block Jacobi on a block diagonal matrix.
			local ii, jj = {}, {}
			for I = 1, nb do
				table.insert(ii, I)
				table.insert(jj, I)
			end
			table.insert(ii, 2)
			table.insert(jj, 2)
			local vals = values(case, b * b * #ii)
			-- make the diagonal blocks dominant
			local re = table.create(b * b * #ii, 0)
			for k = 0, #ii - 1 do
				for e = 0, b * b - 1 do
					re[b * b * k + e + 1] = if e % (b + 1) == 0 then 4 * b else 1
				end
			end
			local scale = if case.complex then M.from(re, table.create(#re, 0)) else M.from(re)
			vals = M.add(nil, vals, scale)
			local D = M.bsparse(b * nb, b * nb, b, ii, jj, vals)
			check(M.bsparse_nnz(D) == b * b * nb, `b={b} repeated block merged`)
			local first = M.bsparse_block(D, 2, 2) :: number
			local okb = true
			for e = 1, b * b do
				local s = M.get(vals, b * b + e)
				local t = M.get(vals, b * b * nb + e)
				local got = M.get(D.values, b * b * (first - 1) + e)
				if type(got) == "vector" then
					okb = okb and math.abs((got :: any).x - ((s :: any).x + (t :: any).x)) <= tol and math.abs((got :: any).y - ((s :: any).y + (t :: any).y)) <= tol
				elseif case.complex then
					local sr, si = M.get(vals, b * b + e)
					local tr, ti = M.get(vals, b * b * nb + e)
					local gr, gi = M.get(D.values, b * b * (first - 1) + e)
					okb = okb and math.abs(gr - (sr + tr)) <= tol and math.abs(gi - (si + ti)) <= tol
				else
					okb = okb and math.abs(got - (s + t)) <= tol * 10
				end
			end
			check(okb, `b={b} repeated positions summed`)
			local P = M.bsparse_jacobi(D)
			local r = values(case, b * nb)
			check(near(M, M.bsparse_matvec(nil, D, P(nil, r)), r, tol), `b={b} block Jacobi inverts the diagonal blocks`)
			D.symmetric = not case.complex
			D.hermitian = case.complex
			-- A marked adjoint is the plain product (the mark is a promise, false here, so this
			-- only checks that the shortcut is taken).
			check(near(M, M.bsparse_matvec_adjoint(nil, D, r), M.bsparse_matvec(nil, D, r), 0), `b={b} marked adjoint`)
		end
	end)
end

return nil
