--!strict

-- Schur factorization and the general (nonsymmetric) eigensolver.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

-- C is the complex module of the same precision, which holds eigenvalues and vectors.
type Case = { name: string, M: any, C: any, complex: boolean, eps: number }

local cases: { Case } = {
	{ name = "f32", M = lib.f32, C = lib.c32, complex = false, eps = h.eps32 },
	{ name = "f64", M = lib.f64, C = lib.c64, complex = false, eps = h.eps64 },
	{ name = "c32", M = lib.c32, C = lib.c32, complex = true, eps = h.eps32 },
	{ name = "c64", M = lib.c64, C = lib.c64, complex = true, eps = h.eps64 },
}

local function random_values(n: number): { number }
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = 2 * math.random() - 1
	end
	return t
end

local function random_matrix(case: Case, n: number): lib.Matrix
	local A = case.M.matrix(n, n)
	A.data = if case.complex
		then case.M.from(random_values(n * n), random_values(n * n))
		else case.M.from(random_values(n * n))
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

-- A copy of A with entry (i, j) set to re + im i (im is ignored for real types).
local function with_entry(case: Case, A: lib.Matrix, i: number, j: number, re: number, im: number): lib.Matrix
	local r, m = case.M.mattotable(A)
	r[i][j] = re
	if m then
		m[i][j] = im
	end
	return if m then case.M.matfrom(r, m) else case.M.matfrom(r)
end

-- A copy of A with everything below the diagonal set to zero.
local function rewrite_upper(case: Case, A: lib.Matrix): lib.Matrix
	local re, im = case.M.mattotable(A)
	for i = 1, A.rows do
		for j = 1, i - 1 do
			re[i][j] = 0
			if im then
				im[i][j] = 0
			end
		end
	end
	return if im then case.M.matfrom(re, im) else case.M.matfrom(re)
end

-- A as a complex matrix of the same precision.
local function as_complex(case: Case, A: lib.Matrix): lib.Matrix
	if case.complex then
		return case.M.matcopy(nil, A)
	end
	local Z = case.C.matrix(A.rows, A.cols)
	Z.data = case.C.complexify(A.data)
	return Z
end

-- Eigenvalues as a list of { re, im }.
local function value_list(case: Case, values: buffer): { { number } }
	local re: { number }, im: { number } = case.C.totable(values)
	local list: { { number } } = {}
	for i = 1, #re do
		list[i] = { re[i], im[i] }
	end
	return list
end

-- Every expected eigenvalue is matched, one to one, by a computed one within tol (order
-- does not matter, so rounding in nearly equal real parts cannot reorder anything).
local function values_close(case: Case, values: buffer, expected: { { number } }, tol: number): boolean
	local got = value_list(case, values)
	if #got ~= #expected then
		return false
	end
	local used: { boolean } = table.create(#got, false)
	for _, e in expected do
		local best, dist = 0, math.huge
		for i, g in got do
			local d = math.max(math.abs(g[1] - e[1]), math.abs(g[2] - e[2]))
			if not used[i] and d < dist then
				best, dist = i, d
			end
		end
		if dist > tol then
			return false
		end
		used[best] = true
	end
	return true
end

-- |A V - V diag(values)| / |A| and the largest deviation of a column norm from 1.
local function residual(case: Case, A: lib.Matrix, F: lib.Eig): (number, number)
	local C = case.C
	local V = F.vectors :: lib.Matrix
	local n = A.rows
	local re, im = C.totable(F.values)
	local dre, dim = {}, {}
	for i = 1, n do
		dre[i], dim[i] = table.create(n, 0), table.create(n, 0)
		dre[i][i], dim[i][i] = re[i], im[i]
	end
	local Ac = as_complex(case, A)
	local lhs = C.matmul(nil, Ac, V)
	local rhs = C.matmul(nil, V, C.matfrom(dre, dim))
	local r = C.norm(C.sub(nil, lhs.data, rhs.data)) / math.max(C.norm(Ac.data), 1e-300)
	local worst = 0
	for j = 1, n do
		local col = C.create(n)
		buffer.copy(col, 0, V.data, buffer.len(col) * (j - 1), buffer.len(col))
		worst = math.max(worst, math.abs(C.norm(col) - 1))
	end
	return r, worst
end

for _, case in cases do
	local M, C, name, eps = case.M, case.C, case.name, case.eps
	local loose = 1e3 * eps

	test(`{name} eig on known matrices`, function()
		local function vals(rows: { { number } }): buffer
			return M.eig_copy(from_rows(case, rows), false).values
		end
		check(values_close(case, vals({ { 0, -1 }, { 1, 0 } }), { { 0, -1 }, { 0, 1 } }, loose), "rotation: +-i")
		check(values_close(case, vals({ { 2, 0, 0 }, { 0, -1, 0 }, { 0, 0, 5 } }), { { -1, 0 }, { 2, 0 }, { 5, 0 } }, loose), "diagonal")
		check(values_close(case, vals({ { 1, 4, 5 }, { 0, 2, 6 }, { 0, 0, 3 } }), { { 1, 0 }, { 2, 0 }, { 3, 0 } }, loose), "triangular")
		-- Companion matrix of (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6.
		check(values_close(case, vals({ { 6, -11, 6 }, { 1, 0, 0 }, { 0, 1, 0 } }), { { 1, 0 }, { 2, 0 }, { 3, 0 } }, 1e4 * eps), "companion")
		check(values_close(case, vals({ { 0, 0 }, { 0, 0 } }), { { 0, 0 }, { 0, 0 } }, 0), "zero matrix")
		check(values_close(case, vals({ { -3 } }), { { -3, 0 } }, 0), "1 by 1")
		check(buffer.len(M.eig(M.matrix(0, 0)).values) == 0, "0 by 0")

		-- Cyclic permutations: the standard shifts stall, so this needs exceptional shifts.
		for _, n in { 3, 6 } do
			local rows = {}
			for i = 1, n do
				rows[i] = table.create(n, 0)
				rows[i][(i % n) + 1] = 1
			end
			local expected: { { number } } = {}
			for k = 0, n - 1 do
				expected[k + 1] = { math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n) }
			end
			local A = from_rows(case, rows)
			local F = M.eig_copy(A)
			check(values_close(case, F.values, expected, loose), `cyclic {n}: roots of unity`)
			local r, unit = residual(case, A, F)
			check(r <= loose and unit <= loose, `cyclic {n}: residual {r}, unit {unit}`)
		end
	end)

	test(`{name} eig on random matrices`, function()
		math.randomseed(61)
		for _, n in { 2, 5, 12, 30 } do
			local A = random_matrix(case, n)
			local A0 = M.matcopy(nil, A)
			local F = M.eig(A)
			local r, unit = residual(case, A0, F)
			local tol = 100 * n * eps
			check(r <= tol, `n={n}: |A V - V L| / |A| = {r}`)
			check(unit <= tol, `n={n}: unit eigenvectors ({unit})`)
			if not case.complex then
				-- Complex eigenvalues of a real matrix come in exact conjugate pairs.
				local re, im = C.totable(F.values)
				local paired = true
				local k = 1
				while k <= n do
					if im[k] ~= 0 then
						paired = paired and k < n and re[k + 1] == re[k] and im[k + 1] == -im[k] and im[k] > 0
						k += 2
					else
						k += 1
					end
				end
				check(paired, `n={n}: conjugate pairs`)
			end
			local only = M.eig_copy(A0, false).values
			local expected = value_list(case, F.values)
			check(values_close(case, only, expected, 100 * tol), `n={n}: values without vectors agree`)
		end
		local A = random_matrix(case, 6)
		local A0 = M.matcopy(nil, A)
		M.eig_copy(A)
		M.schur_copy(A)
		check(M.norm(M.sub(nil, A.data, A0.data)) == 0, "eig_copy and schur_copy leave A alone")
	end)

	test(`{name} Schur factorization`, function()
		math.randomseed(62)
		for _, n in { 1, 4, 15 } do
			local A = random_matrix(case, n)
			local A0 = M.matcopy(nil, A)
			local F = M.schur(A)
			check(F.schur == A, `n={n}: in place`)
			local Z = F.z :: lib.Matrix
			local T = F.schur
			local Zh = if case.complex then M.adjoint(nil, Z) else M.transpose(nil, Z)
			local tol = 100 * n * eps
			local recon = M.norm(M.sub(nil, M.matmul(nil, M.matmul(nil, Z, T), Zh).data, A0.data)) / M.norm(A0.data)
			check(recon <= tol, `n={n}: |Z T Z^H - A| / |A| = {recon}`)
			local orth = M.norm(M.sub(nil, M.matmul(nil, Zh, Z).data, M.identity(n).data))
			check(orth <= tol, `n={n}: |Z^H Z - I| = {orth}`)
			-- Structure: triangular (complex) or quasi-triangular with standardized blocks.
			local re, im = M.mattotable(T)
			local shaped = true
			for i = 1, n do
				for j = 1, i - 1 do
					local nonzero = re[i][j] ~= 0 or (im ~= nil and im[i][j] ~= 0)
					if nonzero then
						local allowed = not case.complex and j == i - 1 and re[i - 1][i - 1] == re[i][i]
							and (i < 3 or re[i - 1][i - 2] == 0) and (i == n or re[i + 1][i] == 0)
						shaped = shaped and allowed
					end
				end
			end
			check(shaped, `n={n}: Schur form has the right shape`)
		end
	end)

	test(`{name} balancing a badly scaled matrix`, function()
		math.randomseed(63)
		local n = 8
		local B = random_matrix(case, n)
		-- A = D B D^-1 with D = diag(2^(10 i)): exactly B's eigenvalues (powers of 2 scale
		-- without rounding), but |A| is about 1e21.
		local re, im = M.mattotable(B)
		for i = 1, n do
			for j = 1, n do
				local f = 2 ^ (10 * (i - j))
				re[i][j] *= f
				if im then
					im[i][j] *= f
				end
			end
		end
		local A = if im then M.matfrom(re, im) else M.matfrom(re)
		local expected = value_list(case, M.eig_copy(B, false).values)
		local tol = 1e3 * eps
		local F = M.eig_copy(A)
		check(values_close(case, F.values, expected, tol), "balanced: eigenvalues accurate")
		check(not values_close(case, M.eig_copy(A, false, false).values, expected, 1e-3),
			"unbalanced: eigenvalues far off (this is what balancing fixes)")
		-- Each eigenvector v of A gives an eigenvector D^-1 v of B; checking that tests the
		-- back-transformation of the vectors.
		local V = F.vectors :: lib.Matrix
		local vr, vi = C.mattotable(V)
		for i = 1, n do
			for j = 1, n do
				vr[i][j] /= 2 ^ (10 * (i - 1))
				vi[i][j] /= 2 ^ (10 * (i - 1))
			end
		end
		local W = C.matfrom(vr, vi)
		local r, _ = residual(case, B, { values = F.values, vectors = W })
		-- The columns of W are not unit vectors, so measure the residual column by column.
		local worst = 0
		local lr, li = C.totable(F.values)
		local Bc = as_complex(case, B)
		for j = 1, n do
			local col = C.create(n)
			buffer.copy(col, 0, W.data, buffer.len(col) * (j - 1), buffer.len(col))
			local Bv = C.matvec(nil, Bc, col)
			local lv = if case.name == "c32" or case.name == "f32"
				then C.cscale(nil, vector.create(lr[j], li[j], 0), col)
				else C.cscale(nil, lr[j], li[j], col)
			worst = math.max(worst, C.norm(C.sub(nil, Bv, lv)) / C.norm(col))
		end
		check(worst <= 100 * n * eps * C.norm(Bc.data), `eigenvectors map back to B's ({worst}, overall {r})`)
		local U = M.eig_copy(B, true, false)
		local ru, unit = residual(case, B, U)
		check(ru <= 100 * n * eps and unit <= 100 * n * eps, "balanced = false still works")
	end)

	test(`{name} balancing permutes reducible matrices`, function()
		math.randomseed(64)
		-- A = P [T1 X Y; 0 B Z; 0 0 T2] P^T with T1 (3 by 3) and T2 (2 by 2) upper triangular
		-- and B (4 by 4) full, scrambled by a random permutation P. Balancing has to find the
		-- isolated eigenvalues (the diagonals of T1 and T2) and leave only B to the QR
		-- algorithm.
		local n, n1, nb = 9, 3, 4
		-- Drawn first, so every type gets the same permutation.
		local perm = {}
		for i = 1, n do
			perm[i] = i
		end
		for i = n, 2, -1 do
			local j = math.random(1, i)
			perm[i], perm[j] = perm[j], perm[i]
		end
		local re, im = {}, {}
		for i = 1, n do
			re[i], im[i] = table.create(n, 0), table.create(n, 0)
			for j = 1, n do
				local inB = i > n1 and i <= n1 + nb and j > n1 and j <= n1 + nb
				if j > i or i == j or inB then
					re[i][j] = 2 * math.random() - 1
					im[i][j] = if case.complex then 2 * math.random() - 1 else 0
				end
			end
		end
		-- Round the entries to the module's precision first, so the expected values are the
		-- stored ones.
		local stored_re, stored_im = M.mattotable(if case.complex then M.matfrom(re, im) else M.matfrom(re))
		re = stored_re
		if stored_im then
			im = stored_im
		end
		-- The isolated eigenvalues are exactly these diagonal entries.
		local isolated = {}
		for i = 1, n do
			if i <= n1 or i > n1 + nb then
				table.insert(isolated, { re[i][i], im[i][i] })
			end
		end
		local pr, pi = {}, {}
		for i = 1, n do
			pr[i], pi[i] = table.create(n, 0), table.create(n, 0)
			for j = 1, n do
				pr[i][j], pi[i][j] = re[perm[i]][perm[j]], im[perm[i]][perm[j]]
			end
		end
		local A = if case.complex then M.matfrom(pr, pi) else M.matfrom(pr)
		local F = M.eig_copy(A)
		local got = value_list(case, F.values)
		local exact = 0
		for _, v in isolated do
			for _, g in got do
				if g[1] == v[1] and g[2] == v[2] then
					exact += 1
					break
				end
			end
		end
		check(exact == #isolated, `isolated eigenvalues come out exactly ({exact} of {#isolated})`)
		local r, unit = residual(case, A, F)
		check(r <= 100 * n * eps and unit <= 100 * n * eps, `eigenvectors (residual {r})`)
		check(values_close(case, M.eig_copy(A, false, false).values, got, 1e4 * eps), "agrees with balanced = false")

		-- Upper triangular: everything is isolated.
		local T = rewrite_upper(case, random_matrix(case, 6))
		local Ft = M.eig_copy(T)
		local tr, ti = M.mattotable(T)
		local diag = {}
		for i = 1, 6 do
			diag[i] = { tr[i][i], if ti then ti[i][i] else 0 }
		end
		check(values_close(case, Ft.values, diag, 0), "triangular: eigenvalues are the diagonal, exactly")
		local rt, ut = residual(case, T, Ft)
		check(rt <= 100 * 6 * eps and ut <= 100 * 6 * eps, "triangular: eigenvectors")

		-- A Markov chain with two absorbing states: eigenvalue 1 at least twice.
		local P = from_rows(case, {
			{ 1, 0, 0, 0, 0 },
			{ 0.2, 0.3, 0.5, 0, 0 },
			{ 0, 0.4, 0.1, 0.5, 0 },
			{ 0, 0, 0.6, 0.1, 0.3 },
			{ 0, 0, 0, 0, 1 },
		})
		local Fm = M.eig_copy(P)
		local ones = 0
		for _, g in value_list(case, Fm.values) do
			if math.abs(g[1] - 1) <= 10 * eps and math.abs(g[2]) <= 10 * eps then
				ones += 1
			end
		end
		check(ones == 2, `absorbing Markov chain: eigenvalue 1 twice ({ones})`)
		local rm, um = residual(case, P, Fm)
		check(rm <= 100 * 5 * eps and um <= 100 * 5 * eps, "Markov chain: eigenvectors")
	end)
	test(`{name} eig and schur return NaN for non-finite input`, function()
		local single = eps == h.eps32
		for _, bad in { 0 / 0, math.huge, -math.huge } do
			local A = random_matrix(case, 5)
			A = with_entry(case, A, 2, 4, bad, 0)
			local F = M.eig_copy(A)
			check(h.all_nan(F.values, single) and h.all_nan((F.vectors :: lib.Matrix).data, single), `eig: {bad} gives NaN`)
			check(h.all_nan(M.eig_copy(A, false).values, single), `eig without vectors: {bad} gives NaN`)
			check(h.all_nan(M.eig_copy(A, true, false).values, single), `unbalanced eig: {bad} gives NaN`)
			local S = M.schur(A)
			check(
				h.all_nan(S.values, single) and h.all_nan(S.schur.data, single) and h.all_nan((S.z :: lib.Matrix).data, single),
				`schur: {bad} gives NaN`
			)
		end
		local B = from_rows(case, { { 1, 2 }, { 3, 4 } })
		B = with_entry(case, B, 1, 1, 1, 0 / 0)
		check(not case.complex or h.all_nan(M.eig(B).values, single), "a NaN imaginary part gives NaN")
	end)

	test(`{name} eig on extreme scales`, function()
		math.randomseed(65)
		local A = random_matrix(case, 6)
		local base = M.eig_copy(A, false).values
		-- Powers of 2, so scaling A is exact; near the ends of the exponent range.
		local scales = if eps == h.eps32 then { 2 ^ 100, 2 ^ -100 } else { 2 ^ 1000, 2 ^ -1000 }
		for _, sc in scales do
			local As = M.matcopy(nil, A)
			As.data = M.scale(nil, sc, As.data)
			local F = M.eig_copy(As)
			check(values_close(case, C.scale(nil, 1 / sc, F.values), value_list(case, base), loose), `scale {sc}: values`)
			local r, unit = residual(case, As, F)
			check(r <= loose and unit <= loose, `scale {sc}: residual {r}, unit {unit}`)
			local S = M.schur(M.matcopy(nil, As))
			local Z = S.z :: lib.Matrix
			local Zh = if case.complex then M.adjoint(nil, Z) else M.transpose(nil, Z)
			local recon = M.norm(M.sub(nil, M.matmul(nil, M.matmul(nil, Z, S.schur), Zh).data, As.data)) / M.norm(As.data)
			check(recon <= loose, `scale {sc}: |Z T Z^H - A| / |A| = {recon}`)
			check(values_close(case, C.scale(nil, 1 / sc, S.values), value_list(case, base), loose), `scale {sc}: Schur values`)
		end
	end)

	test(`{name} eig on a defective matrix`, function()
		math.randomseed(66)
		-- A = Q J Q^H with J a 4 by 4 Jordan block for eigenvalue 2 and Q unitary. The
		-- eigenvalues are only determined to about eps^(1/4), but each computed pair still
		-- has a small residual.
		local n = 4
		local rows = {}
		for i = 1, n do
			rows[i] = table.create(n, 0)
			rows[i][i] = 2
			if i < n then
				rows[i][i + 1] = 1
			end
		end
		local Q = M.qr_q(nil, M.qr(random_matrix(case, n)))
		local Qh = if case.complex then M.adjoint(nil, Q) else M.transpose(nil, Q)
		local A = M.matmul(nil, M.matmul(nil, Q, from_rows(case, rows)), Qh)
		local F = M.eig_copy(A)
		local expected = { { 2, 0 }, { 2, 0 }, { 2, 0 }, { 2, 0 } }
		check(values_close(case, F.values, expected, 10 * eps ^ (1 / 4)), "eigenvalues near 2")
		local r, unit = residual(case, A, F)
		check(r <= loose and unit <= loose, `residual {r}, unit {unit}`)
		local finite = true
		local vre, vim = C.mattotable(F.vectors)
		for i = 1, n do
			for j = 1, n do
				finite = finite and vre[i][j] - vre[i][j] == 0 and vim[i][j] - vim[i][j] == 0
			end
		end
		check(finite, "eigenvectors are finite")
	end)

	test(`{name} eig on a larger matrix`, function()
		math.randomseed(67)
		local n = 80
		local A = random_matrix(case, n)
		local F = M.eig_copy(A)
		local r, unit = residual(case, A, F)
		check(r <= 100 * n * eps and unit <= 100 * n * eps, `n={n}: residual {r}, unit {unit}`)
	end)
end

return nil
