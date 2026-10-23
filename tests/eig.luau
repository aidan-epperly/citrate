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
end

return nil
