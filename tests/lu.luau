--!strict

-- LU in place, and the _copy versions of the decompositions, which leave their input alone.

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

local function random_matrix(case: Case, m: number, n: number): lib.Matrix
	local A = case.M.matrix(m, n)
	A.data = if case.complex
		then case.M.from(random_values(m * n), random_values(m * n))
		else case.M.from(random_values(m * n))
	return A
end

local function same(M: any, A: lib.Matrix, B: lib.Matrix): boolean
	return M.norm(M.sub(nil, A.data, B.data)) == 0
end

-- Splits a table of rows of an n by n LU factor into unit-lower L and upper U rows.
local function split(t: { { number } }, n: number, unit: number): ({ { number } }, { { number } })
	local L, U = {}, {}
	for i = 1, n do
		L[i], U[i] = table.create(n, 0), table.create(n, 0)
		for j = 1, n do
			if i > j then
				L[i][j] = t[i][j]
			elseif i == j then
				L[i][j], U[i][j] = unit, t[i][j]
			else
				U[i][j] = t[i][j]
			end
		end
	end
	return L, U
end

local function permute_rows(t: { { number } }, piv: { number }): { { number } }
	local p = table.clone(t)
	for k, r in piv do
		p[k], p[r] = p[r], p[k]
	end
	return p
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps

	test(`{name} LU in place`, function()
		math.randomseed(11)
		local n = 7
		local A = random_matrix(case, n, n)
		local A0 = M.matcopy(nil, A)
		local F = M.lu(A)
		check(F.lu == A and not same(M, A, A0), "lu overwrites A and returns it")

		-- P A0 = L U, rebuilt from the factors stored in A.
		local re, im = M.mattotable(F.lu)
		local Lr, Ur = split(re, n, 1)
		local L, U
		if case.complex then
			local Li, Ui = split(im, n, 0)
			L, U = M.matfrom(Lr, Li), M.matfrom(Ur, Ui)
		else
			L, U = M.matfrom(Lr), M.matfrom(Ur)
		end
		local are, aim = M.mattotable(A0)
		local PA = if case.complex
			then M.matfrom(permute_rows(are, h.ints(F.piv)), permute_rows(aim, h.ints(F.piv)))
			else M.matfrom(permute_rows(are, h.ints(F.piv)))
		local err = M.norm(M.sub(nil, M.matmul(nil, L, U).data, PA.data)) / M.norm(A0.data)
		check(err <= 30 * n * eps, `|LU - PA| / |A| = {err}`)
	end)

	test(`{name} copying versions leave A alone`, function()
		math.randomseed(12)
		local A = random_matrix(case, 6, 6)
		local A0 = M.matcopy(nil, A)

		local Fl = M.lu_copy(A)
		check(same(M, A, A0) and Fl.lu ~= A, "lu_copy does not modify A")
		check(same(M, Fl.lu, M.lu(M.matcopy(nil, A)).lu), "lu_copy matches lu on a copy")

		local Fq = M.qr_copy(A)
		check(same(M, A, A0) and Fq.qr ~= A, "qr_copy does not modify A")
		check(same(M, Fq.qr, M.qr(M.matcopy(nil, A)).qr), "qr_copy matches qr on a copy")

		local b = M.create(6)
		M.solve(nil, A, b)
		M.det(A)
		check(same(M, A, A0), "solve and det do not modify A")
	end)
	test(`{name} inverse`, function()
		math.randomseed(13)
		for _, n in { 1, 2, 5, 20 } do
			local A = random_matrix(case, n, n)
			local A0 = M.matcopy(nil, A)
			local X = M.inv_copy(A)
			check(same(M, A, A0), `n={n}: inv_copy leaves A alone`)
			local tol = 100 * n * eps * M.norm(A0.data) * M.norm(X.data)
			local left = M.norm(M.sub(nil, M.matmul(nil, X, A0).data, M.identity(n).data))
			local right = M.norm(M.sub(nil, M.matmul(nil, A0, X).data, M.identity(n).data))
			check(left <= tol and right <= tol, `n={n}: |X A - I| = {left}, |A X - I| = {right}`)
			local viaSolve = M.solve(nil, A0, M.identity(n))
			check(M.norm(M.sub(nil, X.data, viaSolve.data)) <= tol * M.norm(X.data), `n={n}: agrees with solve(A, I)`)
			check(M.inv(A) == A and same(M, A, X), `n={n}: inv is in place and matches inv_copy`)
		end
		-- Pivoting is needed (zero diagonal), and the inverse is exact.
		local zeros = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }
		local P = M.matfrom({ { 0, 2, 0 }, { 0, 0, 4 }, { 8, 0, 0 } }, if case.complex then zeros else nil)
		local re = M.mattotable(M.inv(P))
		check(re[1][3] == 0.125 and re[2][1] == 0.5 and re[3][2] == 0.25 and re[1][1] == 0, "scaled permutation: exact inverse")
		if case.complex then
			-- inv([[i]]) = [[-i]]
			local _, im = M.mattotable(M.inv(M.matfrom({ { 0 } }, { { 1 } })))
			check(im[1][1] == -1, "1 by 1 complex")
		end
		check(not pcall(function()
			M.inv(M.matfrom({ { 1, 2 }, { 2, 4 } }, if case.complex then { { 0, 0 }, { 0, 0 } } else nil))
		end), "singular matrix is an error")
		check(M.inv(M.matrix(0, 0)).rows == 0, "0 by 0")
	end)
end

return nil
