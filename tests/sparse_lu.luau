--!strict

-- Sparse LU: solves of nonsymmetric matrices on all four types (orderings, thresholds,
-- zero diagonals that force pivoting), reuse (keeping the pivots, and pivoting afresh when
-- a kept one vanishes), singular matrices, and agreement with the dense LU.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

local seed = 99
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647
end

type Case = { M: any, complex: boolean, eps: number, name: string }
local cases: { Case } = {
	{ M = lib.f64, complex = false, eps = h.eps64, name = "f64" },
	{ M = lib.f32, complex = false, eps = h.eps32, name = "f32" },
	{ M = lib.c64, complex = true, eps = h.eps64, name = "c64" },
	{ M = lib.c32, complex = true, eps = h.eps32, name = "c32" },
}

-- a random nonsymmetric sparse matrix: a local pattern (as a mesh's), random values, and
-- the diagonal scaled by `diag` (0 leaves it out: pivoting must go elsewhere)
local function random_sparse(case: Case, n: number, reach: number, diag: number): any
	local xs, ys = {}, {}
	for i = 1, n do
		xs[i], ys[i] = rnd(), rnd()
	end
	local I, J, re, im = {}, {}, {}, {}
	local function add(i: number, j: number, v: number)
		table.insert(I, i)
		table.insert(J, j)
		table.insert(re, v)
		table.insert(im, if case.complex then 2 * rnd() - 1 else 0)
	end
	for i = 1, n do
		for j = 1, n do
			if i ~= j and (xs[i] - xs[j]) ^ 2 + (ys[i] - ys[j]) ^ 2 < reach * reach and rnd() < 0.7 then
				add(i, j, 2 * rnd() - 1)
			end
		end
		if diag ~= 0 then
			add(i, i, diag * (1 + rnd()))
		end
	end
	local vals = if case.complex then case.M.from(re, im) else case.M.from(re)
	return case.M.sparse(n, n, I, J, vals)
end

local function rhs(case: Case, n: number): buffer
	local re, im = {}, {}
	for i = 1, n do
		re[i] = math.sin(i)
		im[i] = math.cos(2 * i)
	end
	return if case.complex then case.M.from(re, im) else case.M.from(re)
end

local function relres(case: Case, S: any, x: buffer, b: buffer): number
	local M = case.M
	return M.norm(M.sub(nil, M.sparse_matvec(nil, S, x), b)) / M.norm(b)
end

test("sparse LU: solves on all four types", function()
	for _, case in cases do
		local M = case.M
		local n = 150
		for _, diag in { 3, 0.05, 0 } do
			local S = random_sparse(case, n, 0.15, diag)
			local b = rhs(case, n)
			local custom = buffer.create(4 * n)
			for k = 1, n do
				buffer.writei32(custom, 4 * (k - 1), n + 1 - k)
			end
			for _, order in { "amd", "rcm", "natural", custom } do
				for _, tol in { 0.1, 1 } do
					local F = M.sparse_lu(S, { order = order, tol = tol })
					local label = string.format("%s, diagonal %g, order %s, tol %g", case.name, diag, if type(order) == "string" then order else "custom", tol)
					if F.singular then
						-- a random pattern without a diagonal may be structurally singular
						check(diag == 0, label .. ": singular")
					else
						-- against the dense LU's residual on the same matrix (some are ill-conditioned)
						local res = relres(case, S, M.sparse_lu_solve(nil, F, b), b)
						local dense = relres(case, S, M.solve(nil, M.sparse_matrix(S), b), b)
						check(res < 100 * math.max(dense, case.eps), label .. string.format(": residual %.1e (dense LU %.1e)", res, dense))
					end
				end
			end
		end
	end
end)

test("sparse LU: a zero diagonal is pivoted around", function()
	-- a cyclic shift (no diagonal at all) plus a little coupling: P must move every row
	local M = lib.f64
	local n = 50
	local I, J, V = {}, {}, {}
	for i = 1, n do
		table.insert(I, i)
		table.insert(J, i % n + 1)
		table.insert(V, 2 + i / n)
		table.insert(I, i)
		table.insert(J, (i + 1) % n + 1)
		table.insert(V, 0.3)
	end
	local S = M.sparse(n, n, I, J, M.from(V))
	local b = M.from(table.create(n, 1))
	local F = M.sparse_lu(S)
	check(not F.singular, "not singular")
	local x = M.sparse_lu_solve(nil, F, b)
	check(M.norm(M.sub(nil, M.sparse_matvec(nil, S, x), b)) < 1e-12, "solved")
end)

test("sparse LU: agrees with the dense LU", function()
	local M = lib.f64
	local case = cases[1]
	local n = 40
	local S = random_sparse(case, n, 0.3, 0.5)
	local b = rhs(case, n)
	local x = M.sparse_lu_solve(nil, M.sparse_lu(S), b)
	local y = M.solve(nil, M.sparse_matrix(S), b)
	check(M.norm(M.sub(nil, x, y)) < 1e-10 * M.norm(y), "same solution")
end)

test("sparse LU: reuse, repivoting and singular matrices", function()
	for _, case in cases do
		local M = case.M
		local n = 120
		local S = random_sparse(case, n, 0.15, 2)
		local F = M.sparse_lu(S)
		local b = rhs(case, n)
		-- new values on the same pattern: the pivots are kept
		local S2 = { rows = n, cols = n, colptr = S.colptr, rowind = S.rowind, values = M.scale(nil, 3, S.values) }
		local F2 = M.sparse_lu(S2, { reuse = F })
		check(F2 == F and not F2.repivoted, case.name .. ": refactored with the same pivots")
		check(relres(case, S2, M.sparse_lu_solve(nil, F2, b), b) < 1000 * case.eps, case.name .. ": reused solve")
		-- zero the stored diagonal entries: the kept (diagonal) pivots vanish, so it pivots
		-- afresh
		local values = M.copy(nil, S.values)
		for j = 1, n do
			for p = buffer.readu32(S.colptr, 4 * (j - 1)), buffer.readu32(S.colptr, 4 * j) - 1 do
				if buffer.readu32(S.rowind, 4 * p) == j - 1 then
					if case.name == "c32" then
						M.set(values, p + 1, vector.zero)
					elseif case.complex then
						M.set(values, p + 1, 0, 0)
					else
						M.set(values, p + 1, 0)
					end
				end
			end
		end
		local S3 = { rows = n, cols = n, colptr = S.colptr, rowind = S.rowind, values = values }
		local F3 = M.sparse_lu(S3, { reuse = F })
		if not F3.singular then
			check(F3.repivoted and relres(case, S3, M.sparse_lu_solve(nil, F3, b), b) < 1000 * case.eps, case.name .. ": pivoted afresh")
		end
		-- a singular matrix: two equal rows
		local I, J, V = { 1, 1, 2, 2, 3 }, { 1, 2, 1, 2, 3 }, { 1, 2, 1, 2, 1 }
		local T = M.sparse(3, 3, I, J, if case.complex then M.from(V, { 0, 0, 0, 0, 0 }) else M.from(V))
		local Ft = M.sparse_lu(T)
		check(Ft.singular, case.name .. ": singular")
		check(not pcall(function()
			M.sparse_lu_solve(nil, Ft, rhs(case, 3))
		end), case.name .. ": solving a singular matrix errors")
	end
end)

return nil
