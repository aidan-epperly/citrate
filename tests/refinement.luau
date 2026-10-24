--!strict

-- Iterative refinement with a single-precision factorization.

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

local function random_vector(case: Case, n: number): buffer
	return if case.complex then case.M.from(random_values(n), random_values(n)) else case.M.from(random_values(n))
end

local function random_matrix(case: Case, m: number, n: number): lib.Matrix
	local A = case.M.matrix(m, n)
	A.data = random_vector(case, m * n)
	return A
end

for _, case in cases do
	local M, name, eps = case.M, case.name, case.eps
	local double = eps < 1e-10

	test(`{name} iterative refinement`, function()
		math.randomseed(801)
		local n = 40
		local A = random_matrix(case, n, n)
		local A0 = M.matcopy(nil, A)
		local b = random_vector(case, n)
		local b0 = M.copy(nil, b)
		local x, info = M.solve_refined(nil, A, b)
		local function backward(y: buffer): number
			local r = M.sub(nil, b, M.matvec(nil, A, y))
			return M.norm(r) / (M.matnorm(A, "inf") * M.norm(y))
		end
		check(info.converged and not info.fallback, `converged without falling back ({info.iterations} iterations)`)
		-- Single precision: the residual in double makes the forward error reach the floor
		-- set by rounding x to single precision, even at cond(A) = 1e5, where plain solve
		-- loses about 5 digits. The reference solution comes from the double module.
		if not double then
			local D2: any = if case.complex then lib.c64 else lib.f64
			local Q1 = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
			local Q2 = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
			local rows, zeros = {}, {}
			for i = 1, n do
				rows[i], zeros[i] = table.create(n, 0), table.create(n, 0)
				rows[i][i] = 10 ^ (-5 * (i - 1) / (n - 1))
			end
			local Dg = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
			local Hard = M.matmul(nil, M.matmul(nil, Q1, Dg), if case.complex then M.adjoint(nil, Q2) else M.transpose(nil, Q2))
			-- The same values in double precision.
			local function wide(y: buffer): buffer
				return D2.from(M.totable(y))
			end
			local H2 = D2.matrix(n, n)
			H2.data = wide(Hard.data)
			local xref = D2.solve(nil, H2, wide(b))
			local function fwd(y: buffer): number
				return D2.norm(D2.sub(nil, wide(y), xref)) / D2.norm(xref)
			end
			local plain = fwd(M.solve(nil, Hard, b))
			local refined, rinfo = M.solve_refined(nil, Hard, b)
			local fr = fwd(refined)
			check(rinfo.converged and fr <= 20 * eps, `cond 1e5: forward error {fr} at the rounding floor`)
			check(plain >= 100 * fr, `cond 1e5: plain solve is far worse ({plain})`)
		end
		check(backward(x) <= 10 * eps, `backward error {backward(x)} at working precision`)
		check(M.norm(M.sub(nil, A.data, A0.data)) == 0 and M.norm(M.sub(nil, b, b0)) == 0, "A and b are not modified")
		local plain = M.solve(nil, A, b)
		check(M.norm(M.sub(nil, x, plain)) <= 1e3 * eps * M.norm(plain), "agrees with solve")
		if double then
			check(info.iterations >= 1 and info.iterations <= 10, "refinement took a few steps from a single-precision start")
		end
		-- Beyond what single precision can factor (cond ~ 1e12): the double modules fall
		-- back to a double factorization and are still accurate.
		if double then
			local Q1 = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
			local Q2 = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
			local rows, zeros = {}, {}
			for i = 1, n do
				rows[i], zeros[i] = table.create(n, 0), table.create(n, 0)
				rows[i][i] = 10 ^ (-12 * (i - 1) / (n - 1))
			end
			local D = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
			local Hard = M.matmul(nil, M.matmul(nil, Q1, D), if case.complex then M.adjoint(nil, Q2) else M.transpose(nil, Q2))
			local y, hinfo = M.solve_refined(nil, Hard, b)
			local r = M.sub(nil, b, M.matvec(nil, Hard, y))
			check(hinfo.fallback, "ill-conditioned: falls back to a double factorization")
			check(M.norm(r) / (M.matnorm(Hard, "inf") * M.norm(y)) <= 10 * eps, "ill-conditioned: still backward stable")
		end
		-- GMRES-IR: GMRES preconditioned by the factorization reaches much worse
		-- conditioning than plain refinement.
		local function conditioned(cond: number): lib.Matrix
			local Q1 = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
			local Q2 = M.qr_q(nil, M.qr(random_matrix(case, n, n)))
			local rows, zeros = {}, {}
			for i = 1, n do
				rows[i], zeros[i] = table.create(n, 0), table.create(n, 0)
				rows[i][i] = cond ^ (-(i - 1) / (n - 1))
			end
			local Dg = if case.complex then M.matfrom(rows, zeros) else M.matfrom(rows)
			return M.matmul(nil, M.matmul(nil, Q1, Dg), if case.complex then M.adjoint(nil, Q2) else M.transpose(nil, Q2))
		end
		if double then
			-- cond 1e11: plain refinement from single-precision factors has to fall back;
			-- GMRES-IR (with full inner GMRES) does not.
			local H = conditioned(1e11)
			local _, pinfo = M.solve_refined(nil, H, b)
			local y, ginfo = M.solve_refined(nil, H, b, { method = "gmres", inner = n })
			local r = M.sub(nil, b, M.matvec(nil, H, y))
			check(pinfo.fallback, "cond 1e11: plain refinement falls back")
			check(not ginfo.fallback and ginfo.converged, `cond 1e11: GMRES-IR converges from single-precision factors ({ginfo.iterations} iterations)`)
			check(M.norm(r) / (M.matnorm(H, "inf") * M.norm(y)) <= 10 * eps, "cond 1e11: GMRES-IR is backward stable")
		else
			-- cond 1e9 (beyond 1 / eps): plain refinement cannot help, GMRES-IR still gets
			-- several digits right.
			local D2: any = if case.complex then lib.c64 else lib.f64
			local H = conditioned(1e9)
			local H2 = D2.matrix(n, n)
			H2.data = D2.from(M.totable(H.data))
			local xref = D2.solve(nil, H2, D2.from(M.totable(b)))
			local function fwd(y: buffer): number
				return D2.norm(D2.sub(nil, D2.from(M.totable(y)), xref)) / D2.norm(xref)
			end
			local base = fwd(M.solve(nil, H, b))
			local lu = fwd((M.solve_refined(nil, H, b)))
			local gm = fwd((M.solve_refined(nil, H, b, { method = "gmres" })))
			check(lu <= 1.01 * base, `cond 1e9: plain refinement never makes x worse ({lu} vs {base})`)
			check(gm <= 1e-3 and gm * 100 <= base, `cond 1e9: GMRES-IR {gm} against plain solve {base}`)
		end
		check(not pcall(function()
			M.solve_refined(nil, A, b, { method = "qr" })
		end), "an unknown method is an error")
		local o = M.create(n)
		check(M.solve_refined(o, A, b) == o, "writes into out")
		local zx = M.solve_refined(nil, A, M.create(n))
		check(M.norm(zx) == 0, "zero right-hand side")
		check(not pcall(function()
			M.solve_refined(nil, M.matrix(3, 3), random_vector(case, 3))
		end), "a singular matrix is reported")
	end)
end

return nil
