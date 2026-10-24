--!strict

-- Algebraic multigrid (bsparse_amg) on spring lattices: the V-cycle is a symmetric positive
-- definite operator (CG needs that), update matches a fresh build, fixed components act as
-- the identity, and CG converges in far fewer iterations than with block Jacobi.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check
local krylov = lib.krylov

-- A g by g lattice in d = b dimensions (b = 1: a scalar Laplacian; for b > 2 the extra
-- components get their own scalar coupling), A = m I + K, positive definite; the nodes
-- of the first column are pinned when pin is set.
local function lattice(M: any, g: number, b: number, k: number, pin: boolean): (any, { number })
	local ii, jj, vals = {}, {}, {}
	local function add(I: number, J: number, blk: { number })
		table.insert(ii, I)
		table.insert(jj, J)
		for _, v in blk do
			table.insert(vals, v)
		end
	end
	local function block(u: { number }, f: number): { number }
		local blk = table.create(b * b, 0)
		for s = 1, b do
			for r = 1, b do
				-- components beyond the first two couple like a Laplacian
				local uu = if r > 2 or s > 2 then (if r == s then 1 else 0) else u[r] * u[s]
				blk[(s - 1) * b + r] = f * uu
			end
		end
		return blk
	end
	local fixed = {}
	for y = 0, g - 1 do
		for x = 0, g - 1 do
			local I = y * g + x + 1
			local diag = table.create(b * b, 0)
			for r = 1, b do
				diag[(r - 1) * b + r] = 0.01
			end
			add(I, I, diag)
			if pin and x == 0 then
				for c = 1, b do
					table.insert(fixed, b * (I - 1) + c)
				end
			end
			for _, off in { { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 } } do
				local x2, y2 = x + off[1], y + off[2]
				if x2 >= 0 and x2 < g and y2 >= 0 and y2 < g then
					local J = y2 * g + x2 + 1
					local len = math.sqrt(off[1] * off[1] + off[2] * off[2])
					local u = { off[1] / len, off[2] / len }
					if b == 1 then
						u = { 1 }
					end
					local f = k / len
					add(I, I, block(u, f))
					add(J, J, block(u, f))
					add(I, J, block(u, -f))
					add(J, I, block(u, -f))
				end
			end
		end
	end
	local A = M.bsparse(b * g * g, b * g * g, b, ii, jj, M.from(vals))
	A.symmetric = true
	return A, fixed
end

local function random(M: any, n: number, seed: number): buffer
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = math.sin(seed * 1.3 + 2.1 * i)
	end
	return M.from(t)
end

for _, case in { { name = "f64", M = lib.f64, tol = 1e-10 }, { name = "f32", M = lib.f32, tol = 1e-4 } } :: { any } do
	local M = case.M
	test(`{case.name} multigrid: symmetric positive definite V-cycle`, function()
		for _, b in { 1, 2, 3, 5 } do
			for _, smooth in { false, true } do
				local A, fixed = lattice(M, 14, b, 50, b ~= 1)
				local H = M.bsparse_amg(A, { fixed = fixed, smooth = smooth, coarsest = 30 })
				check(#H.sizes >= 2, `b={b} smooth={smooth}: a hierarchy ({table.concat(H.sizes, " ")})`)
				local n = A.rows
				local x, y = random(M, n, 1), random(M, n, 2)
				local Mx, My = H.apply(M.create(n), x), H.apply(M.create(n), y)
				local a, c = M.dot(Mx, y), M.dot(x, My)
				check(math.abs(a - c) <= case.tol * math.max(math.abs(a), 1), `b={b} smooth={smooth}: symmetric ({a} vs {c})`)
				check(M.dot(x, Mx) > 0 and M.dot(y, My) > 0, `b={b} smooth={smooth}: positive`)
				-- out may be the input
				local z = M.copy(nil, x)
				check(M.norm(M.sub(nil, H.apply(z, z), Mx)) <= case.tol * M.norm(Mx), `b={b} smooth={smooth}: in place`)
			end
		end
	end)

	test(`{case.name} multigrid: update matches a fresh build`, function()
		local A, fixed = lattice(M, 16, 3, 50, true)
		local H = M.bsparse_amg(A, { fixed = fixed })
		-- new values on the same pattern: scale the stiffness part
		local A2 = table.clone(A)
		A2.values = M.scale(nil, 1.7, A.values)
		H.update(A2)
		local fresh = M.bsparse_amg(A2, { fixed = fixed })
		local r = random(M, A.rows, 3)
		local u, v = H.apply(M.create(A.rows), r), fresh.apply(M.create(A.rows), r)
		check(M.norm(M.sub(nil, u, v)) <= case.tol * M.norm(v), "same V-cycle after update")
	end)

	test(`{case.name} multigrid: CG iterations`, function()
		local A, fixed = lattice(M, 30, 2, 1e3, true)
		local n = A.rows
		local isfixed = {}
		for _, c in fixed do
			isfixed[c] = true
		end
		local xf = M.create(n)
		local op = function(out: buffer, x: buffer): buffer
			xf = M.copy(xf, x)
			for _, c in fixed do
				M.set(xf, c, 0)
			end
			out = M.bsparse_matvec(out, A, xf)
			for _, c in fixed do
				M.set(out, c, M.get(x, c))
			end
			return out
		end
		local rhs = random(M, n, 4)
		for _, c in fixed do
			M.set(rhs, c, 0)
		end
		local tol = if M == lib.f32 then 1e-4 else 1e-8
		local jac_options: any = { tol = tol, maxiter = 5000, precond = M.bsparse_jacobi(A) }
		local _, jac = krylov.cg(M.space, op, rhs, jac_options)
		local H = M.bsparse_amg(A, { fixed = fixed })
		local amg_options: any = { tol = tol, maxiter = 5000, precond = H.apply }
		local x, info = krylov.cg(M.space, op, rhs, amg_options)
		check(info.converged and jac.converged, "both converge")
		-- measured 2026-10-24 (f64): block Jacobi ~470 iterations, smoothed aggregation ~30
		check(info.iterations * 5 < jac.iterations, `multigrid {info.iterations} vs block Jacobi {jac.iterations} iterations`)
		for _, c in fixed do
			check(M.get(x, c) == 0, "fixed components stay zero")
			break
		end
	end)
end

return nil
