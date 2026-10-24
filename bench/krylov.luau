--!strict
-- Conjugate gradients on a 200 by 200 grid Laplacian (standalone: luau -O2 --codegen
-- bench/krylov.luau): time per iteration with and without the Jacobi preconditioner. The
-- vector updates are fused (Space.cg_update) and the elementwise kernels unrolled.

local root = require("../src")
local f64, krylov = root.f64, root.krylov

local g = 200
local ii: { number }, jj: { number }, vv: { number } = {}, {}, {}
for a = 0, g - 1 do
	for b = 0, g - 1 do
		local j = a * g + b + 1
		local function add(i: number, v: number)
			table.insert(ii, i)
			table.insert(jj, j)
			table.insert(vv, v)
		end
		add(j, 4.01)
		if a > 0 then
			add(j - g, -1)
		end
		if b > 0 then
			add(j - 1, -1)
		end
		if b < g - 1 then
			add(j + 1, -1)
		end
		if a < g - 1 then
			add(j + g, -1)
		end
	end
end
local S = f64.sparse(g * g, g * g, ii, jj, f64.from(vv))
S.symmetric = true
local rhs = f64.fill(f64.create(g * g), 1)
local op = function(out: buffer, x: buffer): buffer
	return f64.sparse_matvec(out, S, x)
end
local preconditioners: { { name: string, apply: ((buffer, buffer) -> buffer)? } } = {
	{ name = "none", apply = nil },
	{ name = "Jacobi", apply = f64.sparse_jacobi(S) },
}
for _, pre in preconditioners do
	local options: any = { tol = 1e-8, maxiter = 5000, precond = pre.apply }
	local _, info = krylov.cg(f64.space, op, rhs, options)
	local best = math.huge
	for _ = 1, 3 do
		local t0 = os.clock()
		krylov.cg(f64.space, op, rhs, options)
		best = math.min(best, os.clock() - t0)
	end
	print(string.format("CG, 200x200 Laplacian, %-6s %4d iterations  %7.2f ms  (%.0f us per iteration)", pre.name, info.iterations, best * 1000, best * 1e6 / info.iterations))
end
