--!strict
-- Sparse products (standalone: luau -O2 --codegen bench/sparse.luau, or in Roblox).
--
-- Cost model: native Luau is not limited by memory bandwidth here but by instructions,
-- roughly b = 1.6 ns per loop iteration plus a = 0.25 ns per buffer read (and a few ns per
-- row), so a kernel handling E entries per iteration with R reads costs (b + a R) / E per
-- entry. The predictions printed next to each time come from it.

local root = require("../src")
local cloth = require("../demos/cloth/cloth")
local f64, krylov, ode = root.f64, root.krylov, root.ode

local A_READ, B_ITER = 0.25, 1.6

local function time(f: () -> (), reps: number): number
	local best = math.huge
	for _ = 1, 5 do
		local t0 = os.clock()
		for _ = 1, reps do
			f()
		end
		best = math.min(best, (os.clock() - t0) / reps)
	end
	return best
end

local function ns_per_entry(f: () -> (), nnz: number): number
	return time(f, math.max(1, math.floor(2e7 / nnz))) * 1e9 / nnz
end

local function vector(n: number): buffer
	local x = f64.create(n)
	for i = 1, n do
		f64.set(x, i, math.sin(i))
	end
	return x
end

local function laplacian(g: number): any
	local ii, jj, vv = {}, {}, {}
	local function add(i: number, j: number, v: number)
		table.insert(ii, i)
		table.insert(jj, j)
		table.insert(vv, v)
	end
	for a = 0, g - 1 do
		for b = 0, g - 1 do
			local j = a * g + b + 1
			add(j, j, 4)
			if a > 0 then
				add(j - g, j, -1)
			end
			if b > 0 then
				add(j - 1, j, -1)
			end
			if b < g - 1 then
				add(j + 1, j, -1)
			end
			if a < g - 1 then
				add(j + g, j, -1)
			end
		end
	end
	return f64.sparse(g * g, g * g, ii, jj, f64.from(vv))
end

print("products (ns per stored entry; model in parentheses)")
do
	local model = cloth.new({ nx = 30, ny = 30, spacing = 1 / 29, k_struct = 5000, k_shear = 1000, k_bend = 200 })
	local K = model.jacobian(0, model.q0, model.v0)
	local BK = model.block_jacobian(0, model.q0, model.v0)
	local n, nnz = K.rows, f64.sparse_nnz(K)
	local x, o = vector(n), f64.create(n)
	local diff = f64.norm(f64.sub(nil, f64.bsparse_matvec(nil, BK, x), f64.sparse_matvec(nil, K, x)))
	assert(diff <= 1e-9 * f64.norm(f64.sparse_matvec(nil, K, x)), "block_jacobian disagrees with jacobian")
	local general = table.clone(K)
	general.symmetric = nil
	print(string.format("  cloth 30x30 stiffness: %d rows, %d entries (%.0f per row)", n, nnz, nnz / n))
	print(string.format("    general (scatter, unrolled by 4)   %5.2f", ns_per_entry(function()
		f64.sparse_matvec(o, general, x)
	end, nnz)))
	print(string.format("    marked symmetric (gather)          %5.2f  (%.2f)", ns_per_entry(function()
		f64.sparse_matvec(o, K, x)
	end, nnz), 3 * A_READ + B_ITER / 4))
	print(string.format("    3 by 3 blocks                      %5.2f  (%.2f)", ns_per_entry(function()
		f64.bsparse_matvec(o, BK, x)
	end, f64.bsparse_nnz(BK)), A_READ * (1 + 1 / 3 + 1 / 9) + B_ITER / 9))
end
do
	local L = laplacian(300)
	local n, nnz = L.rows, f64.sparse_nnz(L)
	local x, o = vector(n), f64.create(n)
	local general = table.clone(L)
	local marked = table.clone(L)
	marked.symmetric = true
	print(string.format("  2D Laplacian 300x300: %d rows, %d entries (%.0f per row; short rows: per-row costs dominate)", n, nnz, nnz / n))
	print(string.format("    general (scatter)                  %5.2f", ns_per_entry(function()
		f64.sparse_matvec(o, general, x)
	end, nnz)))
	print(string.format("    marked symmetric (gather)          %5.2f", ns_per_entry(function()
		f64.sparse_matvec(o, marked, x)
	end, nnz)))
end

print("cloth system M - h D - h^2 K (60 Hz step, k = 5000): CG to 1e-6, Jacobi against block Jacobi")
do
	local h = 1 / 60
	local model = cloth.new({ nx = 30, ny = 30, spacing = 1 / 29, k_struct = 5000, k_shear = 1000, k_bend = 200 })
	-- a deformed state, so springs point in all directions
	local q = f64.copy(nil, model.q0)
	for i = 1, f64.length(q) do
		f64.set(q, i, f64.get(q, i) + 0.02 * math.sin(7 * i))
	end
	local BK, BD = model.block_jacobian(0, q, model.v0)
	local values = f64.axpy(nil, -h, BD.values, f64.scale(nil, -h * h, BK.values))
	local A: any = { rows = BK.rows, cols = BK.cols, block = 3, rowptr = BK.rowptr, colind = BK.colind, values = values, symmetric = true }
	local mass = model.mass
	for p = 1, A.rows // 3 do
		local blk = f64.bsparse_block(A, p, p) :: number
		for r = 0, 2 do
			local e = 9 * (blk - 1) + 4 * r + 1
			f64.set(values, e, f64.get(values, e) + (if type(mass) == "number" then mass else f64.get(mass, 3 * p - 2 + r)))
		end
	end
	local op = function(out: buffer, v: buffer): buffer
		return f64.bsparse_matvec(out, A, v)
	end
	local rhs = vector(A.rows)
	local preconditioners: { { name: string, apply: (buffer, buffer) -> buffer } } = {
		{ name = "Jacobi", apply = f64.sparse_jacobi(f64.bsparse_sparse(A)) },
		{ name = "block Jacobi", apply = f64.bsparse_jacobi(A) },
	}
	for _, pre in preconditioners do
		local options: any = { tol = 1e-6, maxiter = 1000, precond = pre.apply }
		local _, info = krylov.cg(f64.space, op, rhs, options)
		local ms = time(function()
			krylov.cg(f64.space, op, rhs, options)
		end, 5) * 1000
		print(string.format("  %-13s %4d iterations  %6.2f ms", pre.name, info.iterations, ms))
	end
end

print("cloth implicit step (30x30, k = 5000, one 60 Hz frame)")
do
	local function frame(kind: string): (number, number, string)
		local model = cloth.new({ nx = 30, ny = 30, spacing = 1 / 29, k_struct = 5000, k_shear = 1000, k_bend = 200 })
		local o: any = { mass = model.mass, fixed = model.fixed, tol = 1e-4 }
		if kind == "sparse" then
			o.jacobian = model.jacobian
		else
			o.jacobian = model.block_jacobian
			if kind == "multigrid" then
				o.amg = { smooth = false }
			end
		end
		local it = ode.implicit_mechanics(f64.space, model.force, 0, model.q0, model.v0, o)
		for _ = 1, 60 do
			it:step(1 / 60)
		end
		local its0 = it.stats.iterations
		local frames = 120
		local t0 = os.clock()
		for _ = 1, frames do
			it:step(1 / 60)
		end
		return (os.clock() - t0) / frames * 1000, (it.stats.iterations - its0) / frames, it.path
	end
	for _, kind in { "sparse", "block", "multigrid" } do
		local ms, its, path = frame(kind)
		print(string.format("  %-9s %6.2f ms/frame  %5.1f CG its/frame  (path %s)", kind, ms, its, path))
	end
end
