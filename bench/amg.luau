--!strict
-- Multigrid against block Jacobi on several symmetric positive definite problems
-- (standalone: luau -O2 --codegen bench/amg.luau): CG iterations and time to a 1e-8
-- relative residual, setup and update (refill) times; then cloth frames through
-- ode.implicit_mechanics.

local root = require("../src")
local cloth = require("../demos/cloth/cloth")
local f64, krylov = root.f64, root.krylov

type Problem = { name: string, A: any, fixed: { number } }

-- A block matrix from per-block contributions: add(I, J, block) with block a 9-entry
-- (or b^2) column-major table; repeated positions are summed.
local function assembler(nodes: number, b: number): ((number, number, { number }) -> (), () -> any)
	local ii, jj, vals = {}, {}, {}
	local function add(I: number, J: number, blk: { number })
		table.insert(ii, I)
		table.insert(jj, J)
		for _, v in blk do
			table.insert(vals, v)
		end
	end
	local function build(): any
		return f64.bsparse(b * nodes, b * nodes, b, ii, jj, f64.from(vals))
	end
	return add, build
end

-- Scalar 5-point Laplacian on a g by g grid, coefficient ey in y (anisotropic if small),
-- plus a small shift.
local function laplacian(g: number, ey: number): Problem
	local add, build = assembler(g * g, 1)
	for a = 0, g - 1 do
		for c = 0, g - 1 do
			local I = a * g + c + 1
			add(I, I, { 2 + 2 * ey + 1e-4 })
			if c > 0 then
				add(I, I - 1, { -1 })
			end
			if c < g - 1 then
				add(I, I + 1, { -1 })
			end
			if a > 0 then
				add(I, I - g, { -ey })
			end
			if a < g - 1 then
				add(I, I + g, { -ey })
			end
		end
	end
	local A = build()
	A.symmetric = true
	return { name = string.format("Laplacian %dx%d%s (b=1)", g, g, if ey ~= 1 then string.format(", ey=%g", ey) else ""), A = A, fixed = {} }
end

-- A spring lattice in d dimensions linearized at rest: A = M + h^2 K, with K from springs
-- k u u^T between neighbors (and diagonals), nodes on one face pinned.
local function lattice(dims: { number }, k: number, name: string): Problem
	local d = #dims
	local count = 1
	for _, s in dims do
		count *= s
	end
	local function index(c: { number }): number
		local i, mul = 0, 1
		for a = 1, d do
			i += c[a] * mul
			mul *= dims[a]
		end
		return i + 1
	end
	local add, build = assembler(count, d)
	local h, m = 1 / 60, 1 / count
	local function block_of(u: { number }, f: number): { number }
		local blk = table.create(d * d, 0)
		for s = 1, d do
			for r = 1, d do
				blk[(s - 1) * d + r] = f * u[r] * u[s]
			end
		end
		return blk
	end
	-- neighbor offsets: axis steps and face diagonals
	local offsets: { { number } } = {}
	for a = 1, d do
		local o = table.create(d, 0)
		o[a] = 1
		table.insert(offsets, o)
		for c = a + 1, d do
			for _, sgn in { 1, -1 } do
				local q = table.create(d, 0)
				q[a], q[c] = 1, sgn
				table.insert(offsets, q)
			end
		end
	end
	local coord = table.create(d, 0)
	local fixed = {}
	for i = 1, count do
		local rest = i - 1
		for a = 1, d do
			coord[a] = rest % dims[a]
			rest //= dims[a]
		end
		local I = index(coord)
		local diag = table.create(d * d, 0)
		for r = 1, d do
			diag[(r - 1) * d + r] = m
		end
		add(I, I, diag)
		if coord[1] == 0 then
			for c = 1, d do
				table.insert(fixed, d * (I - 1) + c)
			end
		end
		for _, off in offsets do
			local other = table.create(d, 0)
			local ok = true
			local len = 0
			for a = 1, d do
				other[a] = coord[a] + off[a]
				ok = ok and other[a] >= 0 and other[a] < dims[a]
				len += off[a] * off[a]
			end
			if ok then
				local J = index(other)
				local u = table.create(d, 0)
				for a = 1, d do
					u[a] = off[a] / math.sqrt(len)
				end
				local f = h * h * k / math.sqrt(len)
				add(I, I, block_of(u, f))
				add(J, J, block_of(u, f))
				add(I, J, block_of(u, -f))
				add(J, I, block_of(u, -f))
			end
		end
	end
	local A = build()
	A.symmetric = true
	return { name = name, A = A, fixed = fixed }
end

-- The cloth's implicit system M - h D - h^2 K at a deformed state.
local function cloth_problem(size: number): Problem
	local h = 1 / 60
	local model = cloth.new({ nx = size, ny = size, spacing = 1 / (size - 1), k_struct = 5000, k_shear = 1000, k_bend = 200 })
	local q = f64.copy(nil, model.q0)
	for i = 1, f64.length(q) do
		f64.set(q, i, f64.get(q, i) + 0.02 / size * 30 * math.sin(7 * i))
	end
	local K, D = model.block_jacobian(0, q, model.v0)
	local values = f64.axpy(nil, -h, D.values, f64.scale(nil, -h * h, K.values))
	local A: any = { rows = K.rows, cols = K.cols, block = 3, rowptr = K.rowptr, colind = K.colind, values = values, symmetric = true }
	local mass = model.mass
	for p = 1, A.rows // 3 do
		local blk = f64.bsparse_block(A, p, p) :: number
		for r = 0, 2 do
			local e = 9 * (blk - 1) + 4 * r + 1
			f64.set(values, e, f64.get(values, e) + (if type(mass) == "number" then mass else f64.get(mass, 3 * p - 2 + r)))
		end
	end
	return { name = string.format("cloth %dx%d (b=3)", size, size), A = A, fixed = model.fixed }
end

local problems: { Problem } = {
	laplacian(100, 1),
	laplacian(100, 0.01),
	lattice({ 60, 60 }, 1e4, "2D spring lattice 60x60 (b=2)"),
	lattice({ 12, 12, 12 }, 1e4, "3D soft-body lattice 12^3 (b=3)"),
	cloth_problem(30),
	cloth_problem(60),
}

for _, P in problems do
	local A = P.A
	local n = A.rows
	-- The system as the solvers see it: fixed components held (identity rows).
	local isfixed = table.create(n, false)
	for _, c in P.fixed do
		isfixed[c] = true
	end
	local xf = f64.create(n)
	local op = function(out: buffer, x: buffer): buffer
		xf = f64.copy(xf, x)
		for _, c in P.fixed do
			f64.set(xf, c, 0)
		end
		out = f64.bsparse_matvec(out, A, xf)
		for _, c in P.fixed do
			f64.set(out, c, f64.get(x, c))
		end
		return out
	end
	local rhs = f64.create(n)
	for i = 1, n do
		f64.set(rhs, i, if isfixed[i] then 0 else math.sin(0.37 * i) + 0.5)
	end
	print(string.format("%s: %d unknowns, %d stored entries", P.name, n, f64.bsparse_nnz(A)))
	local function run(label: string, setup: () -> ((buffer, buffer) -> buffer, (() -> ())?, string?))
		local t0 = os.clock()
		local M, refill, info = setup()
		local tsetup = os.clock() - t0
		local options: any = { tol = 1e-8, maxiter = 3000, precond = M }
		local _, res = krylov.cg(f64.space, op, rhs, options)
		local best = math.huge
		for _ = 1, 3 do
			local t1 = os.clock()
			krylov.cg(f64.space, op, rhs, options)
			best = math.min(best, os.clock() - t1)
		end
		local tupdate = 0
		if refill then
			local t2 = os.clock()
			refill()
			tupdate = os.clock() - t2
		end
		print(string.format("  %-22s %5d its  solve %8.2f ms  setup %7.2f ms  update %6.2f ms%s%s", label, res.iterations, best * 1000, tsetup * 1000, tupdate * 1000,
			if res.converged then "" else "  (not converged)", if info then "  " .. info else ""))
	end
	run("block Jacobi", function()
		local masked: any = table.clone(A)
		return f64.bsparse_jacobi(masked), nil, nil
	end)
	local variants: { { name: string, smooth: boolean, alpha: number } } = {
		{ name = "plain aggregation", smooth = false, alpha = 1 },
		{ name = "plain, alpha 1.5", smooth = false, alpha = 1.5 },
		{ name = "smoothed aggregation", smooth = true, alpha = 1 },
	}
	for _, variant in variants do
		run(variant.name, function()
			local H = f64.bsparse_amg(A, { fixed = P.fixed, smooth = variant.smooth, alpha = variant.alpha })
			return H.apply, function()
				H.update(A)
			end, "levels " .. table.concat(H.sizes, " ")
		end)
	end
end

local ode = root.ode
local function frame(size: number, amg: any): (number, number)
	local model = cloth.new({ nx = size, ny = size, spacing = 1 / (size - 1), k_struct = 5000, k_shear = 1000, k_bend = 200 })
	local o: any = { mass = model.mass, fixed = model.fixed, tol = 1e-4, amg = amg }
	o.jacobian = model.block_jacobian
	local it = ode.implicit_mechanics(f64.space, model.force, 0, model.q0, model.v0, o)
	for _ = 1, 60 do it:step(1 / 60) end
	local its0 = it.stats.iterations
	local t0 = os.clock()
	for _ = 1, 120 do it:step(1 / 60) end
	return (os.clock() - t0) / 120 * 1000, (it.stats.iterations - its0) / 120
end
for _, size in { 30, 60 } do
	print(string.format("cloth %dx%d, one 60 Hz frame (tol 1e-4):", size, size))
	local runs: { { name: string, amg: any } } = {
		{ name = "block Jacobi", amg = nil },
		{ name = "AMG plain, refresh 1", amg = { smooth = false } },
		{ name = "AMG smoothed, refresh 1", amg = { smooth = true } },
		{ name = "AMG smoothed, refresh 5", amg = { smooth = true, refresh = 5 } },
	}
	for _, v in runs do
		local ms, its = frame(size, v.amg)
		print(string.format("  %-26s %7.2f ms  %5.1f its", v.name, ms, its))
	end
end
