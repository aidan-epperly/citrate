--!strict
-- A membrane hung from a rod: a unit square, pinned (u = 0) along the middle half of its top
-- edge and free (du/dn = 0) elsewhere, sagging under its own weight: Laplacian(u) = -1 in
-- units of the weight over the tension (luau -O2 --codegen bench/lightning.luau).
--
-- Where the pinned part of the edge meets the free part, the solution behaves like
-- r^(1/2). The lightning solver (rational functions with poles clustered there) converges
-- like exp(-C sqrt(N)); a finite-volume grid converges algebraically. Both are compared on
-- the sag at the middle of the bottom edge.

local root = require("../src")
local lightning, f64, krylov = root.lightning, root.f64, root.krylov

local function zero(): number
	return 0
end
local vertices = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0.75, 1 }, { 0.25, 1 }, { 0, 1 } }
local conditions: { root.LightningCondition } = {
	{ kind = "neumann", value = zero },
	{ kind = "neumann", value = zero },
	{ kind = "neumann", value = zero },
	{ kind = "dirichlet", value = zero }, -- the rod
	{ kind = "neumann", value = zero },
	{ kind = "neumann", value = zero },
}

local function lightning_sag(N: number): (number, number, number, number)
	local t0 = os.clock()
	local S = lightning.laplace(vertices, conditions, { poles = N, source = -1 })
	return S.eval(0.5, 0), S.error, S.unknowns, os.clock() - t0
end

local reference = lightning_sag(40)
print(string.format("sag at the bottom middle: %.15f (lightning, 40 poles per vertex)", reference))
print("lightning:")
for _, N in { 4, 8, 12, 16, 20, 24, 28, 32 } do
	local u, res, unknowns, t = lightning_sag(N)
	print(string.format("  %2d poles/vertex  %4d unknowns  error %.1e  residual %.1e  %6.1f ms", N, unknowns, math.abs(u - reference), res, t * 1000))
end

-- Finite volumes on the vertex grid (spacing 1 / m; half cells on the boundary), which give
-- a symmetric positive definite system; CG with the library's multigrid.
local function grid_sag(m: number): (number, number, number)
	local t0 = os.clock()
	local h = 1 / m
	local id = {}
	local n = 0
	local function pinned(i: number, j: number): boolean
		return j == m and i >= m / 4 and i <= 3 * m / 4
	end
	for j = 0, m do
		for i = 0, m do
			if not pinned(i, j) then
				n += 1
				id[j * (m + 1) + i] = n
			end
		end
	end
	local ii, jj, vv = {}, {}, {}
	local rhs = f64.create(n)
	local function add(a: number, b: number, v: number)
		table.insert(ii, a)
		table.insert(jj, b)
		table.insert(vv, v)
	end
	for j = 0, m do
		for i = 0, m do
			local a = id[j * (m + 1) + i]
			if a then
				-- the control volume: halved at each boundary it touches
				local wx = if i == 0 or i == m then 0.5 else 1
				local wy = if j == 0 or j == m then 0.5 else 1
				f64.set(rhs, a, wx * wy * h * h)
				local diag = 0
				for _, d in { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } } do
					local i2, j2 = i + d[1], j + d[2]
					if i2 >= 0 and i2 <= m and j2 >= 0 and j2 <= m then
						-- the face between the cells: its length over the distance h
						local w = if d[1] ~= 0 then wy else wx
						diag += w
						local b = id[j2 * (m + 1) + i2]
						if b then
							add(a, b, -w)
						end
					end
				end
				add(a, a, diag)
			end
		end
	end
	local A = f64.bsparse(n, n, 1, ii, jj, f64.from(vv))
	A.symmetric = true
	local H = f64.bsparse_amg(A)
	local options: any = { tol = 1e-12, maxiter = 500, precond = H.apply }
	local x = krylov.cg(f64.space, function(out: buffer, v: buffer): buffer
		return f64.bsparse_matvec(out, A, v)
	end, rhs, options)
	return f64.get(x, id[m // 2]), n, os.clock() - t0
end

print("finite volumes (CG with multigrid):")
for _, m in { 16, 32, 64, 128, 256 } do
	local u, n, t = grid_sag(m)
	print(string.format("  m = %3d  %6d unknowns  error %.1e  %8.1f ms", m, n, math.abs(u - reference), t * 1000))
end
