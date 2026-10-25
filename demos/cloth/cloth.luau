--!strict
--!native
--!optimize 2

-- A mass-spring cloth: an nx by ny grid of particles joined by structural (grid
-- neighbors), shear (diagonals) and bending (two apart) springs with damping along each
-- spring, plus gravity and air drag. Positions and velocities are f64 arrays of 3 values
-- per particle (x, y, z), particle p = (j - 1) nx + i at components 3p - 2 .. 3p.
--
-- force and block_jacobian plug into ode.implicit_mechanics (with its multigrid option):
-- the Jacobians have a fixed pattern of 3 by 3 blocks (each particle and each spring),
-- built once, and each call refills their values; jacobian gives the same matrices in
-- scalar sparse storage. Both are marked symmetric. In the stiffness Jacobian, a
-- compressed spring's transverse term (1 - L/l)(I - u u^T) is dropped when negative (Choi
-- and Ko 2002), so K stays negative semidefinite and the implicit system positive definite. The damping force's
-- dependence on positions is left out of K (it is small and would break the symmetry).

local root = require("../../src")
local f64 = root.f64

type Sparse = root.Sparse
type BlockSparse = root.BlockSparse

export type Params = {
	nx: number,
	ny: number,
	spacing: number?,
	mass: number?,
	bar: number?,
	pin_top: boolean?,
	hanging: boolean?,
	k_struct: number?,
	k_shear: number?,
	k_bend: number?,
	damping: number?,
	air: number?,
	gravity: number?,
	pinned: { number }?,
}

local cloth = {}

function cloth.new(params: Params): any
	local nx, ny = params.nx, params.ny
	local spacing = params.spacing or 0.1
	local np = nx * ny
	local n = 3 * np
	local mass = params.mass or 0.01
	local damping = params.damping or 0.02
	local air = params.air or 0.001
	local gravity = params.gravity or 9.81
	local function index(i: number, j: number): number
		return (j - 1) * nx + i
	end

	-- Springs: endpoints, rest length, stiffness.
	local sa, sb, rest, stiffness = {}, {}, {}, {}
	local function spring(p: number, q: number, k: number)
		table.insert(sa, p)
		table.insert(sb, q)
		table.insert(stiffness, k)
	end
	local ks, ksh, kb = params.k_struct or 500, params.k_shear or 100, params.k_bend or 20
	for j = 1, ny do
		for i = 1, nx do
			local p = index(i, j)
			if i < nx then
				spring(p, index(i + 1, j), ks)
			end
			if j < ny then
				spring(p, index(i, j + 1), ks)
			end
			if i < nx and j < ny then
				spring(p, index(i + 1, j + 1), ksh)
				spring(index(i + 1, j), index(i, j + 1), ksh)
			end
			if i + 2 <= nx then
				spring(p, index(i + 2, j), kb)
			end
			if j + 2 <= ny then
				spring(p, index(i, j + 2), kb)
			end
		end
	end
	local ns = #sa

	-- The starting state: a flat sheet in the x-z plane at height 0, pinned along the
	-- top edge's two corners by default.
	local q0 = f64.create(n)
	-- params.hanging starts the sheet vertical (in the x-y plane, rows going down) at its
	-- rest lengths instead.
	for j = 1, ny do
		for i = 1, nx do
			local p = index(i, j)
			f64.set(q0, 3 * p - 2, (i - 1) * spacing)
			if params.hanging then
				f64.set(q0, 3 * p - 1, -(j - 1) * spacing)
				f64.set(q0, 3 * p, 0)
			else
				f64.set(q0, 3 * p - 1, 0)
				f64.set(q0, 3 * p, (j - 1) * spacing)
			end
		end
	end
	for s = 1, ns do
		local a, b = sa[s], sb[s]
		local dx = f64.get(q0, 3 * b - 2) - f64.get(q0, 3 * a - 2)
		local dy = f64.get(q0, 3 * b - 1) - f64.get(q0, 3 * a - 1)
		local dz = f64.get(q0, 3 * b) - f64.get(q0, 3 * a)
		rest[s] = math.sqrt(dx * dx + dy * dy + dz * dz)
	end
	-- Pinned: params.pinned, else the whole first row with pin_top, else its two corners.
	local pinned = params.pinned
	if pinned == nil then
		pinned = {}
		if params.pin_top then
			for i = 1, nx do
				table.insert(pinned, index(i, 1))
			end
		else
			pinned = { index(1, 1), index(nx, 1) }
		end
	end
	-- Particle masses: mass each, plus params.bar (a heavy bar's mass) spread over the
	-- last row. pm holds them per particle, masses per component (for the solver).
	local pm = table.create(np, mass)
	local bar = params.bar or 0
	for i = 1, nx do
		pm[index(i, ny)] += bar / nx
	end
	local masses = f64.create(n)
	local inv_masses = f64.create(n)
	for p = 1, np do
		for c = 0, 2 do
			f64.set(masses, 3 * p - 2 + c, pm[p])
			f64.set(inv_masses, 3 * p - 2 + c, 1 / pm[p])
		end
	end
	local fixed = {}
	for _, p in pinned :: { number } do
		for c = 0, 2 do
			table.insert(fixed, 3 * p - 2 + c)
		end
	end

	-- The sparse pattern: every 3 by 3 block (a, a), (a, b), (b, a), (b, b) of each spring,
	-- and each particle's diagonal block.
	local ii, jj = {}, {}
	local function block(p: number, q: number)
		for r = 0, 2 do
			for c = 0, 2 do
				table.insert(ii, 3 * p - 2 + r)
				table.insert(jj, 3 * q - 2 + c)
			end
		end
	end
	for p = 1, np do
		block(p, p)
	end
	for s = 1, ns do
		block(sa[s], sb[s])
		block(sb[s], sa[s])
	end
	local pattern = f64.sparse(n, n, ii, jj, f64.create(#ii))
	-- Position in values (1-based) of entry (r, c): column c's rows are sorted.
	local where: { [number]: number } = {}
	for c = 1, n do
		for p = buffer.readu32(pattern.colptr, 4 * (c - 1)), buffer.readu32(pattern.colptr, 4 * c) - 1 do
			local r = buffer.readu32(pattern.rowind, 4 * p) + 1
			where[(c - 1) * n + r] = p + 1
		end
	end
	-- For each spring, the value positions of its four blocks, in the order (a, a),
	-- (b, b), (a, b), (b, a), 9 each (row-major within the block).
	local slots: { { number } } = {}
	for s = 1, ns do
		local a, b = sa[s], sb[s]
		local list = table.create(36, 0)
		local k = 0
		for _, pair in { { a, a }, { b, b }, { a, b }, { b, a } } do
			for r = 0, 2 do
				for c = 0, 2 do
					k += 1
					list[k] = where[(3 * pair[2] - 3 + c) * n + 3 * pair[1] - 2 + r]
				end
			end
		end
		slots[s] = list
	end
	local diag_slots = table.create(n, 0)
	for i = 1, n do
		diag_slots[i] = where[(i - 1) * n + i]
	end
	local Kvalues, Dvalues = f64.create(#ii), f64.create(#ii)
	local K: Sparse = { rows = n, cols = n, colptr = pattern.colptr, rowind = pattern.rowind, values = Kvalues, symmetric = true }
	local D: Sparse = { rows = n, cols = n, colptr = pattern.colptr, rowind = pattern.rowind, values = Dvalues, symmetric = true }

	-- The same pattern in 3 by 3 blocks: per spring, the block numbers of (a, a), (b, b),
	-- (a, b), (b, a), and each particle's diagonal block.
	local bi, bj = {}, {}
	for q = 1, np do
		table.insert(bi, q)
		table.insert(bj, q)
	end
	for sp = 1, ns do
		table.insert(bi, sa[sp])
		table.insert(bj, sb[sp])
		table.insert(bi, sb[sp])
		table.insert(bj, sa[sp])
	end
	local bpattern = f64.bsparse(n, n, 3, bi, bj, f64.create(9 * #bi))
	local bslots = table.create(4 * ns, 0)
	for sp = 1, ns do
		local a, b = sa[sp], sb[sp]
		bslots[4 * sp - 3] = f64.bsparse_block(bpattern, a, a) :: number
		bslots[4 * sp - 2] = f64.bsparse_block(bpattern, b, b) :: number
		bslots[4 * sp - 1] = f64.bsparse_block(bpattern, a, b) :: number
		bslots[4 * sp] = f64.bsparse_block(bpattern, b, a) :: number
	end
	local bdiag = table.create(np, 0)
	for q = 1, np do
		bdiag[q] = f64.bsparse_block(bpattern, q, q) :: number
	end
	local BKvalues, BDvalues = f64.create(f64.bsparse_nnz(bpattern)), f64.create(f64.bsparse_nnz(bpattern))
	local BK: BlockSparse = { rows = n, cols = n, block = 3, rowptr = bpattern.rowptr, colind = bpattern.colind, values = BKvalues, symmetric = true }
	local BD: BlockSparse = { rows = n, cols = n, block = 3, rowptr = bpattern.rowptr, colind = bpattern.colind, values = BDvalues, symmetric = true }

	local model: any = {
		nx = nx,
		ny = ny,
		particles = np,
		springs = ns,
		q0 = q0,
		v0 = f64.create(n),
		fixed = fixed,
		mass = masses,
		particle_mass = pm,
		min_mass = mass,
		spring_a = sa,
		spring_b = sb,
	}

	-- F = gravity + wind + springs + spring damping + air drag. model.wind is an
	-- acceleration (x, y, z) applied to every particle (zero by default).
	model.wind = { 0, 0, 0 }
	function model.force(_: number, q: buffer, v: buffer, out: buffer): buffer
		local wind = model.wind
		for p = 1, np do
			local o = 3 * (p - 1) * 8
			local m = pm[p]
			buffer.writef64(out, o, m * wind[1] - air * buffer.readf64(v, o))
			buffer.writef64(out, o + 8, m * (wind[2] - gravity) - air * buffer.readf64(v, o + 8))
			buffer.writef64(out, o + 16, m * wind[3] - air * buffer.readf64(v, o + 16))
		end
		for s = 1, ns do
			local oa, ob = 24 * (sa[s] - 1), 24 * (sb[s] - 1)
			local dx = buffer.readf64(q, ob) - buffer.readf64(q, oa)
			local dy = buffer.readf64(q, ob + 8) - buffer.readf64(q, oa + 8)
			local dz = buffer.readf64(q, ob + 16) - buffer.readf64(q, oa + 16)
			local l = math.sqrt(dx * dx + dy * dy + dz * dz)
			if l > 0 then
				local ux, uy, uz = dx / l, dy / l, dz / l
				local rvx = buffer.readf64(v, ob) - buffer.readf64(v, oa)
				local rvy = buffer.readf64(v, ob + 8) - buffer.readf64(v, oa + 8)
				local rvz = buffer.readf64(v, ob + 16) - buffer.readf64(v, oa + 16)
				local mag = stiffness[s] * (l - rest[s]) + damping * (rvx * ux + rvy * uy + rvz * uz)
				local fx, fy, fz = mag * ux, mag * uy, mag * uz
				buffer.writef64(out, oa, buffer.readf64(out, oa) + fx)
				buffer.writef64(out, oa + 8, buffer.readf64(out, oa + 8) + fy)
				buffer.writef64(out, oa + 16, buffer.readf64(out, oa + 16) + fz)
				buffer.writef64(out, ob, buffer.readf64(out, ob) - fx)
				buffer.writef64(out, ob + 8, buffer.readf64(out, ob + 8) - fy)
				buffer.writef64(out, ob + 16, buffer.readf64(out, ob + 16) - fz)
			end
		end
		return out
	end

	-- The mechanical energy: kinetic, gravitational and spring (without wind, which does
	-- work, and damping and drag, which dissipate).
	function model.energy(q: buffer, v: buffer): number
		local e = 0
		for p = 1, np do
			local o = 24 * (p - 1)
			local vx, vy, vz = buffer.readf64(v, o), buffer.readf64(v, o + 8), buffer.readf64(v, o + 16)
			e += pm[p] * (0.5 * (vx * vx + vy * vy + vz * vz) + gravity * buffer.readf64(q, o + 8))
		end
		for s = 1, ns do
			local oa, ob = 24 * (sa[s] - 1), 24 * (sb[s] - 1)
			local dx = buffer.readf64(q, ob) - buffer.readf64(q, oa)
			local dy = buffer.readf64(q, ob + 8) - buffer.readf64(q, oa + 8)
			local dz = buffer.readf64(q, ob + 16) - buffer.readf64(q, oa + 16)
			local stretch = math.sqrt(dx * dx + dy * dy + dz * dz) - rest[s]
			e += 0.5 * stiffness[s] * stretch * stretch
		end
		return e
	end

	-- K = dF/dq and D = dF/dv, refilled in place (the returned matrices are reused).
	-- Scratch tables for the spring blocks, reused for every spring.
	local m = table.create(9, 0)
	local dm = table.create(9, 0)
	local u = table.create(3, 0)
	-- Spring s's stiffness and damping blocks into m and dm (row-major); false if its
	-- ends coincide.
	local function spring_blocks(s: number, q: buffer): boolean
		local oa, ob = 24 * (sa[s] - 1), 24 * (sb[s] - 1)
		local dx = buffer.readf64(q, ob) - buffer.readf64(q, oa)
		local dy = buffer.readf64(q, ob + 8) - buffer.readf64(q, oa + 8)
		local dz = buffer.readf64(q, ob + 16) - buffer.readf64(q, oa + 16)
		local l = math.sqrt(dx * dx + dy * dy + dz * dz)
		if l == 0 then
			return false
		end
		u[1], u[2], u[3] = dx / l, dy / l, dz / l
		local k = stiffness[s]
		local t = math.max(1 - rest[s] / l, 0)
		for r = 1, 3 do
			for c = 1, 3 do
				local uu = u[r] * u[c]
				m[(r - 1) * 3 + c] = k * (t * ((if r == c then 1 else 0) - uu) + uu)
				dm[(r - 1) * 3 + c] = damping * uu
			end
		end
		return true
	end

	function model.jacobian(_: number, q: buffer, _: buffer): (Sparse, Sparse)
		buffer.fill(Kvalues, 0, 0)
		buffer.fill(Dvalues, 0, 0)
		for s = 1, ns do
			if spring_blocks(s, q) then
				local list = slots[s]
				for e = 1, 9 do
					-- (a, a) and (b, b) get -block, (a, b) and (b, a) get +block.
					local pk, pd = m[e], dm[e]
					for blk = 0, 3 do
						local pos = 8 * (list[blk * 9 + e] - 1)
						local sign = if blk < 2 then -1 else 1
						buffer.writef64(Kvalues, pos, buffer.readf64(Kvalues, pos) + sign * pk)
						buffer.writef64(Dvalues, pos, buffer.readf64(Dvalues, pos) + sign * pd)
					end
				end
			end
		end
		for i = 1, n do
			local pos = 8 * (diag_slots[i] - 1)
			buffer.writef64(Dvalues, pos, buffer.readf64(Dvalues, pos) - air)
		end
		return K, D
	end

	-- K and D as 3 by 3 block sparse matrices.
	function model.block_jacobian(_: number, q: buffer, _: buffer): (BlockSparse, BlockSparse)
		buffer.fill(BKvalues, 0, 0)
		buffer.fill(BDvalues, 0, 0)
		-- Written out per entry (the blocks are symmetric): nested loops and table reads
		-- here cost several times the arithmetic.
		local KV, DV = BKvalues, BDvalues
		for s = 1, ns do
			local oa, ob = 24 * (sa[s] - 1), 24 * (sb[s] - 1)
			local dx = buffer.readf64(q, ob) - buffer.readf64(q, oa)
			local dy = buffer.readf64(q, ob + 8) - buffer.readf64(q, oa + 8)
			local dz = buffer.readf64(q, ob + 16) - buffer.readf64(q, oa + 16)
			local l = math.sqrt(dx * dx + dy * dy + dz * dz)
			if l > 0 then
				local ux, uy, uz = dx / l, dy / l, dz / l
				local k = stiffness[s]
				local t = math.max(1 - rest[s] / l, 0)
				-- Kb = k ((1 - t) u u^T + t I), Db = damping u u^T
				local cu, ci = k * (1 - t), k * t
				local xx, xy, xz, yy, yz, zz = ux * ux, ux * uy, ux * uz, uy * uy, uy * uz, uz * uz
				local k11, k12, k13, k22, k23, k33 = cu * xx + ci, cu * xy, cu * xz, cu * yy + ci, cu * yz, cu * zz + ci
				local d11, d12, d13, d22, d23, d33 = damping * xx, damping * xy, damping * xz, damping * yy, damping * yz, damping * zz
				for blk = 0, 3 do
					-- (a, a) and (b, b) get -block, (a, b) and (b, a) get +block.
					local o = 72 * (bslots[4 * s - 3 + blk] - 1)
					local sg = if blk < 2 then -1 else 1
					buffer.writef64(KV, o, buffer.readf64(KV, o) + sg * k11)
					buffer.writef64(KV, o + 8, buffer.readf64(KV, o + 8) + sg * k12)
					buffer.writef64(KV, o + 16, buffer.readf64(KV, o + 16) + sg * k13)
					buffer.writef64(KV, o + 24, buffer.readf64(KV, o + 24) + sg * k12)
					buffer.writef64(KV, o + 32, buffer.readf64(KV, o + 32) + sg * k22)
					buffer.writef64(KV, o + 40, buffer.readf64(KV, o + 40) + sg * k23)
					buffer.writef64(KV, o + 48, buffer.readf64(KV, o + 48) + sg * k13)
					buffer.writef64(KV, o + 56, buffer.readf64(KV, o + 56) + sg * k23)
					buffer.writef64(KV, o + 64, buffer.readf64(KV, o + 64) + sg * k33)
					buffer.writef64(DV, o, buffer.readf64(DV, o) + sg * d11)
					buffer.writef64(DV, o + 8, buffer.readf64(DV, o + 8) + sg * d12)
					buffer.writef64(DV, o + 16, buffer.readf64(DV, o + 16) + sg * d13)
					buffer.writef64(DV, o + 24, buffer.readf64(DV, o + 24) + sg * d12)
					buffer.writef64(DV, o + 32, buffer.readf64(DV, o + 32) + sg * d22)
					buffer.writef64(DV, o + 40, buffer.readf64(DV, o + 40) + sg * d23)
					buffer.writef64(DV, o + 48, buffer.readf64(DV, o + 48) + sg * d13)
					buffer.writef64(DV, o + 56, buffer.readf64(DV, o + 56) + sg * d23)
					buffer.writef64(DV, o + 64, buffer.readf64(DV, o + 64) + sg * d33)
				end
			end
		end
		for p = 1, np do
			for r = 0, 2 do
				local pos = 72 * (bdiag[p] - 1) + 32 * r
				buffer.writef64(BDvalues, pos, buffer.readf64(BDvalues, pos) - air)
			end
		end
		return BK, BD
	end

	-- Symplectic Euler substeps (explicit, as games usually step cloth): v += h F / m,
	-- q += h v, with the pinned components held. Returns q, v.
	function model.explicit_step(q: buffer, v: buffer, h: number, substeps: number): (buffer, buffer)
		local hs = h / substeps
		local F = f64.create(n)
		local free = f64.fill(f64.create(n), 1)
		for _, i in fixed do
			f64.set(free, i, 0)
		end
		for _ = 1, substeps do
			F = model.force(0, q, v, F)
			F = f64.mul(F, inv_masses, F)
			v = f64.axpy(v, hs, F, v)
			v = f64.mul(v, free, v)
			q = f64.axpy(q, hs, v, q)
		end
		return q, v
	end

	return model
end

return cloth
