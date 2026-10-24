--!strict
--!native
--!optimize 2

-- A mass-spring cloth: an nx by ny grid of particles joined by structural (grid
-- neighbors), shear (diagonals) and bending (two apart) springs with damping along each
-- spring, plus gravity and air drag. Positions and velocities are f64 arrays of 3 values
-- per particle (x, y, z), particle p = (j - 1) nx + i at components 3p - 2 .. 3p.
--
-- force and jacobian plug into ode.implicit_mechanics; the Jacobians are sparse with a
-- fixed pattern built once (3 by 3 blocks for each particle and each spring), and each
-- call refills their values. In the stiffness Jacobian, a compressed spring's transverse
-- term (1 - L/l)(I - u u^T) is dropped when negative (Choi and Ko 2002), so K stays
-- negative semidefinite and the implicit system positive definite. The damping force's
-- dependence on positions is left out of K (it is small and would break the symmetry).

local root = require("../../src")
local f64 = root.f64

type Sparse = root.Sparse

export type Params = {
	nx: number,
	ny: number,
	spacing: number?,
	mass: number?,
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
	for j = 1, ny do
		for i = 1, nx do
			local p = index(i, j)
			f64.set(q0, 3 * p - 2, (i - 1) * spacing)
			f64.set(q0, 3 * p - 1, 0)
			f64.set(q0, 3 * p, (j - 1) * spacing)
		end
	end
	for s = 1, ns do
		local a, b = sa[s], sb[s]
		local dx = f64.get(q0, 3 * b - 2) - f64.get(q0, 3 * a - 2)
		local dz = f64.get(q0, 3 * b) - f64.get(q0, 3 * a)
		rest[s] = math.sqrt(dx * dx + dz * dz)
	end
	local pinned = params.pinned or { index(1, 1), index(nx, 1) }
	local fixed = {}
	for _, p in pinned do
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
	local K: Sparse = { rows = n, cols = n, colptr = pattern.colptr, rowind = pattern.rowind, values = Kvalues }
	local D: Sparse = { rows = n, cols = n, colptr = pattern.colptr, rowind = pattern.rowind, values = Dvalues }

	local model: any = {
		nx = nx,
		ny = ny,
		particles = np,
		springs = ns,
		q0 = q0,
		v0 = f64.create(n),
		fixed = fixed,
		mass = mass,
		spring_a = sa,
		spring_b = sb,
	}

	-- F = gravity + wind + springs + spring damping + air drag. model.wind is an
	-- acceleration (x, y, z) applied to every particle (zero by default).
	model.wind = { 0, 0, 0 }
	function model.force(_: number, q: buffer, v: buffer, out: buffer): buffer
		local wx, wy, wz = mass * model.wind[1], mass * model.wind[2], mass * model.wind[3]
		for p = 1, np do
			local o = 3 * (p - 1) * 8
			buffer.writef64(out, o, wx - air * buffer.readf64(v, o))
			buffer.writef64(out, o + 8, wy - mass * gravity - air * buffer.readf64(v, o + 8))
			buffer.writef64(out, o + 16, wz - air * buffer.readf64(v, o + 16))
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

	-- K = dF/dq and D = dF/dv, refilled in place (the returned matrices are reused).
	function model.jacobian(_: number, q: buffer, _: buffer): (Sparse, Sparse)
		buffer.fill(Kvalues, 0, 0)
		buffer.fill(Dvalues, 0, 0)
		local m = table.create(9, 0)
		local dm = table.create(9, 0)
		for s = 1, ns do
			local oa, ob = 24 * (sa[s] - 1), 24 * (sb[s] - 1)
			local dx = buffer.readf64(q, ob) - buffer.readf64(q, oa)
			local dy = buffer.readf64(q, ob + 8) - buffer.readf64(q, oa + 8)
			local dz = buffer.readf64(q, ob + 16) - buffer.readf64(q, oa + 16)
			local l = math.sqrt(dx * dx + dy * dy + dz * dz)
			if l > 0 then
				local u = { dx / l, dy / l, dz / l }
				local k = stiffness[s]
				local t = math.max(1 - rest[s] / l, 0)
				for r = 1, 3 do
					for c = 1, 3 do
						local uu = u[r] * u[c]
						m[(r - 1) * 3 + c] = k * (t * ((if r == c then 1 else 0) - uu) + uu)
						dm[(r - 1) * 3 + c] = damping * uu
					end
				end
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

	-- The implicit system for ode.implicit_mechanics' system option, from the springs
	-- directly: with B_s = h^2 Kb_s + h c u u^T for spring s = (a, b),
	--   A x = (M + h air) x + sum_s B_s (x_a - x_b) on a and B_s (x_b - x_a) on b,
	-- a block-Jacobi preconditioner (each particle's 3 by 3 diagonal block inverted), and
	-- K v. Work arrays are reused across calls.
	local blocks = buffer.create(8 * 9 * ns)
	local kblocks = buffer.create(8 * 9 * ns)
	local pinv = buffer.create(8 * 9 * np)
	function model.system(_: number, q: buffer, v: buffer, h: number): ((buffer, buffer) -> buffer, (buffer, buffer) -> buffer, buffer)
		local base = mass + h * air
		-- Diagonal blocks start as base I.
		buffer.fill(pinv, 0, 0)
		for p = 0, np - 1 do
			local o = 72 * p
			buffer.writef64(pinv, o, base)
			buffer.writef64(pinv, o + 32, base)
			buffer.writef64(pinv, o + 64, base)
		end
		local Kv = f64.create(n)
		for s = 1, ns do
			local a, b = sa[s], sb[s]
			local oa, ob = 24 * (a - 1), 24 * (b - 1)
			local dx = buffer.readf64(q, ob) - buffer.readf64(q, oa)
			local dy = buffer.readf64(q, ob + 8) - buffer.readf64(q, oa + 8)
			local dz = buffer.readf64(q, ob + 16) - buffer.readf64(q, oa + 16)
			local l = math.sqrt(dx * dx + dy * dy + dz * dz)
			local ux, uy, uz = 0, 0, 0
			if l > 0 then
				ux, uy, uz = dx / l, dy / l, dz / l
			end
			local k = stiffness[s]
			local tt = if l > 0 then math.max(1 - rest[s] / l, 0) else 0
			-- Kb = k ((1 - tt) u u^T + tt I) and B = h^2 Kb + h c u u^T, row-major.
			local cu, ci = k * (1 - tt), k * tt
			local bu, bi = h * h * cu + h * damping, h * h * ci
			local o = 72 * (s - 1)
			local xx, xy, xz, yy, yz, zz = ux * ux, ux * uy, ux * uz, uy * uy, uy * uz, uz * uz
			buffer.writef64(kblocks, o, cu * xx + ci)
			buffer.writef64(kblocks, o + 8, cu * xy)
			buffer.writef64(kblocks, o + 16, cu * xz)
			buffer.writef64(kblocks, o + 24, cu * xy)
			buffer.writef64(kblocks, o + 32, cu * yy + ci)
			buffer.writef64(kblocks, o + 40, cu * yz)
			buffer.writef64(kblocks, o + 48, cu * xz)
			buffer.writef64(kblocks, o + 56, cu * yz)
			buffer.writef64(kblocks, o + 64, cu * zz + ci)
			buffer.writef64(blocks, o, bu * xx + bi)
			buffer.writef64(blocks, o + 8, bu * xy)
			buffer.writef64(blocks, o + 16, bu * xz)
			buffer.writef64(blocks, o + 24, bu * xy)
			buffer.writef64(blocks, o + 32, bu * yy + bi)
			buffer.writef64(blocks, o + 40, bu * yz)
			buffer.writef64(blocks, o + 48, bu * xz)
			buffer.writef64(blocks, o + 56, bu * yz)
			buffer.writef64(blocks, o + 64, bu * zz + bi)
			-- Both endpoints' diagonal blocks gain B.
			local pa, pb = 72 * (a - 1), 72 * (b - 1)
			for e = 0, 64, 8 do
				local val = buffer.readf64(blocks, o + e)
				buffer.writef64(pinv, pa + e, buffer.readf64(pinv, pa + e) + val)
				buffer.writef64(pinv, pb + e, buffer.readf64(pinv, pb + e) + val)
			end
			-- K v: row a gets Kb (v_b - v_a), row b the opposite.
			local wx = buffer.readf64(v, ob) - buffer.readf64(v, oa)
			local wy = buffer.readf64(v, ob + 8) - buffer.readf64(v, oa + 8)
			local wz = buffer.readf64(v, ob + 16) - buffer.readf64(v, oa + 16)
			local rx = (cu * xx + ci) * wx + cu * xy * wy + cu * xz * wz
			local ry = cu * xy * wx + (cu * yy + ci) * wy + cu * yz * wz
			local rz = cu * xz * wx + cu * yz * wy + (cu * zz + ci) * wz
			buffer.writef64(Kv, oa, buffer.readf64(Kv, oa) + rx)
			buffer.writef64(Kv, oa + 8, buffer.readf64(Kv, oa + 8) + ry)
			buffer.writef64(Kv, oa + 16, buffer.readf64(Kv, oa + 16) + rz)
			buffer.writef64(Kv, ob, buffer.readf64(Kv, ob) - rx)
			buffer.writef64(Kv, ob + 8, buffer.readf64(Kv, ob + 8) - ry)
			buffer.writef64(Kv, ob + 16, buffer.readf64(Kv, ob + 16) - rz)
		end
		-- Invert each diagonal block (symmetric positive definite, 3 by 3) in place.
		for p = 0, np - 1 do
			local o = 72 * p
			local m11, m12, m13 = buffer.readf64(pinv, o), buffer.readf64(pinv, o + 8), buffer.readf64(pinv, o + 16)
			local m22, m23, m33 = buffer.readf64(pinv, o + 32), buffer.readf64(pinv, o + 40), buffer.readf64(pinv, o + 64)
			local c11 = m22 * m33 - m23 * m23
			local c12 = m13 * m23 - m12 * m33
			local c13 = m12 * m23 - m13 * m22
			local c22 = m11 * m33 - m13 * m13
			local c23 = m12 * m13 - m11 * m23
			local c33 = m11 * m22 - m12 * m12
			local id = 1 / (m11 * c11 + m12 * c12 + m13 * c13)
			buffer.writef64(pinv, o, c11 * id)
			buffer.writef64(pinv, o + 8, c12 * id)
			buffer.writef64(pinv, o + 16, c13 * id)
			buffer.writef64(pinv, o + 24, c12 * id)
			buffer.writef64(pinv, o + 32, c22 * id)
			buffer.writef64(pinv, o + 40, c23 * id)
			buffer.writef64(pinv, o + 48, c13 * id)
			buffer.writef64(pinv, o + 56, c23 * id)
			buffer.writef64(pinv, o + 64, c33 * id)
		end
		local function applyA(out: buffer, x: buffer): buffer
			for i = 0, n - 1 do
				buffer.writef64(out, 8 * i, base * buffer.readf64(x, 8 * i))
			end
			for s = 1, ns do
				local oa, ob = 24 * (sa[s] - 1), 24 * (sb[s] - 1)
				local o = 72 * (s - 1)
				local wx = buffer.readf64(x, oa) - buffer.readf64(x, ob)
				local wy = buffer.readf64(x, oa + 8) - buffer.readf64(x, ob + 8)
				local wz = buffer.readf64(x, oa + 16) - buffer.readf64(x, ob + 16)
				local rx = buffer.readf64(blocks, o) * wx + buffer.readf64(blocks, o + 8) * wy + buffer.readf64(blocks, o + 16) * wz
				local ry = buffer.readf64(blocks, o + 24) * wx + buffer.readf64(blocks, o + 32) * wy + buffer.readf64(blocks, o + 40) * wz
				local rz = buffer.readf64(blocks, o + 48) * wx + buffer.readf64(blocks, o + 56) * wy + buffer.readf64(blocks, o + 64) * wz
				buffer.writef64(out, oa, buffer.readf64(out, oa) + rx)
				buffer.writef64(out, oa + 8, buffer.readf64(out, oa + 8) + ry)
				buffer.writef64(out, oa + 16, buffer.readf64(out, oa + 16) + rz)
				buffer.writef64(out, ob, buffer.readf64(out, ob) - rx)
				buffer.writef64(out, ob + 8, buffer.readf64(out, ob + 8) - ry)
				buffer.writef64(out, ob + 16, buffer.readf64(out, ob + 16) - rz)
			end
			return out
		end
		local function precond(out: buffer, r: buffer): buffer
			for p = 0, np - 1 do
				local o, x = 72 * p, 24 * p
				local r1, r2, r3 = buffer.readf64(r, x), buffer.readf64(r, x + 8), buffer.readf64(r, x + 16)
				buffer.writef64(out, x, buffer.readf64(pinv, o) * r1 + buffer.readf64(pinv, o + 8) * r2 + buffer.readf64(pinv, o + 16) * r3)
				buffer.writef64(out, x + 8, buffer.readf64(pinv, o + 24) * r1 + buffer.readf64(pinv, o + 32) * r2 + buffer.readf64(pinv, o + 40) * r3)
				buffer.writef64(out, x + 16, buffer.readf64(pinv, o + 48) * r1 + buffer.readf64(pinv, o + 56) * r2 + buffer.readf64(pinv, o + 64) * r3)
			end
			return out
		end
		return applyA, precond, Kv
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
			v = f64.axpy(v, hs / mass, F, v)
			v = f64.mul(v, free, v)
			q = f64.axpy(q, hs, v, q)
		end
		return q, v
	end

	return model
end

return cloth
