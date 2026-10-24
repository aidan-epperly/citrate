--!strict
--!native
--!optimize 2

-- An aggregation multigrid preconditioner for spring networks: systems whose matrix is
--   (A x)_a = D_a x_a - sum over edges e = (a, b) of B_e x_b,
-- with 3 by 3 symmetric blocks D_a (each node's diagonal block) and B_e (each edge's
-- coupling), as the implicit step of a mass-spring model produces (A = M - h D - h^2 K).
-- Pinned nodes are left out (their rows are the identity in the solver).
--
-- Levels come from greedy aggregation: an unaggregated node and its unaggregated
-- neighbors form an aggregate, and leftovers join a neighboring aggregate; the coarse
-- system is the Galerkin product P^T A P for the piecewise constant prolongation P, which
-- has the same form: a coarse edge's block is the sum of the fine edges' blocks between
-- the two aggregates, and a coarse node's block is the sum of its members' blocks less
-- twice the blocks of the edges inside it. The topology is built once; update() refills
-- the blocks each step, and apply() is one symmetric V-cycle (damped block Jacobi before
-- and after the coarse correction, a dense Cholesky solve at the coarsest level), so it
-- is a symmetric positive definite preconditioner for conjugate gradients.

local root = require("../../src")
local f64 = root.f64

type Level = {
	nodes: number,
	ea: { number },
	eb: { number },
	free: { boolean },
	blocks: buffer, -- 9 per edge
	diag: buffer, -- 9 per node
	inv: buffer, -- 9 per node: the inverses of the diagonal blocks
	-- Toward the next level: each node's aggregate (0 for pinned nodes), each edge's
	-- coarse edge (0 inside an aggregate).
	agg: { number },
	coarse_edge: { number },
	x: buffer,
	r: buffer,
	t: buffer,
}

local multigrid = {}

local OMEGA = 0.6 -- the block Jacobi damping
local COARSEST = 40 -- nodes at which to stop coarsening

local function new_level(nodes: number, ea: { number }, eb: { number }, free: { boolean }): Level
	return {
		nodes = nodes,
		ea = ea,
		eb = eb,
		free = free,
		blocks = buffer.create(72 * #ea),
		diag = buffer.create(72 * nodes),
		inv = buffer.create(72 * nodes),
		agg = {},
		coarse_edge = {},
		x = buffer.create(24 * nodes),
		r = buffer.create(24 * nodes),
		t = buffer.create(24 * nodes),
	}
end

-- Aggregates the free nodes of a level (joined through the edges marked strong, or all
-- edges) and builds the next level's topology.
local function coarsen(L: Level, strong: { boolean }?): Level
	local n = L.nodes
	local adjacency: { { number } } = {}
	for i = 1, n do
		adjacency[i] = {}
	end
	for e = 1, #L.ea do
		local a, b = L.ea[e], L.eb[e]
		if L.free[a] and L.free[b] and (strong == nil or strong[e]) then
			table.insert(adjacency[a], b)
			table.insert(adjacency[b], a)
		end
	end
	local agg = table.create(n, 0)
	local count = 0
	-- Seeds: a node whose neighbors are all unaggregated takes them.
	for i = 1, n do
		if L.free[i] and agg[i] == 0 then
			local clear = true
			for _, j in adjacency[i] do
				if agg[j] ~= 0 then
					clear = false
					break
				end
			end
			if clear then
				count += 1
				agg[i] = count
				for _, j in adjacency[i] do
					agg[j] = count
				end
			end
		end
	end
	-- Leftovers join a neighboring aggregate (or form their own).
	for i = 1, n do
		if L.free[i] and agg[i] == 0 then
			for _, j in adjacency[i] do
				if agg[j] ~= 0 then
					agg[i] = agg[j]
					break
				end
			end
			if agg[i] == 0 then
				count += 1
				agg[i] = count
			end
		end
	end
	-- Coarse edges: one per pair of distinct aggregates joined by a fine edge.
	local index: { [number]: number } = {}
	local cea, ceb = {}, {}
	local coarse_edge = table.create(#L.ea, 0)
	for e = 1, #L.ea do
		local I, J = agg[L.ea[e]], agg[L.eb[e]]
		if I ~= 0 and J ~= 0 and I ~= J then
			local lo, hi = math.min(I, J), math.max(I, J)
			local key = lo * (count + 1) + hi
			local ce = index[key]
			if ce == nil then
				table.insert(cea, lo)
				table.insert(ceb, hi)
				ce = #cea
				index[key] = ce
			end
			coarse_edge[e] = ce
		end
	end
	L.agg, L.coarse_edge = agg, coarse_edge
	return new_level(count, cea, ceb, table.create(count, true))
end

-- The hierarchy for nodes joined by edges (ea[e], eb[e]); pinned[i] marks pinned nodes,
-- and strong (optional) the edges to aggregate along on the finest level (the stiffest
-- ones: aggregating along every edge of a densely connected network coarsens too fast).
function multigrid.new(nodes: number, ea: { number }, eb: { number }, pinned: { boolean }, strong: { boolean }?): any
	local free = table.create(nodes, true)
	for i = 1, nodes do
		free[i] = not pinned[i]
	end
	local levels: { Level } = { new_level(nodes, ea, eb, free) }
	while levels[#levels].nodes > COARSEST do
		local nextl = coarsen(levels[#levels], if #levels == 1 then strong else nil)
		if nextl.nodes >= levels[#levels].nodes then
			break
		end
		table.insert(levels, nextl)
	end
	local mg: any = { levels = levels }
	local sizes = {}
	for _, L in levels do
		table.insert(sizes, L.nodes)
	end
	mg.sizes = sizes
	-- Smoothing sweeps before and after the coarse correction, and the correction's scale
	-- (above 1 compensates for piecewise constant prolongation's weak coarse correction).
	mg.sweeps = 1
	mg.alpha = 1.5
	local coarse_factor: any = nil

	-- Inverts each node's 3 by 3 diagonal block (symmetric) into L.inv.
	local function invert(L: Level)
		local d, inv = L.diag, L.inv
		for p = 0, L.nodes - 1 do
			local o = 72 * p
			local m11, m12, m13 = buffer.readf64(d, o), buffer.readf64(d, o + 8), buffer.readf64(d, o + 16)
			local m22, m23, m33 = buffer.readf64(d, o + 32), buffer.readf64(d, o + 40), buffer.readf64(d, o + 64)
			local c11 = m22 * m33 - m23 * m23
			local c12 = m13 * m23 - m12 * m33
			local c13 = m12 * m23 - m13 * m22
			local c22 = m11 * m33 - m13 * m13
			local c23 = m12 * m13 - m11 * m23
			local c33 = m11 * m22 - m12 * m12
			local det = m11 * c11 + m12 * c12 + m13 * c13
			local id = if det ~= 0 then 1 / det else 0
			buffer.writef64(inv, o, c11 * id)
			buffer.writef64(inv, o + 8, c12 * id)
			buffer.writef64(inv, o + 16, c13 * id)
			buffer.writef64(inv, o + 24, c12 * id)
			buffer.writef64(inv, o + 32, c22 * id)
			buffer.writef64(inv, o + 40, c23 * id)
			buffer.writef64(inv, o + 48, c13 * id)
			buffer.writef64(inv, o + 56, c23 * id)
			buffer.writef64(inv, o + 64, c33 * id)
		end
	end

	-- Refills the hierarchy from the finest level's blocks (9 per edge, buffer) and
	-- diagonal blocks (9 per node).
	function mg.update(blocks: buffer, diag: buffer)
		local F = levels[1]
		buffer.copy(F.blocks, 0, blocks, 0, 72 * #F.ea)
		buffer.copy(F.diag, 0, diag, 0, 72 * F.nodes)
		invert(F)
		for l = 1, #levels - 1 do
			local L, C = levels[l], levels[l + 1]
			buffer.fill(C.blocks, 0, 0)
			buffer.fill(C.diag, 0, 0)
			for i = 1, L.nodes do
				local I = L.agg[i]
				if I ~= 0 then
					local s, t = 72 * (i - 1), 72 * (I - 1)
					for e = 0, 64, 8 do
						buffer.writef64(C.diag, t + e, buffer.readf64(C.diag, t + e) + buffer.readf64(L.diag, s + e))
					end
				end
			end
			for e = 1, #L.ea do
				local I, J = L.agg[L.ea[e]], L.agg[L.eb[e]]
				if I ~= 0 and J ~= 0 then
					local s = 72 * (e - 1)
					if I == J then
						local t = 72 * (I - 1)
						for k = 0, 64, 8 do
							buffer.writef64(C.diag, t + k, buffer.readf64(C.diag, t + k) - 2 * buffer.readf64(L.blocks, s + k))
						end
					else
						local t = 72 * (L.coarse_edge[e] - 1)
						for k = 0, 64, 8 do
							buffer.writef64(C.blocks, t + k, buffer.readf64(C.blocks, t + k) + buffer.readf64(L.blocks, s + k))
						end
					end
				end
			end
			invert(C)
		end
		-- The coarsest level, factored densely.
		local C = levels[#levels]
		local m = 3 * C.nodes
		local A = f64.matrix(m, m)
		for i = 1, C.nodes do
			for r = 1, 3 do
				for c = 1, 3 do
					f64.matset(A, 3 * i - 3 + r, 3 * i - 3 + c, buffer.readf64(C.diag, 72 * (i - 1) + 24 * (r - 1) + 8 * (c - 1)))
				end
			end
		end
		for e = 1, #C.ea do
			local a, b = C.ea[e], C.eb[e]
			for r = 1, 3 do
				for c = 1, 3 do
					local v = -buffer.readf64(C.blocks, 72 * (e - 1) + 24 * (r - 1) + 8 * (c - 1))
					f64.matset(A, 3 * a - 3 + r, 3 * b - 3 + c, v)
					f64.matset(A, 3 * b - 3 + r, 3 * a - 3 + c, v)
				end
			end
		end
		coarse_factor = f64.chol(A)
	end

	-- out = A x on a level (pinned nodes: nothing).
	local function apply(L: Level, out: buffer, x: buffer)
		local d = L.diag
		for p = 0, L.nodes - 1 do
			local o, v = 72 * p, 24 * p
			if not L.free[p + 1] then
				buffer.writef64(out, v, 0)
				buffer.writef64(out, v + 8, 0)
				buffer.writef64(out, v + 16, 0)
				continue
			end
			local x1, x2, x3 = buffer.readf64(x, v), buffer.readf64(x, v + 8), buffer.readf64(x, v + 16)
			buffer.writef64(out, v, buffer.readf64(d, o) * x1 + buffer.readf64(d, o + 8) * x2 + buffer.readf64(d, o + 16) * x3)
			buffer.writef64(out, v + 8, buffer.readf64(d, o + 24) * x1 + buffer.readf64(d, o + 32) * x2 + buffer.readf64(d, o + 40) * x3)
			buffer.writef64(out, v + 16, buffer.readf64(d, o + 48) * x1 + buffer.readf64(d, o + 56) * x2 + buffer.readf64(d, o + 64) * x3)
		end
		local B, ea, eb, free = L.blocks, L.ea, L.eb, L.free
		for e = 1, #ea do
			local a, b = ea[e], eb[e]
			if free[a] and free[b] then
				local o, va, vb = 72 * (e - 1), 24 * (a - 1), 24 * (b - 1)
				local a1, a2, a3 = buffer.readf64(x, va), buffer.readf64(x, va + 8), buffer.readf64(x, va + 16)
				local b1, b2, b3 = buffer.readf64(x, vb), buffer.readf64(x, vb + 8), buffer.readf64(x, vb + 16)
				local m11, m12, m13 = buffer.readf64(B, o), buffer.readf64(B, o + 8), buffer.readf64(B, o + 16)
				local m22, m23, m33 = buffer.readf64(B, o + 32), buffer.readf64(B, o + 40), buffer.readf64(B, o + 64)
				buffer.writef64(out, va, buffer.readf64(out, va) - (m11 * b1 + m12 * b2 + m13 * b3))
				buffer.writef64(out, va + 8, buffer.readf64(out, va + 8) - (m12 * b1 + m22 * b2 + m23 * b3))
				buffer.writef64(out, va + 16, buffer.readf64(out, va + 16) - (m13 * b1 + m23 * b2 + m33 * b3))
				buffer.writef64(out, vb, buffer.readf64(out, vb) - (m11 * a1 + m12 * a2 + m13 * a3))
				buffer.writef64(out, vb + 8, buffer.readf64(out, vb + 8) - (m12 * a1 + m22 * a2 + m23 * a3))
				buffer.writef64(out, vb + 16, buffer.readf64(out, vb + 16) - (m13 * a1 + m23 * a2 + m33 * a3))
			end
		end
	end

	-- x += OMEGA D^-1 (r - A x), with t a work array (A x is skipped when x is zero).
	local function smooth(L: Level, x: buffer, r: buffer, t: buffer, zero: boolean?)
		if zero then
			buffer.fill(t, 0, 0)
		else
			apply(L, t, x)
		end
		local inv = L.inv
		for p = 0, L.nodes - 1 do
			if L.free[p + 1] then
				local o, v = 72 * p, 24 * p
				local s1 = buffer.readf64(r, v) - buffer.readf64(t, v)
				local s2 = buffer.readf64(r, v + 8) - buffer.readf64(t, v + 8)
				local s3 = buffer.readf64(r, v + 16) - buffer.readf64(t, v + 16)
				buffer.writef64(x, v, buffer.readf64(x, v) + OMEGA * (buffer.readf64(inv, o) * s1 + buffer.readf64(inv, o + 8) * s2 + buffer.readf64(inv, o + 16) * s3))
				buffer.writef64(x, v + 8, buffer.readf64(x, v + 8) + OMEGA * (buffer.readf64(inv, o + 24) * s1 + buffer.readf64(inv, o + 32) * s2 + buffer.readf64(inv, o + 40) * s3))
				buffer.writef64(x, v + 16, buffer.readf64(x, v + 16) + OMEGA * (buffer.readf64(inv, o + 48) * s1 + buffer.readf64(inv, o + 56) * s2 + buffer.readf64(inv, o + 64) * s3))
			end
		end
	end

	-- One V-cycle on level l for A x = r (x starts at zero).
	local function cycle(l: number, x: buffer, r: buffer)
		local L = levels[l]
		buffer.fill(x, 0, 0)
		if l == #levels then
			local rhs = f64.create(3 * L.nodes)
			buffer.copy(rhs, 0, r, 0, 24 * L.nodes)
			local sol = f64.chol_solve(nil, coarse_factor, rhs)
			buffer.copy(x, 0, sol, 0, 24 * L.nodes)
			return
		end
		smooth(L, x, r, L.t, true)
		for _ = 2, mg.sweeps do
			smooth(L, x, r, L.t)
		end
		-- Restrict the residual, correct from the coarse level, prolong.
		apply(L, L.t, x)
		local C = levels[l + 1]
		buffer.fill(C.r, 0, 0)
		for i = 1, L.nodes do
			local I = L.agg[i]
			if I ~= 0 then
				local v, c = 24 * (i - 1), 24 * (I - 1)
				for k = 0, 16, 8 do
					buffer.writef64(C.r, c + k, buffer.readf64(C.r, c + k) + buffer.readf64(r, v + k) - buffer.readf64(L.t, v + k))
				end
			end
		end
		cycle(l + 1, C.x, C.r)
		for i = 1, L.nodes do
			local I = L.agg[i]
			if I ~= 0 then
				local v, c = 24 * (i - 1), 24 * (I - 1)
				for k = 0, 16, 8 do
					buffer.writef64(x, v + k, buffer.readf64(x, v + k) + mg.alpha * buffer.readf64(C.x, c + k))
				end
			end
		end
		for _ = 1, mg.sweeps do
			smooth(L, x, r, L.t)
		end
	end

	-- out = M^-1 r (one V-cycle); pinned nodes' entries are copied from r.
	function mg.apply(out: buffer, r: buffer): buffer
		local F = levels[1]
		cycle(1, F.x, r)
		buffer.copy(out, 0, F.x, 0, 24 * F.nodes)
		for i = 1, F.nodes do
			if not F.free[i] then
				local v = 24 * (i - 1)
				buffer.writef64(out, v, buffer.readf64(r, v))
				buffer.writef64(out, v + 8, buffer.readf64(r, v + 8))
				buffer.writef64(out, v + 16, buffer.readf64(r, v + 16))
			end
		end
		return out
	end

	return mg
end

return multigrid
