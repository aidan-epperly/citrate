--!strict
--!native
--!optimize 2

-- Kernel matrices K_ij = k(|p_i - p_j|) on many points, compressed as HODLR matrices
-- (hierarchically off-diagonal low rank) so that K + lambda I can be factored, solved and
-- its log-determinant found in about n k^2 log n work instead of n^3 / 3.
--
--   local H = hodlr.new(points, function(r) return math.exp(-r * r) end, { tol = 1e-10 })
--   H:factor(1e-9)          -- K + 1e-9 I
--   local x = H:solve(b)    -- (K + 1e-9 I) x = b
--   H:logdet()              -- log det (K + 1e-9 I)
--
-- The points are split in halves recursively (at the median of the widest coordinate)
-- down to leaves of `leaf` points. Each block coupling two halves is low rank for smooth
-- kernels (Gaussian, Matern) in one or two dimensions, and is stored as
-- X_I K(skel_I, skel_J) X_J^T: an interpolative decomposition. To choose the skeleton
-- points skel_I of half I without evaluating the whole block, I's interactions are sampled
-- at the points of its sibling near I and at proxy points spread over the rest of the
-- sibling's box (Xing and Chow, 2020); the samples are sketched by a random matrix, and the
-- skeleton chosen from the sketch by LU with partial pivoting and a small column-pivoted
-- QR (see lu_skeleton). Points beyond the kernel's reach from the sibling are left out.
--
-- The ranks, and so the cost, grow with the accuracy asked (tol) and with the dimension:
-- in 3-D the ranks of these blocks grow like n^(2/3) and the method saves little.
-- Factoring needs K + lambda I to stay positive definite despite the compression error
-- (about tol |K|): lambda = 2 tol |K| (H:norm() estimates |K|) is enough, and with it
-- interpolants were as accurate as a dense solve with the same lambda. Factoring again
-- with another lambda reuses the compression.

local types = require("./types")
local f64 = require("./f64")
local random = require("./random")

type Matrix = types.Matrix

local function fail(who: string, message: string): never
	error("hodlr." .. who .. ": " .. message, 3)
end

export type HodlrOptions = { tol: number?, leaf: number?, proxies: number? }

type Node = {
	lo: number, -- first and last point, in tree order (1-based)
	hi: number,
	left: Node?,
	right: Node?,
	bmin: { number },
	bmax: { number },
	-- this node's skeleton (tree-order indices) and interpolation matrix X (size by k):
	-- its rows of K outside the node are X K(skel, outside)
	skel: { number },
	X: Matrix,
	-- with children: B = K(skel_left, skel_right); after factoring, the factor's Q blocks
	-- and Z = Lc^-1 - I (see apply_inverse)
	B: Matrix?,
	QL: Matrix?,
	QR: Matrix?,
	Z: Matrix?,
	-- a leaf's kernel block, and the inverse of the Cholesky factor of block + lambda I
	K: Matrix?,
	L: Matrix?,
}

export type Hodlr = {
	n: number,
	tol: number,
	order: { number },
	ranks: { number },
	lambda: number?,
	factor: (self: Hodlr, lambda: number) -> (),
	solve: (self: Hodlr, b: buffer, out: buffer?) -> buffer,
	matvec: (self: Hodlr, x: buffer, out: buffer?) -> buffer,
	logdet: (self: Hodlr) -> number,
	norm: (self: Hodlr) -> number,
	[string]: any,
}

local Hodlr = {}
Hodlr.__index = Hodlr

-- The points' coordinates, d numbers a point (point k's from (k - 1) d + 1): points are
-- vectors (d = 3), tables of d numbers or f64 arrays of d numbers, any d.
local function read_points(points: { any }, who: string): ({ number }, number)
	local n = #points
	local first = points[1]
	local d = if type(first) == "vector" then 3
		elseif type(first) == "table" then #first
		elseif type(first) == "buffer" then buffer.len(first) // 8
		else 0
	if d < 1 then
		error(string.format("%s: point 1 is not a vector, a table of numbers or an f64 array", who), 3)
	end
	local X = table.create(n * d, 0) :: { number }
	for k = 1, n do
		local p = points[k]
		local base = (k - 1) * d
		if type(p) == "vector" and d == 3 then
			X[base + 1], X[base + 2], X[base + 3] = p.x, p.y, p.z
		elseif type(p) == "table" and #p == d then
			for a = 1, d do
				X[base + a] = p[a]
			end
		elseif type(p) == "buffer" and buffer.len(p) == 8 * d then
			for a = 1, d do
				X[base + a] = buffer.readf64(p, 8 * (a - 1))
			end
		else
			error(string.format("%s: point %d is not like point 1 (%d coordinates)", who, k, d), 3)
		end
		for a = 1, d do
			local v = X[base + a]
			if type(v) ~= "number" or v ~= v then
				error(string.format("%s: point %d has a coordinate that is not a number", who, k), 3)
			end
		end
	end
	return X, d
end

-- Distance from the point at P[base + 1 .. base + d] to a box.
local function boxdist(bmin: { number }, bmax: { number }, P: { number }, base: number, d: number): number
	local s = 0
	for a = 1, d do
		local x = P[base + a]
		local e = math.max(0, bmin[a] - x, x - bmax[a])
		s += e * e
	end
	return math.sqrt(s)
end

-- The inverse of a lower triangular matrix (in place of L).
local function lower_inverse(L: Matrix): Matrix
	local k = L.rows
	local l = L.data
	local Z = f64.matrix(k, k)
	local z = Z.data
	for c = 0, k - 1 do
		-- column c of L^-1: solve L z = e_c (z is zero above c)
		buffer.writef64(z, 8 * (c * k + c), 1 / buffer.readf64(l, 8 * (c * k + c)))
		for i = c + 1, k - 1 do
			local acc = 0
			for j = c, i - 1 do
				acc += buffer.readf64(l, 8 * (j * k + i)) * buffer.readf64(z, 8 * (c * k + j))
			end
			buffer.writef64(z, 8 * (c * k + i), -acc / buffer.readf64(l, 8 * (i * k + i)))
		end
	end
	return Z
end

-- The skeleton of a sketch Y (s by m; column i stands for point i): LU with partial
-- pivoting of Y^T picks min(s, m) candidate points, and a column-pivoted QR of Y's
-- candidate columns (s by at most s, cheap) picks and orders the skeleton, truncated at
-- tol (Dong and Martinsson, 2023). X^T = R11^-1 Q1^T Y is then the least squares fit
-- Y ~ Y(:, skeleton) X^T, by products. Returns the skeleton (local indices), X and k.
local function lu_skeleton(Y: Matrix, tol: number): ({ number }, Matrix, number)
	local s, m = Y.rows, Y.cols
	local r = math.min(s, m)
	local W = f64.matcopy(nil, Y)
	local w = W.data
	local perm = table.create(m, 0)
	for i = 1, m do
		perm[i] = i
	end
	-- Gaussian elimination on Y^T's rows, which are Y's columns: contiguous
	for j = 0, r - 1 do
		local p, best = j, 0
		for i = j, m - 1 do
			local v = math.abs(buffer.readf64(w, 8 * (i * s + j)))
			if v > best then
				p, best = i, v
			end
		end
		if best == 0 then
			r = j
			break
		end
		if p ~= j then
			for l = 0, s - 1 do
				local a, b = 8 * (j * s + l), 8 * (p * s + l)
				local t = buffer.readf64(w, a)
				buffer.writef64(w, a, buffer.readf64(w, b))
				buffer.writef64(w, b, t)
			end
			perm[j + 1], perm[p + 1] = perm[p + 1], perm[j + 1]
		end
		local pivot = buffer.readf64(w, 8 * (j * s + j))
		local col = 8 * j * s
		for i = j + 1, m - 1 do
			local base = 8 * i * s
			local f = buffer.readf64(w, base + 8 * j) / pivot
			if f ~= 0 then
				for l = j + 1, s - 1 do
					buffer.writef64(w, base + 8 * l, buffer.readf64(w, base + 8 * l) - f * buffer.readf64(w, col + 8 * l))
				end
			end
		end
	end
	if r == 0 then
		return {}, f64.matrix(m, 0), 0
	end
	-- the candidates' columns of Y, and their pivoted QR
	local S = f64.matrix(s, r)
	for j = 1, r do
		buffer.copy(S.data, 8 * (j - 1) * s, Y.data, 8 * (perm[j] - 1) * s, 8 * s)
	end
	local F = f64.qrp(S, nil, tol)
	local k = f64.qrp_rank(F, tol)
	if math.abs(buffer.readf64(S.data, 0)) == 0 then
		k = 0
	end
	local skel = table.create(k, 0)
	for j = 1, k do
		skel[j] = perm[buffer.readi32(F.perm, 4 * (j - 1))]
	end
	if k == 0 then
		return skel, f64.matrix(m, 0), 0
	end
	-- Z = R11^-1 Q1^T Y (k by m): R11^-1 as the transpose of a lower triangular inverse
	local Q = f64.qr_q(nil, F)
	local Q1 = f64.matrix(s, k)
	buffer.copy(Q1.data, 0, Q.data, 0, 8 * s * k)
	local Rt = f64.matrix(k, k) -- R11^T, lower triangular
	for j = 0, k - 1 do
		for i = 0, j do
			buffer.writef64(Rt.data, 8 * (i * k + j), buffer.readf64(S.data, 8 * (j * s + i)))
		end
	end
	local Z = f64.gemm(nil, 1, lower_inverse(Rt), f64.gemm(nil, 1, Q1, Y, 0, "t"), 0, "t")
	local X = f64.transpose(nil, Z)
	-- the skeleton's own rows exactly
	for a = 1, k do
		local row = skel[a] - 1
		for b = 1, k do
			buffer.writef64(X.data, 8 * ((b - 1) * m + row), if a == b then 1 else 0)
		end
	end
	return skel, X, k
end

-- The interpolative decomposition of the rows of a block M = K(I, columns), from M^T (c
-- by m, filled by the caller), truncated at tol: the skeleton (local row indices) and X
-- (m by k) with M ~ X M(skeleton, :). When M has many more columns than its rank, from a
-- random sketch G M^T (s by m, s a little above the rank, found by doubling from the
-- guess) by lu_skeleton; otherwise by a truncated column-pivoted QR of M^T.
local function interpolative(Mt: Matrix, tol: number, guess: number, rng: any): ({ number }, Matrix)
	local c, m = Mt.rows, Mt.cols
	local rows = math.max(guess, 16)
	while rows < c do
		local G = f64.matrix(rows, c)
		for i = 0, rows * c - 1 do
			buffer.writef64(G.data, 8 * i, rng:uniform(-1, 1))
		end
		local skel, X, k = lu_skeleton(f64.matmul(nil, G, Mt), tol)
		if k <= rows - 8 or k >= m then
			return skel, X
		end
		rows *= 2
	end
	local F = f64.qrp(Mt, nil, tol)
	local k = f64.qrp_rank(F, tol)
	local a = Mt.data
	if math.abs(buffer.readf64(a, 0)) == 0 then
		k = 0
	end
	local perm = F.perm
	local skel = table.create(k, 0)
	for j = 1, k do
		skel[j] = buffer.readi32(perm, 4 * (j - 1))
	end
	local X = f64.matrix(m, k)
	local x = X.data
	for j = 1, k do
		buffer.writef64(x, 8 * ((j - 1) * m + skel[j] - 1), 1)
	end
	-- the other rows: R11^-1 R12, column by column (R11 is k by k, upper triangular)
	local t = table.create(k, 0)
	for j = k + 1, m do
		for i = 1, k do
			t[i] = buffer.readf64(a, 8 * ((j - 1) * c + i - 1))
		end
		for i = k, 1, -1 do
			local s = t[i]
			for l = i + 1, k do
				s -= buffer.readf64(a, 8 * ((l - 1) * c + i - 1)) * t[l]
			end
			t[i] = s / buffer.readf64(a, 8 * ((i - 1) * c + i - 1))
		end
		local row = buffer.readi32(perm, 4 * (j - 1)) - 1
		for i = 1, k do
			buffer.writef64(x, 8 * ((i - 1) * m + row), t[i])
		end
	end
	return skel, X
end

-- A compressed kernel matrix on the points (vectors, or tables or f64 arrays of d numbers,
-- any dimension d), for a kernel
-- given as a function of distance. options: tol (default 1e-10, relative, for each
-- block's decomposition), leaf (points in a leaf, default 64), proxies (per block,
-- default 400).
local function new(points: { any }, kernel: (number) -> number, options: HodlrOptions?): Hodlr
	local o: HodlrOptions = options or {}
	local n = #points
	if n < 1 then
		fail("new", "need at least one point")
	end
	if type(kernel) ~= "function" then
		fail("new", "kernel must be a function of distance")
	end
	local tol = o.tol or 1e-10
	local leaf = math.max(o.leaf or 64, 8)
	local nproxy = o.proxies or 400
	local X0, d = read_points(points, "hodlr.new")
	-- the tree: order[t] is the point at tree position t
	local order = table.create(n, 0)
	for k = 1, n do
		order[k] = k
	end
	local all: { Node } = {}
	local function build(lo: number, hi: number): Node
		local bmin, bmax = table.create(d, math.huge) :: { number }, table.create(d, -math.huge) :: { number }
		for t = lo, hi do
			local base = (order[t] - 1) * d
			for a = 1, d do
				local x = X0[base + a]
				bmin[a], bmax[a] = math.min(bmin[a], x), math.max(bmax[a], x)
			end
		end
		local node: Node = { lo = lo, hi = hi, bmin = bmin, bmax = bmax, skel = {}, X = f64.matrix(0, 0) } :: any
		if hi - lo + 1 > leaf then
			local axis = 1
			for a = 2, d do
				if bmax[a] - bmin[a] > bmax[axis] - bmin[axis] then
					axis = a
				end
			end
			local slice = table.move(order, lo, hi, 1, {})
			table.sort(slice, function(i: number, j: number): boolean
				return X0[(i - 1) * d + axis] < X0[(j - 1) * d + axis]
			end)
			table.move(slice, 1, #slice, lo, order)
			local mid = (lo + hi) // 2
			node.left = build(lo, mid)
			node.right = build(mid + 1, hi)
		end
		table.insert(all, node) -- children before parents
		return node
	end
	local root = build(1, n)
	-- the coordinates in tree order: point t's from (t - 1) d + 1
	local P = table.create(n * d, 0) :: { number }
	for t = 1, n do
		table.move(X0, (order[t] - 1) * d + 1, order[t] * d, (t - 1) * d + 1, P)
	end
	local function K(i: number, j: number): number
		local s = 0
		local bi, bj = (i - 1) * d, (j - 1) * d
		for a = 1, d do
			local e = P[bi + a] - P[bj + a]
			s += e * e
		end
		return kernel(math.sqrt(s))
	end

	-- Leaves' blocks, and each child's skeleton from its sibling's near points and proxies.
	-- proxies spread over a box by a Halton sequence (random points past its 32 dimensions)
	local halton = if d <= 32 then random.halton(d) else nil
	local sketch = random.new(1)
	-- The kernel's reach: beyond it, |k(r)| < tol |k(0)| / 10 (checked on 257 distances up
	-- to the points' diameter). Points of a node farther than this from its sibling's box
	-- interact with it below the tolerance: they stay out of its decomposition (their rows
	-- of X are zero). Rapidly decaying kernels (Gaussian) save much; others nothing.
	local reach = math.huge
	do
		local diam = 0
		for a = 1, d do
			diam += (root.bmax[a] - root.bmin[a]) ^ 2
		end
		diam = math.sqrt(diam)
		local limit = tol * math.abs(kernel(0)) / 10
		if diam > 0 and math.abs(kernel(diam)) < limit then
			reach = diam
			for j = 255, 0, -1 do
				local r = diam * j / 256
				if math.abs(kernel(r)) >= limit then
					break
				end
				reach = r
			end
		end
	end
	local proxies: { { number } } = {}
	for k = 1, 4 * nproxy do
		if halton then
			proxies[k] = halton:next()
		else
			local u = table.create(d, 0) :: { number }
			for a = 1, d do
				u[a] = sketch:uniform()
			end
			proxies[k] = u
		end
	end
	local ranks: { number } = {}
	local function compress(I: Node, J: Node, depth: number)
		local m = I.hi - I.lo + 1
		local nj = J.hi - J.lo + 1
		-- J's typical spacing sets the near band: the points of J within 6 spacings of I's box
		local vol, dims = 1, 0
		for a = 1, d do
			local e = J.bmax[a] - J.bmin[a]
			if e > 0 then
				vol *= e
				dims += 1
			end
		end
		local h = if dims > 0 then (vol / nj) ^ (1 / dims) else 0
		local band = 6 * h
		-- the candidates, d coordinates each in cand
		local cand: { number } = {}
		local c = 0
		for t = J.lo, J.hi do
			if boxdist(I.bmin, I.bmax, P, (t - 1) * d, d) < band then
				table.move(P, (t - 1) * d + 1, t * d, c * d + 1, cand)
				c += 1
			end
		end
		if c + nproxy >= nj then
			-- as cheap to use all of J
			table.clear(cand)
			c = nj
			table.move(P, (J.lo - 1) * d + 1, J.hi * d, 1, cand)
		else
			local added = 0
			local u2 = table.create(d, 0) :: { number }
			for _, u in proxies do
				for a = 1, d do
					u2[a] = J.bmin[a] + u[a] * (J.bmax[a] - J.bmin[a])
				end
				if boxdist(I.bmin, I.bmax, u2, 0, d) >= band then
					table.move(u2, 1, d, c * d + 1, cand)
					c += 1
					added += 1
					if added == nproxy then
						break
					end
				end
			end
		end
		-- I's points within reach of J
		local active = table.create(m, 0)
		local ma = 0
		for i = 1, m do
			local t = I.lo + i - 1
			if boxdist(J.bmin, J.bmax, P, (t - 1) * d, d) <= reach then
				ma += 1
				active[ma] = i
			end
		end
		local Mt = f64.matrix(c, ma)
		local a = Mt.data
		for q = 1, ma do
			local bt = (I.lo + active[q] - 2) * d
			local base = 8 * (q - 1) * c
			for l = 1, c do
				local s, bl = 0, (l - 1) * d
				for e = 1, d do
					local diff = P[bt + e] - cand[bl + e]
					s += diff * diff
				end
				buffer.writef64(a, base + 8 * (l - 1), kernel(math.sqrt(s)))
			end
		end
		-- the rank guess: a sibling's at this depth, else twice a child's (ranks grow by
		-- less than 2 per level up)
		local skel, X = interpolative(Mt, tol, (ranks[depth] or 2 * (ranks[depth + 1] or 12)) + 16, sketch)
		if ma < m then
			local k = #skel
			local full = f64.matrix(m, k)
			for b = 0, k - 1 do
				for q = 1, ma do
					buffer.writef64(full.data, 8 * (b * m + active[q] - 1), buffer.readf64(X.data, 8 * (b * ma + q - 1)))
				end
			end
			X = full
			for j = 1, k do
				skel[j] = active[skel[j]]
			end
		end
		for j = 1, #skel do
			skel[j] += I.lo - 1
		end
		I.skel, I.X = skel, X
		ranks[depth] = math.max(ranks[depth] or 0, #skel)
	end
	local function visit(node: Node, depth: number)
		local L, R = node.left, node.right
		if L and R then
			visit(L, depth + 1)
			visit(R, depth + 1)
			compress(L, R, depth + 1)
			compress(R, L, depth + 1)
			local kl, kr = #L.skel, #R.skel
			local B = f64.matrix(kl, kr)
			for j = 1, kr do
				for i = 1, kl do
					buffer.writef64(B.data, 8 * ((j - 1) * kl + i - 1), K(L.skel[i], R.skel[j]))
				end
			end
			node.B = B
		else
			local m = node.hi - node.lo + 1
			local Kb = f64.matrix(m, m)
			for j = 1, m do
				for i = 1, m do
					buffer.writef64(Kb.data, 8 * ((j - 1) * m + i - 1), K(node.lo + i - 1, node.lo + j - 1))
				end
			end
			node.K = Kb
		end
	end
	visit(root, 0)
	return setmetatable({ n = n, tol = tol, order = order, ranks = ranks, root = root, nodes = all }, Hodlr) :: any
end

-- The rows of node of Y (rows from base + 1, leading dimension ld, r columns) as a matrix,
-- and back.
local function rows_of(node: Node, y: buffer, ld: number, r: number, base: number): Matrix
	local m = node.hi - node.lo + 1
	local M = f64.matrix(m, r)
	for c = 0, r - 1 do
		buffer.copy(M.data, 8 * c * m, y, 8 * (c * ld + node.lo - base - 1), 8 * m)
	end
	return M
end
local function put_rows(node: Node, M: Matrix, y: buffer, ld: number, r: number, base: number)
	local m = node.hi - node.lo + 1
	for c = 0, r - 1 do
		buffer.copy(y, 8 * (c * ld + node.lo - base - 1), M.data, 8 * c * m, 8 * m)
	end
end

-- The factor of a node's block A = L L^T: for a leaf, its Cholesky factor; with children,
-- L = diag(L_left, L_right) (I + Q Y Q^T), where Q = diag(Q_left, Q_right) has orthonormal
-- columns spanning L_child^-1 X_child, and I + Y = Lc is the Cholesky factor of the small
-- matrix I + Q^T (L_D^-1 U C U^T L_D^-T) Q (the symmetric factorization of Ambikasaran,
-- O'Neil and Singh). Each step is a triangular solve or a small Cholesky factorization, so
-- it stays accurate when K + lambda I is ill-conditioned, as Gaussian kernel matrices are.
-- The triangular factors' inverses are kept (L^-1 for leaves, Lc^-1 - I for the rest), so
-- that applying them is matrix products.

-- The node's correction I + Q Z Q^T (Z = Lc^-1 - I), or with transpose I + Q Z^T Q^T,
-- applied to its rows of y.
local function correct(node: Node, y: buffer, ld: number, r: number, base: number, transpose: boolean)
	local L, R = node.left :: Node, node.right :: Node
	local QL, QR, Z = node.QL :: Matrix, node.QR :: Matrix, node.Z :: Matrix
	local kl, kr = QL.cols, QR.cols
	local k = kl + kr
	local YL, YR = rows_of(L, y, ld, r, base), rows_of(R, y, ld, r, base)
	local wl = f64.gemm(nil, 1, QL, YL, 0, "t")
	local wr = f64.gemm(nil, 1, QR, YR, 0, "t")
	local w = f64.matrix(k, r)
	for c = 0, r - 1 do
		buffer.copy(w.data, 8 * c * k, wl.data, 8 * c * kl, 8 * kl)
		buffer.copy(w.data, 8 * (c * k + kl), wr.data, 8 * c * kr, 8 * kr)
	end
	local d = f64.gemm(nil, 1, Z, w, 0, if transpose then "t" else "n")
	local dl, dr = f64.matrix(kl, r), f64.matrix(kr, r)
	for c = 0, r - 1 do
		buffer.copy(dl.data, 8 * c * kl, d.data, 8 * c * k, 8 * kl)
		buffer.copy(dr.data, 8 * c * kr, d.data, 8 * (c * k + kl), 8 * kr)
	end
	f64.gemm(YL, 1, QL, dl, 1)
	f64.gemm(YR, 1, QR, dr, 1)
	put_rows(L, YL, y, ld, r, base)
	put_rows(R, YR, y, ld, r, base)
end

-- y := L^-1 y, or L^-T y with transpose, for the node's rows of y (rows from base + 1,
-- leading dimension ld, r columns), in place.
local function apply_inverse(node: Node, y: buffer, ld: number, r: number, base: number, transpose: boolean)
	local L, R = node.left, node.right
	if not (L and R) then
		local Yn = rows_of(node, y, ld, r, base)
		put_rows(node, f64.gemm(nil, 1, node.L :: Matrix, Yn, 0, if transpose then "t" else "n"), y, ld, r, base)
		return
	end
	local coupled = node.Z ~= nil
	if transpose then
		if coupled then
			correct(node, y, ld, r, base, true)
		end
		apply_inverse(L, y, ld, r, base, true)
		apply_inverse(R, y, ld, r, base, true)
	else
		apply_inverse(L, y, ld, r, base, false)
		apply_inverse(R, y, ld, r, base, false)
		if coupled then
			correct(node, y, ld, r, base, false)
		end
	end
end

-- R (k by k) from the first k rows of a QR factorization.
local function r_factor(F: any, k: number): Matrix
	local m = F.qr.rows
	local Rm = f64.matrix(k, k)
	for j = 0, k - 1 do
		for i = 0, j do
			buffer.writef64(Rm.data, 8 * (j * k + i), buffer.readf64(F.qr.data, 8 * (j * m + i)))
		end
	end
	return Rm
end

-- V = Q R with orthonormal Q: by Cholesky QR twice (V^T V = R^T R, Q = V R^-1, then again
-- on Q), which is all matrix products and accurate while V's condition number is below
-- about 1e8; by Householder QR otherwise (when V^T V is not numerically definite).
local function orthonormalize(V: Matrix): (Matrix, Matrix)
	local k = V.cols
	local Q, R = V, nil :: Matrix?
	for _ = 1, 2 do
		local G = f64.gemm(nil, 1, Q, Q, 0, "t")
		if not f64.chol(G).definite then
			local F = f64.qr_copy(V)
			return f64.qr_q(nil, F), r_factor(F, k)
		end
		local Linv = lower_inverse(G)
		Q = f64.gemm(nil, 1, Q, Linv, 0, "n", "t")
		-- this pass's R = L^T
		local Rp = f64.matrix(k, k)
		for j = 0, k - 1 do
			for i = 0, j do
				buffer.writef64(Rp.data, 8 * (j * k + i), buffer.readf64(G.data, 8 * (i * k + j)))
			end
		end
		R = if R then f64.matmul(nil, Rp, R) else Rp
	end
	return Q, R :: Matrix
end

-- Factor K + lambda I (lambda >= 0). Errors if it is not positive definite (lambda too
-- small for the compression's accuracy). Factoring again with another lambda reuses the
-- compression, which is most of the work.
function Hodlr.factor(self: Hodlr, lambda: number)
	if not (lambda >= 0) then
		fail("factor", "lambda must be a nonnegative number")
	end
	local s = self :: any
	s.lambda = nil
	local logdet = 0
	local function indefinite()
		error(string.format("hodlr.factor: K + %g I is not positive definite: increase lambda", lambda), 3)
	end
	for _, node: Node in s.nodes do
		local L, R = node.left, node.right
		if not (L and R) then
			local Kb = node.K :: Matrix
			local m = Kb.rows
			local A = f64.matcopy(nil, Kb)
			for i = 0, m - 1 do
				local off = 8 * (i * m + i)
				buffer.writef64(A.data, off, buffer.readf64(A.data, off) + lambda)
			end
			if not f64.chol(A).definite then
				indefinite()
			end
			for i = 0, m - 1 do
				logdet += 2 * math.log(buffer.readf64(A.data, 8 * (i * m + i)))
			end
			node.L = lower_inverse(A)
			continue
		end
		local B = node.B :: Matrix
		local kl, kr = B.rows, B.cols
		node.Z = nil
		if kl == 0 or kr == 0 then
			continue
		end
		-- V = L_child^-1 X_child = Q R for each child
		local ml, mr = L.hi - L.lo + 1, R.hi - R.lo + 1
		local VL, VR = f64.matcopy(nil, L.X), f64.matcopy(nil, R.X)
		apply_inverse(L, VL.data, ml, kl, L.lo - 1, false)
		apply_inverse(R, VR.data, mr, kr, R.lo - 1, false)
		local QL, RL = orthonormalize(VL)
		local QR, RR = orthonormalize(VR)
		node.QL, node.QR = QL, QR
		-- I + [0, RL B RR^T; (RL B RR^T)^T, 0], factored by Cholesky
		local C12 = f64.matmul(nil, RL, f64.gemm(nil, 1, B, RR, 0, "n", "t"))
		local k = kl + kr
		local M = f64.matrix(k, k)
		local d = M.data
		for i = 0, k - 1 do
			buffer.writef64(d, 8 * (i * k + i), 1)
		end
		for j = 0, kr - 1 do
			for i = 0, kl - 1 do
				local v = buffer.readf64(C12.data, 8 * (j * kl + i))
				buffer.writef64(d, 8 * ((kl + j) * k + i), v)
				buffer.writef64(d, 8 * (i * k + kl + j), v)
			end
		end
		if not f64.chol(M).definite then
			indefinite()
		end
		for i = 0, k - 1 do
			logdet += 2 * math.log(buffer.readf64(d, 8 * (i * k + i)))
		end
		local Z = lower_inverse(M)
		for i = 0, k - 1 do
			buffer.writef64(Z.data, 8 * (i * k + i), buffer.readf64(Z.data, 8 * (i * k + i)) - 1)
		end
		node.Z = Z
	end
	s.lambda, s.logdet_value = lambda, logdet
end

-- x with (K + lambda I) x = b, after factor.
function Hodlr.solve(self: Hodlr, b: buffer, out: buffer?): buffer
	local s = self :: any
	if s.lambda == nil then
		fail("solve", "call factor first")
	end
	local n, order = s.n, s.order
	if buffer.len(b) ~= 8 * n then
		fail("solve", "b must have one value per point")
	end
	local y = f64.create(n)
	for t = 1, n do
		buffer.writef64(y, 8 * (t - 1), buffer.readf64(b, 8 * (order[t] - 1)))
	end
	apply_inverse(s.root, y, n, 1, 0, false)
	apply_inverse(s.root, y, n, 1, 0, true)
	local o = out or f64.create(n)
	for t = 1, n do
		buffer.writef64(o, 8 * (order[t] - 1), buffer.readf64(y, 8 * (t - 1)))
	end
	return o
end

-- log det (K + lambda I), after factor.
function Hodlr.logdet(self: Hodlr): number
	local s = self :: any
	if s.lambda == nil then
		fail("logdet", "call factor first")
	end
	return s.logdet_value
end

-- K x (the compressed K, without lambda).
function Hodlr.matvec(self: Hodlr, x: buffer, out: buffer?): buffer
	local s = self :: any
	local n, order = s.n, s.order
	if buffer.len(x) ~= 8 * n then
		fail("matvec", "x must have one value per point")
	end
	local xt, y = f64.create(n), f64.create(n)
	for t = 1, n do
		buffer.writef64(xt, 8 * (t - 1), buffer.readf64(x, 8 * (order[t] - 1)))
	end
	local function apply(node: Node)
		local L, R = node.left, node.right
		if not (L and R) then
			local Kb = node.K :: Matrix
			local m = Kb.rows
			local first = node.lo - 1
			for j = 0, m - 1 do
				local xj = buffer.readf64(xt, 8 * (first + j))
				for i = 0, m - 1 do
					local off = 8 * (first + i)
					buffer.writef64(y, off, buffer.readf64(y, off) + buffer.readf64(Kb.data, 8 * (j * m + i)) * xj)
				end
			end
			return
		end
		apply(L)
		apply(R)
		local B = node.B :: Matrix
		local kl, kr = B.rows, B.cols
		if kl == 0 or kr == 0 then
			return
		end
		-- y_L += X_L B X_R^T x_R, y_R += X_R B^T X_L^T x_L
		local function project(C: Node, k: number): { number }
			local z = table.create(k, 0)
			local m = C.hi - C.lo + 1
			local xd = C.X.data
			for a = 0, k - 1 do
				local acc = 0
				for i = 0, m - 1 do
					acc += buffer.readf64(xd, 8 * (a * m + i)) * buffer.readf64(xt, 8 * (C.lo - 1 + i))
				end
				z[a + 1] = acc
			end
			return z
		end
		local function lift(C: Node, k: number, w: { number })
			local m = C.hi - C.lo + 1
			local xd = C.X.data
			for a = 0, k - 1 do
				local wa = w[a + 1]
				for i = 0, m - 1 do
					local off = 8 * (C.lo - 1 + i)
					buffer.writef64(y, off, buffer.readf64(y, off) + buffer.readf64(xd, 8 * (a * m + i)) * wa)
				end
			end
		end
		local zl, zr = project(L, kl), project(R, kr)
		local wl, wr = table.create(kl, 0), table.create(kr, 0)
		for j = 0, kr - 1 do
			for i = 0, kl - 1 do
				local bij = buffer.readf64(B.data, 8 * (j * kl + i))
				wl[i + 1] += bij * zr[j + 1]
				wr[j + 1] += bij * zl[i + 1]
			end
		end
		lift(L, kl, wl)
		lift(R, kr, wr)
	end
	apply(s.root)
	local o = out or f64.create(n)
	for t = 1, n do
		buffer.writef64(o, 8 * (order[t] - 1), buffer.readf64(y, 8 * (t - 1)))
	end
	return o
end

-- An estimate of |K| (the largest eigenvalue; K is positive semidefinite for the kernels
-- this is meant for), by power iteration with matvec.
function Hodlr.norm(self: Hodlr): number
	local s = self :: any
	if s.norm_value then
		return s.norm_value
	end
	local n = s.n
	local x = f64.create(n)
	for i = 1, n do
		f64.set(x, i, 1 + 0.25 * math.sin(1.7 * i))
	end
	local y = f64.create(n)
	local lambda = 0
	for _ = 1, 30 do
		x = f64.scale(x, 1 / f64.norm(x), x)
		y = self:matvec(x, y)
		local next = f64.dot(x, y)
		local done = math.abs(next - lambda) <= 1e-3 * math.abs(next)
		lambda = next
		x, y = y, x
		if done then
			break
		end
	end
	s.norm_value = lambda
	return lambda
end

return {
	new = new,
}
