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
-- X_I K(skel_I, skel_J) X_J^T: an interpolative decomposition, the skeleton points skel_I
-- of half I chosen by column-pivoted QR. To choose them without evaluating the whole
-- block, I's interactions are sampled at the points of its sibling near I and at proxy
-- points spread over the rest of the sibling's box (Xing and Chow, 2020).
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

-- A point's coordinates (a vector, or a table { x, y, z } with y and z optional).
local function coords(p: any, k: number): (number, number, number)
	if type(p) == "vector" then
		return p.x, p.y, p.z
	elseif type(p) == "table" and type(p[1]) == "number" then
		return p[1], p[2] or 0, p[3] or 0
	end
	error(string.format("hodlr.new: point %d is not a vector or { x, y, z }", k), 3)
end

-- Distance from point (x, y, z) to a box.
local function boxdist(bmin: { number }, bmax: { number }, x: number, y: number, z: number): number
	local dx = math.max(0, bmin[1] - x, x - bmax[1])
	local dy = math.max(0, bmin[2] - y, y - bmax[2])
	local dz = math.max(0, bmin[3] - z, z - bmax[3])
	return math.sqrt(dx * dx + dy * dy + dz * dz)
end

-- The interpolative decomposition of the rows of a block M = K(I, columns), from the
-- column-pivoted QR of M^T (c by m, filled by the caller), truncated at tol: the skeleton
-- (local row indices) and X (m by k) with M ~ X M(skeleton, :). When M has many more
-- columns than its rank, the QR is of a random sketch G M^T (s by m, s a little above the
-- rank, found by doubling from the guess): the same decomposition for a product (fast)
-- instead of c - s more Householder steps (slow).
local function interpolative(Mt: Matrix, tol: number, guess: number, rng: any): ({ number }, Matrix)
	local c, m = Mt.rows, Mt.cols
	local Y = Mt
	local F: types.QRP, k: number
	local rows = math.max(guess, 16)
	while true do
		if rows >= c then
			Y = Mt
			F = f64.qrp(Y, nil, tol)
			k = f64.qrp_rank(F, tol)
			break
		end
		local G = f64.matrix(rows, c)
		for i = 0, rows * c - 1 do
			buffer.writef64(G.data, 8 * i, rng:uniform(-1, 1))
		end
		Y = f64.matmul(nil, G, Mt)
		F = f64.qrp(Y, nil, tol)
		k = f64.qrp_rank(F, tol)
		if k <= rows - 8 or k >= m then
			break
		end
		rows *= 2
	end
	c = Y.rows
	local a = Y.data
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

-- A compressed kernel matrix on the points (vectors or { x, y, z } tables), for a kernel
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
	local X0, Y0, Z0 = table.create(n, 0), table.create(n, 0), table.create(n, 0)
	for k = 1, n do
		local x, y, z = coords(points[k], k)
		if x ~= x or y ~= y or z ~= z then
			fail("new", string.format("point %d has a NaN coordinate", k))
		end
		X0[k], Y0[k], Z0[k] = x, y, z
	end
	-- the tree: order[t] is the point at tree position t
	local order = table.create(n, 0)
	for k = 1, n do
		order[k] = k
	end
	local all: { Node } = {}
	local function build(lo: number, hi: number): Node
		local bmin, bmax = { math.huge, math.huge, math.huge }, { -math.huge, -math.huge, -math.huge }
		for t = lo, hi do
			local k = order[t]
			local x, y, z = X0[k], Y0[k], Z0[k]
			bmin[1], bmax[1] = math.min(bmin[1], x), math.max(bmax[1], x)
			bmin[2], bmax[2] = math.min(bmin[2], y), math.max(bmax[2], y)
			bmin[3], bmax[3] = math.min(bmin[3], z), math.max(bmax[3], z)
		end
		local node: Node = { lo = lo, hi = hi, bmin = bmin, bmax = bmax, skel = {}, X = f64.matrix(0, 0) } :: any
		if hi - lo + 1 > leaf then
			local axis = 1
			for a = 2, 3 do
				if bmax[a] - bmin[a] > bmax[axis] - bmin[axis] then
					axis = a
				end
			end
			local C = if axis == 1 then X0 elseif axis == 2 then Y0 else Z0
			local slice = table.move(order, lo, hi, 1, {})
			table.sort(slice, function(i: number, j: number): boolean
				return C[i] < C[j]
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
	local PX, PY, PZ = table.create(n, 0), table.create(n, 0), table.create(n, 0)
	for t = 1, n do
		local k = order[t]
		PX[t], PY[t], PZ[t] = X0[k], Y0[k], Z0[k]
	end
	local function K(i: number, j: number): number
		local dx, dy, dz = PX[i] - PX[j], PY[i] - PY[j], PZ[i] - PZ[j]
		return kernel(math.sqrt(dx * dx + dy * dy + dz * dz))
	end

	-- Leaves' blocks, and each child's skeleton from its sibling's near points and proxies.
	local halton = random.halton(3)
	local sketch = random.new(1)
	local proxies: { { number } } = {}
	for k = 1, 4 * nproxy do
		proxies[k] = halton:next()
	end
	local ranks: { number } = {}
	local function compress(I: Node, J: Node, depth: number)
		local m = I.hi - I.lo + 1
		local nj = J.hi - J.lo + 1
		-- J's typical spacing sets the near band: the points of J within 6 spacings of I's box
		local vol, dims = 1, 0
		for a = 1, 3 do
			local e = J.bmax[a] - J.bmin[a]
			if e > 0 then
				vol *= e
				dims += 1
			end
		end
		local h = if dims > 0 then (vol / nj) ^ (1 / dims) else 0
		local band = 6 * h
		local cx, cy, cz = {}, {}, {}
		for t = J.lo, J.hi do
			if boxdist(I.bmin, I.bmax, PX[t], PY[t], PZ[t]) < band then
				table.insert(cx, PX[t])
				table.insert(cy, PY[t])
				table.insert(cz, PZ[t])
			end
		end
		if #cx + nproxy >= nj then
			-- as cheap to use all of J
			table.clear(cx)
			table.clear(cy)
			table.clear(cz)
			for t = J.lo, J.hi do
				table.insert(cx, PX[t])
				table.insert(cy, PY[t])
				table.insert(cz, PZ[t])
			end
		else
			local added = 0
			for _, u in proxies do
				local x = J.bmin[1] + u[1] * (J.bmax[1] - J.bmin[1])
				local y = J.bmin[2] + u[2] * (J.bmax[2] - J.bmin[2])
				local z = J.bmin[3] + u[3] * (J.bmax[3] - J.bmin[3])
				if boxdist(I.bmin, I.bmax, x, y, z) >= band then
					table.insert(cx, x)
					table.insert(cy, y)
					table.insert(cz, z)
					added += 1
					if added == nproxy then
						break
					end
				end
			end
		end
		local c = #cx
		local Mt = f64.matrix(c, m)
		local a = Mt.data
		for i = 1, m do
			local t = I.lo + i - 1
			local x, y, z = PX[t], PY[t], PZ[t]
			local base = 8 * (i - 1) * c
			for l = 1, c do
				local dx, dy, dz = x - cx[l], y - cy[l], z - cz[l]
				buffer.writef64(a, base + 8 * (l - 1), kernel(math.sqrt(dx * dx + dy * dy + dz * dz)))
			end
		end
		-- the rank guess: a sibling's at this depth, else twice a child's (ranks grow by
		-- less than 2 per level up)
		local skel, X = interpolative(Mt, tol, (ranks[depth] or 2 * (ranks[depth + 1] or 12)) + 16, sketch)
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
