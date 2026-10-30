--!strict
-- A drum: the Dirichlet Laplacian on a polygon whose edges run along a unit grid or the
-- diagonals of its squares (such as the Gordon-Webb-Wolpert isospectral pair, made of
-- half-squares), discretized by linear finite elements on a criss-cross mesh: each grid
-- square is cut into m by m subsquares and each subsquare into four triangles by both
-- diagonals. The mass matrix is lumped (diagonal), so the frequencies are the square roots
-- of the eigenvalues of D^-1/2 K D^-1/2, which eigsh finds by Lanczos: on its inverse
-- (shift-invert), applied by a banded Cholesky factorization after a reverse
-- Cuthill-McKee ordering of the mesh, or (method = "lanczos") on the matrix itself.
--
-- The mesh is unchanged by the reflections and quarter turns of the grid, so the
-- transplantation that proves the GWW drums isospectral (each eigenfunction of one drum,
-- restricted to its triangles, gives one of the other by signed sums) works for the
-- discrete problem too: the two discrete drums have the same eigenvalues to rounding.
--
-- Usable outside Roblox: drum.new(vertices, m) and drum.modes(model, k).

local citrate = require("../../src")
local f64, krylov, graph, pde = citrate.f64, citrate.krylov, citrate.graph, citrate.pde

local drum = {}

-- The Gordon-Webb-Wolpert drums (as in Driscoll, SIAM Review 1997), in grid units: each
-- is seven half-squares of the unit grid, area 3.5. Driscoll's eigenvalues are for
-- twice this size, so a quarter of these.
drum.gww = {
	{ { 0, 0 }, { 0, 1 }, { 2, 3 }, { 2, 2 }, { 3, 2 }, { 2, 1 }, { 1, 1 }, { 1, 0 } },
	{ { 0, 2 }, { 2, 2 }, { 2, 3 }, { 3, 2 }, { 2, 1 }, { 1, 1 }, { 1, 0 }, { 0, 1 } },
}

-- The value at (px, py) of a function given at the nodes (0 outside the drum), by linear
-- interpolation on the mesh triangle containing the point.
function drum.interpolate(model: Model, values: { number }, px: number, py: number): number
	local x: { number }, y: { number }, T = model.x, model.y, model.triangles
	for t = 0, #T // 3 - 1 do
		local a, b, c = T[3 * t + 1], T[3 * t + 2], T[3 * t + 3]
		local d = (y[b] - y[c]) * (x[a] - x[c]) + (x[c] - x[b]) * (y[a] - y[c])
		local l1 = ((y[b] - y[c]) * (px - x[c]) + (x[c] - x[b]) * (py - y[c])) / d
		local l2 = ((y[c] - y[a]) * (px - x[c]) + (x[a] - x[c]) * (py - y[c])) / d
		local l3 = 1 - l1 - l2
		if l1 >= -1e-12 and l2 >= -1e-12 and l3 >= -1e-12 then
			return l1 * values[a] + l2 * values[b] + l3 * values[c]
		end
	end
	return 0
end

local function inside(poly: { { number } }, x: number, y: number): boolean
	local c = false
	local n = #poly
	for i = 1, n do
		local a, b = poly[i], poly[i % n + 1]
		if (a[2] > y) ~= (b[2] > y) and x < a[1] + (y - a[2]) * (b[1] - a[1]) / (b[2] - a[2]) then
			c = not c
		end
	end
	return c
end

export type Model = {
	n: number, -- unknowns (interior nodes)
	K: any, -- stiffness, n by n sparse
	mass: buffer, -- lumped masses of the unknowns
	-- every node (boundary ones too), as positions, with the unknown each is (0 on the boundary)
	x: { number },
	y: { number },
	unknown: { number },
	-- triangles as node triples (1-based, into x and y)
	triangles: { number },
	area: number,
}

function drum.new(poly: { { number } }, m: number): Model
	local xmin: number, xmax: number, ymin: number, ymax: number = math.huge, -math.huge, math.huge, -math.huge
	for _, p in poly do
		local px: number, py: number = p[1], p[2]
		xmin, xmax = math.min(xmin, px), math.max(xmax, px)
		ymin, ymax = math.min(ymin, py), math.max(ymax, py)
	end
	local h = 1 / m
	local nx, ny = math.round((xmax - xmin) * m), math.round((ymax - ymin) * m)
	-- Nodes: subsquare corners (i, j), 0 <= i <= nx, and centers. A node is an unknown when
	-- every mesh triangle around it is inside (8 around a corner, 4 around a center).
	local W = nx + 1
	local function corner(i: number, j: number): number
		return j * W + i + 1
	end
	local ncorner = W * (ny + 1)
	local function center(i: number, j: number): number
		return ncorner + j * nx + i + 1
	end
	local count: { number } = table.create(ncorner + nx * ny, 0)
	local tris: { number } = {}
	for j = 0, ny - 1 do
		for i = 0, nx - 1 do
			local cx, cy = xmin + (i + 0.5) * h, ymin + (j + 0.5) * h
			local c = center(i, j)
			local ring = { corner(i, j), corner(i + 1, j), corner(i + 1, j + 1), corner(i, j + 1) }
			local rx = { cx - h / 2, cx + h / 2, cx + h / 2, cx - h / 2 }
			local ry = { cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2 }
			for e = 1, 4 do
				local f = e % 4 + 1
				-- the triangle's centroid decides (edges of the polygon are mesh edges)
				if inside(poly, (rx[e] + rx[f] + cx) / 3, (ry[e] + ry[f] + cy) / 3) then
					table.insert(tris, ring[e])
					table.insert(tris, ring[f])
					table.insert(tris, c)
					count[ring[e]] += 1
					count[ring[f]] += 1
					count[c] += 1
				end
			end
		end
	end
	-- the nodes in use, renumbered compactly, and their positions
	local id = table.create(#count, 0)
	local points = {}
	for k = 1, #count do
		if count[k] > 0 then
			if k <= ncorner then
				local j, i = (k - 1) // W, (k - 1) % W
				table.insert(points, { xmin + i * h, ymin + j * h })
			else
				local j, i = (k - ncorner - 1) // nx, (k - ncorner - 1) % nx
				table.insert(points, { xmin + (i + 0.5) * h, ymin + (j + 0.5) * h })
			end
			id[k] = #points
		end
	end
	local triangles = {}
	for t = 1, #tris do
		tris[t] = id[tris[t]]
	end
	for t = 0, #tris // 3 - 1 do
		table.insert(triangles, { tris[3 * t + 1], tris[3 * t + 2], tris[3 * t + 3] })
	end
	-- linear finite elements (pde.fem): the unknowns are the points off the boundary, in
	-- point order; K and the lumped mass restricted to them
	local D = pde.fem(points, triangles)
	local n = #D.interior
	local K = D._Kii
	local mass = f64.create(n)
	for a, k in D.interior do
		f64.set(mass, a, D.mass[k])
	end
	local unknown = D._index
	return { n = n, K = K, mass = mass, x = D.x, y = D.y, unknown = unknown, triangles = tris, area = D.volume }
end

-- The k lowest eigenvalues (ascending, as a table) and their modes (f64 arrays over the
-- unknowns, orthonormal in the mass inner product, each positive at its largest entry),
-- by thick-restart Lanczos on A = D^-1/2 K D^-1/2: by default on A^-1, whose largest
-- eigenvalues are A's smallest, well separated, so Lanczos converges in a few dozen
-- products; each is a solve with A's Cholesky factor, banded once the mesh is reordered
-- (reverse Cuthill-McKee: 1697 unknowns have a band of 872 as numbered and 62 after).
-- method = "lanczos" works on A itself instead (no factorization; many more products).
-- options.yield, if given, is called every 20 products (Roblox: task.wait).
type ModeOptions = { tol: number?, yield: (() -> ())?, method: string? }
function drum.modes(model: Model, k: number, options: ModeOptions?): ({ number }, { buffer })
	local o: ModeOptions = options or {}
	local yield = o.yield
	local n = model.n
	local dinv = f64.create(n)
	for i = 1, n do
		f64.set(dinv, i, 1 / math.sqrt(f64.get(model.mass, i)))
	end
	local tmp = f64.create(n)
	local products = 0
	local v0 = f64.create(n)
	for i = 1, n do
		f64.set(v0, i, 1 + 0.1 * math.sin(i))
	end
	local values: buffer, vectors: { buffer }
	if o.method == "lanczos" then
		values, vectors = krylov.eigsh(f64.space, function(out: buffer, v: buffer): buffer
			products += 1
			if yield and products % 20 == 0 then
				yield()
			end
			f64.mul(tmp, dinv, v)
			f64.sparse_matvec(out, model.K, tmp)
			return f64.mul(out, dinv, out)
		end, v0, k, { which = "SA", tol = o.tol or 1e-10, ncv = math.max(2 * k + 10, 30), maxiter = 2000 })
	else
		-- A = D^-1/2 K D^-1/2 and its Cholesky factorization: sparse (ordered by approximate
		-- minimum degree), or (method = "banded") in band storage after reverse Cuthill-McKee
		local K = model.K
		local scaled = f64.create(f64.length(K.values))
		for j = 0, n - 1 do
			for p = buffer.readu32(K.colptr, 4 * j), buffer.readu32(K.colptr, 4 * j + 4) - 1 do
				local i = buffer.readu32(K.rowind, 4 * p)
				buffer.writef64(scaled, 8 * p, buffer.readf64(K.values, 8 * p) * f64.get(dinv, i + 1) * f64.get(dinv, j + 1))
			end
		end
		local A: citrate.Sparse = { rows = n, cols = n, colptr = K.colptr, rowind = K.rowind, values = scaled, symmetric = true }
		local solve: (buffer, buffer) -> buffer
		if o.method == "banded" then
			local perm = graph.rcm(graph.of(A))
			local F = f64.band_chol(f64.sparse_symband(A, perm))
			assert(F.definite, "drum.modes: the stiffness matrix is not positive definite")
			local pb = f64.create(n)
			solve = function(out: buffer, v: buffer): buffer
				-- A^-1 v = P^T (P A P^T)^-1 P v
				for r = 0, n - 1 do
					buffer.writef64(pb, 8 * r, buffer.readf64(v, 8 * (buffer.readi32(perm, 4 * r) - 1)))
				end
				f64.band_chol_solve(pb, F, pb)
				for r = 0, n - 1 do
					buffer.writef64(out, 8 * (buffer.readi32(perm, 4 * r) - 1), buffer.readf64(pb, 8 * r))
				end
				return out
			end
		else
			local F = f64.sparse_chol(A)
			assert(F.definite, "drum.modes: the stiffness matrix is not positive definite")
			solve = function(out: buffer, v: buffer): buffer
				return f64.sparse_chol_solve(out, F, v)
			end
		end
		values, vectors = krylov.eigsh(f64.space, function(out: buffer, v: buffer): buffer
			products += 1
			if yield and products % 20 == 0 then
				yield()
			end
			return solve(out, v)
		end, v0, k, { which = "LA", tol = o.tol or 1e-10, ncv = math.max(2 * k + 10, 30), maxiter = 2000 })
		for i = 1, f64.length(values) do
			f64.set(values, i, 1 / f64.get(values, i))
		end
	end
	-- sort ascending; the modes are D^-1/2 times the eigenvectors
	local order = {}
	for i = 1, f64.length(values) do
		order[i] = i
	end
	table.sort(order, function(a: number, b: number): boolean
		return f64.get(values, a) < f64.get(values, b)
	end)
	local lambdas, modes = {}, {}
	for r, i in order do
		lambdas[r] = f64.get(values, i)
		local u = f64.mul(f64.create(n), dinv, vectors[i])
		local big = f64.amax(u)
		-- sign: positive at its largest entry
		local s = 1
		for j = 1, n do
			if math.abs(f64.get(u, j)) == big then
				s = if f64.get(u, j) > 0 then 1 else -1
				break
			end
		end
		modes[r] = if s < 0 then f64.scale(u, -1, u) else u
	end
	return lambdas, modes
end

return drum
