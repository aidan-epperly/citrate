--!strict
--!native
--!optimize 2

-- Computational geometry in any dimension: exact orientation and in-sphere tests, Delaunay
-- triangulations and convex hulls.
--
--   geometry.orient(a, b, c)         -- 1 if a, b, c turn counterclockwise, -1 clockwise, 0
--                                    -- if they are collinear (exactly)
--   geometry.insphere(a, b, c, p)    -- 1 if p is inside the circle through a, b, c
--   local T = geometry.delaunay(points)
--
-- Points are vectors (3-D) or tables or f64 arrays of d numbers, any d.
--
-- Exactness. The algorithms branch on the signs of determinants, and a sign computed in
-- floating point can be wrong for nearly degenerate input (points nearly on a line, or
-- nearly on a circle: regular grids of points are full of exactly degenerate ones). A wrong
-- sign can make a triangulation overlap itself or loop forever. So the signs are exact:
-- each determinant is computed in floating point with a bound on its rounding error
-- (Shewchuk's for 2-D and 3-D, a general one otherwise), and when the bound does not settle
-- the sign, again exactly, in floating-point expansions (Shewchuk, "Adaptive precision
-- floating-point arithmetic and fast robust geometric predicates", 1997: a number as a sum
-- of doubles that do not overlap, with exact sums and products). The exact path costs far
-- more, but is rare except on degenerate input. Coordinates must stay below about 1e150 in
-- size (so that products do not overflow).

local random = require("./random")

local function fail(who: string, message: string): never
	error("geometry." .. who .. ": " .. message, 3)
end

-- Exact arithmetic on expansions: tables of doubles, increasing in magnitude, nonoverlapping;
-- their sum is the number. Zero components are dropped (zero is the empty table).

local SPLITTER = 2 ^ 27 + 1
local U = 2 ^ -53 -- the unit roundoff

-- a + b = x + y exactly.
local function two_sum(a: number, b: number): (number, number)
	local x = a + b
	local bv = x - a
	local av = x - bv
	return x, (a - av) + (b - bv)
end

-- a * b = x + y exactly (Dekker's product, with Veltkamp's split).
local function two_product(a: number, b: number): (number, number)
	local x = a * b
	local c = SPLITTER * a
	local ahi = c - (c - a)
	local alo = a - ahi
	c = SPLITTER * b
	local bhi = c - (c - b)
	local blo = b - bhi
	local err = x - ahi * bhi
	err -= alo * bhi
	err -= ahi * blo
	return x, alo * blo - err
end

-- e + f (Shewchuk's fast-expansion-sum, zeros eliminated: the components of both merged
-- by magnitude and accumulated in one pass).
local function add(e: { number }, f: { number }): { number }
	local ne, nf = #e, #f
	if ne == 0 then
		return table.clone(f)
	elseif nf == 0 then
		return table.clone(e)
	end
	local h = table.create(ne + nf) :: { number }
	local ei, fi = 1, 1
	local enow, fnow = e[1], f[1]
	local q: number
	if (fnow > enow) == (fnow > -enow) then
		q = enow
		ei += 1
		enow = e[ei]
	else
		q = fnow
		fi += 1
		fnow = f[fi]
	end
	if ei <= ne and fi <= nf then
		local x, y
		-- the first sum is of the two smallest components: Dekker's fast two-sum
		if (fnow > enow) == (fnow > -enow) then
			x = enow + q
			y = q - (x - enow)
			ei += 1
			enow = e[ei]
		else
			x = fnow + q
			y = q - (x - fnow)
			fi += 1
			fnow = f[fi]
		end
		q = x
		if y ~= 0 then
			table.insert(h, y)
		end
		while ei <= ne and fi <= nf do
			if (fnow > enow) == (fnow > -enow) then
				x, y = two_sum(q, enow)
				ei += 1
				enow = e[ei]
			else
				x, y = two_sum(q, fnow)
				fi += 1
				fnow = f[fi]
			end
			q = x
			if y ~= 0 then
				table.insert(h, y)
			end
		end
	end
	while ei <= ne do
		local x, y = two_sum(q, enow)
		ei += 1
		enow = e[ei]
		q = x
		if y ~= 0 then
			table.insert(h, y)
		end
	end
	while fi <= nf do
		local x, y = two_sum(q, fnow)
		fi += 1
		fnow = f[fi]
		q = x
		if y ~= 0 then
			table.insert(h, y)
		end
	end
	if q ~= 0 then
		table.insert(h, q)
	end
	return h
end

-- e * b (Shewchuk's scale-expansion, zeros eliminated).
local function scale(e: { number }, b: number): { number }
	local h = {}
	if #e == 0 or b == 0 then
		return h
	end
	local q, y = two_product(e[1], b)
	if y ~= 0 then
		table.insert(h, y)
	end
	for i = 2, #e do
		local p1, p0 = two_product(e[i], b)
		local s, t = two_sum(q, p0)
		if t ~= 0 then
			table.insert(h, t)
		end
		q, t = two_sum(p1, s)
		if t ~= 0 then
			table.insert(h, t)
		end
	end
	if q ~= 0 then
		table.insert(h, q)
	end
	return h
end

-- e * f.
local function mul(e: { number }, f: { number }): { number }
	local h = {}
	for _, fi in f do
		h = add(h, scale(e, fi))
	end
	return h
end

-- The sign of an expansion: its largest component's.
local function sign(e: { number }): number
	local top = e[#e]
	return if top == nil or top == 0 then 0 elseif top > 0 then 1 else -1
end

-- The exact difference a - b as an expansion.
local function diff(a: number, b: number): { number }
	local x = a - b
	local bv = a - x
	local av = x + bv
	local y = (a - av) + (bv - b)
	if y == 0 then
		return if x == 0 then {} else { x }
	end
	return { y, x }
end

-- The determinant of an m by m matrix, exactly or in floating point.
--
-- By Laplace expansion along the rows, over subsets of columns: D[S] is the determinant of
-- the first |S| rows restricted to the columns in S, and
--   D[S] = sum over j in S of (-1)^(|S| + rank of j in S) a(|S|, j) D[S - {j}],
-- 2^m m products in all (the matrices here are small: m is the dimension plus one or two).

-- Floating point, with A[S] the same sum of absolute values: the rounding error is at most
-- gamma_k A[full] with k = m (m + 1) / 2 (a product and k - 1 additions at level k).
local function det_float(M: { number }, m: number, D: { number }, A: { number }): (number, number)
	D[1], A[1] = 1, 1 -- the empty set (index 0 + 1)
	local full = bit32.lshift(1, m) - 1
	for S = 1, full do
		local k = 0 -- |S|
		local s = S
		while s ~= 0 do
			k += 1
			s = bit32.band(s, s - 1)
		end
		local val, abs = 0, 0
		local rank = 0
		local row = (k - 1) * m
		for j = 1, m do
			local bit = bit32.lshift(1, j - 1)
			if bit32.band(S, bit) ~= 0 then
				rank += 1
				local a = M[row + j]
				local sub = S - bit + 1
				local term = a * D[sub]
				if (k + rank) % 2 == 1 then
					term = -term
				end
				val += term
				abs += math.abs(a) * A[sub]
			end
		end
		D[S + 1], A[S + 1] = val, abs
	end
	return D[full + 1], A[full + 1]
end

-- Exactly, the entries being expansions.
local function det_exact(M: { { number } }, m: number): { number }
	local D: { { number } } = { {1} }
	local full = bit32.lshift(1, m) - 1
	for S = 1, full do
		local k = 0
		local s = S
		while s ~= 0 do
			k += 1
			s = bit32.band(s, s - 1)
		end
		local val: { number } = {}
		local rank = 0
		local row = (k - 1) * m
		for j = 1, m do
			local bit = bit32.lshift(1, j - 1)
			if bit32.band(S, bit) ~= 0 then
				rank += 1
				local term = mul(M[row + j], D[S - bit + 1])
				if (k + rank) % 2 == 1 then
					for i = 1, #term do
						term[i] = -term[i]
					end
				end
				val = add(val, term)
			end
		end
		D[S + 1] = val
	end
	return D[full + 1]
end

-- Predicates on points stored as coordinates: point i's d coordinates are X[(i - 1) d + 1 ..
-- i d].

-- Scratch for the general predicates, by size.
local scratch_m: { [number]: { number } } = {}
local scratch_d: { [number]: { number } } = {}
local scratch_a: { [number]: { number } } = {}
local function scratch(m: number): ({ number }, { number }, { number })
	local M = scratch_m[m]
	if M == nil then
		M = table.create(m * m, 0)
		scratch_m[m] = M
		scratch_d[m] = table.create(bit32.lshift(1, m), 0)
		scratch_a[m] = table.create(bit32.lshift(1, m), 0)
	end
	return M, scratch_d[m], scratch_a[m]
end

-- A shortcut to exactness, for grids: when every coordinate difference, scaled by 1 or 16,
-- is an integer (so the differences are exact), and the determinant's terms in absolute
-- value add up to less than 2^53, every product and sum in the floating-point determinant
-- is an integer below 2^53, so it is exact. Returns the sign, or nil when it does not apply.
-- M holds the m by m entries (the lifted column, if any, computed from the scaled
-- differences by the caller through lift = true).
local TWO53 = 2 ^ 53
local function integer_sign(X: { number }, d: number, rows: { number }, p0: number, m: number, lift: boolean): number?
	local M, D, A = scratch(m)
	for _, s in { 1, 16 } do
		local ok = true
		for i = 1, m do
			local bi = (rows[i] - 1) * d
			local sum = 0
			for a = 1, d do
				local xa, xb = X[bi + a], X[p0 + a]
				local x = xa - xb
				-- the difference must be exact: its rounding error (as in diff) zero
				local bv = xa - x
				local e = x * s
				if (xa - (x + bv)) + (bv - xb) ~= 0 or e ~= math.floor(e) or math.abs(e) >= 2 ^ 26 then
					ok = false
					break
				end
				M[(i - 1) * m + a] = e
				sum += e * e
			end
			if not ok then
				break
			end
			if lift then
				M[(i - 1) * m + m] = sum
			end
		end
		if ok then
			local det, abs = det_float(M, m, D, A)
			if abs < TWO53 then
				return if det > 0 then 1 elseif det < 0 then -1 else 0
			end
			return nil
		end
	end
	return nil
end

-- The orientation of the d + 1 points ids[1 .. d + 1]: the sign of det[p_i - p_1], i = 2 ..
-- d + 1 (rows). In 2-D, 1 for a counterclockwise turn.
local rows_scratch: { number } = {}
local function orient_exact(X: { number }, d: number, ids: { number }): number
	for i = 1, d do
		rows_scratch[i] = ids[i + 1]
	end
	local quick = integer_sign(X, d, rows_scratch, (ids[1] - 1) * d, d, false)
	if quick then
		return quick
	end
	local M: { { number } } = table.create(d * d) :: any
	local b0 = (ids[1] - 1) * d
	for i = 1, d do
		local bi = (ids[i + 1] - 1) * d
		for a = 1, d do
			M[(i - 1) * d + a] = diff(X[bi + a], X[b0 + a])
		end
	end
	return sign(det_exact(M, d))
end

local function orient_general(X: { number }, d: number, ids: { number }): number
	local M, D, A = scratch(d)
	local b0 = (ids[1] - 1) * d
	for i = 1, d do
		local bi = (ids[i + 1] - 1) * d
		for a = 1, d do
			M[(i - 1) * d + a] = X[bi + a] - X[b0 + a]
		end
	end
	local det, abs = det_float(M, d, D, A)
	-- the entries' own rounding (a relative u each) adds d to the count
	local k = d * (d + 1) / 2 + d + 1
	local bound = k * U / (1 - k * U) * abs
	if det > bound then
		return 1
	elseif -det > bound then
		return -1
	end
	return orient_exact(X, d, ids)
end

-- The in-sphere test of point p against the sphere through the d + 1 points ids[1 .. d + 1]:
-- the sign of det[p_i - p, |p_i - p|^2] (rows i = 1 .. d + 1), which is the orientation's
-- sign times (positive inside, negative outside); see insphere below.
local function lifted_exact(X: { number }, d: number, ids: { number }, p: number): number
	local m = d + 1
	local quick = integer_sign(X, d, ids, (p - 1) * d, m, true)
	if quick then
		return quick
	end
	local M: { { number } } = table.create(m * m) :: any
	local bp = (p - 1) * d
	for i = 1, m do
		local bi = (ids[i] - 1) * d
		local lift: { number } = {}
		for a = 1, d do
			local e = diff(X[bi + a], X[bp + a])
			M[(i - 1) * m + a] = e
			lift = add(lift, mul(e, e))
		end
		M[(i - 1) * m + m] = lift
	end
	return sign(det_exact(M, m))
end

local function lifted_general(X: { number }, d: number, ids: { number }, p: number): number
	local m = d + 1
	local M, D, A = scratch(m)
	local bp = (p - 1) * d
	for i = 1, m do
		local bi = (ids[i] - 1) * d
		local lift = 0
		for a = 1, d do
			local e = X[bi + a] - X[bp + a]
			M[(i - 1) * m + a] = e
			lift += e * e
		end
		M[(i - 1) * m + m] = lift
	end
	local det, abs = det_float(M, m, D, A)
	-- the differences' rounding (u) and the lifted column's (2u + gamma_d) add 2 d + 4
	local k = m * (m + 1) / 2 + 2 * d + 4
	local bound = k * U / (1 - k * U) * abs
	if det > bound then
		return 1
	elseif -det > bound then
		return -1
	end
	return lifted_exact(X, d, ids, p)
end

-- 2-D and 3-D: the determinants written out, with Shewchuk's error bounds (from his
-- predicates.c: ccwerrboundA, o3derrboundA, iccerrboundA, isperrboundA); the exact path is
-- the general one.

local function orient2(X: { number }, ids: { number }): number
	local i1, i2, i3 = 2 * ids[1], 2 * ids[2], 2 * ids[3]
	local ax, ay = X[i1 - 1], X[i1]
	local bx, by = X[i2 - 1], X[i2]
	local cx, cy = X[i3 - 1], X[i3]
	-- det[b - a; c - a], as Shewchuk's (a - c) x (b - c)
	local l = (ax - cx) * (by - cy)
	local r = (ay - cy) * (bx - cx)
	local det = l - r
	local bound = (3 * U + 16 * U * U) * (math.abs(l) + math.abs(r))
	if det > bound then
		return 1
	elseif -det > bound then
		return -1
	end
	return orient_exact(X, 2, ids)
end

local function orient3(X: { number }, ids: { number }): number
	local i1, i2, i3, i4 = 3 * ids[1], 3 * ids[2], 3 * ids[3], 3 * ids[4]
	-- det[b - a; c - a; d - a] = -det[a - d; b - d; c - d]
	local dx, dy, dz = X[i4 - 2], X[i4 - 1], X[i4]
	local adx, ady, adz = X[i1 - 2] - dx, X[i1 - 1] - dy, X[i1] - dz
	local bdx, bdy, bdz = X[i2 - 2] - dx, X[i2 - 1] - dy, X[i2] - dz
	local cdx, cdy, cdz = X[i3 - 2] - dx, X[i3 - 1] - dy, X[i3] - dz
	local bdxcdy, cdxbdy = bdx * cdy, cdx * bdy
	local cdxady, adxcdy = cdx * ady, adx * cdy
	local adxbdy, bdxady = adx * bdy, bdx * ady
	local det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady)
	local perm = (math.abs(bdxcdy) + math.abs(cdxbdy)) * math.abs(adz)
		+ (math.abs(cdxady) + math.abs(adxcdy)) * math.abs(bdz)
		+ (math.abs(adxbdy) + math.abs(bdxady)) * math.abs(cdz)
	local bound = (7 * U + 56 * U * U) * perm
	if det > bound then
		return -1
	elseif -det > bound then
		return 1
	end
	return orient_exact(X, 3, ids)
end

local function lifted2(X: { number }, ids: { number }, p: number): number
	local i1, i2, i3, ip = 2 * ids[1], 2 * ids[2], 2 * ids[3], 2 * p
	local px, py = X[ip - 1], X[ip]
	local adx, ady = X[i1 - 1] - px, X[i1] - py
	local bdx, bdy = X[i2 - 1] - px, X[i2] - py
	local cdx, cdy = X[i3 - 1] - px, X[i3] - py
	local bdxcdy, cdxbdy = bdx * cdy, cdx * bdy
	local alift = adx * adx + ady * ady
	local cdxady, adxcdy = cdx * ady, adx * cdy
	local blift = bdx * bdx + bdy * bdy
	local adxbdy, bdxady = adx * bdy, bdx * ady
	local clift = cdx * cdx + cdy * cdy
	local det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady)
	local perm = (math.abs(bdxcdy) + math.abs(cdxbdy)) * alift
		+ (math.abs(cdxady) + math.abs(adxcdy)) * blift
		+ (math.abs(adxbdy) + math.abs(bdxady)) * clift
	local bound = (10 * U + 96 * U * U) * perm
	if det > bound then
		return 1
	elseif -det > bound then
		return -1
	end
	return lifted_exact(X, 2, ids, p)
end

local function lifted3(X: { number }, ids: { number }, p: number): number
	local i1, i2, i3, i4, ip = 3 * ids[1], 3 * ids[2], 3 * ids[3], 3 * ids[4], 3 * p
	local px, py, pz = X[ip - 2], X[ip - 1], X[ip]
	local aex, aey, aez = X[i1 - 2] - px, X[i1 - 1] - py, X[i1] - pz
	local bex, bey, bez = X[i2 - 2] - px, X[i2 - 1] - py, X[i2] - pz
	local cex, cey, cez = X[i3 - 2] - px, X[i3 - 1] - py, X[i3] - pz
	local dex, dey, dez = X[i4 - 2] - px, X[i4 - 1] - py, X[i4] - pz
	local aexbey, bexaey = aex * bey, bex * aey
	local ab = aexbey - bexaey
	local bexcey, cexbey = bex * cey, cex * bey
	local bc = bexcey - cexbey
	local cexdey, dexcey = cex * dey, dex * cey
	local cd = cexdey - dexcey
	local dexaey, aexdey = dex * aey, aex * dey
	local da = dexaey - aexdey
	local aexcey, cexaey = aex * cey, cex * aey
	local ac = aexcey - cexaey
	local bexdey, dexbey = bex * dey, dex * bey
	local bd = bexdey - dexbey
	local abc = aez * bc - bez * ac + cez * ab
	local bcd = bez * cd - cez * bd + dez * bc
	local cda = cez * da + dez * ac + aez * cd
	local dab = dez * ab + aez * bd + bez * da
	local alift = aex * aex + aey * aey + aez * aez
	local blift = bex * bex + bey * bey + bez * bez
	local clift = cex * cex + cey * cey + cez * cez
	local dlift = dex * dex + dey * dey + dez * dez
	local det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd)
	local aezplus, bezplus, cezplus, dezplus = math.abs(aez), math.abs(bez), math.abs(cez), math.abs(dez)
	local aexbeyplus, bexaeyplus = math.abs(aexbey), math.abs(bexaey)
	local bexceyplus, cexbeyplus = math.abs(bexcey), math.abs(cexbey)
	local cexdeyplus, dexceyplus = math.abs(cexdey), math.abs(dexcey)
	local dexaeyplus, aexdeyplus = math.abs(dexaey), math.abs(aexdey)
	local aexceyplus, cexaeyplus = math.abs(aexcey), math.abs(cexaey)
	local bexdeyplus, dexbeyplus = math.abs(bexdey), math.abs(dexbey)
	local perm = ((cexdeyplus + dexceyplus) * bezplus + (dexbeyplus + bexdeyplus) * cezplus + (bexceyplus + cexbeyplus) * dezplus) * alift
		+ ((dexaeyplus + aexdeyplus) * cezplus + (aexceyplus + cexaeyplus) * dezplus + (cexdeyplus + dexceyplus) * aezplus) * blift
		+ ((aexbeyplus + bexaeyplus) * dezplus + (bexdeyplus + dexbeyplus) * aezplus + (dexaeyplus + aexdeyplus) * bezplus) * clift
		+ ((bexceyplus + cexbeyplus) * aezplus + (cexaeyplus + aexceyplus) * bezplus + (aexbeyplus + bexaeyplus) * cezplus) * dlift
	local bound = (16 * U + 224 * U * U) * perm
	-- (Shewchuk's insphere determinant is det[p_i - p, |p_i - p|^2] itself)
	if det > bound then
		return 1
	elseif -det > bound then
		return -1
	end
	return lifted_exact(X, 3, ids, p)
end

-- The predicates for dimension d: orient(X, ids) and lifted(X, ids, p), and the sign that
-- turns lifted * orient into inside (+1) / outside (-1), found once per dimension from a
-- simplex whose inside is known.
type Predicates = {
	orient: (X: { number }, ids: { number }) -> number,
	lifted: (X: { number }, ids: { number }, p: number) -> number,
	inside: number,
}
local predicates_by_d: { [number]: Predicates } = {}
local function predicates(d: number): Predicates
	local P = predicates_by_d[d]
	if P then
		return P
	end
	local orient = if d == 2 then orient2
		elseif d == 3 then orient3
		else function(X: { number }, ids: { number }): number
			return orient_general(X, d, ids)
		end
	local lifted = if d == 2 then lifted2
		elseif d == 3 then lifted3
		else function(X: { number }, ids: { number }, p: number): number
			return lifted_general(X, d, ids, p)
		end
	-- the unit simplex 0, e_1, ..., e_d and a point inside it
	local X = table.create((d + 2) * d, 0) :: { number }
	for i = 1, d do
		X[i * d + i] = 1
	end
	for a = 1, d do
		X[(d + 1) * d + a] = 1 / (4 * d)
	end
	local ids = table.create(d + 1, 0) :: { number }
	for i = 1, d + 1 do
		ids[i] = i
	end
	local inside = orient(X, ids) * lifted(X, ids, d + 2)
	assert(inside ~= 0, "geometry: the predicates' calibration failed")
	P = { orient = orient, lifted = lifted, inside = inside }
	predicates_by_d[d] = P
	return P
end

-- Points

-- The coordinates of a list of points, d numbers each, into one table; returns it and d.
local function read_points(points: { any }, who: string): ({ number }, number)
	local n = #points
	local first = points[1]
	local d = if type(first) == "vector" then 3
		elseif type(first) == "table" then #first
		elseif type(first) == "buffer" then buffer.len(first) // 8
		else 0
	if d < 1 then
		fail(who, "point 1 is not a vector, a table of numbers or an f64 array")
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
			fail(who, string.format("point %d is not like point 1 (%d coordinates)", k, d))
		end
		for a = 1, d do
			local v = X[base + a]
			if type(v) ~= "number" or v ~= v or math.abs(v) == math.huge then
				fail(who, string.format("point %d has a coordinate that is not a finite number", k))
			end
		end
	end
	return X, d
end

-- Triangulations

-- Delaunay triangulations by Bowyer and Watson's algorithm, in any dimension d >= 2: points
-- are inserted one at a time; each new point removes the simplices whose circumsphere
-- holds it (a cavity, found from the simplex containing it by following neighbors) and
-- joins itself to the cavity's boundary. The triangulation is closed off by "ghost"
-- simplices joining each facet of the convex hull to a point at infinity, so a point
-- outside the hull is inserted the same way (a ghost holds p when p lies beyond its facet,
-- or, with p on the facet's hyperplane, when the finite simplex across the facet holds
-- it), and collinear or coplanar points need no special cases. Points are inserted in a
-- randomized spatial order (Amenta, Choi and Rote's BRIO, rounds of doubling size, each
-- sorted along a k-d tree's order) and located by walking from the last new simplex,
-- so each insertion costs about a constant.
--
-- With hull = true, finite simplices never hold a point: only points outside the current
-- hull are inserted, and the result is the convex hull (with a triangulation of its
-- inside that is not Delaunay).
--
-- Storage, for speed in any dimension: simplex s has vertices V[(s - 1) m + 1 .. s m]
-- (m = d + 1; 0 is the point at infinity) and neighbors N[...] (N[(s - 1) m + i] is across
-- the facet opposite vertex i). Finite simplices are positively oriented; a ghost is
-- oriented so that replacing its infinite vertex by a point beyond its facet gives a
-- positive orientation.

type Mesh = {
	d: number,
	m: number,
	X: { number },
	V: { number },
	N: { number },
	inf: { number }, -- the position of each simplex's infinite vertex (0: finite)
	alive: { boolean },
	free: { number },
	count: number,
	stamp: { number },
	tick: number,
	P: Predicates,
	hull: boolean,
	ids: { number }, -- scratch: a simplex's vertices
	last: number, -- the simplex to start walks from
	seed: number, -- for the walk's choices
	base: number, -- facet keys: vertex numbers in base n + 1, when they fit in 53 bits
	numeric: boolean,
}

local function new_simplex(T: Mesh): number
	local s: number = table.remove(T.free) or 0
	if s == 0 then
		T.count += 1
		s = T.count
		T.stamp[s] = 0
	end
	T.alive[s] = true
	return s
end

-- Whether simplex s conflicts with point p (p would remove it): finite, p strictly inside
-- its circumsphere; a ghost, p beyond its facet, or on the facet's hyperplane and in
-- conflict with the finite simplex across it.
local function conflicts(T: Mesh, s: number, p: number): boolean
	local m, V, ids = T.m, T.V, T.ids
	local j = T.inf[s]
	if j ~= 0 then
		local base = (s - 1) * m
		for k = 1, m do
			ids[k] = if k == j then p else V[base + k]
		end
		local o = T.P.orient(T.X, ids)
		if o ~= 0 then
			return o > 0
		end
		s = T.N[base + j]
	end
	if T.hull then
		return false
	end
	local base = (s - 1) * m
	for k = 1, m do
		ids[k] = V[base + k]
	end
	return T.P.lifted(T.X, ids, p) == T.P.inside
end

-- The simplex containing p (a finite one, closed), or a ghost whose facet p lies strictly
-- beyond: a walk from T.last, crossing any facet p lies beyond, never straight back, and
-- trying the facets from a random one (the remembering stochastic walk of Devillers, Pion
-- and Teillaud: a fixed order can cycle in a triangulation that is not Delaunay).
local function locate(T: Mesh, p: number): number
	local m, V, N, inf, ids, X = T.m, T.V, T.N, T.inf, T.ids, T.X
	local orient = T.P.orient
	local s = T.last
	if inf[s] ~= 0 then
		s = N[(s - 1) * m + inf[s]]
	end
	local from = 0
	local seed = T.seed
	while inf[s] == 0 do
		local base = (s - 1) * m
		for k = 1, m do
			ids[k] = V[base + k]
		end
		seed = (seed * 16807) % 2147483647
		local turn = seed % m
		local moved = false
		for q = 0, m - 1 do
			local i = (turn + q) % m + 1
			local nb = N[base + i]
			if nb ~= from then
				ids[i] = p
				local o = orient(X, ids)
				ids[i] = V[base + i]
				if o < 0 then
					from = s
					s = nb
					moved = true
					break
				end
			end
		end
		if not moved then
			break
		end
	end
	T.seed = seed
	return s
end

-- Inserts point p; false if it duplicates a vertex (or lies inside the hull, in hull mode).
local function insert(T: Mesh, p: number, cavity: { number }, boundary: { number }, waiting: { [any]: number }, sorted: { number }): boolean
	local s = locate(T, p)
	if not conflicts(T, s, p) then
		return false
	end
	local m, V, N, inf, stamp = T.m, T.V, T.N, T.inf, T.stamp
	T.tick += 2
	local inside, outside = T.tick, T.tick - 1 -- stamps: in the cavity, checked and not
	table.clear(cavity)
	table.clear(boundary)
	cavity[1] = s
	stamp[s] = inside
	local top = 1
	while top <= #cavity do
		local c = cavity[top]
		top += 1
		local base = (c - 1) * m
		for i = 1, m do
			local nb = N[base + i]
			local st = stamp[nb]
			if st ~= inside then
				if st ~= outside and conflicts(T, nb, p) then
					stamp[nb] = inside
					table.insert(cavity, nb)
				else
					stamp[nb] = outside
					table.insert(boundary, c * m + i - 1)
				end
			end
		end
	end
	-- a new simplex on each boundary facet: the cavity simplex with p for the vertex across
	-- the facet, keeping its orientation; the new simplices' facets through p are joined by
	-- matching their other d - 1 vertices (sorted, as a key)
	table.clear(waiting)
	local made = 0
	local nbase, numeric = T.base, T.numeric
	for _, f in boundary do
		local c, i = f // m, f % m + 1
		local t = new_simplex(T)
		local cb, tb = (c - 1) * m, (t - 1) * m
		local ghost = 0
		for k = 1, m do
			local v = if k == i then p else V[cb + k]
			V[tb + k] = v
			if v == 0 then
				ghost = k
			end
		end
		inf[t] = ghost
		local out = N[cb + i]
		N[tb + i] = out
		local ob = (out - 1) * m
		for k = 1, m do
			if N[ob + k] == c then
				N[ob + k] = t
				break
			end
		end
		for k = 1, m do
			if k ~= i then
				-- the key of the facet opposite k: the vertices but p and the k-th
				local cnt = 0
				for r = 1, m do
					if r ~= i and r ~= k then
						local v = V[tb + r]
						local q = cnt
						while q > 0 and sorted[q] > v do
							sorted[q + 1] = sorted[q]
							q -= 1
						end
						sorted[q + 1] = v
						cnt += 1
					end
				end
				local key: any
				if numeric then
					key = 0
					for q = cnt, 1, -1 do
						key = key * nbase + sorted[q]
					end
				else
					key = table.concat(sorted, ",", 1, cnt)
				end
				local other = waiting[key]
				if other == nil then
					waiting[key] = t * m + k - 1
				else
					waiting[key] = nil
					local os = other // m
					N[tb + k] = os
					N[(os - 1) * m + other % m + 1] = t
				end
			end
		end
		made = t
	end
	for _, c in cavity do
		T.alive[c] = false
		stamp[c] = 0
		table.insert(T.free, c)
	end
	T.last = made
	return true
end

-- The order of insertion: BRIO rounds (the last half of a random permutation, then the
-- last half of the rest, ...), each sorted along a k-d tree's median order.
local function insertion_order(X: { number }, d: number, ids: { number }): { number }
	local n = #ids
	local rng = random.new(12345)
	for i = n, 2, -1 do
		local j = rng:integer(1, i)
		ids[i], ids[j] = ids[j], ids[i]
	end
	local function kd_sort(lo: number, hi: number, depth: number)
		if hi - lo < 2 then
			return
		end
		-- the widest axis of the box
		local wa, widest = 1, -1
		for a = 1, d do
			local mn, mx = math.huge, -math.huge
			for k = lo, hi do
				local v = X[(ids[k] - 1) * d + a]
				mn, mx = math.min(mn, v), math.max(mx, v)
			end
			if mx - mn > widest then
				wa, widest = a, mx - mn
			end
		end
		local mid = (lo + hi) // 2
		local l, h = lo, hi
		while l < h do
			local pv = X[(ids[(l + h) // 2] - 1) * d + wa]
			local i, j = l, h
			while i <= j do
				while X[(ids[i] - 1) * d + wa] < pv do
					i += 1
				end
				while X[(ids[j] - 1) * d + wa] > pv do
					j -= 1
				end
				if i <= j then
					ids[i], ids[j] = ids[j], ids[i]
					i += 1
					j -= 1
				end
			end
			if mid <= j then
				h = j
			elseif mid >= i then
				l = i
			else
				break
			end
		end
		kd_sort(lo, mid, depth + 1)
		kd_sort(mid + 1, hi, depth + 1)
	end
	local hi = n
	while hi >= 1 do
		local lo = if hi > 64 then hi // 2 + 1 else 1
		kd_sort(lo, hi, 0)
		hi = lo - 1
	end
	return ids
end

-- Whether the points ids (fewer than d + 2) are affinely independent, exactly: some
-- projection onto |ids| - 1 coordinate axes has a nonzero orientation.
local function independent(X: { number }, d: number, ids: { number }): boolean
	local k = #ids - 1
	if k == 0 then
		return true
	end
	local P = predicates(k)
	local Y = table.create(#ids * k, 0) :: { number }
	local local_ids = table.create(#ids, 0) :: { number }
	for i = 1, #ids do
		local_ids[i] = i
	end
	-- each set of k axes, in lexicographic order
	local axes = table.create(k, 0) :: { number }
	for a = 1, k do
		axes[a] = a
	end
	while true do
		for i, id in ids do
			for a = 1, k do
				Y[(i - 1) * k + a] = X[(id - 1) * d + axes[a]]
			end
		end
		if (if k == 1 then (if Y[1] ~= Y[2] then 1 else 0) else P.orient(Y, local_ids)) ~= 0 then
			return true
		end
		local a = k
		while a >= 1 and axes[a] == d - k + a do
			a -= 1
		end
		if a < 1 then
			return false
		end
		axes[a] += 1
		for b = a + 1, k do
			axes[b] = axes[b - 1] + 1
		end
	end
end

export type TriangulationOptions = { hull: boolean? }
export type Triangulation = {
	d: number,
	n: number,
	simplices: { { number } },
	hull: { { number } },
	vertices: { number },
	duplicates: { [number]: number },
	[string]: any,
}

local function triangulate(points: { any }, options: TriangulationOptions?, who: string): Triangulation
	local o: TriangulationOptions = options or {}
	local X, d = read_points(points, who)
	local n = #points
	if d < 2 then
		fail(who, "the points need at least 2 coordinates (1-D: sort them)")
	end
	if n < d + 1 then
		fail(who, string.format("%d-dimensional points need at least %d", d, d + 1))
	end
	local m = d + 1
	local order = table.create(n, 0) :: { number }
	for i = 1, n do
		order[i] = i
	end
	insertion_order(X, d, order)
	-- the first simplex: d + 1 affinely independent points, taken in order
	local base: { number } = {}
	for _, i in order do
		table.insert(base, i)
		if not independent(X, d, base) then
			table.remove(base)
		elseif #base == m then
			break
		end
	end
	if #base < m then
		fail(who, string.format("the points lie in a %d-dimensional subspace (drop a coordinate, or rotate them into fewer)", #base - 1))
	end
	local T: Mesh = {
		d = d,
		m = m,
		X = X,
		V = {},
		N = {},
		alive = {},
		free = {},
		count = 0,
		stamp = { [0] = 0 } :: any,
		tick = 0,
		P = predicates(d),
		hull = o.hull == true,
		ids = table.create(m, 0),
		last = 1,
		seed = 12345,
		inf = {},
		base = n + 1,
		numeric = (n + 1) ^ (d - 1) < 2 ^ 53,
	}
	if T.P.orient(X, base) < 0 then
		base[1], base[2] = base[2], base[1]
	end
	local first = new_simplex(T)
	for k = 1, m do
		T.V[k] = base[k]
		T.N[k] = 0
	end
	T.inf[first] = 0
	-- a ghost on each facet: the facet with infinity across from the vertex it drops, two
	-- other vertices swapped to turn it outward; ghosts are joined across their facets
	-- through infinity by matching the facets' vertices
	local waiting: { [any]: number } = {}
	local sorted = table.create(m, 0) :: { number }
	local ghosts = {}
	for i = 1, m do
		local g = new_simplex(T)
		local gb = (g - 1) * m
		for k = 1, m do
			T.V[gb + k] = if k == i then 0 else base[k]
			T.N[gb + k] = 0
		end
		local a, b = if i == 1 then 2 else 1, if i <= 2 then 3 else 2
		T.V[gb + a], T.V[gb + b] = T.V[gb + b], T.V[gb + a]
		T.inf[g] = i
		T.N[gb + i] = first
		T.N[i] = g
		ghosts[i] = g
	end
	local facets: { [string]: number } = {}
	for _, g in ghosts do
		local gb = (g - 1) * m
		for k = 1, m do
			if k ~= T.inf[g] then
				local list = {}
				for r = 1, m do
					if r ~= k then
						table.insert(list, T.V[gb + r])
					end
				end
				table.sort(list)
				local key = table.concat(list, ",")
				local other = facets[key]
				if other == nil then
					facets[key] = g * m + k - 1
				else
					local os = other // m
					T.N[gb + k] = os
					T.N[(os - 1) * m + other % m + 1] = g
				end
			end
		end
	end
	T.last = first
	local in_base: { [number]: boolean } = {}
	for _, i in base do
		in_base[i] = true
	end
	local cavity, boundary = {}, {}
	local duplicates: { [number]: number } = {}
	local skipped: { number } = {}
	for _, p in order do
		if not in_base[p] then
			if not insert(T, p, cavity, boundary, waiting, sorted) then
				table.insert(skipped, p)
			end
		end
	end
	-- the results: finite simplices, hull facets (each ghost's finite vertices, ordered so
	-- that they and a point inside make a positive simplex), vertices
	local simplices: { { number } } = {}
	local hull: { { number } } = {}
	local used = table.create(n, false)
	local index = table.create(T.count, 0) :: { number }
	for s = 1, T.count do
		if T.alive[s] then
			local sb = (s - 1) * m
			local j = T.inf[s]
			if j == 0 then
				local simplex = table.create(m, 0) :: { number }
				for k = 1, m do
					simplex[k] = T.V[sb + k]
					used[simplex[k]] = true
				end
				table.insert(simplices, simplex)
				index[s] = #simplices
			else
				-- swapping infinity for the point across the facet must stay positive: the
				-- facet's vertices in the ghost's order with that point at j
				local facet = table.create(d, 0) :: { number }
				local c = 0
				for k = 1, m do
					if k ~= j then
						c += 1
						facet[c] = T.V[sb + k]
					end
				end
				-- the ghost with an inside point x for infinity is negative; moving x to the
				-- end takes m - j transpositions, so facet .. x is positive when m - j is
				-- odd: swap two when even
				if (m - j) % 2 == 0 then
					facet[1], facet[2] = facet[2], facet[1]
				end
				table.insert(hull, facet)
			end
		end
	end
	-- duplicates: a skipped point equal to a vertex (in hull mode, interior points are
	-- skipped too, and only exact copies are listed)
	if #skipped > 0 then
		local seen: { [string]: number } = {}
		local function key(i: number): string
			local parts = table.create(d, "")
			for a = 1, d do
				parts[a] = string.format("%.17g", X[(i - 1) * d + a])
			end
			return table.concat(parts, ",")
		end
		for i = 1, n do
			if used[i] then
				seen[key(i)] = i
			end
		end
		for _, i in skipped do
			local k = seen[key(i)]
			if k then
				duplicates[i] = k
			end
		end
	end
	local vertices: { number } = {}
	if T.hull then
		for _, f in hull do
			for _, v in f do
				used[v] = true
			end
		end
		-- the corners: a vertex v on the boundary is not one when some boundary neighbor u
		-- lies on the hyperplane of every facet through v (then the line uv lies in a face
		-- of the hull, with v inside it; for a corner, the facets through it meet only there)
		local through: { [number]: { number } } = {}
		for k, f in hull do
			for _, v in f do
				local list = through[v]
				if list == nil then
					list = {}
					through[v] = list
				end
				table.insert(list, k)
			end
		end
		local ids = table.create(m, 0) :: { number }
		for v = 1, n do
			local list = through[v]
			if list then
				local corner = true
				local tried: { [number]: boolean } = {}
				for _, k in list do
					for _, u in hull[k] do
						if u ~= v and not tried[u] then
							tried[u] = true
							local flat = true
							for _, k2 in list do
								local f = hull[k2]
								for q = 1, d do
									ids[q] = f[q]
								end
								ids[m] = u
								if T.P.orient(X, ids) ~= 0 then
									flat = false
									break
								end
							end
							if flat then
								corner = false
								break
							end
						end
					end
					if not corner then
						break
					end
				end
				if corner then
					table.insert(vertices, v)
				end
			end
		end
	else
		for i = 1, n do
			if used[i] then
				table.insert(vertices, i)
			end
		end
	end
	local R: any = {
		d = d,
		n = n,
		simplices = simplices,
		hull = hull,
		vertices = vertices,
		duplicates = duplicates,
		_mesh = T,
		_index = index,
	}
	return R
end

-- The Delaunay triangulation of points in d >= 2 dimensions (vectors are 3-D; tables or f64
-- arrays of d numbers): no point lies strictly inside any simplex's circumsphere. Returns
-- a table: simplices (each a table of the d + 1 indices of its vertices, positively
-- oriented: counterclockwise triangles in 2-D), hull (the convex hull's facets: d indices
-- each, in 2-D its edges, counterclockwise around the hull), vertices (the points used:
-- all but duplicates) and duplicates (index of a point -> the index of the point it
-- repeats). Points on a common circle (sphere) are triangulated in one of the valid ways.
-- Errors if all the points lie in a hyperplane (a line, in 2-D).
local function delaunay(points: { any }): Triangulation
	return triangulate(points, nil, "delaunay")
end

-- 2-D: Andrew's monotone chain (the points sorted by x, then y; the lower and upper hulls
-- built by keeping left turns, by the exact orientation), the same result as the general
-- hull (corners only, and a fan triangulation of the inside) at less than half the cost
-- (10,000 random points: 10 ms against 26, most of it the sort).
local function hull2(points: { any }): Triangulation
	local X, d = read_points(points, "hull")
	local n = #points
	local order = table.create(n, 0) :: { number }
	for i = 1, n do
		order[i] = i
	end
	table.sort(order, function(a: number, b: number): boolean
		local xa, xb = X[2 * a - 1], X[2 * b - 1]
		if xa ~= xb then
			return xa < xb
		end
		local ya, yb = X[2 * a], X[2 * b]
		if ya ~= yb then
			return ya < yb
		end
		return a < b
	end)
	local orient = predicates(2).orient
	local ids = { 0, 0, 0 }
	local H: { number } = {}
	local function chain(from: number, to: number, step: number, keep: number)
		for k = from, to, step do
			local p = order[k]
			while #H > keep do
				ids[1], ids[2], ids[3] = H[#H - 1], H[#H], p
				if orient(X, ids) > 0 then
					break
				end
				table.remove(H)
			end
			table.insert(H, p)
		end
	end
	-- lower hull left to right, then the upper right to left (each ends where the other
	-- starts)
	chain(1, n, 1, 1)
	local lower = #H
	chain(n - 1, 1, -1, lower)
	table.remove(H) -- the first point again
	-- exact copies of a corner are not corners
	local corners: { number } = {}
	local seen: { [number]: boolean } = {}
	for _, v in H do
		if not seen[v] then
			seen[v] = true
			table.insert(corners, v)
		end
	end
	if #corners < 3 then
		fail("hull", "the points lie in a 1-dimensional subspace (a line)")
	end
	local facets: { { number } } = {}
	local simplices: { { number } } = {}
	for k = 1, #corners do
		table.insert(facets, { corners[k], corners[k % #corners + 1] })
		if k >= 2 and k < #corners then
			table.insert(simplices, { corners[1], corners[k], corners[k + 1] })
		end
	end
	local duplicates: { [number]: number } = {}
	local at: { [string]: number } = {}
	for _, v in corners do
		at[string.format("%.17g,%.17g", X[2 * v - 1], X[2 * v])] = v
	end
	for i = 1, n do
		if not seen[i] then
			local k = at[string.format("%.17g,%.17g", X[2 * i - 1], X[2 * i])]
			if k then
				duplicates[i] = k
			end
		end
	end
	local R: any = { d = d, n = n, simplices = simplices, hull = facets, vertices = corners, duplicates = duplicates }
	return R
end

-- The convex hull of points in d >= 2 dimensions: a table with hull (the facets: d indices
-- each, oriented so that each facet with a point inside the hull is positively oriented;
-- in 2-D the edges, counterclockwise), vertices (the indices of its corners) and
-- simplices (a triangulation of its inside). Facets may be coplanar (a square face of a
-- cube comes as two triangles), and may have points on the hull's boundary that are not
-- corners among their vertices (a point inserted before the hull grew past it).
local function hull(points: { any }): Triangulation
	local first = points[1]
	if type(first) == "table" and #first == 2 or type(first) == "buffer" and buffer.len(first) == 16 then
		return hull2(points)
	end
	return triangulate(points, { hull = true }, "hull")
end

-- The orientation of d + 1 points in d dimensions (as separate arguments): 1, -1 or 0
-- exactly. In 2-D, 1 when a, b, c turn counterclockwise; in 3-D, 1 when d lies on the side
-- of the plane through a, b, c from which they turn clockwise (the sign of det[b - a; c -
-- a; d - a]); in general the sign of det[p_2 - p_1; ...; p_(d+1) - p_1].
local function orient(...: any): number
	local points = { ... }
	local X, d = read_points(points, "orient")
	if #points ~= d + 1 then
		fail("orient", string.format("%d-dimensional points need %d of them, not %d", d, d + 1, #points))
	end
	local ids = table.create(d + 1, 0) :: { number }
	for i = 1, d + 1 do
		ids[i] = i
	end
	return predicates(d).orient(X, ids)
end

-- Whether the last point lies inside (1), on (0) or outside (-1) the sphere through the
-- other d + 1 (in 2-D, the circle through three points), exactly. The d + 1 points must not
-- lie in a hyperplane (three on a line, in 2-D).
local function insphere(...: any): number
	local points = { ... }
	local X, d = read_points(points, "insphere")
	if #points ~= d + 2 then
		fail("insphere", string.format("%d-dimensional points need %d of them, not %d", d, d + 2, #points))
	end
	local ids = table.create(d + 1, 0) :: { number }
	for i = 1, d + 1 do
		ids[i] = i
	end
	local P = predicates(d)
	local o = P.orient(X, ids)
	if o == 0 then
		fail("insphere", "the first points lie in a hyperplane: they have no sphere through them")
	end
	return o * P.lifted(X, ids, d + 2) * P.inside
end

return {
	orient = orient,
	insphere = insphere,
	delaunay = delaunay,
	hull = hull,
	_predicates = predicates,
	_triangulate = triangulate,
	_read_points = read_points,
	_general = { orient = orient_general, lifted = lifted_general, orient_exact = orient_exact, lifted_exact = lifted_exact },
}
