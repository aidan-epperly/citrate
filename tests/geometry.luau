--!strict

-- Computational geometry: the exact predicates against floating point where it is reliable
-- and against exactly known answers where it is not (nearly collinear points, cocircular
-- points, hyperplanes), in 2 to 5 dimensions.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check
local geometry = lib.geometry

local seed = 777
local function rnd(): number
	seed = (seed * 16807) % 2147483647
	return seed / 2147483647
end

-- A determinant by Gaussian elimination with pivoting (reliable for random matrices).
local function det(M: { number }, m: number): number
	local A = table.clone(M)
	local s = 1
	for k = 1, m do
		local piv = k
		for i = k + 1, m do
			if math.abs(A[(i - 1) * m + k]) > math.abs(A[(piv - 1) * m + k]) then
				piv = i
			end
		end
		if piv ~= k then
			for j = 1, m do
				A[(k - 1) * m + j], A[(piv - 1) * m + j] = A[(piv - 1) * m + j], A[(k - 1) * m + j]
			end
			s = -s
		end
		s *= A[(k - 1) * m + k]
		for i = k + 1, m do
			local f = A[(i - 1) * m + k] / A[(k - 1) * m + k]
			for j = k, m do
				A[(i - 1) * m + j] -= f * A[(k - 1) * m + j]
			end
		end
	end
	return s
end

test("geometry: exact predicates", function()
	local G = geometry._general
	-- random points: the fast (2-D, 3-D), general and exact paths agree with floating point
	for d = 2, 5 do
		local P = geometry._predicates(d)
		local good = true
		for _ = 1, 300 do
			local X = {}
			for i = 1, (d + 2) * d do
				X[i] = rnd() * 10 - 5
			end
			local ids = {}
			for i = 1, d + 1 do
				ids[i] = i
			end
			local M = {}
			for i = 1, d do
				for a = 1, d do
					M[(i - 1) * d + a] = X[i * d + a] - X[a]
				end
			end
			local m = d + 1
			local L = {}
			for i = 1, m do
				local lift = 0
				for a = 1, d do
					local e = X[(i - 1) * d + a] - X[(d + 1) * d + a]
					L[(i - 1) * m + a] = e
					lift += e * e
				end
				L[(i - 1) * m + m] = lift
			end
			local o, l = math.sign(det(M, d)), math.sign(det(L, m))
			good = good
				and P.orient(X, ids) == o
				and G.orient(X, d, ids) == o
				and G.orient_exact(X, d, ids) == o
				and P.lifted(X, ids, d + 2) == l
				and G.lifted(X, d, ids, d + 2) == l
				and G.lifted_exact(X, d, ids, d + 2) == l
		end
		check(good, string.format("%d-D: fast, general and exact paths agree", d))
	end
	-- Kettner et al.'s classroom example: p = (0.5 + i u, 0.5 + j u) against the line through
	-- (12, 12) and (24, 24); the exact orientation is the sign of j - i, which floating
	-- point gets wrong for many (i, j)
	local u = 2 ^ -53
	local good, naive_wrong = true, 0
	for i = 0, 15 do
		for j = 0, 15 do
			local px, py = 0.5 + i * u, 0.5 + j * u
			local o = geometry.orient({ px, py }, { 12, 12 }, { 24, 24 })
			good = good and o == math.sign(j - i)
			local naive = math.sign((12 - px) * (24 - py) - (12 - py) * (24 - px))
			if naive ~= math.sign(j - i) then
				naive_wrong += 1
			end
		end
	end
	check(good and naive_wrong > 0, string.format("nearly collinear points (floating point is wrong on %d of 256)", naive_wrong))
	-- cocircular points (radius 5 about a far-off center) and points just off the circle
	local c = 2 ^ 20 + 0.5
	local A, B, C = { c + 3, c + 4 }, { c + 5, c }, { c, c + 5 }
	check(geometry.insphere(A, B, C, { c - 4, c - 3 }) == 0, "cocircular")
	local just_in = { c - 4, c - 3 + 2 ^ -32 }
	local just_out = { c - 4, c - 3 - 2 ^ -32 }
	check(geometry.insphere(A, B, C, just_in) == 1 and geometry.insphere(A, B, C, just_out) == -1, "just inside, just outside")
	check(geometry.insphere(B, A, C, just_in) == 1, "either orientation")
	-- 3-D: a sphere of radius 3 (Pythagorean quadruples 1^2 + 2^2 + 2^2 = 9)
	local S = { { 1, 2, 2 }, { 2, -1, 2 }, { -2, 2, 1 }, { 2, 2, -1 } }
	check(geometry.insphere(S[1], S[2], S[3], S[4], { -1, -2, -2 }) == 0, "cospherical in 3-D")
	check(geometry.insphere(S[1], S[2], S[3], S[4], { 0, 0, 2.999999 }) == 1 and geometry.insphere(S[1], S[2], S[3], S[4], { 0, 0, 3.000001 }) == -1, "inside and outside a sphere")
	-- 4-D: points on a hyperplane, and on the unit sphere
	local H = {}
	for i = 1, 5 do
		local x, y, z = math.floor(rnd() * 100), math.floor(rnd() * 100), math.floor(rnd() * 100)
		H[i] = { x, y, z, 3 * x - 2 * y + z + 7 }
	end
	check(geometry.orient(H[1], H[2], H[3], H[4], H[5]) == 0, "4-D: five points on a hyperplane")
	local E = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 }, { -1, 0, 0, 0 } }
	check(geometry.insphere(E[1], E[2], E[3], E[4], E[5], { 0.5, 0.5, 0.5, 0.5 }) == 0, "4-D: cospherical")
	check(geometry.insphere(E[1], E[2], E[3], E[4], E[5], { 0.5, 0.5, 0.5, 0.5 - 2 ^ -40 }) == 1, "4-D: just inside")
	check(geometry.orient(vector.create(0, 0, 0), vector.create(1, 0, 0), vector.create(0, 1, 0), vector.create(0, 0, 1)) == 1, "vectors")
	check(not pcall(function()
		geometry.insphere({ 0, 0 }, { 1, 1 }, { 2, 2 }, { 0, 1 })
	end), "insphere with collinear points")
end)

-- Checks a triangulation by brute force: every simplex positively oriented and no point
-- strictly inside its circumsphere (unless hull), the simplices' volumes adding up to the
-- hull's (the hull facets with an inside point make cones that add up the same), every
-- point inside or on every hull facet, and every point used or listed as a duplicate.
local function verify(points: { { number } }, T: any, delaunay: boolean): (boolean, string)
	local d = #points[1]
	local P = geometry._predicates(d)
	local X: { number } = {}
	for i, p in points do
		for a = 1, d do
			X[(i - 1) * d + a] = p[a]
		end
	end
	local function volume(ids: { number }): number
		local M = {}
		for i = 1, d do
			for a = 1, d do
				M[(i - 1) * d + a] = X[(ids[i + 1] - 1) * d + a] - X[(ids[1] - 1) * d + a]
			end
		end
		local v = det(M, d)
		for k = 2, d do
			v /= k
		end
		return v
	end
	local total = 0
	for k, s in T.simplices do
		if P.orient(X, s) ~= 1 then
			return false, "simplex " .. k .. " is not positively oriented"
		end
		total += volume(s)
		if delaunay then
			for i = 1, #points do
				if P.lifted(X, s, i) == P.inside then
					return false, string.format("point %d inside simplex %s's sphere", i, tostring(k))
				end
			end
		end
	end
	-- the hull: a center inside, every point on the inner side of every facet
	local center = table.create(d, 0)
	for _, v in T.simplices[1] do
		for a = 1, d do
			center[a] += X[(v - 1) * d + a] / (d + 1)
		end
	end
	local Xc = table.clone(X)
	local c = #points + 1
	for a = 1, d do
		Xc[(c - 1) * d + a] = center[a]
	end
	local cones = 0
	for k, f in T.hull do
		local ids = table.clone(f)
		table.insert(ids, 0)
		for i = 1, #points do
			ids[d + 1] = i
			if P.orient(Xc, ids) < 0 then
				return false, string.format("point %d outside hull facet %s", i, tostring(k))
			end
		end
		ids[d + 1] = c
		if P.orient(Xc, ids) ~= 1 then
			return false, "hull facet " .. k .. " faces inward"
		end
		local M = {}
		for i = 1, d do
			for a = 1, d do
				M[(i - 1) * d + a] = Xc[(f[i] - 1) * d + a] - center[a]
			end
		end
		local v = math.abs(det(M, d))
		for q = 2, d do
			v /= q
		end
		cones += v
	end
	if math.abs(total - cones) > 1e-9 * math.max(1, cones) then
		return false, string.format("volumes %.12g and %.12g", total, cones)
	end
	local covered = 0
	for _ in T.duplicates do
		covered += 1
	end
	if delaunay and covered + #T.vertices ~= #points then
		return false, string.format("%d vertices and %d duplicates for %d points", #T.vertices, covered, #points)
	end
	return true, ""
end

test("geometry: Delaunay triangulations and hulls", function()
	local function random_points(n: number, d: number): { { number } }
		local P = {}
		for i = 1, n do
			local p = {}
			for a = 1, d do
				p[a] = rnd()
			end
			P[i] = p
		end
		return P
	end
	local cases: { { any } } = {}
	table.insert(cases, { "2-D random", random_points(300, 2) })
	-- a grid: every square's four corners on a circle, and every row collinear
	local grid: { { number } } = {}
	for i = 0, 11 do
		for j = 0, 9 do
			table.insert(grid, { i, j })
		end
	end
	table.insert(cases, { "2-D grid", grid })
	-- the same grid at spacings 1/16 (exact by the integer shortcut) and 0.1 (by expansions)
	for _, h in { 0.0625, 0.1 } do
		local scaled: { { number } } = {}
		for i, p in grid do
			scaled[i] = { 1000 + p[1] * h, -50 + p[2] * h }
		end
		table.insert(cases, { "2-D grid spaced " .. h, scaled })
	end
	-- points on a circle (all cocircular), with the center
	local circle: { { number } } = { { 0, 0 } }
	for k = 0, 15 do
		local t = 2 * math.pi * k / 16
		table.insert(circle, { 5 * math.cos(t), 5 * math.sin(t) })
	end
	table.insert(circle, { 3, 4 })
	table.insert(circle, { -3, 4 })
	table.insert(circle, { 4, -3 })
	table.insert(cases, { "2-D cocircular", circle })
	-- collinear points but one, and duplicates
	local line: { { number } } = {}
	for i = 0, 20 do
		table.insert(line, { i, 2 * i })
	end
	table.insert(line, { 3, 0 })
	table.insert(line, { 5, 10 })
	table.insert(line, { 3, 0 })
	table.insert(cases, { "2-D collinear and duplicates", line })
	table.insert(cases, { "3-D random", random_points(200, 3) })
	local cube = {}
	for i = 0, 4 do
		for j = 0, 4 do
			for k = 0, 3 do
				table.insert(cube, { i, j, k })
			end
		end
	end
	table.insert(cases, { "3-D lattice", cube })
	table.insert(cases, { "4-D random", random_points(80, 4) })
	local tess = {}
	for i = 0, 2 do
		for j = 0, 2 do
			for k = 0, 2 do
				for l = 0, 1 do
					table.insert(tess, { i, j, k, l })
				end
			end
		end
	end
	table.insert(cases, { "4-D lattice", tess })
	for _, case in cases do
		local name, pts = case[1], case[2]
		local T = geometry.delaunay(pts)
		local ok, why = verify(pts, T, true)
		check(ok, string.format("%s: Delaunay (%d simplices) %s", name, #T.simplices, why))
		local H = geometry.hull(pts)
		local okh, whyh = verify(pts, H, false)
		check(okh, string.format("%s: hull (%d facets) %s", name, #H.hull, whyh))
	end
	-- 2-D counts: 2 n - h - 2 triangles; duplicates found
	local pts = random_points(500, 2)
	local T: any = geometry.delaunay(pts)
	check(#T.simplices == 2 * 500 - #T.hull - 2, "2-D: 2n - h - 2 triangles")
	local L = geometry.delaunay(line)
	check(L.duplicates[24] == 22 and (L.duplicates[23] == 6 or L.duplicates[6] == 23) and #L.vertices == 22, "duplicates")
	-- the 2-D hull (monotone chain) has the general hull's corners
	for _, set in { pts, grid, circle, line } do
		local fast = geometry.hull(set)
		local general = geometry._triangulate(set, { hull = true }, "hull")
		local a: { string }, b: { string } = {}, {}
		for _, v in fast.vertices do
			table.insert(a, set[v][1] .. "," .. set[v][2])
		end
		for _, v in general.vertices do
			table.insert(b, set[v][1] .. "," .. set[v][2])
		end
		table.sort(a)
		table.sort(b)
		local same = #a == #b
		for k = 1, #a do
			same = same and a[k] == b[k]
		end
		check(same, string.format("2-D hull: %d corners, as the general hull's %d", #a, #b))
	end
	-- the hull of a square's points: its 4 corners
	local H = geometry.hull(grid)
	table.sort(H.vertices)
	check(#H.vertices == 4 and H.vertices[1] == 1 and H.vertices[4] == 120, "a grid's hull is its corners")
	-- vectors are 3-D; points in a plane are refused
	check(#geometry.delaunay({ vector.create(0, 0, 0), vector.create(1, 0, 0), vector.create(0, 1, 0), vector.create(0, 0, 1), vector.create(1, 1, 1) }).simplices >= 2, "vectors")
	check(not pcall(function()
		geometry.delaunay({ { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } })
	end), "coplanar points in 3-D")
end)

-- Twice the signed area of a polygon (shoelace).
local function area2(poly: { { number } }): number
	local s = 0
	for i = 1, #poly do
		local a, b = poly[i], poly[i % #poly + 1]
		s += a[1] * b[2] - a[2] * b[1]
	end
	return s
end

-- Shapes for the polygon tests: convex and not, with collinear points, holes, a spiral and
-- a comb.
local function shapes(): { { any } }
	local out: { { any } } = {}
	table.insert(out, { "L-shape", { { 0, 0 }, { 2, 0 }, { 2, 1 }, { 1, 1 }, { 1, 2 }, { 0, 2 } }, {} })
	local star: { { number } } = {}
	for k = 0, 9 do
		local r = if k % 2 == 0 then 5 else 2
		local t = math.pi * k / 5
		table.insert(star, { r * math.cos(t), r * math.sin(t) })
	end
	table.insert(out, { "star", star, {} })
	-- a square with collinear points along its sides, clockwise, with two holes
	local sq: { { number } } = {}
	for k = 0, 3 do
		table.insert(sq, { 0, 4 - k })
	end
	for k = 0, 3 do
		table.insert(sq, { k, 0 })
	end
	for k = 0, 3 do
		table.insert(sq, { 4, k })
	end
	for k = 0, 3 do
		table.insert(sq, { 4 - k, 4 })
	end
	table.insert(out, { "square with holes", sq, { { { 0.5, 0.5 }, { 1.5, 0.5 }, { 1.5, 1.5 }, { 0.5, 1.5 } }, { { 2.5, 2.5 }, { 3.5, 2.6 }, { 3, 3.5 } } } })
	local comb: { { number } } = { { 0, 0 }, { 10, 0 } }
	for k = 4, 0, -1 do
		table.insert(comb, { 2 * k + 2, 5 })
		table.insert(comb, { 2 * k + 1.5, 5 })
		table.insert(comb, { 2 * k + 1.5, 1 })
		table.insert(comb, { 2 * k + 0.5, 1 })
		table.insert(comb, { 2 * k + 0.5, 5 })
	end
	table.remove(comb)
	table.insert(comb, { 0, 5 })
	table.insert(out, { "comb", comb, {} })
	local spiral: { { number } } = {}
	local inner: { { number } } = {}
	for k = 0, 40 do
		local t = 0.3 * k
		table.insert(spiral, { (1 + 0.3 * t) * math.cos(t), (1 + 0.3 * t) * math.sin(t) })
		table.insert(inner, 1, { (0.8 + 0.3 * t) * math.cos(t), (0.8 + 0.3 * t) * math.sin(t) })
	end
	for _, p in inner do
		table.insert(spiral, p)
	end
	table.insert(out, { "spiral", spiral, {} })
	return out
end

test("geometry: polygons", function()
	for _, shape in shapes() do
		local name, poly, holes = shape[1], shape[2], shape[3]
		local points, triangles = geometry.triangulate(poly, { holes = holes })
		local count = #poly
		local want = math.abs(area2(poly))
		for _, h in holes do
			count += #h
			want -= math.abs(area2(h))
		end
		-- n + 2 (holes) - 2 triangles, positively oriented, covering the area, each inside
		local got = 0
		local ok = #triangles == count + 2 * #holes - 2
		for _, tri in triangles do
			local a, b, c = points[tri[1]], points[tri[2]], points[tri[3]]
			local t2 = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
			ok = ok and t2 > 0 and geometry.orient(a, b, c) == 1
			got += t2
			local centroid = { (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3 }
			ok = ok and geometry.inside(centroid, poly) == 1
			for _, h in holes do
				ok = ok and geometry.inside(centroid, h) == -1
			end
		end
		-- every polygon edge is a triangle edge
		local edges: { [string]: boolean } = {}
		for _, tri in triangles do
			for k = 1, 3 do
				local u, w = tri[k], tri[k % 3 + 1]
				edges[math.min(u, w) .. "," .. math.max(u, w)] = true
			end
		end
		local first = 0
		for _, loop in { poly, table.unpack(holes) } do
			for k = 1, #loop do
				local u, w = first + k, first + k % #loop + 1
				ok = ok and edges[math.min(u, w) .. "," .. math.max(u, w)] == true
			end
			first += #loop
		end
		check(ok and math.abs(got - want) < 1e-9 * want, string.format("%s: %d triangles, area %.6f of %.6f", name, #triangles, got / 2, want / 2))
		-- quality meshes: the angle and size asked for, the area kept, all inside
		for _, req: { number } in { { 20 }, { 30, 0.25 } } :: { { number } } do
			local size: number? = req[2]
			local mp, mt, info = geometry.mesh(poly, { holes = holes, angle = req[1], size = size })
			local area, longest, good = 0, 0, true
			for _, tri in mt do
				local a, b, c = mp[tri[1]], mp[tri[2]], mp[tri[3]]
				local t2 = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
				good = good and t2 > 0
				area += t2
				for k = 1, 3 do
					local p, q = mp[tri[k]], mp[tri[k % 3 + 1]]
					longest = math.max(longest, math.sqrt((p[1] - q[1]) ^ 2 + (p[2] - q[2]) ^ 2))
				end
				local centroid = { (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3 }
				good = good and geometry.inside(centroid, poly) == 1
			end
			-- the star's and spiral's input angles are sharp: those corners keep them
			local sharp = name == "star" or name == "spiral" or name == "comb"
			local want_angle: number = req[1]
			local angle_ok = info.min_angle >= want_angle - 1e-9 or sharp
			check(good and angle_ok and info.warning == nil and math.abs(area - want) < 1e-6 * want and (size == nil or longest <= (size :: number) + 1e-12),
				string.format("%s: mesh at %d degrees%s: %d points, smallest angle %.1f, longest edge %.3f", name, want_angle, if size then ", size " .. size else "", info.points, info.min_angle, longest))
		end
	end
	-- a thin wedge (an input angle of 5 degrees): refinement still ends
	local wedge = { { 0, 0 }, { 10, 0 }, { 10 * math.cos(math.rad(5)), 10 * math.sin(math.rad(5)) } }
	local _, wt, winfo = geometry.mesh(wedge, { angle = 25 })
	check(winfo.warning == nil and #wt > 0, string.format("a 5-degree wedge: %d triangles, smallest angle %.1f", #wt, winfo.min_angle))
	-- point in polygon: inside, outside, on edges and vertices, exactly
	local L = { { 0, 0 }, { 2, 0 }, { 2, 1 }, { 1, 1 }, { 1, 2 }, { 0, 2 } }
	check(geometry.inside({ 0.5, 0.5 }, L) == 1 and geometry.inside({ 1.5, 1.5 }, L) == -1 and geometry.inside({ 1, 1.5 }, L) == 0
		and geometry.inside({ 2, 1 }, L) == 0 and geometry.inside({ 0, 1 }, L) == 0 and geometry.inside({ 1.5, 0 }, L) == 0 and geometry.inside({ 3, 0 }, L) == -1, "point in polygon")
	-- errors: crossing edges, repeated points
	check(not pcall(function()
		geometry.triangulate({ { 0, 0 }, { 2, 2 }, { 2, 0 }, { 0, 2 } })
	end), "a self-crossing polygon is refused")
	check(not pcall(function()
		geometry.triangulate({ { 0, 0 }, { 1, 0 }, { 1, 1 }, { 1, 0 } })
	end), "a repeated point is refused")
end)

test("geometry: locate, graphs and closest points", function()
	-- locate: barycentric weights that rebuild the point, in 2-D and 3-D; nil outside
	for _, d in { 2, 3 } do
		local pts: { { number } } = {}
		for i = 1, 300 do
			local p = {}
			for a = 1, d do
				p[a] = rnd()
			end
			pts[i] = p
		end
		local T: any = geometry.delaunay(pts)
		local good = true
		for _ = 1, 50 do
			local q = {}
			for a = 1, d do
				q[a] = 0.25 + 0.5 * rnd()
			end
			local k, w = T:locate(q)
			good = good and k ~= nil
			if k and w then
				local sum = 0
				local back = table.create(d, 0)
				for i, v in T.simplices[k] do
					good = good and w[i] >= -1e-12
					sum += w[i]
					for a = 1, d do
						back[a] += w[i] * pts[v][a]
					end
				end
				for a = 1, d do
					good = good and math.abs(back[a] - q[a]) < 1e-12
				end
				good = good and math.abs(sum - 1) < 1e-12
			end
		end
		local far = table.create(d, 5)
		check(good and T:locate(far) == nil, string.format("%d-D: locate", d))
		-- the graph: 3n - 3 - h edges in 2-D
		local G = T:graph()
		if d == 2 then
			check(buffer.len(G.ind) // 4 == 2 * (3 * 300 - 3 - #T.hull), "2-D: the Delaunay graph has 3n - 3 - h edges")
		end
	end
	-- closest points against brute-force sampling, as vectors (3-D) and f64 arrays (5-D)
	local function brute_segment(p: { number }, a: { number }, b: { number }): number
		local best = math.huge
		for k = 0, 2000 do
			local t = k / 2000
			local s2 = 0
			for i = 1, #p do
				s2 += (a[i] + t * (b[i] - a[i]) - p[i]) ^ 2
			end
			best = math.min(best, s2)
		end
		return math.sqrt(best)
	end
	local function brute_triangle(p: { number }, a: { number }, b: { number }, c: { number }): number
		local best = math.huge
		local m = 200
		for i = 0, m do
			for j = 0, m - i do
				local u, v = i / m, j / m
				local s2 = 0
				for k = 1, #p do
					s2 += (a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]) - p[k]) ^ 2
				end
				best = math.min(best, s2)
			end
		end
		return math.sqrt(best)
	end
	local f64 = lib.f64
	local good = true
	for trial = 1, 20 do
		local d = if trial % 2 == 0 then 3 else 5
		local function point(): { number }
			local p = {}
			for a = 1, d do
				p[a] = rnd() * 4 - 2
			end
			return p
		end
		local p, a, b, c, e = point(), point(), point(), point(), point()
		local function as(x: { number }): any
			return if d == 3 then vector.create(x[1], x[2], x[3]) else f64.from(x)
		end
		local function dist(x: any, y: { number }): number
			local s2 = 0
			for k = 1, d do
				s2 += ((if d == 3 then (x :: any)[({ "x", "y", "z" })[k]] else f64.get(x, k)) - y[k]) ^ 2
			end
			return math.sqrt(s2)
		end
		local tol = if d == 3 then 1e-5 else 1e-9 -- vectors are single precision
		local q, t = geometry.closest_on_segment(as(p), as(a), as(b))
		good = good and t >= 0 and t <= 1 and dist(q, p) <= brute_segment(p, a, b) + tol + 1e-6
		local q2, u, v, w = geometry.closest_on_triangle(as(p), as(a), as(b), as(c))
		good = good and math.abs(u + v + w - 1) < 1e-12 and math.min(u, v, w) >= -1e-12 and dist(q2, p) <= brute_triangle(p, a, b, c) + tol + 1e-6
		-- segments: no pair of sampled points closer
		local c1, c2, s1, t1 = geometry.closest_between_segments(as(a), as(b), as(c), as(e))
		local got = 0
		for k = 1, d do
			local x1 = if d == 3 then (c1 :: any)[({ "x", "y", "z" })[k]] else f64.get(c1, k)
			local x2 = if d == 3 then (c2 :: any)[({ "x", "y", "z" })[k]] else f64.get(c2, k)
			got += (x1 - x2) ^ 2
		end
		got = math.sqrt(got)
		local best = math.huge
		for i = 0, 100 do
			for j = 0, 100 do
				local s2 = 0
				for k = 1, d do
					s2 += (a[k] + i / 100 * (b[k] - a[k]) - c[k] - j / 100 * (e[k] - c[k])) ^ 2
				end
				best = math.min(best, s2)
			end
		end
		good = good and s1 >= 0 and s1 <= 1 and t1 >= 0 and t1 <= 1 and got <= math.sqrt(best) + tol + 1e-9
	end
	check(good, "closest points on segments and triangles, and between segments (3-D vectors, 5-D arrays)")
end)

test("pde: meshes of polygons", function()
	-- Poisson's equation on an L-shape with the harmonic solution sin(x) e^y: second order
	local L = { { 0, 0 }, { 2, 0 }, { 2, 1 }, { 1, 1 }, { 1, 2 }, { 0, 2 } }
	local exact = function(x: number, y: number): number
		return math.sin(x) * math.exp(y)
	end
	local errs = {}
	for i, size in { 0.2, 0.1 } do
		local points, triangles, info = lib.pde.mesh(L, { size = size, angle = 25 })
		local D = lib.pde.fem(points, triangles)
		local u = lib.pde.poisson(D, 0, exact)
		local e = 0
		for p = 1, D.n do
			e = math.max(e, math.abs(u[p] - exact(D.x[p], D.y[p])))
		end
		errs[i] = e
		check(info.min_angle >= 25 and math.abs(D.volume - 3) < 1e-12, string.format("size %.1f: %d points, smallest angle %.1f", size, info.points, info.min_angle))
	end
	check(errs[1] / errs[2] > 2.5, string.format("second order on a polygon mesh: %.2e then %.2e", errs[1], errs[2]))
end)

return nil
