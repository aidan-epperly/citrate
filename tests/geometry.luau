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

return nil
