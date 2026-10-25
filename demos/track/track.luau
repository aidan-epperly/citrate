--!strict
--!native
-- A closed track of cubic Bézier segments and two ways to move along it at a given
-- distance: citrate's curve.at_length, and the lookup table popular Roblox Bézier modules
-- use (bstummer's Bezier, for one: chord lengths at 101 parameters per curve, a linear scan
-- for the entry below the wanted length, and linear interpolation in t). Usable outside
-- Roblox.
local citrate = require("../../src")
local curve = citrate.curve

local track = {}

-- The control points of the loop (studs): hills, a dip and a turn that climbs.
track.POINTS = {
	vector.create(0, 6, -70),
	vector.create(55, 10, -55),
	vector.create(80, 30, -5),
	vector.create(65, 12, 45),
	vector.create(20, 4, 70),
	vector.create(-25, 22, 60),
	vector.create(-55, 40, 25),
	vector.create(-80, 14, -10),
	vector.create(-70, 4, -45),
	vector.create(-35, 18, -75),
}

-- The cubic Bézier control points (3n + 1, the last equal to the first) of the closed
-- uniform Catmull-Rom spline through pts.
function track.bezier_points(pts: { vector }): { vector }
	local n = #pts
	local out = {}
	for i = 1, n do
		local p0, p1 = pts[(i - 2) % n + 1], pts[i]
		local p2, p3 = pts[i % n + 1], pts[(i + 1) % n + 1]
		table.insert(out, p1)
		table.insert(out, p1 + (p2 - p0) / 6)
		table.insert(out, p2 - (p3 - p1) / 6)
	end
	table.insert(out, pts[1])
	return out
end

local function cubic(b: { vector }, k: number, t: number): vector
	local p0, p1, p2, p3 = b[k], b[k + 1], b[k + 2], b[k + 3]
	local u = 1 - t
	return p0 * (u * u * u) + p1 * (3 * u * u * t) + p2 * (3 * u * t * t) + p3 * (t * t * t)
end

-- citrate: the same segments as one composite curve, with arclength tables built to a
-- tolerance of 1e-8 (positions within about 1e-5 studs, near what a Vector3 holds; 1e-6
-- costs as much per point and is ten times less accurate).
export type CitrateTrack = { curve: any, length: number }
function track.citrate(b: { vector }): CitrateTrack
	local C = curve.bezier_path(b, { closed = true })
	return { curve = C, length = curve.arclength(C, { tol = 1e-8 }) }
end
function track.citrate_at(T: CitrateTrack, s: number): vector
	return curve.at_length(T.curve, s % T.length)
end

-- The lookup table: per segment, chord lengths at `samples` evenly spaced parameters.
export type Table = { b: { vector }, segments: number, samples: number, lens: { { number } }, starts: { number }, length: number }
function track.lookup(b: { vector }, samples: number): Table
	local segments = (#b - 1) // 3
	local lens, starts, total = {}, {}, 0
	for seg = 1, segments do
		local k = 3 * seg - 2
		local lut, len, last = table.create(samples, 0), 0, cubic(b, k, 0)
		lut[1] = 0
		for i = 2, samples do
			local p = cubic(b, k, (i - 1) / (samples - 1))
			len += vector.magnitude(p - last)
			lut[i] = len
			last = p
		end
		lens[seg], starts[seg] = lut, total
		total += len
	end
	return { b = b, segments = segments, samples = samples, lens = lens, starts = starts, length = total }
end
function track.lookup_at(T: Table, s: number): vector
	s %= T.length
	-- the segment, by a scan over the segment lengths (as PolyBezier does)
	local seg = 1
	while seg < T.segments and T.starts[seg + 1] <= s do
		seg += 1
	end
	local lut = T.lens[seg]
	local target = s - T.starts[seg]
	-- the largest entry below the target, by a scan over the table (as ConvertT does)
	local i = 1
	for j = 2, T.samples do
		if lut[j] <= target then
			i = j
		end
	end
	local t = (i - 1) / (T.samples - 1)
	if i < T.samples then
		t += (target - lut[i]) / (lut[i + 1] - lut[i]) / (T.samples - 1)
	end
	return cubic(T.b, 3 * seg - 2, t)
end

return track
