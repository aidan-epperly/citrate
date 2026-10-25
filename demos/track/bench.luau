--!strict
-- The track outside Roblox (luau -O2 [--codegen] demos/track/bench.luau): how evenly each
-- method spaces points by arclength (the speed of a cart moving at a constant rate, relative
-- to that rate), its largest position error, and the time per point, for tables of 101,
-- 1001 and 4701 samples per segment.
local track = require("./track")
local citrate = require("../../src")

local b = track.bezier_points(track.POINTS)
-- the reference: citrate at its default (machine-precision) tolerance
local ref = citrate.curve.bezier_path(b, { closed = true })
local L = citrate.curve.length(ref)

local N = 4000
local function stats(at: (number) -> vector): (number, number, number)
	local lo, hi, perr = math.huge, -math.huge, 0
	local prev = at(0)
	for k = 1, N do
		local s = k / N * L
		local p = at(s)
		perr = math.max(perr, vector.magnitude(p - citrate.curve.at_length(ref, s)))
		local v = vector.magnitude(p - prev) / (L / N)
		lo = math.min(lo, v)
		hi = math.max(hi, v)
		prev = p
	end
	return lo, hi, perr
end
local function per_point(at: (number) -> vector): number
	local best = math.huge
	for _ = 1, 3 do
		local t0 = os.clock()
		for k = 1, 500 do
			at((k * 7919 % 500) / 500 * L)
		end
		best = math.min(best, (os.clock() - t0) / 500)
	end
	return best
end
print(string.format("track length %.4f studs, %d segments", L, (#b - 1) // 3))
print("method                 speed / set speed     max position error   time per point")
local function row(name: string, at: (number) -> vector)
	local lo, hi, perr = stats(at)
	print(string.format("%-22s [%.4f, %.4f]    %.1e studs        %.2f us", name, lo, hi, perr, per_point(at) * 1e6))
end
for _, tol in { 1e-6, 1e-8, 1e-10 } do
	local Ct = citrate.curve.bezier_path(b, { closed = true })
	local Lt = citrate.curve.arclength(Ct, { tol = tol })
	row(`citrate, tol {tol}`, function(s)
		return citrate.curve.at_length(Ct, s % Lt)
	end)
end
for _, n in { 101, 1001, 4701 } do
	local T = track.lookup(b, n)
	row(`table, {n} samples`, function(s)
		return track.lookup_at(T, s)
	end)
end
