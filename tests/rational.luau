--!strict

-- AAA rational approximation: smooth functions to machine precision with few support
-- points, poles and residues of meromorphic functions, the slow but steady convergence for
-- |x|, complex data on a circle, and the removal of Froissart doublets.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check
local c64, rational = lib.c64, lib.rational

local function interval(a: number, b: number, n: number): buffer
	local re = table.create(n, 0)
	for i = 1, n do
		re[i] = a + (b - a) * (i - 1) / (n - 1)
	end
	return c64.from(re, table.create(n, 0))
end

local function values(Z: buffer, f: (number, number) -> (number, number)): buffer
	local n = c64.length(Z)
	local re, im = table.create(n, 0), table.create(n, 0)
	for i = 1, n do
		local x, y = c64.get(Z, i)
		re[i], im[i] = f(x, y)
	end
	return c64.from(re, im)
end

test("aaa: exp and tan on intervals", function()
	local Z = interval(-1, 1, 1000)
	local R = rational.aaa(Z, values(Z, function(x: number): (number, number)
		return math.exp(x), 0
	end))
	-- as in Nakatsukasa, Sete and Trefethen: type (6, 6) for exp on [-1, 1]
	check(c64.length(R.support) <= 8 and R.error < 1e-14, `exp: {c64.length(R.support)} support points, error {R.error}`)
	local r = R.eval(0.123)
	check(math.abs(r - math.exp(0.123)) < 1e-14, "exp between the samples")
	local Z2 = interval(-1.5, 1.5, 1000)
	local T = rational.aaa(Z2, values(Z2, function(x: number): (number, number)
		return math.tan(x), 0
	end))
	local best = math.huge
	for k = 1, c64.length(T.poles) do
		local pr, pi = c64.get(T.poles, k)
		if math.abs(pr - math.pi / 2) + math.abs(pi) < best then
			best = math.abs(pr - math.pi / 2) + math.abs(pi)
			-- the residue of tan at pi/2 is -1
			local rr, ri = c64.get(T.residues, k)
			check(math.abs(rr + 1) < 1e-10 and math.abs(ri) < 1e-10, "residue of tan at pi/2")
		end
	end
	check(best < 1e-12, `pole of tan at pi/2: off by {best}`)
	check(#R.errors == c64.length(R.support), "one error per step")
end)

test("aaa: |x|, complex data, Froissart doublets", function()
	local Z = interval(-1, 1, 1000)
	local A = rational.aaa(Z, values(Z, function(x: number): (number, number)
		return math.abs(x), 0
	end))
	-- measured: 47 support points to 4e-14 with 2000 samples
	check(A.error < 1e-12, `|x|: error {A.error} with {c64.length(A.support)} support points`)
	-- 1 / (z - (0.5 + 0.5i)) + z^2 on the unit circle: one pole, residue 1
	local n = 200
	local re, im = table.create(n, 0), table.create(n, 0)
	for i = 1, n do
		re[i], im[i] = math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)
	end
	local Zc = c64.from(re, im)
	local C = rational.aaa(Zc, values(Zc, function(x: number, y: number): (number, number)
		local ar, ai = x - 0.5, y - 0.5
		local d = ar * ar + ai * ai
		return ar / d + x * x - y * y, -ai / d + 2 * x * y
	end))
	-- One pole near the data. The z^2 term is represented by poles far away (where depends
	-- on rounding: effectively infinite on one machine, about 6e7 in Roblox).
	local near: { number } = {}
	for k = 1, c64.length(C.poles) do
		local qr, qi = c64.get(C.poles, k)
		if math.sqrt(qr * qr + qi * qi) < 10 then
			table.insert(near, k)
		else
			check(math.sqrt(qr * qr + qi * qi) > 1e5, "other poles far away")
		end
	end
	check(#near == 1, `one pole near the data ({#near})`)
	local pr, pi = c64.get(C.poles, near[1] or 1)
	local rr, ri = c64.get(C.residues, near[1] or 1)
	check(math.abs(pr - 0.5) + math.abs(pi - 0.5) < 1e-12 and math.abs(rr - 1) + math.abs(ri) < 1e-12, "pole and residue")
	-- the zero nearest the data is a zero
	local zr, zi, best = 0, 0, math.huge
	for k = 1, c64.length(C.zeros) do
		local qr, qi = c64.get(C.zeros, k)
		if qr * qr + qi * qi < best then
			zr, zi, best = qr, qi, qr * qr + qi * qi
		end
	end
	local vr, vi = C.eval(zr, zi)
	check(math.abs(vr) + math.abs(vi) < 1e-10, "a zero is a zero")
	-- Fitting exp past convergence (tol 0) leaves spurious poles with tiny residues on the
	-- interval; cleanup removes them (measured: 18 of 24 poles, back to 7 support points).
	local F = values(Z, function(x: number): (number, number)
		return math.exp(x), 0
	end)
	local raw = rational.aaa(Z, F, { tol = 0, mmax = 25, cleanup = false })
	local clean = rational.aaa(Z, F, { tol = 0, mmax = 25 })
	check(raw.removed == 0 and clean.removed >= 10, `doublets removed: {clean.removed}`)
	check(clean.error < 1e-14 and c64.length(clean.support) < c64.length(raw.support), "clean and accurate")
	for k = 1, c64.length(clean.poles) do
		local qr, qi = c64.get(clean.poles, k)
		check(not (math.abs(qi) < 0.05 and math.abs(qr) < 1.02), "no poles left on the interval")
	end
end)

return nil
