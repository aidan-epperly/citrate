--!strict

-- The elementwise kernels (unrolled by the generator) at every length up to 9, so every
-- remainder of the unrolling is covered, against plain loops over get/set; and the fused
-- conjugate gradient update against its definition.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

local function values(M: any, complex: boolean, n: number, seed: number): buffer
	local re, im = table.create(n, 0), table.create(n, 0)
	for i = 1, n do
		re[i] = math.sin(seed + 1.7 * i)
		im[i] = math.cos(seed - 0.9 * i)
	end
	return if complex then M.from(re, im) else M.from(re)
end

-- Entry i as (re, im).
local function at(M: any, x: buffer, i: number): (number, number)
	local re, im = M.get(x, i)
	if type(re) == "vector" then
		return (re :: any).x, (re :: any).y
	end
	return re, im or 0
end

local cases = {
	{ name = "f32", M = lib.f32, complex = false, tol = 1e-6 },
	{ name = "f64", M = lib.f64, complex = false, tol = 1e-15 },
	{ name = "c32", M = lib.c32, complex = true, tol = 1e-6 },
	{ name = "c64", M = lib.c64, complex = true, tol = 1e-15 },
}

for _, case in cases do
	local M, complex, tol = case.M, case.complex, case.tol
	test(`{case.name} unrolled elementwise kernels`, function()
		for n = 0, 9 do
			local x, y = values(M, complex, n, 1), values(M, complex, n, 2)
			local function each(name: string, got: buffer, want: (number) -> (number, number))
				local ok = M.length(got) == n
				for i = 1, n do
					local gr, gi = at(M, got, i)
					local wr, wi = want(i)
					ok = ok and math.abs(gr - wr) <= tol and math.abs(gi - wi) <= tol
				end
				check(ok, `{name}, length {n}`)
			end
			each("add", M.add(nil, x, y), function(i)
				local a, b = at(M, x, i)
				local c, d = at(M, y, i)
				return a + c, b + d
			end)
			each("sub", M.sub(nil, x, y), function(i)
				local a, b = at(M, x, i)
				local c, d = at(M, y, i)
				return a - c, b - d
			end)
			each("scale", M.scale(nil, 0.5, x), function(i)
				local a, b = at(M, x, i)
				return 0.5 * a, 0.5 * b
			end)
			each("axpy", M.axpy(nil, 0.5, x, y), function(i)
				local a, b = at(M, x, i)
				local c, d = at(M, y, i)
				return 0.5 * a + c, 0.5 * b + d
			end)
			each("mul", M.mul(nil, x, y), function(i)
				local a, b = at(M, x, i)
				local c, d = at(M, y, i)
				return a * c - b * d, a * d + b * c
			end)
			each("fill", if M == lib.c32 then M.fill(M.create(n), vector.create(2, 3, 0)) elseif complex then M.fill(M.create(n), 2, 3) else M.fill(M.create(n), 2), function()
				return 2, if complex then 3 else 0
			end)
			-- reductions: the real inner product, the sum and the norm
			local d, s, q = 0, 0, 0
			for i = 1, n do
				local a, b = at(M, x, i)
				local c, e = at(M, y, i)
				d += a * c + b * e
				s += a
				q += a * a + b * b
			end
			check(math.abs(M.space.dot(x, y) - d) <= tol * 10, `dot, length {n}`)
			check(math.abs(M.norm(x) - math.sqrt(q)) <= tol * 10, `norm, length {n}`)
			local sr = M.sum(x)
			if type(sr) == "vector" then
				sr = (sr :: any).x
			end
			check(math.abs((sr :: number) - s) <= tol * 10, `sum, length {n}`)

			-- cg_update: x + a p, r - a q, and |r|^2
			local xx, r = M.copy(nil, x), M.copy(nil, y)
			local p, qv = values(M, complex, n, 3), values(M, complex, n, 4)
			local rr = M.space.cg_update(xx, r, p, qv, 0.25)
			local wantx = M.axpy(nil, 0.25, p, x)
			local wantr = M.axpy(nil, -0.25, qv, y)
			check(M.norm(M.sub(nil, xx, wantx)) <= tol * 10 and M.norm(M.sub(nil, r, wantr)) <= tol * 10, `cg_update vectors, length {n}`)
			check(math.abs(rr - M.space.dot(r, r)) <= tol * 10 * math.max(1, rr), `cg_update norm, length {n}`)
		end
	end)
end

return nil
