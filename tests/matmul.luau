--!strict

-- Matrix products against a plain reference: sizes that exercise every part of the
-- register-blocked kernels (8 by 16 and 8 by 8 blocks, the 4 by 4 blocks and the single
-- entries left over), every op(A) and op(B), and beta.

local h = require("./harness")
local lib = require("../src")

local test, check = h.test, h.check

-- Entry (i, j) of op(X) as (re, im), for any module.
local function entry(M: any, X: any, op: string, i: number, j: number): (number, number)
	local r, c = if op == "n" then i else j, if op == "n" then j else i
	local re, im = M.matget(X, r, c)
	if type(re) == "vector" then
		re, im = (re :: any).x, (re :: any).y
	end
	im = im or 0
	if op == "c" then
		im = -im
	end
	return re, im
end

local function fill(M: any, rows: number, cols: number, seed: number): any
	local X = M.matrix(rows, cols)
	local complex = M == lib.c64 or M == lib.c32
	for j = 1, cols do
		for i = 1, rows do
			local re = math.sin(seed + 1.3 * i + 0.7 * j)
			local im = math.cos(seed + 0.4 * i - 1.1 * j)
			if M == lib.c32 then
				M.matset(X, i, j, vector.create(re, im, 0))
			elseif complex then
				M.matset(X, i, j, re, im)
			else
				M.matset(X, i, j, re)
			end
		end
	end
	return X
end

test("matrix products: register-blocked kernels against a reference", function()
	local sizes = { { 1, 1, 1 }, { 3, 5, 2 }, { 8, 9, 16 }, { 17, 6, 33 }, { 20, 13, 19 }, { 33, 40, 35 } }
	for _, M in { lib.f64, lib.f32, lib.c64, lib.c32 } :: { any } do
		local complex = M == lib.c64 or M == lib.c32
		local tol = if M == lib.f32 or M == lib.c32 then 1e-5 else 1e-12
		local ops = if complex then { "n", "t", "c" } else { "n", "t" }
		for _, size in sizes do
			local m, p, n = size[1], size[2], size[3]
			-- matmul
			local A, B = fill(M, m, p, 1), fill(M, p, n, 2)
			local C = M.matmul(nil, A, B)
			local worst = 0
			for i = 1, m do
				for j = 1, n do
					local sr, si = 0, 0
					for k = 1, p do
						local ar, ai = entry(M, A, "n", i, k)
						local br, bi = entry(M, B, "n", k, j)
						sr += ar * br - ai * bi
						si += ar * bi + ai * br
					end
					local cr, ci = entry(M, C, "n", i, j)
					worst = math.max(worst, math.abs(cr - sr) + math.abs(ci - si))
				end
			end
			check(worst <= tol * p, `matmul {m}x{p}x{n}: error {worst}`)
			-- gemm with every op and beta = 0 or 0.5
			for _, opa in ops do
				for _, opb in ops do
					local Ag = if opa == "n" then fill(M, m, p, 3) else fill(M, p, m, 3)
					local Bg = if opb == "n" then fill(M, p, n, 4) else fill(M, n, p, 4)
					for _, beta in { 0, 0.5 } do
						local C0 = fill(M, m, n, 5)
						local Cg = M.gemm(if beta ~= 0 then M.matcopy(nil, C0) else nil, 2, Ag, Bg, beta, opa, opb)
						local err = 0
						for i = 1, m do
							for j = 1, n do
								local sr, si = 0, 0
								for k = 1, p do
									local ar, ai = entry(M, Ag, opa, i, k)
									local br, bi = entry(M, Bg, opb, k, j)
									sr += ar * br - ai * bi
									si += ar * bi + ai * br
								end
								local c0r, c0i = entry(M, C0, "n", i, j)
								local er, ei = 2 * sr + beta * c0r, 2 * si + beta * c0i
								local cr, ci = entry(M, Cg, "n", i, j)
								err = math.max(err, math.abs(cr - er) + math.abs(ci - ei))
							end
						end
						check(err <= 2 * tol * p + tol, `gemm {opa}{opb} beta {beta} {m}x{p}x{n}: error {err}`)
					end
				end
			end
		end
	end
end)

return nil
