--!strict

-- FFTs (complex modules), real FFTs and DCTs (real modules).

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close

local function random_values(n: number): { number }
	local t = table.create(n, 0)
	for i = 1, n do
		t[i] = 2 * math.random() - 1
	end
	return t
end

-- Direct O(n^2) DFT in double precision, as the reference.
local function naive_dft(re: { number }, im: { number }, sign: number): ({ number }, { number })
	local n = #re
	local outr, outi = table.create(n, 0), table.create(n, 0)
	for k = 0, n - 1 do
		local sr, si = 0, 0
		for j = 0, n - 1 do
			local a = sign * 2 * math.pi * ((j * k) % n) / n
			local c, s = math.cos(a), math.sin(a)
			sr += re[j + 1] * c - im[j + 1] * s
			si += re[j + 1] * s + im[j + 1] * c
		end
		outr[k + 1], outi[k + 1] = sr, si
	end
	return outr, outi
end

local function max_diff(ar: { number }, ai: { number }, br: { number }, bi: { number }): number
	local d = 0
	for k = 1, #ar do
		d = math.max(d, math.abs(ar[k] - br[k]), math.abs(ai[k] - bi[k]))
	end
	return d
end

-- Lengths covering every pass: radix 4, 2, 3, 5, the generic odd primes 7 to 13, mixed
-- products, and Bluestein (primes above 13 and lengths with such a factor).
local SIZES = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 30, 34, 49, 60, 64, 97, 128, 143, 210 }

for _, case in { { name = "c32", M = lib.c32 :: any, eps = h.eps32 }, { name = "c64", M = lib.c64 :: any, eps = h.eps64 } } do
	local M, name, eps = case.M, case.name, case.eps

	test(`{name} fft matches the direct DFT`, function()
		math.randomseed(81)
		for _, n in SIZES do
			local re, im = random_values(n), random_values(n)
			local x = M.from(re, im)
			check(M.fft(x) == x, `n={n}: in place, returns x`)
			local gr, gi = M.totable(x)
			local rr, ri = naive_dft(re, im, -1)
			local tol = 20 * eps * math.max(1, math.log(math.max(n, 2), 2)) * math.sqrt(math.max(n, 1))
			check(max_diff(gr, gi, rr, ri) <= tol, `n={n}: fft error {max_diff(gr, gi, rr, ri)}`)
			M.ifft(x)
			local br, bi = M.totable(x)
			check(max_diff(br, bi, re, im) <= tol, `n={n}: ifft(fft(x)) = x`)
		end
	end)

	test(`{name} fft properties and plans`, function()
		math.randomseed(82)
		local n = 60
		local x = M.from(random_values(n), random_values(n))
		local x0 = M.copy(nil, x)
		local X = M.fft_copy(x)
		check(M.norm(M.sub(nil, x, x0)) == 0, "fft_copy leaves x alone")
		local X0 = M.copy(nil, X)
		local back = M.ifft_copy(X)
		check(M.norm(M.sub(nil, X, X0)) == 0, "ifft_copy leaves X alone")
		check(M.norm(M.sub(nil, back, x)) <= 100 * eps * M.norm(x), "ifft_copy inverts fft_copy")
		-- Parseval: |X|^2 = n |x|^2
		close(M.norm(X) ^ 2, n * M.norm(x) ^ 2, 100 * eps * n, "Parseval")
		-- Linearity
		local y = M.from(random_values(n), random_values(n))
		local lhs = M.fft(M.axpy(nil, 2, x, y))
		local rhs = M.axpy(nil, 2, M.fft_copy(x), M.fft_copy(y))
		check(M.norm(M.sub(nil, lhs, rhs)) <= 100 * eps * M.norm(lhs), "linearity")
		-- A delta transforms to all ones, a constant to a delta.
		local d = M.create(8)
		M.set(d, 1, if name == "c32" then vector.create(1, 0, 0) else 1, if name == "c32" then nil else 0)
		local dr, di = M.totable(M.fft(d))
		local ones = true
		for k = 1, 8 do
			ones = ones and dr[k] == 1 and di[k] == 0
		end
		check(ones, "delta -> ones")

		local plan = M.fft_plan(n)
		check(plan.n == n and M.fft_plan(n) == plan, "plans are cached")
		local z = M.copy(nil, x)
		M.fft(z, plan)
		check(M.norm(M.sub(nil, z, X)) == 0, "explicit plan gives the same result")
		check(not pcall(function()
			M.fft(M.create(10), plan)
		end), "a plan of the wrong length is an error")
	end)

	test(`{name} fft2`, function()
		math.randomseed(83)
		local m, n = 6, 5
		local A = M.matrix(m, n)
		A.data = M.from(random_values(m * n), random_values(m * n))
		local A0 = M.matcopy(nil, A)
		check(M.fft2(A) == A, "in place")
		-- Reference: the 2D DFT is the 1D DFT along columns, then along rows.
		local re, im = M.mattotable(A0)
		local Rr, Ri = {}, {}
		for i = 1, m do
			Rr[i], Ri[i] = table.create(n, 0), table.create(n, 0)
		end
		for j = 1, n do
			local cr, ci = {}, {}
			for i = 1, m do
				cr[i], ci[i] = re[i][j], im[i][j]
			end
			local tr, ti = naive_dft(cr, ci, -1)
			for i = 1, m do
				Rr[i][j], Ri[i][j] = tr[i], ti[i]
			end
		end
		for i = 1, m do
			Rr[i], Ri[i] = naive_dft(Rr[i], Ri[i], -1)
		end
		local gr, gi = M.mattotable(A)
		local worst = 0
		for i = 1, m do
			worst = math.max(worst, max_diff(gr[i], gi[i], Rr[i], Ri[i]))
		end
		check(worst <= 200 * eps, `fft2 error {worst}`)
		M.ifft2(A)
		check(M.norm(M.sub(nil, A.data, A0.data)) <= 200 * eps * M.norm(A0.data), "ifft2(fft2(A)) = A")
	end)
end

for _, case in { { name = "f32", R = lib.f32 :: any, C = lib.c32 :: any, eps = h.eps32 }, { name = "f64", R = lib.f64 :: any, C = lib.c64 :: any, eps = h.eps64 } } do
	local R, C, name, eps = case.R, case.C, case.name, case.eps

	test(`{name} rfft and irfft`, function()
		math.randomseed(84)
		for _, n in { 1, 2, 3, 4, 5, 8, 9, 17, 30, 64, 97 } do
			local x = R.from(random_values(n))
			local x0 = R.copy(nil, x)
			local X = R.rfft(nil, x)
			check(R.norm(R.sub(nil, x, x0)) == 0, `n={n}: rfft leaves x alone`)
			check(C.length(X) == n // 2 + 1, `n={n}: n // 2 + 1 coefficients`)
			local full = C.fft(C.complexify(x))
			local fr, fi = C.totable(full)
			local gr, gi = C.totable(X)
			local d = 0
			for k = 1, n // 2 + 1 do
				d = math.max(d, math.abs(fr[k] - gr[k]), math.abs(fi[k] - gi[k]))
			end
			local tol = 50 * eps * math.sqrt(n) * math.max(1, math.log(n + 1, 2))
			check(d <= tol, `n={n}: rfft agrees with the complex fft ({d})`)
			local y = R.irfft(nil, X, n)
			check(R.norm(R.sub(nil, y, x)) <= tol * R.norm(x), `n={n}: irfft(rfft(x)) = x`)
		end
	end)

	test(`{name} DCT types 1 to 4`, function()
		math.randomseed(85)
		-- DCT-II of a constant c: only y_0 = 2 N c is nonzero.
		local y = R.totable(R.dct(R.from({ 3, 3, 3, 3, 3 })))
		local zeros = true
		for k = 2, 5 do
			zeros = zeros and math.abs(y[k]) <= 100 * eps
		end
		close(y[1], 30, 100 * eps, "DCT-II of a constant")
		check(zeros, "DCT-II of a constant has one nonzero coefficient")

		for _, n in { 2, 3, 4, 5, 8, 13, 16, 31 } do
			local x = R.from(random_values(n))
			for kind = 1, 4 do
				for _, norm in { "none", "ortho" } do
					local nm: lib.DctNorm? = if norm == "ortho" then "ortho" else nil
					local tol = 200 * eps * n
					local back = R.idct(R.dct_copy(x, kind, nm), kind, nm)
					check(R.norm(R.sub(nil, back, x)) <= tol * R.norm(x), `n={n} type {kind} {norm}: idct(dct(x)) = x`)
					if nm == "ortho" then
						close(R.norm(R.dct_copy(x, kind, nm)), R.norm(x), tol, `n={n} type {kind}: orthonormal`)
					end
				end
			end
		end
		-- DCT-III straight from SciPy's definition: y_k = x_0 + 2 sum x_n cos(pi n (2k + 1) / (2N)).
		local xs = { 0.5, -1, 2, 0.25 }
		local got = R.totable(R.dct(R.from(xs), 3))
		for k = 0, 3 do
			local want = xs[1]
			for n = 1, 3 do
				want += 2 * xs[n + 1] * math.cos(math.pi * n * (2 * k + 1) / 8)
			end
			close(got[k + 1], want, 50 * eps, `DCT-III by definition, k={k}`)
		end
		check(not pcall(function()
			R.dct(R.from({ 1 }), 1)
		end), "type 1 needs at least 2 points")
		local x = R.from({ 1, 2, 3 })
		R.dct_copy(x, 4)
		check(R.get(x, 2) == 2, "dct_copy leaves x alone")
		local inv = R.idct_copy(x, 4)
		check(R.get(x, 2) == 2, "idct_copy leaves x alone")
		check(R.norm(R.sub(nil, R.dct(inv, 4), x)) <= 100 * eps * R.norm(x), "idct_copy inverts dct")
	end)
end

return nil
