--!strict
-- Newton's method for systems F(x) = 0, three ways (standalone: luau -O2 [--codegen]
-- bench/newton.luau):
--   inverse   form J^-1 every iteration and step x -= J^-1 F (the textbook formula);
--   solve     solve J p = F by LU every iteration instead (no inverse formed);
--   citrate   optimize.root: LU solves too, into reused storage, and a backtracking line
--             search on |F|^2 / 2 when the full step does not decrease it.
-- Robustness on the Moré-Garbow-Hillstrom systems from their standard starts and from 10
-- and 100 times farther, two systems where full steps fail, then time per solve on
-- Chandrasekhar's H-equation (a dense Jacobian) by size.

local lib = require("../src")
local f64, opt = lib.f64, lib.optimize
local mgh = require("../tests/mgh")

local TOL, MAXITER = 1e-10, 100

local function tab(b: buffer): { number }
	local t = {}
	for k = 1, buffer.len(b) // 8 do
		t[k] = buffer.readf64(b, 8 * (k - 1))
	end
	return t
end

type System = { n: number, F: (buffer, buffer) -> buffer, J: (buffer, any) -> any }

-- an MGH problem as buffers
local function adapt(p: any): System
	return {
		n = p.n,
		F = function(x: buffer, out: buffer): buffer
			local v = p.r(tab(x))
			for i = 1, p.n do
				buffer.writef64(out, 8 * (i - 1), v[i])
			end
			return out
		end,
		J = function(x: buffer, J: any): any
			local v = p.jac(tab(x))
			for i = 1, p.n do
				for j = 1, p.n do
					buffer.writef64(J.data, 8 * ((j - 1) * p.n + i - 1), v[i][j])
				end
			end
			return J
		end,
	}
end

type Result = { converged: boolean, iterations: number, evaluations: number, norm: number }

-- Plain Newton (full steps), the step by the inverse or by an LU solve.
local function plain(S: System, x0: buffer, inverse: boolean): (buffer, Result)
	local n = S.n
	local x, fx = f64.copy(nil, x0), f64.create(n)
	local J = f64.matrix(n, n)
	local res: Result = { converged = false, iterations = 0, evaluations = 1, norm = 0 }
	S.F(x, fx)
	for it = 1, MAXITER do
		res.iterations = it
		if f64.amax(fx) <= TOL then
			res.converged = true
			break
		end
		if not (f64.amax(fx) < math.huge) then
			break -- diverged
		end
		S.J(x, J)
		local p
		if inverse then
			local Ji = f64.inv_copy(J)
			p = f64.matvec(nil, Ji, fx)
		else
			local F = f64.lu_copy(J)
			if F.singular then
				break
			end
			p = f64.lu_solve(nil, F, fx)
		end
		f64.axpy(x, -1, p, x)
		S.F(x, fx)
		res.evaluations += 1
	end
	res.norm = f64.norm(fx)
	res.converged = res.converged or f64.amax(fx) <= TOL
	return x, res
end

local function citrate(S: System, x0: buffer): (buffer, Result)
	local x, info = opt.root(f64.space, S.F, x0, { jac = S.J, tol = TOL, maxiter = MAXITER })
	return x, { converged = info.converged, iterations = info.iterations, evaluations = info.evaluations, norm = info.norm :: number }
end

local function cell(r: Result): string
	if r.converged then
		return string.format("%d its", r.iterations)
	end
	return if r.norm == r.norm and r.norm < math.huge then string.format("fails (|F| %.1e)", r.norm) else "diverges"
end

-- Two systems where full Newton steps fail: a cycle (x^3 - 2x + 2 from 0: Newton jumps
-- 0, 1, 0, 1, ...) and arctangents from 3, where each full step overshoots further.
local extra: { { name: string, S: System, x0: { number } } } = {
	{
		name = "x^3 - 2x + 2 (a cycle)",
		x0 = { 0, 0 },
		S = {
			n = 2,
			F = function(x: buffer, out: buffer): buffer
				local a, b = buffer.readf64(x, 0), buffer.readf64(x, 8)
				buffer.writef64(out, 0, a * a * a - 2 * a + 2)
				buffer.writef64(out, 8, b - a)
				return out
			end,
			J = function(x: buffer, J: any): any
				local a = buffer.readf64(x, 0)
				f64.matset(J, 1, 1, 3 * a * a - 2)
				f64.matset(J, 1, 2, 0)
				f64.matset(J, 2, 1, -1)
				f64.matset(J, 2, 2, 1)
				return J
			end,
		},
	},
	{
		name = "arctangents from 3, n = 5",
		x0 = { 3, 3, 3, 3, 3 },
		S = {
			n = 5,
			F = function(x: buffer, out: buffer): buffer
				for i = 0, 4 do
					buffer.writef64(out, 8 * i, math.atan(buffer.readf64(x, 8 * i)) + 0.05 * buffer.readf64(x, 8 * ((i + 1) % 5)))
				end
				return out
			end,
			J = function(x: buffer, J: any): any
				buffer.fill(J.data, 0, 0)
				for i = 0, 4 do
					local xi = buffer.readf64(x, 8 * i)
					f64.matset(J, i + 1, i + 1, 1 / (1 + xi * xi))
					f64.matset(J, i + 1, (i + 1) % 5 + 1, 0.05)
				end
				return J
			end,
		},
	},
}

print("## Robustness: iterations to max |F_i| <= 1e-10, or failure (100 iterations at most)\n")
print("| problem, start | inverse | solve | citrate |")
print("| --- | --- | --- | --- |")
local tally = { 0, 0, 0 }
local total = 0
for _, p in mgh.problems do
	if p.root then
		local S = adapt(p)
		for _, scale in { 1, 10, 100 } do
			local x0 = f64.from(p.x0)
			f64.scale(x0, scale, x0)
			local rows = {}
			for k, m in { "inverse", "solve", "citrate" } do
				local ok, _, r = pcall(function()
					if m == "citrate" then
						return citrate(S, x0)
					end
					return plain(S, x0, m == "inverse")
				end)
				local res: Result = if ok then r else { converged = false, iterations = 0, evaluations = 0, norm = 0 / 0 }
				if res.converged then
					tally[k] += 1
				end
				rows[k] = if ok then cell(res) else "error"
			end
			total += 1
			print(string.format("| %s, %s x0 | %s | %s | %s |", p.name, if scale == 1 then "" else tostring(scale), rows[1], rows[2], rows[3]))
		end
	end
end
for _, e in extra do
	local rows = {}
	for k, m in { "inverse", "solve", "citrate" } do
		local x0 = f64.from(e.x0)
		local _, r
		if m == "citrate" then
			_, r = citrate(e.S, x0)
		else
			_, r = plain(e.S, x0, m == "inverse")
		end
		if r.converged then
			tally[k] += 1
		end
		rows[k] = cell(r)
	end
	total += 1
	print(string.format("| %s | %s | %s | %s |", e.name, rows[1], rows[2], rows[3]))
end
print(string.format("| solved | %d of %d | %d of %d | %d of %d |", tally[1], total, tally[2], total, tally[3], total))

-- Chandrasekhar's H-equation (radiative transfer; a standard test in Kelley's book),
-- discretized by the midpoint rule on n nodes mu_i: F_i(H) = H_i - 1 / (1 - sum_j a_ij H_j),
-- a_ij = c mu_i / (2 n (mu_i + mu_j)), c = 0.9. Its Jacobian is dense.
local function hequation(n: number): System
	local c = 0.9
	local A = f64.matrix(n, n)
	for i = 1, n do
		local mi = (i - 0.5) / n
		for j = 1, n do
			local mj = (j - 0.5) / n
			f64.matset(A, i, j, c * mi / (2 * n * (mi + mj)))
		end
	end
	local s = f64.create(n)
	return {
		n = n,
		F = function(x: buffer, out: buffer): buffer
			f64.matvec(s, A, x)
			for i = 0, n - 1 do
				buffer.writef64(out, 8 * i, buffer.readf64(x, 8 * i) - 1 / (1 - buffer.readf64(s, 8 * i)))
			end
			return out
		end,
		J = function(x: buffer, J: any): any
			f64.matvec(s, A, x)
			for j = 0, n - 1 do
				for i = 0, n - 1 do
					local d = 1 - buffer.readf64(s, 8 * i)
					buffer.writef64(J.data, 8 * (j * n + i), (if i == j then 1 else 0) - buffer.readf64(A.data, 8 * (j * n + i)) / (d * d))
				end
			end
			return J
		end,
	}
end

local function time(f: () -> ()): number
	local reps, t0 = 0, os.clock()
	repeat
		f()
		reps += 1
	until os.clock() - t0 > 0.2
	return (os.clock() - t0) / reps
end
local function ms(s: number): string
	return if s < 1e-3 then string.format("%.0f us", s * 1e6) else string.format("%.1f ms", s * 1e3)
end

print("\n## Speed: the H-equation from H = 1 (time per solve, iterations, final |F|)\n")
print("| n | inverse | solve | citrate |")
print("| --- | --- | --- | --- |")
for _, n in { 10, 50, 200, 500 } do
	local S = hequation(n)
	local x0 = f64.fill(f64.create(n), 1)
	local rows = {}
	for k, m in { "inverse", "solve", "citrate" } do
		local run = function(): (buffer, Result)
			if m == "citrate" then
				return citrate(S, x0)
			end
			return plain(S, x0, m == "inverse")
		end
		local _, r = run()
		local t = time(function()
			run()
		end)
		rows[k] = string.format("%s, %d its, |F| %.0e", ms(t), r.iterations, r.norm)
	end
	print(string.format("| %d | %s | %s | %s |", n, rows[1], rows[2], rows[3]))
end
