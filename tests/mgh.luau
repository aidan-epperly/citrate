--!strict

-- Test problems from Moré, Garbow and Hillstrom, "Testing unconstrained optimization
-- software" (1981), as residuals r(x) (m of them, x of size n) with their Jacobians and
-- standard starting points. Least squares minimizes |r|^2 / 2; minimization uses
-- f = |r|^2 / 2 with gradient J^T r; the square ones with a zero residual are systems of
-- equations. fstar is the known minimum of |r|^2 / 2. bench/optimize.py has the same
-- problems for SciPy.

export type Problem = {
	name: string,
	n: number,
	m: number,
	x0: { number },
	fstar: number,
	root: boolean, -- square with a zero residual: also a test for root
	r: (x: { number }) -> { number },
	jac: (x: { number }) -> { { number } }, -- rows
}

local problems: { Problem } = {}
local function add(p: Problem)
	table.insert(problems, p)
end

add({
	name = "Rosenbrock",
	n = 2,
	m = 2,
	x0 = { -1.2, 1 },
	fstar = 0,
	root = true,
	r = function(x)
		return { 10 * (x[2] - x[1] ^ 2), 1 - x[1] }
	end,
	jac = function(x)
		return { { -20 * x[1], 10 }, { -1, 0 } }
	end,
})

add({
	name = "Freudenstein-Roth",
	n = 2,
	m = 2,
	x0 = { 0.5, -2 },
	fstar = 24.4921, -- the local minimum methods reach from x0 (the global one is 0)
	root = false,
	r = function(x)
		return {
			-13 + x[1] + ((5 - x[2]) * x[2] - 2) * x[2],
			-29 + x[1] + ((x[2] + 1) * x[2] - 14) * x[2],
		}
	end,
	jac = function(x)
		return { { 1, 10 * x[2] - 3 * x[2] ^ 2 - 2 }, { 1, 3 * x[2] ^ 2 + 2 * x[2] - 14 } }
	end,
})

add({
	name = "Powell badly scaled",
	n = 2,
	m = 2,
	x0 = { 0, 1 },
	fstar = 0,
	root = true,
	r = function(x)
		return { 1e4 * x[1] * x[2] - 1, math.exp(-x[1]) + math.exp(-x[2]) - 1.0001 }
	end,
	jac = function(x)
		return { { 1e4 * x[2], 1e4 * x[1] }, { -math.exp(-x[1]), -math.exp(-x[2]) } }
	end,
})

add({
	name = "Brown badly scaled",
	n = 2,
	m = 3,
	x0 = { 1, 1 },
	fstar = 0,
	root = false,
	r = function(x)
		return { x[1] - 1e6, x[2] - 2e-6, x[1] * x[2] - 2 }
	end,
	jac = function(x)
		return { { 1, 0 }, { 0, 1 }, { x[2], x[1] } }
	end,
})

add({
	name = "Beale",
	n = 2,
	m = 3,
	x0 = { 1, 1 },
	fstar = 0,
	root = false,
	r = function(x)
		local y = { 1.5, 2.25, 2.625 }
		local out = {}
		for i = 1, 3 do
			out[i] = y[i] - x[1] * (1 - x[2] ^ i)
		end
		return out
	end,
	jac = function(x)
		local out = {}
		for i = 1, 3 do
			out[i] = { -(1 - x[2] ^ i), x[1] * i * x[2] ^ (i - 1) }
		end
		return out
	end,
})

add({
	name = "Jennrich-Sampson",
	n = 2,
	m = 10,
	x0 = { 0.3, 0.4 },
	fstar = 124.362 / 2,
	root = false,
	r = function(x)
		local out = {}
		for i = 1, 10 do
			out[i] = 2 + 2 * i - (math.exp(i * x[1]) + math.exp(i * x[2]))
		end
		return out
	end,
	jac = function(x)
		local out = {}
		for i = 1, 10 do
			out[i] = { -i * math.exp(i * x[1]), -i * math.exp(i * x[2]) }
		end
		return out
	end,
})

local function theta(x1: number, x2: number): number
	local t = math.atan(x2 / x1) / (2 * math.pi)
	return if x1 < 0 then t + 0.5 else t
end
add({
	name = "Helical valley",
	n = 3,
	m = 3,
	x0 = { -1, 0, 0 },
	fstar = 0,
	root = true,
	r = function(x)
		return { 10 * (x[3] - 10 * theta(x[1], x[2])), 10 * (math.sqrt(x[1] ^ 2 + x[2] ^ 2) - 1), x[3] }
	end,
	jac = function(x)
		local q = x[1] ^ 2 + x[2] ^ 2
		local s = math.sqrt(q)
		-- d theta / dx1 = -x2 / (2 pi q), d theta / dx2 = x1 / (2 pi q)
		return {
			{ 100 * x[2] / (2 * math.pi * q), -100 * x[1] / (2 * math.pi * q), 10 },
			{ 10 * x[1] / s, 10 * x[2] / s, 0 },
			{ 0, 0, 1 },
		}
	end,
})

local BARD_Y = { 0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39 }
add({
	name = "Bard",
	n = 3,
	m = 15,
	x0 = { 1, 1, 1 },
	fstar = 8.21487e-3 / 2,
	root = false,
	r = function(x)
		local out = {}
		for i = 1, 15 do
			local u, v = i, 16 - i
			local w = math.min(u, v)
			out[i] = BARD_Y[i] - (x[1] + u / (v * x[2] + w * x[3]))
		end
		return out
	end,
	jac = function(x)
		local out = {}
		for i = 1, 15 do
			local u, v = i, 16 - i
			local w = math.min(u, v)
			local d = (v * x[2] + w * x[3]) ^ 2
			out[i] = { -1, u * v / d, u * w / d }
		end
		return out
	end,
})

add({
	name = "Box 3D",
	n = 3,
	m = 10,
	x0 = { 0, 10, 20 },
	fstar = 0,
	root = false,
	r = function(x)
		local out = {}
		for i = 1, 10 do
			local t = 0.1 * i
			out[i] = math.exp(-t * x[1]) - math.exp(-t * x[2]) - x[3] * (math.exp(-t) - math.exp(-10 * t))
		end
		return out
	end,
	jac = function(x)
		local out = {}
		for i = 1, 10 do
			local t = 0.1 * i
			out[i] = { -t * math.exp(-t * x[1]), t * math.exp(-t * x[2]), -(math.exp(-t) - math.exp(-10 * t)) }
		end
		return out
	end,
})

local S5, S10 = math.sqrt(5), math.sqrt(10)
add({
	name = "Powell singular",
	n = 4,
	m = 4,
	x0 = { 3, -1, 0, 1 },
	fstar = 0,
	root = true,
	r = function(x)
		return { x[1] + 10 * x[2], S5 * (x[3] - x[4]), (x[2] - 2 * x[3]) ^ 2, S10 * (x[1] - x[4]) ^ 2 }
	end,
	jac = function(x)
		local a, b = 2 * (x[2] - 2 * x[3]), 2 * S10 * (x[1] - x[4])
		return { { 1, 10, 0, 0 }, { 0, 0, S5, -S5 }, { 0, a, -2 * a, 0 }, { b, 0, 0, -b } }
	end,
})

local S90 = math.sqrt(90)
add({
	name = "Wood",
	n = 4,
	m = 6,
	x0 = { -3, -1, -3, -1 },
	fstar = 0,
	root = false,
	r = function(x)
		return {
			10 * (x[2] - x[1] ^ 2),
			1 - x[1],
			S90 * (x[4] - x[3] ^ 2),
			1 - x[3],
			S10 * (x[2] + x[4] - 2),
			(x[2] - x[4]) / S10,
		}
	end,
	jac = function(x)
		return {
			{ -20 * x[1], 10, 0, 0 },
			{ -1, 0, 0, 0 },
			{ 0, 0, -2 * S90 * x[3], S90 },
			{ 0, 0, -1, 0 },
			{ 0, S10, 0, S10 },
			{ 0, 1 / S10, 0, -1 / S10 },
		}
	end,
})

local KO_Y = { 0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246 }
local KO_U = { 4, 2, 1, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625 }
add({
	name = "Kowalik-Osborne",
	n = 4,
	m = 11,
	x0 = { 0.25, 0.39, 0.415, 0.39 },
	fstar = 3.07505e-4 / 2,
	root = false,
	r = function(x)
		local out = {}
		for i = 1, 11 do
			local u = KO_U[i]
			out[i] = KO_Y[i] - x[1] * (u * u + u * x[2]) / (u * u + u * x[3] + x[4])
		end
		return out
	end,
	jac = function(x)
		local out = {}
		for i = 1, 11 do
			local u = KO_U[i]
			local num, den = u * u + u * x[2], u * u + u * x[3] + x[4]
			out[i] = { -num / den, -x[1] * u / den, x[1] * num * u / den ^ 2, x[1] * num / den ^ 2 }
		end
		return out
	end,
})

local function extended_rosenbrock(n: number): Problem
	local x0 = {}
	for i = 1, n, 2 do
		x0[i], x0[i + 1] = -1.2, 1
	end
	return {
		name = `Extended Rosenbrock {n}`,
		n = n,
		m = n,
		x0 = x0,
		fstar = 0,
		root = true,
		r = function(x)
			local out = {}
			for i = 1, n, 2 do
				out[i], out[i + 1] = 10 * (x[i + 1] - x[i] ^ 2), 1 - x[i]
			end
			return out
		end,
		jac = function(x)
			local out = {}
			for i = 1, n do
				out[i] = table.create(n, 0)
			end
			for i = 1, n, 2 do
				out[i][i], out[i][i + 1] = -20 * x[i], 10
				out[i + 1][i] = -1
			end
			return out
		end,
	}
end
add(extended_rosenbrock(10))

local function trigonometric(n: number): Problem
	return {
		name = `Trigonometric {n}`,
		n = n,
		m = n,
		x0 = table.create(n, 1 / n),
		fstar = 2.79506e-5 / 2, -- n = 10 (MGH: a local minimum; not a zero)
		root = false,
		r = function(x)
			local s = 0
			for j = 1, n do
				s += math.cos(x[j])
			end
			local out = {}
			for i = 1, n do
				out[i] = n - s + i * (1 - math.cos(x[i])) - math.sin(x[i])
			end
			return out
		end,
		jac = function(x)
			local out = {}
			for i = 1, n do
				local row = {}
				for j = 1, n do
					row[j] = math.sin(x[j])
				end
				row[i] += i * math.sin(x[i]) - math.cos(x[i])
				out[i] = row
			end
			return out
		end,
	}
end
add(trigonometric(10))

local function discrete_bv(n: number): Problem
	local h = 1 / (n + 1)
	local x0 = {}
	for i = 1, n do
		local t = i * h
		x0[i] = t * (t - 1)
	end
	return {
		name = `Discrete boundary value {n}`,
		n = n,
		m = n,
		x0 = x0,
		fstar = 0,
		root = true,
		r = function(x)
			local out = {}
			for i = 1, n do
				local t = i * h
				out[i] = 2 * x[i] - (x[i - 1] or 0) - (x[i + 1] or 0) + h * h * (x[i] + t + 1) ^ 3 / 2
			end
			return out
		end,
		jac = function(x)
			local out = {}
			for i = 1, n do
				local row = table.create(n, 0)
				local t = i * h
				row[i] = 2 + 1.5 * h * h * (x[i] + t + 1) ^ 2
				if i > 1 then
					row[i - 1] = -1
				end
				if i < n then
					row[i + 1] = -1
				end
				out[i] = row
			end
			return out
		end,
	}
end
add(discrete_bv(10))

local function broyden_tridiagonal(n: number): Problem
	return {
		name = `Broyden tridiagonal {n}`,
		n = n,
		m = n,
		x0 = table.create(n, -1),
		fstar = 0,
		root = true,
		r = function(x)
			local out = {}
			for i = 1, n do
				out[i] = (3 - 2 * x[i]) * x[i] - (x[i - 1] or 0) - 2 * (x[i + 1] or 0) + 1
			end
			return out
		end,
		jac = function(x)
			local out = {}
			for i = 1, n do
				local row = table.create(n, 0)
				row[i] = 3 - 4 * x[i]
				if i > 1 then
					row[i - 1] = -1
				end
				if i < n then
					row[i + 1] = -2
				end
				out[i] = row
			end
			return out
		end,
	}
end
add(broyden_tridiagonal(10))

return {
	problems = problems,
	-- the scalable ones, for other sizes
	extended_rosenbrock = extended_rosenbrock,
	trigonometric = trigonometric,
	discrete_bv = discrete_bv,
	broyden_tridiagonal = broyden_tridiagonal,
}
