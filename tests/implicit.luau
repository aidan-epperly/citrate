--!strict

-- Implicit mechanics: the linearized backward Euler step against its closed form, stability
-- at stiffness far beyond explicit limits, the three ways of giving the system (sparse
-- Jacobians, operators, a model's own system), pinned components and f32 arrays.

local h = require("./harness")
local lib = require("../src")

local test, check, close = h.test, h.check, h.close
local ode, f64, f32 = lib.ode, lib.f64, lib.f32

-- A chain of n unit masses joined by springs of stiffness k (the first end fixed to a
-- wall by another spring), with damping c on each spring: F = -K0 q - C0 v, with K0, C0
-- the tridiagonal stiffness matrices.
local function chain(M: any, n: number, k: number, c: number): any
	local ii, jj, kv, cv = {}, {}, {}, {}
	local function entry(i: number, j: number, a: number, b: number)
		table.insert(ii, i)
		table.insert(jj, j)
		table.insert(kv, a)
		table.insert(cv, b)
	end
	for i = 1, n do
		local deg = if i < n then 2 else 1
		entry(i, i, -deg * k, -deg * c)
		if i < n then
			entry(i, i + 1, k, c)
			entry(i + 1, i, k, c)
		end
	end
	local K = M.sparse(n, n, ii, jj, M.from(kv))
	local D = M.sparse(n, n, ii, jj, M.from(cv))
	return {
		K = K,
		D = D,
		force = function(_: number, q: buffer, v: buffer, out: buffer): buffer
			out = M.sparse_matvec(out, K, q)
			local dv = M.sparse_matvec(nil, D, v)
			return M.add(out, out, dv)
		end,
	}
end

test("implicit mechanics: the backward Euler step", function()
	-- One unit mass, q'' = -k q - c v: the linearized step is exact for linear forces,
	-- (1 + h c + h^2 k) dv = h (-k q - c v - h k v).
	local k, c, hs = 50, 0.5, 0.1
	local K = f64.sparse(1, 1, { 1 }, { 1 }, f64.from({ -k }))
	local D = f64.sparse(1, 1, { 1 }, { 1 }, f64.from({ -c }))
	local it = ode.implicit_mechanics(f64.space, function(_: number, q: buffer, v: buffer, out: buffer): buffer
		f64.set(out, 1, -k * f64.get(q, 1) - c * f64.get(v, 1))
		return out
	end, 0, f64.from({ 1 }), f64.from({ 0 }), {
		jacobian = function(): (any, any)
			return K, D
		end,
		tol = 1e-12,
	})
	local q, v = 1, 0
	for _ = 1, 20 do
		local dv = hs * (-k * q - c * v - hs * k * v) / (1 + hs * c + hs * hs * k)
		v += dv
		q += hs * v
		it:step(hs)
	end
	close(f64.get(it.q, 1), q, 1e-10, "position")
	close(f64.get(it.v, 1), v, 1e-10, "velocity")
	close(it.t, 20 * hs, 1e-12, "time")
	check(it.stats.steps == 20 and it.stats.nfev == 20, "statistics")
end)

test("implicit mechanics: stiff chains, pins and representations", function()
	-- A chain with k = 1e8 (periods near 1e-4) stepped at 1/60: explicit methods would need
	-- thousands of substeps; the implicit step stays bounded and settles. Each way of
	-- giving the system agrees.
	local n, k = 20, 1e8
	local model = chain(f64, n, k, 10)
	local q0 = f64.create(n)
	for i = 1, n do
		f64.set(q0, i, math.sin(i))
	end
	local function run(options: any): any
		options.tol = 1e-10
		options.maxiter = 2000
		local it = ode.implicit_mechanics(f64.space, model.force, 0, q0, f64.create(n), options)
		for _ = 1, 60 do
			it:step(1 / 60)
		end
		return it
	end
	local sparse = run({
		jacobian = function(): (any, any)
			return model.K, model.D
		end,
	})
	local operators = run({
		jacobian = function(): (any, any)
			return function(out: buffer, x: buffer): buffer
				return f64.sparse_matvec(out, model.K, x)
			end, function(out: buffer, x: buffer): buffer
				return f64.sparse_matvec(out, model.D, x)
			end
		end,
	})
	local system = run({
		system = function(_: number, _: buffer, v: buffer, hs: number): (any, any, any)
			local apply = function(out: buffer, x: buffer): buffer
				out = f64.copy(out, x)
				out = f64.axpy(out, -hs * hs, f64.sparse_matvec(nil, model.K, x), out)
				return f64.axpy(out, -hs, f64.sparse_matvec(nil, model.D, x), out)
			end
			local diag = f64.create(n)
			for i = 1, n do
				f64.set(diag, i, 1 - hs * hs * f64.sparse_get(model.K, i, i) - hs * f64.sparse_get(model.D, i, i))
			end
			return apply, diag, f64.sparse_matvec(nil, model.K, v)
		end,
	})
	-- The same matrices marked symmetric, and in 2 by 2 blocks (combined, with block
	-- Jacobi).
	local Ks, Ds = table.clone(model.K), table.clone(model.D)
	Ks.symmetric, Ds.symmetric = true, true
	local marked = run({
		jacobian = function(): (any, any)
			return Ks, Ds
		end,
	})
	local BK, BD = f64.bsparse_from_sparse(model.K, 2), f64.bsparse_from_sparse(model.D, 2)
	local blocks = run({
		jacobian = function(): (any, any)
			return BK, BD
		end,
		solver = "cg",
	})
	-- and solved directly (banded Cholesky): the default for this small band
	local direct = run({
		jacobian = function(): (any, any)
			return BK, BD
		end,
	})
	local block_ops = run({
		jacobian = function(): (any, any)
			return BK, function(out: buffer, x: buffer): buffer
				return f64.sparse_matvec(out, model.D, x)
			end
		end,
	})
	check(f64.norm(sparse.q) < f64.norm(q0) and f64.norm(sparse.q) == f64.norm(sparse.q), `stiff chain stays bounded: {f64.norm(sparse.q)}`)
	-- K and D built separately with the same pattern still combine into one matrix.
	check(sparse.path == "sparse" and operators.path == "operators" and system.path == "system", `paths: {sparse.path}, {operators.path}, {system.path}`)
	check(marked.path == "sparse" and blocks.path == "block" and block_ops.path == "operators" and direct.path == "direct", `paths: {marked.path}, {blocks.path}, {block_ops.path}, {direct.path}`)
	for name, it in ({ operators = operators, system = system, marked = marked, blocks = blocks, block_ops = block_ops, direct = direct } :: { [string]: any }) do
		local d = f64.norm(f64.sub(nil, it.q, sparse.q)) / f64.norm(sparse.q)
		check(d <= 1e-6, `{name} agrees with sparse Jacobians: {d}`)
	end

	-- Pinned components keep their positions and zero velocity.
	local pinned = ode.implicit_mechanics(f64.space, model.force, 0, q0, f64.fill(f64.create(n), 1), {
		jacobian = function(): (any, any)
			return model.K, model.D
		end,
		fixed = { 1, n },
	})
	pinned:advance(1, 1 / 60)
	close(pinned.t, 1, 1e-12, "advance: time")
	check(pinned.stats.steps == 60, `advance: {pinned.stats.steps} steps`)
	check(f64.get(pinned.q, 1) == f64.get(q0, 1) and f64.get(pinned.q, n) == f64.get(q0, n), "pinned positions")
	check(f64.get(pinned.v, 1) == 0 and f64.get(pinned.v, n) == 0, "pinned velocities")

	-- f32 arrays.
	local m32 = chain(f32, n, 1e4, 1)
	local q32 = f32.create(n)
	for i = 1, n do
		f32.set(q32, i, math.sin(i))
	end
	local it32 = ode.implicit_mechanics(f32.space, m32.force, 0, q32, f32.create(n), {
		jacobian = function(): (any, any)
			return m32.K, m32.D
		end,
	})
	it32:advance(1, 1 / 60)
	check(f32.norm(it32.q) < f32.norm(q32), "f32: stiff chain settles")

	check(not pcall(function()
		ode.implicit_mechanics(f64.space, model.force, 0, q0, f64.create(n), {} :: any)
	end), "a jacobian or system is required")
end)

test("implicit mechanics: the rational step is R(hL)", function()
	-- On a linear damped oscillator one step multiplies (q, v) by R(hL) exactly, R the
	-- third-order (2, 2) rational function with R(infinity) = rho_inf.
	local function mm(A: { number }, B: { number }): { number }
		return { A[1] * B[1] + A[2] * B[3], A[1] * B[2] + A[2] * B[4], A[3] * B[1] + A[4] * B[3], A[3] * B[2] + A[4] * B[4] }
	end
	local function lin(A: { number }, B: { number }, a: number, b: number): { number }
		return { a * A[1] + b * B[1], a * A[2] + b * B[2], a * A[3] + b * B[3], a * A[4] + b * B[4] }
	end
	local function inv(A: { number }): { number }
		local d = A[1] * A[4] - A[2] * A[3]
		return { A[4] / d, -A[2] / d, -A[3] / d, A[1] / d }
	end
	local I = { 1, 0, 0, 1 }
	local h = 0.1
	for _, ri in { 0, 0.5, 1 } do
		local q1 = -(2 + ri) / (3 * (1 + ri))
		local q2 = -(1 + 3 * q1) / 6
		local p1, p2 = 1 + q1, 1 / 3 + q1 / 2
		for _, w in { 0.5, 3, 100 } do
			for _, g in { 0, 0.3 } do
				for _, solver in { "cg", "direct" } do
				local K = f64.bsparse(1, 1, 1, { 1 }, { 1 }, f64.from({ -w * w }))
				local D = f64.bsparse(1, 1, 1, { 1 }, { 1 }, f64.from({ -g }))
				local it = ode.implicit_mechanics(f64.space, function(_: number, q: buffer, v: buffer, out: buffer): buffer
					f64.set(out, 1, -w * w * f64.get(q, 1) - g * f64.get(v, 1))
					return out
				end, 0, f64.from({ 1 }), f64.from({ 0.5 }), {
					jacobian = function(): (any, any)
						return K, D
					end,
					rho_inf = ri,
					tol = 1e-14,
					solver = solver,
				})
				it:step(h)
				local Z = { 0, h, -h * w * w, -h * g }
				local Rm = mm(lin(lin(I, Z, 1, p1), mm(Z, Z), 1, p2), inv(lin(lin(I, Z, 1, q1), mm(Z, Z), 1, q2)))
				local eq, ev = Rm[1] + 0.5 * Rm[2], Rm[3] + 0.5 * Rm[4]
				check(math.abs(f64.get(it.q, 1) - eq) + math.abs(f64.get(it.v, 1) - ev) < 1e-12, `rho_inf {ri}, omega {w}, gamma {g}, {solver}`)
				check(it.path == (if solver == "cg" then "rational" else "rational direct"), `path {it.path}`)
				end
			end
		end
	end
	-- third order: halving h divides the error after t = 1 by about 8 (rho_inf = 0)
	local function err(steps: number): number
		local K = f64.bsparse(1, 1, 1, { 1 }, { 1 }, f64.from({ -4 }))
		local it = ode.implicit_mechanics(f64.space, function(_: number, q: buffer, _: buffer, out: buffer): buffer
			f64.set(out, 1, -4 * f64.get(q, 1))
			return out
		end, 0, f64.from({ 1 }), f64.from({ 0 }), {
			jacobian = function(): (any, any)
				return K, nil
			end,
			rho_inf = 0,
			tol = 1e-14,
		})
		for _ = 1, steps do
			it:step(1 / steps)
		end
		return math.abs(f64.get(it.q, 1) - math.cos(2))
	end
	local ratio = err(20) / err(40)
	check(ratio > 7 and ratio < 9, `third order: error ratio {ratio}`)
end)

return nil
