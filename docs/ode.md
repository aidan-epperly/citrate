# Differential equations

Reference: [ode](reference/ode.md).

## Initial value problems

```luau
local ode = citrate.ode
-- A damped spring on a vector state (position, velocity): it stays a vector throughout.
local spring = function(t, y, _) return vector.create(y.y, -4 * y.x - 0.1 * y.y, 0) end
local sol = ode.solve(citrate.vec.space, spring, { 0, 10 }, vector.create(1, 0, 0), { rtol = 1e-6 })
print(sol.at(2.5))                       -- dense output anywhere in [0, 10]

-- Advancing a simulation by the frame time, with error control inside each frame:
local it = ode.integrator(citrate.vec.space, spring, 0, vector.create(1, 0, 0))
it:advance(1 / 60)                        -- it.t, it.y
```

Buffers work the same way with their module's space (`citrate.f64.space`, ...); the
right-hand side may write into its `out` argument instead of allocating.

Explicit methods: Euler, midpoint, Heun, RK4, and the adaptive Bogacki-Shampine 3(2),
Dormand-Prince 5(4) (`"dp5"`, the default) and DOP853, with per-component error control,
dense output, events, output times and backward integration.

## Stiff problems

Stiff problems (chemical kinetics, discretized diffusion, circuits) take a stiff method:
`"rodas5p"` (Rosenbrock, the fastest on small and medium systems), `"radau"` (Radau IIA 5,
the most robust, also for stiff oscillations) or `"bdf"` (variable-order NDF/BDF, cheapest
per step on large systems; sparse Jacobians via `jac_sparsity` and GMRES with ILU(0), or
matrix-free Newton-Krylov with `linear = "krylov"`). Each takes an optional Jacobian `jac`,
its band `jac_band = { kl, ku }` and a mass matrix `mass` for M y' = f(t, y), including
differential-algebraic equations (singular M). `method = "auto"` starts with
Dormand-Prince and switches to a stiff method (`stiff_method`, default `"rodas5p"`) when it
detects stiffness, and back when it passes.

```luau
local f64 = citrate.f64
local robertson = function(t, y, out)
	local a, b, c = f64.get(y, 1), f64.get(y, 2), f64.get(y, 3)
	f64.set(out, 1, -0.04 * a + 1e4 * b * c)
	f64.set(out, 2, 0.04 * a - 1e4 * b * c - 3e7 * b * b)
	f64.set(out, 3, 3e7 * b * b)
	return out
end
local sol = ode.solve(f64.space, robertson, { 0, 40 }, f64.from({ 1, 0, 0 }), { method = "rodas5p", rtol = 1e-6, atol = 1e-10 })
local auto = ode.solve(f64.space, robertson, { 0, 40 }, f64.from({ 1, 0, 0 }), { method = "auto" })
```

## Second-order equations and mechanics

`ode.solve_second_order` solves y'' = f(t, y, y') and returns positions and velocities. For
forces that do not depend on the velocity, the Runge-Kutta-Nystrom pairs `"dprkn6"`,
`"dprkn8"` and `"dprkn12"` reach a given accuracy with fewer evaluations than a
Runge-Kutta method on the first-order system; any other method (including the stiff ones,
for arrays) works by that reduction.

```luau
local orbit = ode.solve_second_order(citrate.vec.space, function(t, q, v, _)
	return -q / vector.magnitude(q) ^ 3
end, { 0, 100 }, vector.create(1, 0, 0), vector.create(0, 1, 0), { method = "dprkn8", rtol = 1e-10 })
local q, v = orbit.at(50)
```

For long runs of mechanics q'' = a(t, q) (orbits, springs, game physics), the symplectic
integrators keep the energy error bounded: `ode.mechanics` and `ode.solve_mechanics` with
velocity Verlet, Yoshida 4 and 6, or PEFRL.

## Implicit mechanics

Stiff mechanical systems stepped at a fixed frame time (cloth, ropes, stiff springs) suit
`ode.implicit_mechanics`: one linearized backward Euler step per frame (Baraff and Witkin),
solved by conjugate gradients, stable however stiff the forces. It takes the force and its
Jacobians K = dF/dq and D = dF/dv as sparse or block sparse matrices or as operators, or a
model's own operator for M - h D - h^2 K. Block sparse Jacobians can be preconditioned by
multigrid with the `amg` option. `demos/cloth` is a Roblox demo of implicit against
explicit cloth. For small stiff systems, a Rodas5P integrator's `step(h)` takes one fixed
step with no iterations.

## Boundary value problems

Boundary value problems are solved by Chebyshev collocation with damped Newton iteration
and adaptive resolution. Each unknown function u_j satisfies an equation of its own order,
u_j^(k_j) = f_j(x, z), where z lists every function with its derivatives below their
orders; the boundary conditions are bc(z(a), z(b)) = 0. Higher-order equations are solved
directly, without introducing derivatives as unknowns, and each solution comes back as a
Chebfun (see [Chebyshev approximation](chebyshev.md)). Passing a number m instead of the
orders means a first-order system of m equations.

```luau
-- Bratu's problem u'' + e^u = 0, u(0) = u(1) = 0, as one second-order equation: z = (u, u').
local sol = ode.bvp(function(x, z, out)
	f64.set(out, 1, -math.exp(f64.get(z, 1)))
	return out
end, function(za, zb, out)
	f64.set(out, 1, f64.get(za, 1))
	f64.set(out, 2, f64.get(zb, 1))
	return out
end, { 0, 1 }, { 2 })
print(citrate.cheb.eval(sol.y[1], 0.5))  -- u(1/2)
```

The solution is resolved together with its derivatives below each equation's order, so
those derivatives (by `cheb.diff`) are as accurate as the solution itself.
