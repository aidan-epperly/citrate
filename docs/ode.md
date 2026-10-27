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
right-hand side may write into its `out` argument instead of allocating. An integrator's
`it.y` is then one array, updated in place by each step, so stepping allocates nothing
(no garbage in a game loop; `bench/alloc.luau` measures it): copy it to keep a state.
`ode.solve` stores a copy of the state at every step. Its dense output (`sol.at`, on by
default; `dense = false` turns it off) keeps each step's interpolation coefficients
packed into one buffer, not a table per step: on a 3-variable system 170 to 270 bytes a
step with the states, against 640 to 1000 before (`dense = false`: about 70).

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
step with no iterations. It linearizes once per step, so it suits stiff decay (chemistry,
diffusion) better than stiff springs that turn: on a 6-link chain with rods of stiffness
2e4, one step per 60 Hz frame blew up within 12 frames, while `advance(dt)` (error
controlled, rtol 1e-3) stayed exact at about 0.35 ms per frame and `implicit_mechanics`
(one solve per frame) took 0.03 ms.

Small systems with a narrow band (chains, ropes, tails: block tridiagonal) are solved
directly instead of by conjugate gradients: the system is copied into banded storage and
factored (Cholesky, or complex LU for the rational step below), which `solver = "auto"`
(the default) chooses when the half bandwidth kl is at most 48 and n kl^2 at most 2e5,
after reordering the blocks by reverse Cuthill-McKee if that narrows the band (once per
pattern; see [graphs](graphs.md)), so the numbering of a chain does not matter.
On a 6-bone tail (18 unknowns) a rational step took 11 us natively and 61 interpreted
(with its force and Jacobian), against 35 and 160 by COCG.

### Position-based dynamics: XPBD

`citrate.xpbd` steps the same kind of system differently: particles and constraints
(rods, springs toward a rest shape), each constraint with a compliance (the inverse of
its stiffness) and a damping. Each substep predicts the positions and then moves the
particles onto each constraint in turn, by the update of extended position-based
dynamics (Macklin, Müller and Chentanez 2016): one Gauss-Seidel sweep on the backward
Euler step written for the constraints' multipliers. The compliance enters as alpha / h^2,
so stiffness means the same at every substep size, and a weight on a spring rests at
Hooke's stretch m g alpha whatever the substeps. Small substeps with one sweep each are
more accurate than few with many sweeps (Macklin et al. 2019).

```luau
local world = citrate.xpbd.new({ vector.zero, vector.create(1, 0, 0), vector.create(2, 0, 0) }, { gravity = vector.create(0, -9.8, 0), drag = 1 })
world:pin(1)                     -- moved by world:move, not by the constraints
world:distance(1, 2, 1e-5, 50)   -- a stiff rod with a dashpot
world:distance(2, 3, 1e-5, 50)
world:step(1 / 60, 4)            -- four substeps of 1/240 s
```

Particles can collide with spheres, capsules and planes (`world:sphere`, `capsule`,
`plane`; with friction), and drag pulls them toward a wind's velocity. The solver keeps
the caller's representation: points given as vectors are stored and updated as vectors,
and with `space = f64.space` (or `f32.space`) in a buffer of that precision; the core is
generated once per storage from one template.

On the 6-bone tail of `demos/tails`, in substeps of 1/240 s, it moved as the implicit
integrator does at every update rate, for 2.1 us per 60 Hz update natively and 11
interpreted (implicit: 14 and 66). Its cost grows with the time simulated, not the
updates: an update at 15 Hz takes four times one at 60 Hz. With a fixed number of
substeps per update instead it is cheaper at low rates but no longer the same at every
rate (the tail drooped 0.81 studs at 60 Hz and 1.18 at 15 Hz with two substeps).

### Less numerical damping: the rational step

Backward Euler multiplies a mode of frequency omega by 1 / (1 - i h omega) each step, so it
damps slow, visible motion too: at h omega = 1 a mode keeps 71% of its amplitude per step.
With `rho_inf` set (between 0 and 1), each step instead applies a third-order rational
function R of type (2, 2), with |R| = rho_inf for the fastest modes: 0 removes them like
Radau IIA, 1 keeps all energy like the Gauss method, and every value in between is
A-stable. Slow modes are damped only to fourth order in h omega (98.8% survives a step at
h omega = 1 with rho_inf = 0). The step costs one complex solve, COCG preconditioned on the
real part, with block sparse K and D; on cloth that was about 4 times a backward Euler step.

Measured on a 30 by 30 cloth with no physical damping, swinging from its pinned top edge
(`bench/rational_step.luau`), as the fraction of the drop's energy lost:

| springs | method | after 1 s | after 4 s |
| --- | --- | --- | --- |
| soft (stiffness 5000) | backward Euler | 0.50 | 0.87 |
| | rational, rho_inf = 0 | 0.15 | 0.59 |
| stiff (stiffness 1e6) | backward Euler | 0.91 | 0.99 |
| | rational, rho_inf = 0 | 0.85 | 0.97 |

With soft springs the rational step keeps much more of the motion. With stiff ones both
lose most of it, and the cause is not the rational function: each step linearizes the
forces once, and a stiff spring that turns by an angle theta in a step gains a spurious
stretch of about L theta^2 / 2, whose energy (about 5 J per frame here, against a 44 J
drop) any damped step then removes. Values of rho_inf near 1 are unsafe for the same
reason: undamped fast modes absorb that error and the energy grows (17 times the drop in
the first second at rho_inf = 1). Use rho_inf of at most 0.5.

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
