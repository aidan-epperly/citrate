# Differential equations

Initial value problems (explicit, stiff, second-order, symplectic), implicit mechanics and boundary value problems.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `solve`

```luau
solve<T>(S: Space<T>, f: OdeFunction<T>, tspan: { number }, y0: T, options: OdeOptions<T>?): OdeSolution<T>
```

Solves y' = f(t, y), y(tspan[1]) = y0, over tspan (forward or backward). options set the method (default "dp5"), tolerances, output times, events and dense output (see types.OdeOptions); the result holds the steps and at(t) (see types.OdeSolution).

### `integrator`

```luau
integrator<T>(S: Space<T>, f: OdeFunction<T>, t0: number, y0: T, options: OdeOptions<T>?): any
```

An integrator: a state (it.t, it.y) you advance yourself, for simulations that step by a frame time. options as for solve (method, rtol, atol, max_step; h is the step for fixed-step methods' advance). it:step(h) takes exactly one step of size h (for an adaptive method, its main formula without error control); it:advance(dt) moves exactly dt ahead: adaptive methods take error-controlled steps, remembering the step size for next time; fixed-step methods take steps of options.h (or one step of dt). Both return it.y. Stiff methods and "auto" only advance, except rodas5p, whose step(h) takes one step of exactly h without error control (a fixed cost per step, for frame-locked simulation).

### `methods`

```luau
methods(): { [string]: boolean }
```

The names of the available methods, and whether each is adaptive.

### `mechanics`

```luau
mechanics<T>(S: Space<T>, accel: OdeFunction<T>, t0: number, q0: T, v0: T, options: { method: string?, h: number? }?): any
```

A mechanics integrator: positions it.q and velocities it.v (in the same space) at time it.t, advanced by it:step(h) (one step) or it:advance(dt) (steps of options.h, or one step of dt). options.method is "verlet" (default), "yoshida4", "pefrl" or "yoshida6"; accel(t, q, out) returns the acceleration.

### `solve_mechanics`

```luau
solve_mechanics<T>(S: Space<T>, accel: OdeFunction<T>, tspan: { number }, q0: T, v0: T, options: { method: string?, h: number?, t_eval: { number }? }): any
```

Integrates q'' = accel(t, q) over tspan with fixed steps options.h (required), by a symplectic method (see mechanics). Returns { t, q, v (the steps), stats, at(t) -> (q, v) (cubic Hermite interpolation, from the velocities and accelerations at the step ends), and q_eval, v_eval at options.t_eval }.

### `jacobian`

```luau
jacobian(S: any, f: OdeFunction<buffer>, t: number, y: buffer, options: any?): any
```

The Jacobian df/dy of f at (t, y) by finite differences, as the stiff solvers estimate it: a matrix, or with options.jac_band = { kl, ku } a banded matrix, or with options.jac_sparsity a sparse matrix of that pattern (one evaluation of f per group of columns that share no row). y is an f32, f64, c32 or c64 array.

### `bvp`

```luau
bvp(f: (number, buffer, buffer) -> buffer, bc: (buffer, buffer, buffer) -> buffer, domain: { number }, orders: { number } | number, options: BvpOptions?): any
```

Solves the boundary value problem described above on domain = { a, b }. orders is the list { k_1, .., k_m } of the equations' orders, or a number m for a first-order system of m equations (y' = f(x, y) with bc(y(a), y(b)) = 0). options (all optional): guess (default zero; nonlinear problems with several solutions converge to the one nearest the guess), tol (default 1e-10: the relative size of the Chebyshev coefficients counted as resolved, and the Newton tolerance), max_points (default 513), max_newton (iterations per attempt, default 30), jac and bc_jac (see BvpOptions). Returns { y = { Chebfun per unknown function }, status ("success" or "failed"), message, points (the final N), newton (the total iterations), residual (the largest |u_j^(k_j) - f_j| at the collocation points, relative to the size of f) }.

### `solve_second_order`

```luau
solve_second_order<T>(S: Space<T>, f: (number, T, T, T) -> T, tspan: { number }, y0: T, v0: T, options: OdeOptions<T>?): any
```

Second-order equations y'' = f(t, y, y')  ode.solve_second_order(S, f, tspan, y0, v0, options) solves y'' = f(t, y, v) with y(t0) = y0 and y'(t0) = v0, where f(t, y, v, out) -> the acceleration. options are as for solve, and method is either a Runge-Kutta-Nystrom pair "dprkn6", "dprkn8" or "dprkn12" (see odenystrom.luau): fewer evaluations for the same accuracy, but only for f that does not depend on v; or any method of solve ("dp5", the default, "dop853", stiff ones, "auto", ...), applied to the first-order system (y, v)' = (v, f(t, y, v)). For arrays the system state is one array of length 2n ([y; v]), so the stiff methods apply (with a dense finite-difference Jacobian; jac, jac_band, jac_sparsity and mass are not supported here); other spaces (vectors) use the pair (y, v) with the explicit methods. Event functions are g(t, y, v). Returns { t, y, v (the steps), status, message, stats, events (a list of { t, y, v, index }), t_eval, y_eval, v_eval, at }, where sol.at(t) returns (y(t), v(t)).

### `implicit_mechanics`

```luau
implicit_mechanics(S: any, force: (number, buffer, buffer, buffer) -> buffer, t0: number, q0: buffer, v0: buffer, options: ImplicitOptions): any
```

An implicit mechanics integrator for M q'' = F(t, q, v): it.t, it.q, it.v, advanced by it:step(h) (one step) or it:advance(dt, h?) (steps of h, default dt). force(t, q, v, out) returns F. options: jacobian(t, q, v) -> K, D?   dF/dq and dF/dv (sparse or block sparse matrices, or operators; D may be omitted when F does not depend on v). K and D of one format with the same pattern (with the diagonal stored) are combined into one matrix, so each CG iteration makes a single product; block sparse ones are preconditioned by block Jacobi. Mark K and D symmetric (as a conservative force's are) for the faster product. system(t, q, v, h) -> apply, preconditioner, Kv   instead of jacobian, for models that can apply A = M - h D - h^2 K faster themselves (cloth, from its springs): the operator (out, x) -> out, A's diagonal (for the Jacobi preconditioner) or a preconditioner (out, r) -> out (symmetric positive definite, such as block Jacobi), and K v mass      a number or an array (per component; default 1) fixed     components (1-based) held in place: their velocity stays zero tol, maxiter  for CG (default 1e-5 relative residual, 200 iterations) amg       with block sparse K and D: precondition by algebraic multigrid (bsparse_amg, these are its options, fixed added) instead of block Jacobi. Its hierarchy is refilled every `refresh` steps (default 1); in between it is stale, which stays a valid (positive definite) preconditioner but can cost many iterations when the system changes quickly (swinging cloth: 9 iterations refilled every step, 27 every 5th, 133 every 20th). For refills every step, plain aggregation (smooth = false) was fastest on cloth. rho_inf   in [0, 1]: instead of backward Euler, a third-order rational step that damps slow modes far less (see the rational scheme below), with |R| = rho_inf for the fastest: 0 removes them (L-stable), 1 keeps all energy. Needs block sparse K and D; costs one complex solve per step (COCG). solver    "auto" (default), "direct" or "cg": with block sparse K and D on one pattern, "direct" factors the system in banded storage (Cholesky, or complex LU for rho_inf) instead of iterating. "auto" does so when that is cheap (half bandwidth kl <= 48 and n kl^2 <= 2e5: chains and ropes, not cloth) and amg is not set. On a 6-link tail a rational step took 14 us natively against 35 by COCG. it.stats counts steps, force and Jacobian evaluations and CG iterations; it.path says how the last step applied the system ("sparse" or "block": K and D combined into one matrix, "operators", or "system").
