# Root finding and optimization

Reference: [optimize](reference/optimize.md).

```luau
local opt = citrate.optimize
-- a scalar root: Brent's method in a bracket, or Newton's from a guess
local x = opt.root_scalar(function(x) return math.cos(x) - x end, 0, 1)
local y = opt.newton(function(x) return x * x - 2 end, 1)
-- every root in an interval, from a Chebyshev interpolant
local all = opt.roots(function(x) return math.sin(10 * x) end, 0, 1)
-- the global minimum on an interval (Brent's default is local)
local xm = opt.minimize_scalar(f, -2, 2, { method = "chebyshev" })

-- least squares: r(x, out) writes the residuals into out (an f64 array, new if nil)
local x, info = opt.least_squares(citrate.f64.space, r, x0, { jac = jacobian })
-- minimization: L-BFGS with a gradient, or finite differences without one
local x, fx = opt.minimize(citrate.vec.space, f, vector.zero, { grad = g })
-- a system F(x) = 0 by Newton's method
local x = opt.root(citrate.f64.space, F, x0)
```

Unknowns are values of any `Space`: `vector`s (as in the second example), f64 or f32
arrays; Jacobians, finite differences and Nelder-Mead need arrays or vectors, and the
L-BFGS iteration itself works in any space given a gradient. Every function returns an
`info` table with its evaluation counts.

## Methods, and their Chebyshev and AAA alternatives

Wherever a method solves a one-dimensional problem, it can do so classically or through
an interpolant: a Chebyshev polynomial (as in [Chebyshev approximation](chebyshev.md)) or
an AAA rational (as in [rational approximation](rational.md)). Both were measured against
the classical choice (`bench/optimize.luau`, with SciPy's counts on the same problems in
`bench/optimize.py`); the defaults are what measured best.

| problem | default | alternatives | when an alternative pays |
| --- | --- | --- | --- |
| a scalar root, `root_scalar` | `"brent"` (needs a sign change) | `"chebyshev"`, `"aaa"` | when there is no bracket, or every root is wanted (`roots`) |
| a scalar minimum, `minimize_scalar` | `"brent"` (local) | `"chebyshev"`, `"aaa"` (global) | when the global minimum on the interval is wanted |
| L-BFGS's line search | `"wolfe"` | `"chebyshev"`, `"aaa"`, `"hybrid"` | nonconvex problems; finite-difference gradients at large n |
| Newton's line search, `root` | `"backtracking"` | `"chebyshev"`, `"aaa"` | not measured to pay |
| Levenberg-Marquardt's damping | `"newton"` (Moré's) | `"aaa"`, `"chebyshev"` | not measured to pay |

Measured (evaluations; the full tables are `bench/optimize.luau`'s output):

- Scalar roots: Brent takes 3 to 14 evaluations on six test functions, as SciPy's
  `brentq` does (the same algorithm, the same counts). A Chebyshev interpolant takes 20
  to 118 on smooth ones, 247 with a pole near the interval and 4091 across a steep front;
  AAA takes 29 for the pole and 280 for the front, since a rational function resolves
  them with few degrees of freedom. The interpolants find every root at that price.
- Scalar minima: Brent 6 to 14 evaluations, Chebyshev 21 to 119 and AAA 26 to 67, the
  latter two the global minimum on the interval.
- Minimization with exact gradients, 16 Moré-Garbow-Hillstrom problems: the Wolfe search
  totals 586 function and 586 gradient evaluations and solves 13 (it stops at Wood's
  stationary point, as SciPy's L-BFGS-B does, whose counts it follows closely); the
  Chebyshev search totals 2950 and 341 and solves 15, Wood included: seeing the whole
  line, it steps past the saddle. AAA matches Chebyshev's counts at 5 times its time
  (fitting costs more than interpolating). Nelder-Mead solves 11, at 13790 evaluations
  (its counts match SciPy's).
- With finite-difference gradients a gradient costs n + 1 evaluations, so a search that
  uses fewer gradients gains with n: on the extended Rosenbrock function the Chebyshev
  search used 752, 1912 and 5602 evaluations at n = 10, 30 and 100 against Wolfe's
  732, 2144 and 8466 (SciPy: 792, 2015 and 6464). On the discrete boundary value problem
  at n = 30 it took more iterations and lost (6752 against 3136).
- Newton's method for systems: the unit step is usually accepted; where it is not, the
  backtracking search needs 1 to 3 evaluations and an interpolating one 9, which never
  paid (Rosenbrock: 27 against 123).
- Levenberg-Marquardt: the damping parameter only needs to be found to 10%, and Moré's
  Newton iteration does so in 2 to 3 trials (702 over the 16 problems); AAA took 13109
  and Chebyshev 221429, for the same steps and iterations. All 16 problems are solved by
  each, in about MINPACK's evaluation counts (357 against SciPy's `lm` 309).
