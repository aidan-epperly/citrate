# Root finding and optimization

Scalar roots and minima (Brent, Newton, and global ones through Chebyshev and AAA interpolants), systems (Newton), least squares (Levenberg-Marquardt) and minimization (L-BFGS, Nelder-Mead).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `root_scalar`

```luau
root_scalar(f: (number) -> number, a: number, b: number, options: ScalarOptions?): (number?, ScalarInfo)
```

A root of f in [a, b]. "brent" (the default) needs f(a) and f(b) of opposite signs and finds a root between them, with few evaluations; "chebyshev" and "aaa" interpolate f on [a, b] (Chebyshev polynomial or AAA rational) and return its first root there, which needs no bracket but more evaluations (all roots: see roots). Returns the root (nil if none was found) and info.

### `roots`

```luau
roots(f: (number) -> number, a: number, b: number, options: ScalarOptions?): ({ number }, ScalarInfo)
```

Every root of f in [a, b], ascending, from a Chebyshev ("chebyshev", the default) or AAA ("aaa") interpolant of f there, and info.

### `newton`

```luau
newton(f: (number) -> number, x0: number, options: NewtonOptions?): (number, ScalarInfo)
```

A root of f near x0: Newton's method with f' (options.df), else the secant method (from x0 and a small step). Fast from a good guess (such as last frame's root), but local: it may diverge or find another root. Returns the root and info (converged false if it did not settle).

### `minimize_scalar`

```luau
minimize_scalar(f: (number) -> number, a: number, b: number, options: ScalarOptions?): (number, number, ScalarInfo)
```

A minimum of f on [a, b]: "brent" (the default) finds a local minimum with few evaluations; "chebyshev" and "aaa" interpolate f on [a, b] and return the interpolant's global minimum there. Returns x, f(x) (as evaluated) and info.

### `minimize`

```luau
minimize<T>(S: Space<T>, f: (T) -> number, x0: T, options: MinimizeOptions<T>?): (T, number, Info)
```

A local minimum of f (a number from a value of the space) from x0: L-BFGS (the default; options.grad gives the gradient, else forward differences) or Nelder-Mead (options.method = "nelder-mead": no derivatives, for small or noisy problems). Returns x, f(x) and info (evaluations, gradients, iterations, converged, message).

### `root`

```luau
root<T>(S: Space<T>, F: (T, T) -> T, x0: T, options: RootOptions<T>?): (T, Info)
```

A root of F (F(x, out) writes F(x), of x's size, into out) near x0: Newton's method, the Jacobian from options.jac or forward differences (each step an LU solve, never an inverse), its step damped by a line search on |F|^2 / 2 when it does not decrease that enough. The test is nonmonotone (against the largest of the last few values, Grippo, Lampariello and Lucidi), and a full step that raises |F| gets one more full step from where it lands before the search (the watchdog of Chamberlain, Powell, Lemarechal and Pedersen): so full Newton steps are kept wherever they converge, and the search still stops cycles and overshoots. Returns x and info (evaluations, jacobians, iterations, converged, norm).

### `least_squares`

```luau
least_squares<T>(S: Space<T>, r: (T, buffer?) -> buffer, x0: T, options: LeastSquaresOptions<T>?): (T, Info)
```

Minimizes |r(x)|^2 / 2 (r(x, out) writes the m residuals into the f64 array out, a new one when out is nil) from x0: Levenberg-Marquardt (see above). Returns x and info (evaluations, jacobians, iterations, converged, message, cost, lambda_trials, and state: pass info as options.reuse to start a later call from its Jacobian).
