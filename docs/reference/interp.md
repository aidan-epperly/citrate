# Interpolation

Values on regular grids (bilinear, bicubic) and at scattered points (radial basis functions).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `grid`

```luau
grid(values: any, options: GridOptions?): Grid
```

An interpolant of values on a regular grid: values[i][j] (or a Matrix's entry (i, j)) is the value at (x0 + (i - 1) dx, y0 + (j - 1) dy) (x0, y0 default 0; dx, dy default 1). options.method: "cubic" (the default)  Keys' cubic convolution (a = -1/2, as Catmull-Rom): it passes through the values, its slope is continuous, and its error falls as the cube of the spacing; it can overshoot a little near sharp steps; "linear"               bilinear: never overshoots, but the slope jumps at grid lines. Points outside the grid are clamped to its edge. Needs at least 2 points each way (3 for cubic; with 2, linear is used).

### `rbf`

```luau
rbf(points: { any }, values: { number }, options: RbfOptions?): (Rbf, RbfInfo)
```

An interpolant of values at scattered points (vectors, or tables { x, y } / { x, y, z }) by radial basis functions: f(p) = sum of w_k phi(|p - p_k|) plus a linear polynomial in the coordinates the points span. Returns the interpolant and info: method ("dense" or "hodlr"), kernel, epsilon, smoothing (the lambda used) and residual (the largest misfit at the points, relative to the largest value; check it when the system is ill-conditioned). options.kernel: "thin_plate" (the default)  r^2 log r: in 2-D the surface through the points that bends least (in 3-D that is phi = r); reproduces linear data; "cubic"                     r^3; "gaussian"                  exp(-(epsilon r)^2); "matern"                    Matern with smoothness options.nu = 0.5, 1.5 or 2.5 (the default; a kink at each point for 0.5), scale 1 / epsilon; "multiquadric"              sqrt(1 + (epsilon r)^2). epsilon defaults to 1 / (3 h), h the points' typical spacing (scale-free kernels ignore it). The weights solve a dense linear system (n^3 work: hundreds of points); for gaussian and matern in one or two dimensions above 1500 points (or options.method = "hodlr"), a HODLR compression (citrate.hodlr, accurate to options.tol, default 1e-10) instead, which handles thousands. options.smoothing: lambda added to the diagonal, fitting the values approximately (larger: smoother), for noisy values; "auto" chooses it by maximum likelihood (the values as a Gaussian process with this kernel; about 20 factorizations). Gaussian and matern systems always get at least the lambda their factorization needs to stay positive definite: n eps |K| dense, 2 tol |K| with HODLR.
