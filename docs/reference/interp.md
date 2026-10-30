# Interpolation

Values on regular grids (bilinear, bicubic) and at scattered points (radial basis functions).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `grid`

```luau
grid(values: any, options: GridOptions?): Grid
```

An interpolant of values on a regular grid in any number of dimensions d. values is a nested table d deep (values[i][j] in 2-D, values[i][j][k] in 3-D, ...), an f64 Matrix (2-D), or an f64 array with options.shape = { n1, ..., nd } (the first index fastest). The value with indices (i1, ..., id) is at origin[a] + (i_a - 1) spacing[a] on each axis a: options.origin and options.spacing (default 0 and 1), or for the first three axes x0, y0, z0 and dx, dy, dz. options.method: "cubic" (the default)  Keys' cubic convolution (a = -1/2, as Catmull-Rom) along each axis (a tensor product: 4^d values a point): it passes through the values, its slope is continuous, and its error falls as the cube of the spacing; it can overshoot a little near sharp steps; "linear"               multilinear (2^d values a point): never overshoots, but the slope jumps at grid lines. Points outside the grid are clamped to its edge. Needs at least 2 points along each axis (3 for cubic; with 2, linear is used). Evaluation takes the d coordinates: G:at(x, y), G:at(x, y, z), ...; gradient and sample return the d partial derivatives. 2-D grids have their own evaluation (the general one gives the same values: tested).

### `rbf`

```luau
rbf(points: { any }, values: { number }, options: RbfOptions?): (Rbf, RbfInfo)
```

An interpolant of values at scattered points in any dimension d (vectors, d = 3, or tables or f64 arrays of d numbers) by radial basis functions: f(p) = sum of w_k phi(|p - p_k|) plus a linear polynomial in the coordinates the points span. Returns the interpolant and info: method ("dense" or "hodlr"), kernel, epsilon, smoothing (the lambda used) and residual (the largest misfit at the points, relative to the largest value; check it when the system is ill-conditioned). options.kernel: "thin_plate" (the default)  r^2 log r: in 2-D the surface through the points that bends least (in 3-D that is phi = r); reproduces linear data; "cubic"                     r^3; "gaussian"                  exp(-(epsilon r)^2); "matern"                    Matern with smoothness options.nu = 0.5, 1.5 or 2.5 (the default; a kink at each point for 0.5), scale 1 / epsilon; "multiquadric"              sqrt(1 + (epsilon r)^2). epsilon defaults to 1 / (3 h), h the points' typical spacing (scale-free kernels ignore it). The weights solve a dense linear system (n^3 work: hundreds of points); for gaussian and matern in one or two dimensions above 1500 points (or options.method = "hodlr"), a HODLR compression (citrate.hodlr, accurate to options.tol, default 1e-10) instead, which handles thousands. options.smoothing: lambda added to the diagonal, fitting the values approximately (larger: smoother), for noisy values; "auto" chooses it by maximum likelihood (the values as a Gaussian process with this kernel; about 20 factorizations). Gaussian and matern systems always get at least the lambda their factorization needs to stay positive definite: n eps |K| dense, 2 tol |K| with HODLR.
