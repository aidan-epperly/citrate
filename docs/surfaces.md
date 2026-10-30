# Surfaces and interpolation

Reference: [surface](reference/surface.md), [interp](reference/interp.md).

## Surfaces through grids of points

```luau
local surface = citrate.surface
-- grid[i][j]: a vector; rows i = 1..m, columns j = 1..n
local S = surface.new(grid, { kind = "spline" })
local p = surface.eval(S, u, v)
local n = surface.normal(S, u, v)
local points, triangles, normals = surface.mesh(S, 32, 32)   -- for an EditableMesh
```

| kind | the surface | parameters |
| --- | --- | --- |
| `"bspline"` (default) | smooth, following the grid loosely, through its four corners | u in [0, m - p], v in [0, n - p] (degree p = 3 by default) |
| `"bezier"` | one patch of degree (m - 1, n - 1) | [0, 1] by [0, 1] |
| `"spline"` | the cubic spline surface through every grid point | u = i - 1 and v = j - 1 at point (i, j) |

A tensor-product surface is a curve of curves: each grid row is a curve in v, and the
surface at (u, v) is the curve through the rows' points at v, evaluated at u. So these
surfaces are built from [curves](curves.md), with their accuracy, and work in any space
the curves do (`options.space`); normals and meshes need `vector`s.

The curve across the rows is the costly part of a point: about 29 us for a 10 by 10
B-spline. It is kept for the last v, so points along u at a fixed v cost 0.2 us, and
`surface.mesh` builds it once per row of the mesh: 2.7 ms for a 33 by 33 mesh with normals
(about 2.5 us a point).

## Values on a grid

```luau
local interp = citrate.interp
-- heights[i][j] at (x0 + (i - 1) dx, y0 + (j - 1) dy)
local H = interp.grid(heights, { dx = 4, dy = 4 })
local z = H:at(x, y)
local zx, zy = H:gradient(x, y)       -- the slope, for normals or movement
local z, zx, zy = H:sample(x, y)      -- both, for the price of one

-- any number of dimensions: nested tables d deep, or an f64 array with its shape
local V = interp.grid(density, { spacing = { 2, 2, 1 } })   -- density[i][j][k]
local rho, rx, ry, rz = V:sample(x, y, z)
```

A grid has any number of axes: values as nested tables (`values[i][j][k]` in 3-D), or an
f64 array with `options.shape = { n1, ..., nd }` (the first index fastest), and positions
`options.origin` and `options.spacing` per axis (or `x0`, `y0`, `z0` and `dx`, `dy`, `dz`).
Each method is the one-dimensional rule along every axis in turn (4^d values a cubic
point, 2^d a linear one). Two-dimensional grids have their own evaluation, which the
general one matches to rounding (tested).

`method = "cubic"` (the default) is Keys' cubic convolution, the Catmull-Rom rule in two
dimensions: it passes through the values, its slope is continuous, it reproduces quadratic
data exactly (Keys' edge rule included), and its error falls as the cube of the spacing.
`method = "linear"` is bilinear: it never overshoots, but its slope jumps at grid lines.
Outside the grid the coordinates are clamped to its edge. On a 100 by 100 grid a cubic
value costs 0.12 us natively (0.42 interpreted) and a bilinear one 0.05 (0.13).

The difference in slope matters for anything moving on the surface. In `demos/terrain`
marbles roll without friction over the same grid of heights, both ways: on the bilinear
surface each grid line they cross is a small bump that costs them speed, whatever the time
step, and they stop within seconds (81% of their energy lost in 10 s); on the cubic one
they keep rolling (8% lost in 10 s at 4 substeps a frame, halving as the substeps double).

## Values at scattered points

```luau
local R, info = interp.rbf(points, values)   -- points: vectors, or tables or f64 arrays of d numbers
local v = R:at(vector.create(1, 2, 0))
local gx, gy, gz = R:gradient(vector.create(1, 2, 0))   -- one derivative per coordinate
-- info.method ("dense" or "hodlr"), info.smoothing (the lambda used), info.residual
```

Radial basis function interpolation: f(p) is a weighted sum of a function of the distance
to each point, plus a linear polynomial in the coordinates the points span (points along a
diagonal span one), with the weights from a linear system. Points have any number of
coordinates d (vectors 3); evaluation points take the same form. Two and three coordinates
have their own evaluation loops (20 to 25% faster than the general one).

| `kernel` | phi(r) | notes |
| --- | --- | --- |
| `"thin_plate"` (default) | r^2 log r | in 2-D, the surface through the points that bends least; no scale to choose |
| `"cubic"` | r^3 | no scale to choose |
| `"gaussian"` | exp(-(epsilon r)^2) | very accurate on smooth data, very ill-conditioned |
| `"matern"` | Matern, smoothness `nu` = 0.5, 1.5 or 2.5 (default) | scale 1 / epsilon; a kink at each point for 0.5 |
| `"multiquadric"` | sqrt(1 + (epsilon r)^2) | |

`epsilon` defaults to 1 / (3 h), h the points' typical spacing, so it follows the data's
units: on 100 points spaced about 10 studs apart a fixed epsilon = 1 put a narrow bump at
each point and gave 98% error between them, against 0.15% at the default.

The system is ill-conditioned for the Gaussian especially (condition number 1e16 or more
at a thousand points), so check `info.residual`, the largest misfit at the points relative
to the largest value. Gaussian and Matern systems get the small ridge their factorization
needs to stay positive definite (n eps |K| for a dense solve), so they always solve.

`smoothing = lambda > 0` fits noisy values approximately (larger: smoother), and
`smoothing = "auto"` chooses lambda by maximum likelihood, treating the values as a
Gaussian process with this kernel (Gaussian and Matern only; about 18 factorizations). On
400 points with noise of 0.05 the auto fit's error was 0.027, against 0.12 (Matern) and
1.6 (Gaussian) for fitting the noise exactly.

### Many points: HODLR

A dense solve costs n^3 / 3 (1.3 ms for 100 points, 0.47 s for 1000, 3.6 s for 2000), so
above 1500 points Gaussian and Matern interpolants in one or two dimensions are solved
through a HODLR compression of the kernel matrix instead (`citrate.hodlr`; force either
with `method = "dense"` or `"hodlr"`). The matrix's off-diagonal blocks, between halves of
the points split recursively, are nearly low rank for these kernels, and are stored as
interpolative decompositions. Each half's skeleton points are chosen from a random sketch
of its interactions with the nearby points of the other half and with proxy points spread
over the rest of it, by LU with partial pivoting and then a small column-pivoted QR (as good
as a column-pivoted QR of the sketch, to within 1.4 times the error, and cheaper). For a
Gaussian, points too far from the other half to interact above the tolerance are left out. Factoring is symmetric (each step a triangular solve or a small Cholesky
factorization), which keeps it accurate on Gaussian matrices; a first version using the
Sherman-Morrison-Woodbury formula left residuals of 1e-2 there, or failed to factor.

Compression to `tol` (default 1e-10) perturbs the matrix by about tol |K|, so the ridge is
at least 2 tol |K|. With it, fits were as accurate as dense solves with the same ridge
(to the digits shown, Gaussian and Matern, tol 1e-8 to 1e-12), and refitting with
another `smoothing` reuses the compression. Natively, on random points in a square:

| points | Gaussian: dense / HODLR | Matern 5/2: dense / HODLR |
| --- | --- | --- |
| 1000 | 0.47 / 0.78 s | 0.47 / 0.50 s |
| 1500 | 1.6 / 1.5 s | 1.6 / 1.1 s |
| 2000 | 3.6 / 2.5 s | 3.6 / 1.7 s |
| 3000 | - / 4.7 s | - / 3.4 s |

At 4000 points a dense Cholesky factorization alone takes about 27 s and its matrix 122 MB;
HODLR took 1.3 to 7.1 s and 24 to 81 MB, the larger figures for the narrower kernels (a
scale of 3 spacings), whose blocks have higher rank. In 3-D the ranks grow like n^(2/3) and
compression saves little, so 3-D points stay dense. Evaluating the interpolant still costs
a kernel evaluation per point (70 us for 4000 points).

Distances are computed in double precision even when the points are `vector`s.
