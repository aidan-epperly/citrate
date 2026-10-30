# Getting started

## Installing

citrate is plain Luau with no dependencies, and runs in Roblox and in standalone Luau.

- **Roblox:** sync `default.project.json` with [Rojo](https://rojo.space); it maps the `src`
  folder to one ModuleScript. Then `local citrate = require(path.to.citrate)`.
- **Standalone Luau:** `local citrate = require("./citrate/src")`.

```luau
local f64 = citrate.f64

local A = f64.matfrom({ { 0, 1 }, { 2, 3 } })
local x = f64.solve(nil, A, f64.from({ 1, 8 }))
print(f64.totable(x)) --> { 2.5, 1 }
```

## Modules

| Module | What it is |
| --- | --- |
| `f32`, `f64` | Real arrays (buffers) and column-major matrices, with dense linear algebra, structured and sparse matrices, real FFTs and cosine transforms |
| `c32`, `c64` | The same for complex arrays (interleaved real and imaginary parts), with FFTs of any length |
| `linalg` | Start here: solve equations, least squares, eigenvalues; methods chosen and answers checked |
| `pde` | Start here: grid Laplacians in any dimension; Poisson's equation, heat flow, vibration modes on meshes of any dimension (regions, solids, curves and surfaces) |
| `krylov` | Iterative solvers (CG, MINRES, GMRES, BiCGSTAB) and eigensolvers (Lanczos, Krylov-Schur) |
| `ode` | Initial value problems (explicit, stiff, second-order, symplectic), implicit mechanics, boundary value problems |
| `cheb` | Chebyshev approximation in the spirit of Chebfun |
| `rational` | Rational approximation by the AAA algorithm |
| `lightning` | Laplace and Poisson problems on polygons (Gopal and Trefethen's lightning solver) |
| `curve` | Bézier, spline, B-spline and NURBS curves in any representation |
| `optimize` | Scalar roots and minima, systems of equations, least squares and minimization |
| `random`, `stats` | Seeded random numbers, distributions, quasi-random points; descriptive statistics |
| `special`, `quad` | Special functions (gamma, erf, Bessel); adaptive integration |
| `graph` | Graphs as sparsity patterns: traversal, orderings for banded solvers, shortest paths |
| `spatial` | Uniform grids and k-d trees on point sets: radius, nearest and k nearest queries |
| `rotation` | Rotations in any representation (quaternions, frames, CFrames): interpolation, splines, averaging, fitting, integration |
| `surface`, `interp`, `hodlr` | Surfaces through grids of points; interpolation on grids and at scattered points; compressed kernel matrices for thousands of points |
| `xpbd` | Particles and compliant constraints by extended position-based dynamics |
| `complex64`, `complex32` | Complex scalars: `(re, im)` pairs, or a `vector` |
| `vec` | `vector` as a small real state space |

For Roblox, two components in `roblox/` (outside `src`, since they use Roblox's API):
`roblox/bones.luau` simulates `Bone`s on top of `xpbd` (dynamic tails, hair and antennae;
see `demos/bones`), and `roblox/path.luau` makes a smooth path through Attachments or Parts
and moves things along it by distance, with twist-free frames (see `demos/path`).

Each array module combines parts that can also be required on their own: for `f64`,
`array_f64`, `matrix_f64`, `gemm_f64`, `factor_f64`, `eigen_f64`, `matfun_f64`, `fft_f64`,
`structured_f64` and `sparse_f64`. The [reference](reference/README.md) follows the same
division.

## Representations

Algorithms are written once and work in whatever representation you hand them: a
`vector` (computed with native single-precision arithmetic), or buffers of single- or
double-precision real or complex numbers. The data stays in that representation for the
whole computation. The solvers in `krylov`, `ode` and `curve` take a `Space` (the module's
`space`, or `citrate.vec.space` for vectors) that says how to add, scale and measure states;
see [Writing generic algorithms](generic-algorithms.md).

## Conventions

- Indices are 1-based. Matrices are `{ rows, cols, data }`, stored column-major.
- Every operation that produces an array or matrix takes an `out` argument first. Pass a
  buffer to write into it (it may alias an input for elementwise operations) or `nil` to
  allocate. Always use the returned value.
- Scalars such as step sizes and tolerances are plain (double-precision) numbers. Complex
  scalars are `(re, im)` pairs for `c64` and `vector`s for `c32`.
- Buffer kernels compute in double precision and round once per stored value.
- Decompositions work in place (`lu(A)` overwrites `A` with its factors, in LAPACK's
  layouts), and each has a `_copy` version that leaves `A` alone. See
  [Dense linear algebra](linear-algebra.md).
- Transforms follow NumPy and SciPy: `fft` is unscaled and `ifft` divides by n; `dct` and
  `idct` use SciPy's definitions, unnormalized or with `norm = "ortho"`.
