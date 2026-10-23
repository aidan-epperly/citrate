# citrate

A numerical math library for Luau that runs in Roblox and in standalone Luau.

Algorithms are written once and work in whatever representation you hand them: a
`vector` (computed with native single-precision arithmetic), or buffers of single- or
double-precision real or complex numbers. The data stays in that representation for the
whole computation.

## Contents

| Module | What it is |
| --- | --- |
| `f32`, `f64` | Real arrays (buffers) and column-major matrices: elementwise kernels, `axpy`, `dot`, `norm`, `matvec`, `matmul`, `transpose`, LU with partial pivoting, Cholesky, `solve` (one or many right-hand sides), `det`, `inv`, Householder QR, column-pivoted QR with numerical rank, least squares, Hessenberg reduction, symmetric eigenvalues and eigenvectors (`eigh`), real Schur form and general eigenvalues and eigenvectors (`schur`, `eig`, with complex results; `eig` balances the matrix first (permuting and scaling), as LAPACK does), singular value decomposition (`svd`, with pseudoinverse solves by `svd_solve` and the pseudoinverse `pinv`), the polar decomposition (`polar`, by the QDWH iteration), real FFTs (`rfft`, `irfft`) and discrete cosine transforms of types 1 to 4 (`dct`, `idct`) |
| `c32`, `c64` | The same for complex arrays (interleaved real and imaginary parts; `eigh` is Hermitian, `schur` is the complex Schur form), plus `eigvecs_from_schur`, FFTs of any length (`fft`, `ifft`, `fft2`, `ifft2`, reusable `fft_plan`s), `conj`, `adjoint`, `dotc`, complex `cscale`/`caxpy` |
| Structured matrices (all four array modules) | Triangular solve and multiply on dense matrices (one triangle read); tridiagonal matrices (`tridiagonal`) with pivoted LU and LDL^H; banded matrices in LAPACK band storage (`banded`) with pivoted LU and Cholesky; circulant (`circulant`) and Toeplitz (`toeplitz`) matrices with FFT-based products, FFT circulant solves and Levinson Toeplitz solves |
| Sparse matrices (all four array modules) | Compressed sparse column storage (`sparse` from triplets, duplicates summed): products with vectors and dense matrices, adjoint products, transpose/adjoint, add, diagonal; preconditioners: Jacobi, SSOR, ILU(0) and IC(0) |
| `krylov` | Iterative solvers generic over the representation and the operator: `cg`, `minres` (any `Space`, including `vector`; symmetric positive definite preconditioners), `gmres` and `bicgstab` (array modules, which provide the complex extension of `Space`; right preconditioning) |
| `cheb` | Chebyshev approximation in the spirit of Chebfun: Chebyshev points and barycentric interpolation (`bary`, `chebweights`, `bary_weights`), coefficient transforms, Clenshaw evaluation, adaptive construction to machine precision (`fun`, with Aurentz and Trefethen's `standard_chop`), arithmetic, derivatives, integrals, norms, roots (colleague matrices with balancing and recursive splitting), extrema, the Chebyshev differentiation matrix, Clenshaw-Curtis and Gauss-Legendre quadrature |
| `curve` | Curves with points in any `Space` (`vector`s natively, buffers of any dimension): Bézier curves of any degree, polynomial or rational, and composite paths; cubic Hermite curves; Catmull-Rom (uniform, centripetal, chordal) and Kochanek-Bartels splines; interpolating cubic splines (natural, clamped, not-a-knot, periodic); B-splines and NURBS. Evaluation and derivatives, speed, curvature, torsion, Frenet and rotation-minimizing frames, arclength reparameterization by Chebyshev interpolation, even sampling by arclength, closest points (a global search, or a fast local Newton search for tracking), tight bounding boxes and splitting |
| `complex64` | Double-precision complex scalars as `(re, im)` pairs of numbers |
| `complex32` | Single-precision complex scalars stored in a `vector` |
| `vec` | `vector` as a real state space of dimension at most 3 |

```luau
local citrate = require("./citrate/src")
local f64 = citrate.f64

local A = f64.matfrom({ { 0, 1 }, { 2, 3 } })
local x = f64.solve(nil, A, f64.from({ 1, 8 }))
print(f64.totable(x)) --> { 2.5, 1 }
```

## Conventions

- Indices are 1-based. Matrices are `{ rows, cols, data }`, stored column-major.
- Every operation that produces an array or matrix takes an `out` argument first. Pass a
  buffer to write into it (it may alias an input for elementwise operations) or `nil` to
  allocate. Always use the returned value.
- Scalars such as step sizes and tolerances are plain (double-precision) numbers. Complex
  scalars are `(re, im)` pairs for `c64` and `vector`s for `c32`.
- Buffer kernels compute in double precision and round once per stored value.
- Transforms follow NumPy and SciPy: `fft` is unscaled and `ifft` divides by n; `dct` and
  `idct` use SciPy's definitions, unnormalized or with `norm = "ortho"`. `fft`, `ifft`,
  `dct` and `idct` work in place, with `_copy` versions.
- Decompositions work in place: `lu(A)`, `chol(A)`, `qr(A)`, `qrp(A)` and `hess(A)` overwrite `A` with their factors
  (LAPACK's layouts) and return it inside the factorization. `lu_copy`, `chol_copy`,
  `qr_copy`, `qrp_copy` and `hess_copy` factor a copy and leave `A` alone. `eigh(A)` overwrites `A` with the eigenvectors
  (`eigh_copy` does not); `schur(A)` and `eig(A)` overwrite it with the Schur form, and
  `svd(A)` with U when A has at least as many rows as columns. `det` never modifies `A`.
  `qr_apply_q` and `qr_apply_qh` work in place on their vector.
- `eigh`, `schur`, `eig` and `svd` return all-NaN results (values, vectors and the
  overwritten `A`) when the input holds a NaN or an infinity (for `eigh`, in the lower
  triangle it reads). Like LAPACK's drivers, they first scale a matrix whose entries are
  near the ends of the exponent range by a power of 2, so tiny or huge matrices lose no
  accuracy.
- Structured factorizations follow the same rule: `tridiag_lu`, `tridiag_ldl`, `band_lu`
  and `band_chol` work in place, with `_copy` versions and `_solve` functions.
- `solve(out, A, b, form)` takes a matrix or a factorization you already have; `form`
  says which (`"matrix"`, the default, `"lu"`, `"chol"`, `"qr"`, `"qrp"` or `"svd"`). A square matrix is solved by
  LU, a tall one by QR least squares and a wide one by column-pivoted QR (the basic
  solution). An SVD (from `svd` with vectors) gives the pseudoinverse solution A⁺b, the
  shortest least-squares solution for any shape and rank (`svd_solve` takes a tolerance
  and also returns the residual norm and rank). b may be a vector or a matrix of right-hand sides (giving a matrix X with A X = B,
  from one factorization). Nothing passed to `solve` is modified. `inv(A)` overwrites A
  with its inverse (LU, as LAPACK's getri; `inv_copy` does not), and `pinv(A, tol?)`
  returns the pseudoinverse; solving is cheaper and more accurate when only A⁻¹b is needed.

## Iterative solvers

```luau
local f64, krylov = citrate.f64, citrate.krylov
local S = f64.sparse(n, n, rows, cols, f64.from(values))
local x, info = krylov.cg(f64.space, function(out, v)
	return f64.sparse_matvec(out, S, v)
end, b, { tol = 1e-10, precond = f64.sparse_jacobi(S) })
-- info.converged, info.iterations, info.residual (the true |b - A x| / |b|)

-- Incomplete factorizations make stronger preconditioners:
local F = f64.sparse_ic0_copy(S)
krylov.cg(f64.space, A, b, { precond = function(out, r) return f64.sparse_ic0_solve(out, F, r) end })
```

## Chebyshev approximation

```luau
local cheb = citrate.cheb
local F = cheb.fun(function(x) return math.exp(x) * math.sin(5 * x) end, { 0, 2 })
print(cheb.length(F), F.happy)   --> the degree + 1 needed for machine precision, true
print(cheb.eval(F, 1.3))         --> F(1.3)
print(cheb.sum(F))               --> the integral over [0, 2]
local r = cheb.roots(F)          --> its roots, as an f64 array
local hi, where = cheb.max(F)
```

## Curves

```luau
local curve = citrate.curve
local path = curve.catmull_rom({ vector.create(0, 0, 0), vector.create(4, 2, 0), vector.create(8, 0, 3) })
local L = curve.length(path)
local p = curve.at_length(path, 0.25 * L)    -- the point a quarter of the way along
local points, params = curve.sample(path, 20) -- 20 points evenly spaced by arclength
local frames = curve.frames(path, params)     -- twist-free frames there, for a tube or a road
local t, q, distance = curve.closest(path, vector.create(3, 3, 3))
-- Tracking something that moves a little each frame: start from last frame's answer.
-- (closest_near is local and about 25 times faster; closest is global.)
t = curve.closest_near(path, vector.create(3.1, 3, 3), t)
```

Every family is converted to one piecewise Bézier form, so all of these work the same way
on every curve. With `vector`s, points are computed in double precision and rounded once.
Other representations pass their space: `curve.cubic_spline(points, { space = citrate.f64.space })`.
Arclength is built on first use: on each segment the speed is a Chebyshev interpolant
(split at cusps), its integral is the arclength, and the inverse t(s) is a Chebyshev
interpolant too, so `at_length` costs a binary search and a short Clenshaw sum.

## Writing generic algorithms

An algorithm that should run on every representation takes a `Space<T>` (see
`src/types.luau`) and touches its state only through it:

```luau
local function combination<T>(S: citrate.Space<T>, x: T, y: T): T
	local acc = S.copy(S.similar(x), x)
	return S.axpy(acc, -2, y, acc)
end

combination(citrate.vec.space, vector.create(1, 2, 3), vector.create(0, 1, 0))
combination(citrate.f32.space, citrate.f32.from({ 1, 2, 3 }), citrate.f32.from({ 0, 1, 0 }))
```

## Development

`src/f32.luau`, `src/f64.luau`, `src/c32.luau` and `src/c64.luau` are generated from
`templates/` by `python tools/generate.py`; edit the templates. One module per element
type is needed because Luau only optimizes `buffer.readf64` and friends into fast calls
when they are written out literally.

Run the tests and the type checker with the [Luau](https://github.com/luau-lang/luau/releases)
command-line tools:

```sh
python tools/generate.py --check
luau-analyze src/*.luau tests/*.luau bench/*.luau
luau tests/run.luau
```

## AI acknowledgement

citrate was written with Claude, Anthropic's AI model, working in Claude Code. Aidan
Epperly set the goals and made the design decisions: algorithms that are generic over how
data is represented, the precision and storage conventions, in-place decompositions with
copying versions, the `solve` interface and which algorithms to build. He also chose
between alternatives using the benchmarks in `bench/`. Claude wrote the code, the tests,
the benchmarks and this documentation.

**The code has not yet been verified by a human.** The test suite passes, but until that
review is done, check results that matter independently.

## License

citrate is licensed under the [Mozilla Public License 2.0](LICENSE).
