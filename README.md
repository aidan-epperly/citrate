# citrate

A numerical math library for Luau that runs in Roblox and in standalone Luau.

Algorithms are written once and work in whatever representation you hand them: a
`vector` (computed with native single-precision arithmetic), or buffers of single- or
double-precision real or complex numbers. The data stays in that representation for the
whole computation.

## Contents

| Module | What it is |
| --- | --- |
| `f32`, `f64` | Real arrays (buffers) and column-major matrices: elementwise kernels, `axpy`, `dot`, `norm`, `matvec`, `matmul`, `transpose`, LU with partial pivoting, Cholesky, `solve` (one or many right-hand sides), `det`, `inv`, Householder QR, column-pivoted QR with numerical rank, least squares, Hessenberg reduction, symmetric eigenvalues and eigenvectors (`eigh`), real Schur form and general eigenvalues and eigenvectors (`schur`, `eig`, with complex results; `eig` balances the matrix first (permuting and scaling), as LAPACK does), singular value decomposition (`svd`, with pseudoinverse solves by `svd_solve` and the pseudoinverse `pinv`), the polar decomposition (`polar`, by the QDWH iteration), the symmetric indefinite factorization (`ldl`, Bunch-Kaufman, with `ldl_solve` and `ldl_inertia`), generalized eigenproblems (`eigh_gen` for symmetric-definite pencils; `eig_gen` by the QZ algorithm, with infinite eigenvalues), Sylvester and Lyapunov equations (`sylvester`, `lyapunov`, Bartels-Stewart; `dlyapunov` for the discrete-time equation), iterative refinement from a single-precision factorization (`solve_refined`, plain or GMRES-IR), the symmetric/Hermitian tridiagonal eigensolver (`tridiag_eigh`), updatable factorizations (`qr_full` with `qr_update`, `qr_insert_col` and `qr_delete_col`; `chol_update` and `chol_downdate`), the randomized SVD (`rsvd`), randomly pivoted Cholesky and QR for low-rank approximation (`rpcholesky`, matrix-free `rpcholesky_columns` for kernel matrices, `rpqr`), the matrix exponential, square root and logarithm (`expm` by scaling and squaring, `sqrtm` and `logm` by the Schur method), matrix norms (`matnorm`: 1, 2, inf, Frobenius), condition numbers (`cond`, and the cheap estimate `rcond`), `rank`, `orth`, `null_space`, `slogdet`, utilities (`trace`, `diag`, `diagm`, `kron`, `outer`, `ger`, `submatrix`, `setsubmatrix`, `hstack`, `vstack`, `matvec_adjoint`, `gemm`, `lu_solve_adjoint`), real FFTs (`rfft`, `irfft`) and discrete cosine transforms of types 1 to 4 (`dct`, `idct`) |
| `c32`, `c64` | The same for complex arrays (interleaved real and imaginary parts; `eigh` is Hermitian, `schur` is the complex Schur form), plus `eigvecs_from_schur`, FFTs of any length (`fft`, `ifft`, `fft2`, `ifft2`, reusable `fft_plan`s), `conj`, `adjoint`, `dotc`, complex `cscale`/`caxpy` |
| Structured matrices (all four array modules) | Triangular solve and multiply on dense matrices (one triangle read); tridiagonal matrices (`tridiagonal`) with pivoted LU and LDL^H; banded matrices in LAPACK band storage (`banded`) with pivoted LU and Cholesky; circulant (`circulant`) and Toeplitz (`toeplitz`) matrices with FFT-based products, FFT circulant solves and Levinson Toeplitz solves |
| Sparse matrices (all four array modules) | Compressed sparse column storage (`sparse` from triplets, duplicates summed): products with vectors and dense matrices, adjoint products, transpose/adjoint, add, diagonal; preconditioners: Jacobi, SSOR, ILU(0) and IC(0) |
| `krylov` | Iterative solvers generic over the representation and the operator: `cg`, `minres` (any `Space`, including `vector`; symmetric positive definite preconditioners), `gmres` and `bicgstab` (array modules, which provide the complex extension of `Space`; right preconditioning); eigensolvers for a few eigenvalues of large or matrix-free operators: `eigsh` (thick-restart Lanczos, Hermitian, any `Space`) and `eigs` (Krylov-Schur, general; real spaces get complex eigenvectors as (re, im) pairs) |
| `cheb` | Chebyshev approximation in the spirit of Chebfun: Chebyshev points and barycentric interpolation (`bary`, `chebweights`, `bary_weights`), coefficient transforms, Clenshaw evaluation, adaptive construction to machine precision (`fun`, with Aurentz and Trefethen's `standard_chop`), arithmetic, derivatives, integrals, norms, roots (colleague matrices with balancing and recursive splitting), extrema, the Chebyshev differentiation matrix, Clenshaw-Curtis and Gauss-Legendre quadrature |
| `curve` | Curves with points in any `Space` (`vector`s natively, buffers of any dimension): Bézier curves of any degree, polynomial or rational, and composite paths; cubic Hermite curves; Catmull-Rom (uniform, centripetal, chordal) and Kochanek-Bartels splines; interpolating cubic splines (natural, clamped, not-a-knot, periodic); B-splines and NURBS. Evaluation and derivatives, speed, curvature, torsion, Frenet and rotation-minimizing frames, arclength reparameterization by Chebyshev interpolation, even sampling by arclength, closest points (a global search, or a fast local Newton search for tracking), tight bounding boxes and splitting |
| `ode` | Ordinary differential equations on any `Space` (`vector`s stay vectors): explicit Runge-Kutta methods (Euler, midpoint, Heun, RK4; adaptive Bogacki-Shampine 3(2), Dormand-Prince 5(4) and DOP853) with per-component error control, stiff solvers (Radau IIA 5, the Rosenbrock method Rodas5P, and variable-order NDF/BDF with dense, banded, sparse or matrix-free Newton-Krylov linear algebra and complex states; Jacobians from the user or by grouped finite differences; mass matrices and index-1 DAEs), automatic switching between Dormand-Prince and a stiff method as stiffness comes and goes, boundary value problems of any order by Chebyshev collocation returning Chebfuns (`bvp`), second-order equations y'' = f(t, y, y') (`solve_second_order`: Runge-Kutta-Nystrom pairs DPRKN6, DPRKN8 and DPRKN12, or any method by reduction), dense output, events, output times and backward integration (`solve`), integrators that advance by a frame time (`integrator`), and symplectic integrators for mechanics q'' = a(t, q) (`mechanics`, `solve_mechanics`: velocity Verlet, Yoshida 4 and 6, PEFRL), whose energy error stays bounded over long runs |
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
  says which (`"matrix"`, the default, `"lu"`, `"chol"`, `"ldl"`, `"qr"`, `"qrp"` or `"svd"`). A square matrix is solved by
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

## Differential equations

```luau
local ode = citrate.ode
-- A damped spring on a vector state (position, velocity): it stays a vector throughout.
local spring = function(t, y, _) return vector.create(y.y, -4 * y.x - 0.1 * y.y, 0) end
local sol = ode.solve(citrate.vec.space, spring, { 0, 10 }, vector.create(1, 0, 0), { rtol = 1e-6 })
print(sol.at(2.5))                       -- dense output anywhere in [0, 10]

-- Advancing a simulation by the frame time, with error control inside each frame:
local it = ode.integrator(citrate.vec.space, spring, 0, vector.create(1, 0, 0))
it:advance(1 / 60)                        -- it.t, it.y

-- Mechanics (orbits, springs, game physics): symplectic steps keep the energy bounded.
local orbit = ode.mechanics(citrate.vec.space, function(t, q, _) return -q / vector.magnitude(q) ^ 3 end,
	0, vector.create(1, 0, 0), vector.create(0, 1, 0), { method = "pefrl" })
orbit:step(0.01)                          -- orbit.q, orbit.v
```

Buffers work the same way with their module's space (`citrate.f64.space`, ...); the right-hand
side may write into its `out` argument instead of allocating.

Stiff problems (chemical kinetics, discretized diffusion, circuits) take a stiff method:
`"rodas5p"` (Rosenbrock, the fastest on small and medium systems), `"radau"` (the most
robust, also for stiff oscillations) or `"bdf"` (cheapest per step on large systems; sparse
Jacobians via `jac_sparsity` and GMRES with ILU(0), or matrix-free Newton-Krylov with
`linear = "krylov"`). Each takes an optional Jacobian `jac`, its band `jac_band = { kl, ku }`
and a mass matrix `mass` for M y' = f(t, y), including differential-algebraic equations
(singular M). `method = "auto"` starts with Dormand-Prince and switches to a stiff method
(`stiff_method`, default `"rodas5p"`) when it detects stiffness, and back when it passes:

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

Second-order equations y'' = f(t, y, y') go through `ode.solve_second_order`, which returns
positions and velocities. For forces that do not depend on the velocity, the
Runge-Kutta-Nystrom pairs `"dprkn6"`, `"dprkn8"` and `"dprkn12"` reach a given accuracy with
fewer evaluations than a Runge-Kutta method on the first-order system; any other method
(including the stiff ones, for arrays) works by that reduction:

```luau
local orbit = ode.solve_second_order(citrate.vec.space, function(t, q, v, _)
	return -q / vector.magnitude(q) ^ 3
end, { 0, 100 }, vector.create(1, 0, 0), vector.create(0, 1, 0), { method = "dprkn8", rtol = 1e-10 })
local q, v = orbit.at(50)
```

Boundary value problems are solved by Chebyshev collocation with damped Newton iteration and
adaptive resolution. Each unknown function u_j satisfies an equation of its own order,
u_j^(k_j) = f_j(x, z), where z lists every function with its derivatives below their orders;
the boundary conditions are bc(z(a), z(b)) = 0. Higher-order equations are solved directly,
without introducing derivatives as unknowns, and each solution comes back as a Chebfun (see
`cheb`). Passing a number m instead of the orders means a first-order system of m equations.

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

The same suite runs inside Roblox through Open Cloud: `tools/roblox_test.py` builds a test
place with [Rojo](https://rojo.space) (`tests.project.json`), saves it as a version of a place
you own, runs the tests there with the Luau Execution API and prints the results. It reads
`ROBLOX_API_KEY` (a key with universe-places write and luau-execution-sessions read and write
on that experience), `ROBLOX_UNIVERSE_ID` and `ROBLOX_PLACE_ID` from the environment; saving
replaces the place's contents, so use a place set aside for testing. `--script file.luau`
runs another script instead. Roblox servers flush subnormal numbers to zero, so the few
checks of behavior at subnormal magnitudes are skipped there. To use citrate in a Roblox
project, sync `default.project.json` (the `src` folder as one ModuleScript).

## Third-party coefficients

Some method coefficients are copied from reference implementations rather than derived
here; each table says so in a comment:

- from [SciPy](https://github.com/scipy/scipy) (BSD-3-Clause; its notice is in
  [`licenses/SciPy-LICENSE.txt`](licenses/SciPy-LICENSE.txt)): the DOP853 coefficients
  (`dop853_coefficients.py`), the dense-output coefficients of Dormand-Prince 5(4) and
  Bogacki-Shampine 3(2) (`rk.py`), the Radau IIA 5 constants (`radau.py`, from Hairer
  and Wanner's RADAU5) and the NDF coefficients and difference updates (`bdf.py`), all in
  `src/ode.luau` and `src/odestiff.luau`;
- from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) (MIT; its notice is in
  [`licenses/OrdinaryDiffEq.jl-LICENSE.md`](licenses/OrdinaryDiffEq.jl-LICENSE.md)): the
  Rodas5P tableau and dense-output coefficients (Steinebach 2023;
  `rosenbrock_tableaus.jl`), in `src/odestiff.luau`, and the Runge-Kutta-Nystrom pairs
  DPRKN6 (with its dense output), DPRKN8 and DPRKN12 of Dormand, El-Mikkawy and Prince
  (`rkn_tableaus.jl`, evaluated in high precision from their exact forms), in
  `src/odenystrom.luau`.

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
