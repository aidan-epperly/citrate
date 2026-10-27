# Dense linear algebra

Everything here is in the array modules (`f32`, `f64`, `c32`, `c64`); see the reference
pages for [factorizations](reference/factor.md), [eigenvalues](reference/eigen.md),
[matrix functions](reference/matfun.md), [structured matrices](reference/structured.md) and
[transforms](reference/fft.md).

## Solving

`solve(out, A, b, form)` takes a matrix or a factorization you already have; `form` says
which (`"matrix"`, the default, `"lu"`, `"chol"`, `"ldl"`, `"qr"`, `"qrp"` or `"svd"`).

- A square matrix is solved by LU, a tall one by QR least squares and a wide one by
  column-pivoted QR (the basic solution).
- An SVD (from `svd` with vectors) gives the pseudoinverse solution A⁺b, the shortest
  least-squares solution for any shape and rank (`svd_solve` takes a tolerance and also
  returns the residual norm and rank).
- `b` may be a vector or a matrix of right-hand sides, giving X with A X = B from one
  factorization. Nothing passed to `solve` is modified.
- `inv(A)` overwrites A with its inverse (`inv_copy` does not) and `pinv(A, tol?)` returns
  the pseudoinverse; solving is cheaper and more accurate when only A⁻¹b is needed.
- `solve_refined` refines a single-precision factorization to double-precision accuracy
  (plain, or GMRES-IR with `method = "gmres"`).

```luau
local F = f64.lu_copy(A)                 -- factor once
local x = f64.solve(nil, F, b, "lu")     -- solve many times
local X = f64.solve(nil, A, B)           -- or a matrix of right-hand sides at once
```

## Factorizations work in place

`lu(A)`, `chol(A)`, `qr(A)`, `qrp(A)`, `ldl(A)` and `hess(A)` overwrite `A` with their
factors (LAPACK's layouts) and return them inside the factorization; `lu_copy`,
`chol_copy` and so on factor a copy and leave `A` alone. `eigh(A)` overwrites `A` with the
eigenvectors (`eigh_copy` does not); `schur(A)` and `eig(A)` overwrite it with the Schur
form, and `svd(A)` with U when A has at least as many rows as columns. `det` never modifies
`A`. Structured factorizations (`tridiag_lu`, `tridiag_ldl`, `band_lu`, `band_chol`) follow
the same rule, with `_copy` versions and `_solve` functions.

Factoring again and again (a simulation's matrix each frame) need not allocate: every
factorization takes an earlier one of the same size as a last argument, `reuse`, and
refills its arrays instead of making new ones (`F = f64.lu_copy(A, F)`; the result is that
same `F`). Pivots and permutations (`F.piv`, `F.perm`) are buffers of 32-bit integers,
1-based as in LAPACK: entry k is `buffer.readi32(F.piv, 4 * (k - 1))`.

`qr_full` keeps an explicit Q for updates (`qr_update`, `qr_insert_col`,
`qr_delete_col`), and `chol_update` and `chol_downdate` change a Cholesky factor by a rank
one term.

## Eigenvalues and singular values

- `eigh`: symmetric (Hermitian) eigenvalues and eigenvectors, reading the lower triangle.
- `schur`, `eig`: the Schur form and general eigenvalues and eigenvectors. Real matrices get
  complex results; `eig` balances first (permuting and scaling), as LAPACK does.
- `svd`, `svd_solve`, `pinv`; `rank`, `orth`, `null_space`, `cond`, `rcond` (an estimate).
- `polar` (by the QDWH iteration), `ldl` (Bunch-Kaufman, with `ldl_inertia`).
- `eigh_gen` (symmetric-definite pencils) and `eig_gen` (QZ, with infinite eigenvalues).
- `tridiag_eigh` for symmetric/Hermitian tridiagonal matrices.
- Low rank: `rsvd` (randomized SVD), `rpcholesky` and matrix-free `rpcholesky_columns`
  (randomly pivoted Cholesky, for kernel matrices), `rpqr`.

`eigh`, `schur`, `eig` and `svd` return all-NaN results when the input holds a NaN or an
infinity (for `eigh`, in the lower triangle it reads). Like LAPACK's drivers, they first
scale a matrix whose entries are near the ends of the exponent range by a power of 2, so
tiny or huge matrices lose no accuracy.

## Matrix functions and equations

`expm` (scaling and squaring), `sqrtm` and `logm` (Schur method), `sylvester` and
`lyapunov` (Bartels-Stewart), `dlyapunov` (the discrete-time equation).

## Structured matrices

Triangular solve and multiply on dense matrices (one triangle read); tridiagonal matrices
(`tridiagonal`) with pivoted LU and LDL^H; banded matrices in LAPACK band storage
(`banded`) with pivoted LU and Cholesky; symmetric (Hermitian) banded matrices in the
storage of LAPACK's pbtrf (`symbanded`: the diagonal and kd subdiagonals, kd + 1 rows
against a `banded`'s 3 kd + 1, which keeps room for LU's fill-in), for `band_chol`; circulant (`circulant`) and Toeplitz (`toeplitz`)
matrices with FFT-based products, FFT circulant solves and Levinson Toeplitz solves.

## Fourier and cosine transforms

The complex modules have FFTs of any length (`fft`, `ifft`, `fft2`, `ifft2`, with reusable
`fft_plan`s; mixed radix 2, 3, 4, 5, generic small primes, Bluestein otherwise). The real
modules have `rfft`, `irfft` and discrete cosine transforms of types 1 to 4 (`dct`,
`idct`). `fft`, `ifft`, `dct` and `idct` work in place, with `_copy` versions.
