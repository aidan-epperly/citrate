# Structured matrices

Triangular, tridiagonal, banded, circulant and Toeplitz matrices. Parts `structured_f32`, `structured_f64`, `structured_c32` and `structured_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `triangular_solve`

```luau
triangular_solve(out: buffer?, A: Matrix, b: buffer, uplo: Uplo, unit: boolean?): buffer
```

Solves T x = b, where T is the upper or lower triangle of A (the other triangle is not read); with unit, the diagonal is taken to be 1 and not read either. In place if out is b.

### `triangular_matvec`

```luau
triangular_matvec(out: buffer?, A: Matrix, x: buffer, uplo: Uplo, unit: boolean?): buffer
```

T x for the upper or lower triangle T of A (as in triangular_solve).

### `tridiagonal`

```luau
tridiagonal(lower: buffer, diag: buffer, upper: buffer): Tridiagonal
```

A tridiagonal matrix from its subdiagonal (n-1), diagonal (n) and superdiagonal (n-1). The arrays are used as they are (not copied).

### `tridiag_copy`

```luau
tridiag_copy(T: Tridiagonal): Tridiagonal
```

A copy of T.

### `tridiag_matrix`

```luau
tridiag_matrix(T: Tridiagonal): Matrix
```

The dense matrix.

### `tridiag_matvec`

```luau
tridiag_matvec(out: buffer?, T: Tridiagonal, x: buffer): buffer
```

T x

### `tridiag_lu`

```luau
tridiag_lu(T: Tridiagonal): TridiagonalLU
```

LU factorization with partial pivoting (LAPACK's gttrf), in place: T's arrays are overwritten with the factors. tridiag_lu_copy leaves T unchanged.

### `tridiag_lu_copy`

```luau
tridiag_lu_copy(T: Tridiagonal): TridiagonalLU
```

tridiag_lu of a copy of T; T is not modified.

### `tridiag_lu_solve`

```luau
tridiag_lu_solve(out: buffer?, F: TridiagonalLU, b: buffer): buffer
```

Solves T x = b given F = tridiag_lu(T). In place if out is b.

### `tridiag_ldl`

```luau
tridiag_ldl(T: Tridiagonal): TridiagonalLDL
```

LDL^T factorization of a symmetric positive definite tridiagonal matrix (LAPACK's pttrf), in place: only T.diag and T.lower are read; they are overwritten with D and the subdiagonal of L. definite is false if T is not positive definite (T is then partly overwritten). tridiag_ldl_copy leaves T unchanged.

### `tridiag_ldl_copy`

```luau
tridiag_ldl_copy(T: Tridiagonal): TridiagonalLDL
```

tridiag_ldl of a copy of T; T is not modified.

### `tridiag_ldl_solve`

```luau
tridiag_ldl_solve(out: buffer?, F: TridiagonalLDL, b: buffer): buffer
```

Solves T x = b given F = tridiag_ldl(T). In place if out is b.

### `banded`

```luau
banded(n: number, kl: number, ku: number): Banded
```

A zero n by n banded matrix with kl subdiagonals and ku superdiagonals.

### `band_copy`

```luau
band_copy(B: Banded): Banded
```

A copy of B.

### `band_get`

```luau
band_get(B: Banded, i: number, j: number): number
-- complex modules:
band_get(B: Banded, i: number, j: number): (number, number)
```

Entry (i, j), 1-based; 0 outside the band.

### `band_set`

```luau
band_set(B: Banded, i: number, j: number, v: number)
-- complex modules:
band_set(B: Banded, i: number, j: number, re: number, im: number)
```

Sets entry (i, j), 1-based, which must lie in the band.

### `band_from_matrix`

```luau
band_from_matrix(A: Matrix, kl: number, ku: number): Banded
```

The band of the square matrix A (entries outside it are dropped).

### `band_matrix`

```luau
band_matrix(B: Banded): Matrix
```

The dense matrix.

### `band_matvec`

```luau
band_matvec(out: buffer?, B: Banded, x: buffer): buffer
```

B x

### `band_lu`

```luau
band_lu(B: Banded): BandedLU
```

Banded LU with partial pivoting (LAPACK's gbtf2), in place: B is overwritten with the factors, U getting up to kl + ku superdiagonals in B's extra rows. band_lu_copy leaves B unchanged.

### `band_lu_copy`

```luau
band_lu_copy(B: Banded): BandedLU
```

band_lu of a copy of B; B is not modified.

### `band_lu_solve`

```luau
band_lu_solve(out: buffer?, F: BandedLU, b: buffer): buffer
```

Solves B x = b given F = band_lu(B). In place if out is b.

### `band_chol`

```luau
band_chol(B: Banded): BandedCholesky
```

Banded Cholesky A = L L^T of a symmetric positive definite matrix (LAPACK's pbtf2, lower), in place: only the diagonal and the kl subdiagonals are read and overwritten with L. definite is false if B is not positive definite (B is then partly overwritten). band_chol_copy leaves B unchanged.

### `band_chol_copy`

```luau
band_chol_copy(B: Banded): BandedCholesky
```

band_chol of a copy of B; B is not modified.

### `band_chol_solve`

```luau
band_chol_solve(out: buffer?, F: BandedCholesky, b: buffer): buffer
```

Solves B x = b given F = band_chol(B). In place if out is b.

### `circulant`

```luau
circulant(c: buffer): Circulant
```

The circulant matrix with first column c (its eigenvalues are computed now).

### `circulant_matvec`

```luau
circulant_matvec(out: buffer?, C: Circulant, x: buffer): buffer
```

C x, by FFT.

### `circulant_solve`

```luau
circulant_solve(out: buffer?, C: Circulant, b: buffer): buffer
```

Solves C x = b, by FFT. In place if out is b.

### `circulant_matrix`

```luau
circulant_matrix(c: buffer): Matrix
```

The dense matrix.

### `toeplitz`

```luau
toeplitz(col: buffer, row: buffer?): Toeplitz
```

The Toeplitz matrix with first column col and first row row (default: col, which makes it symmetric). row[1] is ignored, as in SciPy. Both arrays are copied.

### `toeplitz_matrix`

```luau
toeplitz_matrix(T: Toeplitz): Matrix
```

The dense matrix.

### `toeplitz_matvec`

```luau
toeplitz_matvec(out: buffer?, T: Toeplitz, x: buffer): buffer
```

T x, by FFT through the circulant embedding.

### `toeplitz_solve`

```luau
toeplitz_solve(out: buffer?, T: Toeplitz, b: buffer): buffer
```

Solves T x = b by Levinson's recursion, O(n^2) in double precision. It needs every leading principal submatrix of T to be nonsingular (it raises an error otherwise) and is reliable for well-conditioned, especially symmetric positive definite, matrices; for other Toeplitz systems use solve on toeplitz_matrix(T). In place if out is b.
