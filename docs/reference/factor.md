# Factorizations and solves

LU, Cholesky, QR, column-pivoted QR, LDL^T, `solve`, determinants, factorization updates and iterative refinement. Parts `factor_f32`, `factor_f64`, `factor_c32` and `factor_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `lu`

```luau
lu(A: Matrix, reuse: LU?): LU
```

LU factorization with partial pivoting, in place: A is overwritten with L and U (see types.LU) and returned as F.lu. lu_copy leaves A unchanged.

### `lu_copy`

```luau
lu_copy(A: Matrix, reuse: LU?): LU
```

LU factorization of a copy of A; A is not modified. With reuse (an LU from an earlier call, of a matrix of A's size), its arrays are refilled and it is returned: repeated factorizations then allocate nothing.

### `lu_solve`

```luau
lu_solve(out: buffer?, F: LU, b: buffer): buffer
```

Solves A x = b given F = lu(A).

### `inv`

```luau
inv(A: Matrix): Matrix
```

The inverse of a square matrix, in place (LAPACK's getrf + getri): A is overwritten with A^-1 and returned. An error if A is exactly singular. Solving with solve (or a factorization) is faster and more accurate than forming A^-1 and multiplying; use this when the inverse itself is needed. inv_copy leaves A unchanged.

### `inv_copy`

```luau
inv_copy(A: Matrix): Matrix
```

The inverse of a copy of A; A is not modified.

### `det`

```luau
det(A: Matrix): number
-- complex modules:
det(A: Matrix): (number, number)
```

The determinant, from an LU factorization of a copy (A is not modified); see slogdet for large or small determinants.

### `chol`

```luau
chol(A: Matrix, reuse: Cholesky?): Cholesky
```

Cholesky factorization A = L L^T of a symmetric positive definite matrix, in place: the lower triangle of A is overwritten with L and returned as F.chol. Only the lower triangle of A is read and the strictly upper triangle is left as it was. If A is not positive definite, F.definite is false and A is partly overwritten. chol_copy leaves A unchanged.

### `chol_copy`

```luau
chol_copy(A: Matrix, reuse: Cholesky?): Cholesky
```

Cholesky factorization of a copy of A; A is not modified.

### `chol_l`

```luau
chol_l(out: Matrix?, F: Cholesky): Matrix
```

The lower triangular factor L, with zeros above the diagonal.

### `chol_solve`

```luau
chol_solve(out: buffer?, F: Cholesky, b: buffer): buffer
```

Solves A x = b given F = chol(A). b is not modified unless it is passed as out.

### `qr`

```luau
qr(A: Matrix, reuse: QR?): QR
```

Householder QR factorization, in place: A is overwritten (see types.QR for the layout) and returned as F.qr. qr_copy leaves A unchanged.

### `qr_copy`

```luau
qr_copy(A: Matrix, reuse: QR?): QR
```

Householder QR factorization of a copy of A; A is not modified.

### `qr_apply_qh`

```luau
qr_apply_qh(F: QR, x: buffer): buffer
```

x := Q^T x, in place. Returns x. Also accepts a QRP factorization.

### `qr_apply_q`

```luau
qr_apply_q(F: QR, x: buffer): buffer
```

x := Q x, in place. Returns x. Also accepts a QRP factorization.

### `qr_r`

```luau
qr_r(out: Matrix?, F: QR): Matrix
```

The min(rows, cols) by cols upper triangular factor R. Also accepts a QRP factorization.

### `qr_q`

```luau
qr_q(out: Matrix?, F: QR): Matrix
```

The rows by min(rows, cols) factor Q with orthonormal columns (the thin Q). Also accepts a QRP factorization.

### `qr_solve`

```luau
qr_solve(out: buffer?, F: QR, b: buffer): (buffer, number)
```

Least squares: the x minimizing |A x - b| for A with rows >= cols and full column rank, given F = qr(A). b is not modified. Also returns the residual norm |A x - b|.

### `qrp`

```luau
qrp(A: Matrix, reuse: QRP?, tol: number?): QRP
```

Householder QR with column pivoting, A P = Q R, in place: A is overwritten as in qr and returned as F.qr; column j of A P is column perm[j] of A. Each step moves the remaining column of largest norm to the front, so |R_11| >= |R_22| >= ... and small trailing diagonal entries reveal the numerical rank. qrp_copy leaves A unchanged. With tol, it stops once every remaining column's norm is at most tol |R_11| (a truncated factorization, as for an interpolative decomposition): F.rank is the number of steps taken, and only the first F.rank rows of R are final.

### `qrp_copy`

```luau
qrp_copy(A: Matrix, reuse: QRP?, tol: number?): QRP
```

Column-pivoted QR of a copy of A; A is not modified.

### `qrp_rank`

```luau
qrp_rank(F: QRP, tol: number?): number
```

Numerical rank: the number of leading diagonal entries of R with |R_kk| > tol |R_11|. The default tol is max(rows, cols) times the unit roundoff of the storage precision.

### `qrp_solve`

```luau
qrp_solve(out: buffer?, F: QRP, b: buffer, tol: number?): (buffer, number, number)
```

Least squares for any shape and rank, given F = qrp(A): the basic solution, which uses only the first r = qrp_rank(F, tol) pivot columns and sets the other variables to zero. b is not modified. Also returns the residual norm |A x - b| and r.

### `svd_solve`

```luau
svd_solve(out: buffer?, F: Svd, b: buffer, tol: number?): (buffer, number, number)
```

The minimum-norm least-squares solution x = A^+ b for any shape and rank, given F = svd(A) with its vectors: x = sum over i <= r of v_i (u_i^H b) / s_i, where r counts the singular values above tol * s_1 (default tol = max(rows, cols) * eps, as for qrp_rank); smaller ones are treated as zero. Unlike qrp_solve's basic solution, x is the shortest of all least-squares solutions. b is not modified. Also returns the residual norm |A x - b| and r. NaN singular values (from non-finite input) give a NaN x.

### `slogdet`

```luau
slogdet(A: Matrix): (number, number)
-- complex modules:
slogdet(A: Matrix): (number, number, number)
```

The sign and the log of the absolute value of det(A), without overflow: returns (sign, log |det|), with sign 0 and log -inf for a singular matrix.

### `lu_solve_adjoint`

```luau
lu_solve_adjoint(out: buffer?, F: LU, b: buffer): buffer
```

Solves A^T x = b given F = lu(A) (P A = L U, so A^T = U^T L^T P).

### `ldl`

```luau
ldl(A: Matrix, reuse: LDL?): LDL
```

The Bunch-Kaufman factorization P A P^T = L D L^T of a symmetric (possibly indefinite) matrix, in place (LAPACK's sytf2, lower): only the lower triangle of A is read and it is overwritten with L (unit lower triangular, below the diagonal) and the block diagonal D (1 by 1 and 2 by 2 blocks). piv records the interchanges as LAPACK's ipiv does, 1-based: pget(piv, k) > 0 for a 1 by 1 block (row k was swapped with pget(piv, k)), pset(piv, k, pget(piv, k+1) < 0 for a 2 by 2 block (row k+1 was swapped with -pget(piv, k)). The) pivoting (alpha = (1 + sqrt 17) / 8) keeps the growth bounded, so this is stable for any symmetric matrix, unlike Cholesky (definite only) or LU (which ignores symmetry). ldl_copy leaves A unchanged.

### `ldl_copy`

```luau
ldl_copy(A: Matrix, reuse: LDL?): LDL
```

The factorization of a copy of A; A is not modified.

### `ldl_solve`

```luau
ldl_solve(out: buffer?, F: LDL, b: buffer): buffer
```

Solves A x = b given F = ldl(A) (LAPACK's sytrs).

### `ldl_inertia`

```luau
ldl_inertia(F: LDL): (number, number, number)
```

The inertia of A from F = ldl(A): the numbers of positive, negative and zero eigenvalues (Sylvester's law of inertia: D has the same). Each 2 by 2 block has one of each sign.

### `qr_full`

```luau
qr_full(A: Matrix): QRFull
```

The full QR factorization A = Q R as explicit matrices: Q (rows by rows) unitary and R (rows by cols) upper triangular, the form the update functions below work on. A is not modified.

### `qr_update`

```luau
qr_update(F: QRFull, u: buffer, v: buffer): QRFull
```

The rank-one update A + u v^H = Q' R', in place on F = qr_full(A) (Golub and Van Loan 6.5.1): rotations take Q^H u to a multiple of e_1, making R upper Hessenberg; the update then changes only R's first row, and rotations restore the triangle. O(m^2 + m n). Returns F.

### `qr_delete_col`

```luau
qr_delete_col(F: QRFull, j: number): QRFull
```

Removes column j (1-based) from the factored matrix: R loses that column, which leaves it upper Hessenberg from column j on, and rotations restore the triangle. Q is updated in place; F.r is replaced. Returns F.

### `qr_insert_col`

```luau
qr_insert_col(F: QRFull, j: number, x: buffer): QRFull
```

Inserts x as column j (1-based; j = cols + 1 appends) of the factored matrix: the new column of R is Q^H x, whose entries below row j are rotated away from the bottom up. Q is updated in place; F.r is replaced. Returns F.

### `chol_update`

```luau
chol_update(F: Cholesky, x: buffer): Cholesky
```

The rank-one update A + x x^H of a Cholesky factorization F = chol(A), in place on F.chol's lower triangle (LINPACK's chud): one rotation per column, O(n^2). x is not modified. Returns F.

### `chol_downdate`

```luau
chol_downdate(F: Cholesky, x: buffer): Cholesky
```

The rank-one downdate A - x x^H of a Cholesky factorization F = chol(A), in place (LINPACK's chdd, with hyperbolic rotations). An error, leaving F unchanged, if A - x x^H is not positive definite. x is not modified. Returns F.

### `solve_refined`

```luau
solve_refined(out: buffer?, A: Matrix, b: buffer, options: RefineOptions?): (buffer, { iterations: number, converged: boolean, fallback: boolean })
```

Solves the square system A x = b by iterative refinement, factoring A once (LU with partial pivoting) and correcting x by solves with the residual b - A x. Returns x and { iterations, converged, fallback }. A and b are not modified.  Double modules (LAPACK's dsgesv): the factorization is in single precision and the residual in double, until |b - A x| <= sqrt(n) eps |A| |x|: a double-precision solution from single-precision factors (half the memory) while cond(A) is below about 1e7. Beyond that, or if the single-precision LU is singular, it falls back to a double factorization. (Luau computes in double either way, so unlike in compiled code this is not faster than solve.)  Single modules (as LAPACK's extra-precise refinement): the residual is computed in double precision before it is rounded, and the corrections continue until they are below eps |x| or stop shrinking. That improves the forward error from about cond(A) eps to about eps while cond(A) is below about 1 / eps (1e7); converged says whether the corrections reached eps |x|.  options.method = "gmres" solves each correction equation by GMRES preconditioned with the factorization (GMRES-IR, Carson and Higham 2017), up to options.inner (default 30) steps, instead of by one solve with it. GMRES copes with a much rougher preconditioner, so refinement keeps converging to about cond(A) = 1 / eps_single^2 (about 1e14) instead of 1 / eps_single; in the double modules the single-precision factors are then applied in double arithmetic.
