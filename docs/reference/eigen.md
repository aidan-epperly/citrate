# Eigenvalues and singular values

Hessenberg and Schur forms, Hermitian and general eigenproblems, the SVD, the polar decomposition, generalized eigenproblems, rank and condition numbers, randomized low-rank approximations. Parts `eigen_f32`, `eigen_f64`, `eigen_c32` and `eigen_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `hess`

```luau
hess(A: Matrix): Hessenberg
```

Upper Hessenberg reduction A = Q H Q^H (^T for real A), in place: A is overwritten with H on and above the subdiagonal and, below it, the reflectors H_k = I - tau_k v_k v_k^H (v_k has a 1 in row k+1, not stored, and is stored below that in column k). Q = H_0 ... H_n-3. hess_copy leaves A unchanged.

### `hess_copy`

```luau
hess_copy(A: Matrix): Hessenberg
```

Hessenberg reduction of a copy of A; A is not modified.

### `hess_h`

```luau
hess_h(out: Matrix?, F: Hessenberg): Matrix
```

The upper Hessenberg factor H, with zeros below the subdiagonal.

### `hess_q`

```luau
hess_q(out: Matrix?, F: Hessenberg): Matrix
```

The orthogonal factor Q.

### `eigh`

```luau
eigh(A: Matrix, vectors: boolean?, reuse: Eigh?): Eigh
```

Eigenvalues (ascending) and, unless vectors is false, orthonormal eigenvectors of a symmetric matrix, in place: only the lower triangle of A is read, and A is overwritten with the eigenvectors (column j belongs to values[j]), or with workspace if vectors is false. Tridiagonalization, then implicit QR with Wilkinson shifts in double precision. If the lower triangle holds a NaN or an infinity, the values and A are all NaN. eigh_copy leaves A unchanged. reuse, an earlier result for a matrix of A's size, has its arrays and workspace refilled: repeated decompositions then allocate nothing.

### `eigh_copy`

```luau
eigh_copy(A: Matrix, vectors: boolean?, reuse: Eigh?): Eigh
```

Symmetric eigendecomposition of a copy of A; A is not modified. With reuse (as eigh's), the copy goes into its vectors (or kept workspace): nothing is allocated.

### `schur`

```luau
schur(A: Matrix, vectors: boolean?): Schur
```

Real Schur factorization A = Z T Z^T, in place: A is overwritten with the quasi-triangular T, whose 2 by 2 diagonal blocks (standardized, with equal diagonal entries) hold the complex conjugate eigenvalue pairs. Z is computed unless vectors is false. values is a complex array (c64) of the eigenvalues in the order of T's diagonal. If A holds a NaN or an infinity, T, Z and the values are all NaN. schur_copy leaves A unchanged.

### `schur_copy`

```luau
schur_copy(A: Matrix, vectors: boolean?): Schur
```

Real Schur factorization of a copy of A; A is not modified.

### `eig`

```luau
eig(A: Matrix, vectors: boolean?, balanced: boolean?, reuse: Eig?): Eig
```

Eigenvalues and, unless vectors is false, unit eigenvectors of a general real matrix. Both are complex (c64): values is a complex array and column k of vectors (a complex matrix) belongs to values[k]. Unless balanced is false, A is balanced first (permuted and scaled; see balance), as LAPACK's geev does by default. A is overwritten with the (balanced) real Schur form, or with workspace if vectors is false; the eigenvectors are a new matrix. If A holds a NaN or an infinity, the values, the vectors and A are all NaN. eig_copy leaves A unchanged.

### `eig_copy`

```luau
eig_copy(A: Matrix, vectors: boolean?, balanced: boolean?, reuse: Eig?): Eig
```

Eigendecomposition of a copy of A; A is not modified. With reuse (an earlier result for a matrix of A's size), its arrays and workspace are refilled: nothing is allocated.

### `svd`

```luau
svd(A: Matrix, vectors: boolean?, reuse: Svd?): Svd
```

Singular value decomposition A = U diag(values) V^T: values (descending, nonnegative) and, unless vectors is false, U (rows by k) and V (cols by k) with orthonormal columns, k = min(rows, cols). Golub-Kahan bidiagonalization, then implicit QR on the bidiagonal in double precision, keeping the small singular values relatively accurate. In place when rows >= cols: A is overwritten with U (or with workspace if vectors is false). A wide A is not modified (its transpose is decomposed instead). If A holds a NaN or an infinity, every output is NaN. svd_copy never modifies A. reuse, an earlier result for a matrix of A's shape, has its arrays and workspace refilled.

### `svd_copy`

```luau
svd_copy(A: Matrix, vectors: boolean?, reuse: Svd?): Svd
```

Singular value decomposition of a copy of A; A is not modified. With reuse (as svd's), the copy goes into its u (or kept workspace): nothing is allocated.

### `pinv`

```luau
pinv(A: Matrix, tol: number?): Matrix
```

The pseudoinverse A^+ = V S^+ U^H (cols by rows), by the SVD of a copy of A; singular values at or below tol * s_1 (default max(rows, cols) * eps, as for svd_solve) count as zero. A is not modified. NaN or infinite input gives an all-NaN result. To apply A^+ to vectors, solving with form "svd" is cheaper than forming it.

### `matnorm`

```luau
matnorm(A: Matrix, kind: (number | string)?): number
```

Matrix norms: kind 1 (largest column sum of |a_ij|), "inf" (largest row sum), 2 (largest singular value) or "fro" (Frobenius; the default).

### `tridiag_eigh`

```luau
tridiag_eigh(T: Tridiagonal, vectors: boolean?): Eigh
```

Eigenvalues (ascending) and, unless vectors is false, orthonormal eigenvectors of the symmetric tridiagonal matrix T: its diagonal and T.lower are read (T.upper should equal T.lower and is ignored). Implicit QR with Wilkinson shifts in double precision, as inside eigh: O(n^2) for the values, O(n^3) with the vectors. Uses include Gauss quadrature from a Jacobi matrix (Golub-Welsch) and one-dimensional discretizations. T is not modified; the vectors are a new matrix.

### `polar`

```luau
polar(A: Matrix): Polar
```

Polar decomposition A = U H, in place: U has orthonormal columns (rows, if A is wide) and overwrites A; H = U^H A is Hermitian positive semidefinite (cols by cols), new. F.iterations counts the QDWH steps (see qdwh). For full-rank A both factors are unique; U is then the closest matrix with orthonormal columns (or rows) to A in every unitarily invariant norm. If A is exactly rank deficient, U is the canonical partial isometry (U^H U projects onto the range of H); a numerically rank-deficient A, whose tiny singular values rounding has made nonzero, gets an orthonormal U in a few more steps. A wide A is handled through its adjoint. A NaN or infinite entry gives NaN factors. polar_copy leaves A unchanged.

### `polar_copy`

```luau
polar_copy(A: Matrix): Polar
```

The polar decomposition of a copy of A; A is not modified.

### `rank`

```luau
rank(A: Matrix, tol: number?): number
```

The numerical rank (see svd_rank for tol).

### `orth`

```luau
orth(A: Matrix, tol: number?): Matrix
```

An orthonormal basis of the range of A (rows by rank), from the SVD.

### `null_space`

```luau
null_space(A: Matrix, tol: number?): Matrix
```

An orthonormal basis of the null space of A (cols by cols - rank): the orthogonal complement of the first rank right singular vectors, completed through the full Q of their QR factorization.

### `cond`

```luau
cond(A: Matrix, kind: (number | string)?): number
```

The condition number in the given norm (see matnorm): kind 2 (the default) is s_max / s_min from the SVD, for any shape (inf if rank deficient); kinds 1, "inf" and "fro" use the inverse of a square A (inf if singular).

### `rcond`

```luau
rcond(A: Matrix): number
```

An estimate of the reciprocal condition number 1 / (|A|_1 |A^-1|_1) from the LU factorization, in O(n^2) after it (LAPACK's gecon): |A^-1|_1 by Hager's method with Higham's refinements, which is almost always within a factor of 3 and never an overestimate. Returns 0 for a singular matrix. A is not modified.

### `eigh_gen`

```luau
eigh_gen(A: Matrix, B: Matrix, vectors: boolean?): Eigh
```

Eigenvalues (ascending) and, unless vectors is false, eigenvectors of A x = lambda B x for Hermitian A and Hermitian positive definite B (LAPACK's sygv): with B = L L^H it solves the standard problem for C = L^-1 A L^-H and maps the vectors back by x = L^-H y, so they are B-orthonormal (X^H B X = I). A must be Hermitian (both triangles are read); only B's lower triangle is read. In place: B is overwritten with its Cholesky factor and A with the eigenvectors (or workspace). An error if B is not positive definite. eigh_gen_copy leaves both unchanged.

### `eigh_gen_copy`

```luau
eigh_gen_copy(A: Matrix, B: Matrix, vectors: boolean?): Eigh
```

The generalized problem on copies; A and B are not modified.

### `eig_gen`

```luau
eig_gen(A: Matrix, B: Matrix, vectors: boolean?): GenEig
```

Generalized eigenvalues and eigenvectors of A x = lambda B x (see the complex module's eig_gen, whose QZ algorithm runs on complex copies; values, alpha, beta and vectors are complex). Complex eigenvalues come in conjugate pairs only up to rounding. A and B are not modified, so eig_gen_copy is the same function.

### `eig_gen_copy`

```luau
eig_gen_copy(A: Matrix, B: Matrix, vectors: boolean?): GenEig
```

Generalized eigenvalues and eigenvectors of A x = lambda B x (see the complex module's eig_gen, whose QZ algorithm runs on complex copies; values, alpha, beta and vectors are complex). Complex eigenvalues come in conjugate pairs only up to rounding. A and B are not modified, so eig_gen_copy is the same function.

### `rsvd`

```luau
rsvd(A: Matrix, k: number, options: RsvdOptions?): Svd
```

A rank-k approximation A ~ U diag(values) V^H by the randomized SVD (Halko, Martinsson and Tropp 2011): an orthonormal basis Q of the range of A Omega for a Gaussian Omega with k + options.oversample (default 10) columns, refined by options.power (default 2) power iterations (each re-orthonormalized), then the SVD of the small Q^H A. options.seed seeds math.random (the global generator) for reproducible results. Accurate when the singular values decay; the power iterations help when they decay slowly. A is not modified.

### `rpcholesky_columns`

```luau
rpcholesky_columns(n: number, diag: { number }, column: (j: number) -> buffer, k: number, options: RandOptions?): RPCholesky
```

Randomly pivoted Cholesky (Chen, Epperly, Tropp and Webber 2022) for a Hermitian positive semidefinite matrix given by its size n, its diagonal (a table of n real numbers) and a function column(j) returning column j (1-based) as an array. Builds A ~ F F^H, where F has one column per step: each step draws a pivot s with probability proportional to the diagonal of the residual A - F F^H, then takes the residual's column s, scaled by its pivot's square root. Only the chosen columns of A are ever requested, so A need not be formed (kernel matrices, for instance): about k^2 n work and k columns. The result is a Nystrom approximation, near-optimal in trace norm: it stops after k steps, or once trace(A - F F^H) <= options.tol (default 0) times trace(A). options.seed seeds math.random (the global generator). Returns { factor = F (n by rank), pivots (the chosen columns, 1-based), rank, residual = trace(A - F F^H) }.

### `rpcholesky`

```luau
rpcholesky(A: Matrix, k: number, options: RandOptions?): RPCholesky
```

Randomly pivoted Cholesky of a Hermitian positive semidefinite matrix A (see rpcholesky_columns; only the diagonal and the chosen columns are read). A is not modified.

### `rpqr`

```luau
rpqr(A: Matrix, k: number, options: RandOptions?): RPQR
```

Randomly pivoted QR (adaptive sampling; Deshpande and Vempala 2006): A ~ Q R from k columns of A chosen at random, each with probability proportional to its squared norm after projecting out the columns chosen so far. Q (rows by rank) is an orthonormal basis of the chosen columns, made by modified Gram-Schmidt run twice, and R = Q^H A, so Q R is the projection of A onto them. It is RPCholesky applied to A^H A, and near-optimal in the Frobenius norm. Stops after k steps, or once |A - Q R|_F^2 <= options.tol (default 0) times |A|_F^2; options.seed seeds math.random. About k m n work. Returns { q, r, pivots (the chosen columns, 1-based), rank, residual = |A - Q R|_F^2 }. A is not modified.

### `eigvecs_from_schur`

```luau
eigvecs_from_schur(T: Matrix, Z: Matrix, work: any?)
```

Overwrites the columns of Z with unit eigenvectors of A = Z T Z^H, given the upper triangular T of a complex Schur factorization: column k belongs to T's k-th diagonal entry. (LAPACK's trevc followed by the back-transformation, normalized to unit length.)

### `qz`

```luau
qz(A: Matrix, B: Matrix, want: boolean): (Matrix?, Matrix?)
```

The generalized Schur form A = Q S Z^H, B = Q P Z^H with S and P upper triangular, in place (A becomes S and B becomes P), by Moler and Stewart's QZ algorithm with single Wilkinson-type shifts: B is made triangular by a QR factorization, A Hessenberg by rotations that keep B triangular, then QZ steps deflate eigenvalues from the bottom. A (numerically) zero diagonal entry of P in the active block is chased to the top of the block with rotations that keep the Hessenberg-triangular form, where it splits off an infinite eigenvalue. Returns Q and Z (if wanted).
