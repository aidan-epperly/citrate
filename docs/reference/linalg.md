# Linear algebra, simple

Solve equations, least squares, eigenvalues, singular values, determinants: methods chosen and answers checked for you (real numbers).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `matrix`

```luau
matrix(rows: { { number } }): Matrix
```

A matrix from a table of rows, checked: linalg.matrix({ { 1, 2 }, { 3, 4 } }). The other functions take tables directly; this is for building a matrix once and using it often.

### `sparse`

```luau
sparse(rows: number, cols: number, entries: { { number } }): Sparse
```

A sparse matrix (one storing only its nonzero entries) of the given size from a list of entries { row, column, value }: linalg.sparse(3, 3, { { 1, 1, 4 }, { 2, 1, -1 }, ... }). Repeated positions are added together.

### `totable`

```luau
totable(A: any): any
```

A matrix as a table of rows (a sparse one as its list of entries { row, column, value }), or a vector as a table of numbers.

### `solve`

```luau
solve(A: any, b: any): (any, Info)
```

The solution x of A x = b for a square matrix A. Picks the method: for a symmetric matrix, Cholesky factorization (f64.chol, f64.sparse_chol), falling back to LU if it is not positive definite; otherwise LU with pivoting (f64.lu, f64.sparse_lu). The answer is checked (info.residual: |A x - b| / |b|) and refined by up to two steps of iterative refinement when it can be improved. For a dense matrix info.rcond estimates how close A is to singular (1: far, 0: singular), and info.warning is set when the answer may be inaccurate. Errors if A is singular.

### `lstsq`

```luau
lstsq(A: any, b: any): (any, Info)
```

The least-squares solution of A x = b for any shape of A: the x making |A x - b| smallest (for a tall A, the best fit; for a wide or rank-deficient one, a solution with as many zeros as the rank requires). Column-pivoted QR (f64.qrp, f64.qrp_solve). info.rank is the numerical rank and info.residual |A x - b| / |b|. A sparse A is made dense.

### `eig`

```luau
eig(A: any, options: EigOptions?): ({ number }, { any }, Info)
```

The eigenvalues (ascending) and eigenvectors of a symmetric matrix. options.count asks for only the smallest few (options.largest: the largest few). A dense matrix: all of them by f64.eigh. A sparse one: count of them (default 6) by Lanczos (krylov.eigsh) on the inverse of A shifted past its eigenvalues (sparse Cholesky of A - s I, s from Gershgorin's bound), which finds the end of the spectrum quickly; info.residual is the largest |A v - lambda v| relative to |A|'s scale. A sparse matrix of 100 rows or fewer, or one asked for more than a quarter of its eigenvalues, is made dense instead (f64.eigh costs n^3: 46 ms at 200 rows, where Lanczos found 4 eigenvalues of 1521 rows in 22). Errors if A is not symmetric: a nonsymmetric matrix can have complex eigenvalues (see f64.eig). Returns the values (a table) and the vectors (a table of vectors, each in the form A came in).

### `svd`

```luau
svd(A: any): ({ number }, { any }, { any }, Info)
```

The singular values (descending) and the singular vectors: A = U diag(s) V^T, as tables of columns (u_k, v_k: vectors in the form A came in). f64.svd; a sparse A is made dense.

### `det`

```luau
det(A: any): number
```

The determinant of a square matrix (f64.det; a sparse one from its sparse LU factors: the product of the pivots, signed by the row and column permutations). May overflow to infinity for large matrices.

### `inv`

```luau
inv(A: any): any
```

The inverse of a square matrix (f64.inv). Solving with linalg.solve is faster and more accurate than multiplying by the inverse; use this when the inverse itself is wanted. A sparse A is made dense (its inverse is dense). Errors if A is singular.

### `norm`

```luau
norm(x: any): number
```

The length (2-norm) of a vector, or the 2-norm of a matrix (its largest singular value; a sparse one made dense).

### `cond`

```luau
cond(A: any): number
```

The condition number of a matrix: the ratio of its largest to smallest singular value (f64.cond). Large means nearly singular: solving with it loses about log10(cond) digits. A sparse A is made dense.
