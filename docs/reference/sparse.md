# Sparse matrices

Compressed sparse column and block sparse storage, products, preconditioners and algebraic multigrid. Parts `sparse_f32`, `sparse_f64`, `sparse_c32` and `sparse_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `sparse`

```luau
sparse(rows: number, cols: number, ii: { number }, jj: { number }, values: buffer): Sparse
```

A rows by cols sparse matrix from triplets: entry (ii[k], jj[k]) (1-based) gets values[k]; repeated positions are summed, as in SciPy. Explicit zeros are kept.

### `sparse_nnz`

```luau
sparse_nnz(S: Sparse): number
```

Number of stored entries.

### `sparse_from_matrix`

```luau
sparse_from_matrix(A: Matrix): Sparse
```

The nonzero entries of the dense matrix A.

### `sparse_matrix`

```luau
sparse_matrix(S: Sparse): Matrix
```

The dense matrix.

### `sparse_get`

```luau
sparse_get(S: Sparse, i: number, j: number): number
-- complex modules:
sparse_get(S: Sparse, i: number, j: number): (number, number)
```

Entry (i, j), 1-based (0 if not stored).

### `sparse_diag`

```luau
sparse_diag(S: Sparse): buffer
```

The diagonal, as an array of min(rows, cols) entries.

### `sparse_matvec`

```luau
sparse_matvec(out: buffer?, S: Sparse, x: buffer): buffer
```

S x, accumulated in double precision. A general S is applied column by column (a scatter into a double accumulator); one marked symmetric as S^T x, a gather down its columns, about 10% faster still since each sum stays in a local.

### `sparse_matvec_adjoint`

```luau
sparse_matvec_adjoint(out: buffer?, S: Sparse, x: buffer): buffer
```

S^T x.

### `sparse_matmul`

```luau
sparse_matmul(out: Matrix?, S: Sparse, B: Matrix): Matrix
```

S B for a dense matrix B, as a dense matrix.

### `sparse_transpose`

```luau
sparse_transpose(S: Sparse): Sparse
```

S^T.

### `sparse_chol`

```luau
sparse_chol(S: Sparse, options: types.SparseCholOptions?): types.SparseCholesky
```

Sparse Cholesky factorization P S P^T = L L^T of a symmetric positive definite sparse matrix (both triangles stored, as Sparse matrices are; the lower one is read after ordering). options.order chooses P: "amd" (approximate minimum degree: the least fill, the default), "rcm", "natural" or a permutation (as graph.rcm returns). L is stored by supernodes, dense blocks of columns with the same rows (graph.chol_analysis; options.relax = false keeps only exact ones). options.reuse, an earlier factorization of a matrix with the same pattern (the same colptr and rowind buffers), keeps its ordering, analysis and storage: only the values are recomputed, with nothing allocated. F.definite is false if a pivot was not positive (the factorization stops there); F.nnz is the number of entries stored and F.madds the multiply-adds of a factorization.

### `sparse_chol_solve`

```luau
sparse_chol_solve(out: buffer?, F: types.SparseCholesky, b: buffer): buffer
```

The solution of S x = b from sparse_chol's factorization (out may be b).

### `sparse_permute`

```luau
sparse_permute(S: Sparse, perm: buffer): Sparse
```

P S P^T for the permutation perm (a buffer of 32-bit integers, as graph.rcm returns): entry (k, l) is S's entry (perm[k], perm[l]).

### `sparse_symband`

```luau
sparse_symband(S: Sparse, perm: buffer?, into: SymBanded?): SymBanded
```

The band of the symmetric matrix P S P^T (perm as for sparse_permute; default the identity) in symmetric band storage, for band_chol: kd is the permuted pattern's half bandwidth, and the lower triangle is read. into is reused when it has that shape.

### `sparse_add`

```luau
sparse_add(A: Sparse, B: Sparse): Sparse
```

A + B.

### `sparse_jacobi`

```luau
sparse_jacobi(S: Sparse): (out: buffer, r: buffer) -> buffer
```

The Jacobi (diagonal) preconditioner of a square S, for the iterative solvers: out := r ./ diag(S). Every diagonal entry must be nonzero.

### `sparse_adjoint`

```luau
sparse_adjoint(S: Sparse): Sparse
```

For real matrices the adjoint is the transpose.

### `sparse_ssor`

```luau
sparse_ssor(S: Sparse, omega: number?): (out: buffer, r: buffer) -> buffer
```

The SSOR preconditioner of a square S with relaxation omega (0 < omega < 2; the default 1 is symmetric Gauss-Seidel): M = (D + w L) D^-1 (D + w U) / (w (2 - w)), with D, L, U the diagonal and strictly lower and upper parts. Returns out := M^-1 r. If S is symmetric positive definite, so is M (usable with CG). Every diagonal entry must be nonzero.

### `sparse_ilu0`

```luau
sparse_ilu0(S: Sparse): SparseILU
```

Incomplete LU factorization ILU(0) of a square S, in place: S's values are overwritten with L and U in S's own sparsity pattern (no fill-in). Every diagonal entry must be stored; singular is true if one is missing or a pivot is zero. sparse_ilu0_copy leaves S unchanged.

### `sparse_ilu0_copy`

```luau
sparse_ilu0_copy(S: Sparse): SparseILU
```

sparse_ilu0 of a copy of S (the index arrays are shared, the values copied).

### `sparse_ilu0_solve`

```luau
sparse_ilu0_solve(out: buffer?, F: SparseILU, r: buffer): buffer
```

out := (L U)^-1 r given F = sparse_ilu0(S); use it as a preconditioner with function(out, r) return sparse_ilu0_solve(out, F, r) end.

### `sparse_ic0`

```luau
sparse_ic0(S: Sparse): SparseIC
```

Incomplete Cholesky factorization IC(0) of a symmetric positive definite S, in place: only the lower triangle is read, and it is overwritten with L in its own sparsity pattern (no fill-in). It can break down even for positive definite S (definite is then false); it cannot for M-matrices such as discrete Laplacians. sparse_ic0_copy leaves S unchanged.

### `sparse_ic0_copy`

```luau
sparse_ic0_copy(S: Sparse): SparseIC
```

sparse_ic0 of a copy of S (the index arrays are shared, the values copied).

### `sparse_ic0_solve`

```luau
sparse_ic0_solve(out: buffer?, F: SparseIC, r: buffer): buffer
```

out := (L L^T)^-1 r given F = sparse_ic0(S); use it as a preconditioner with function(out, r) return sparse_ic0_solve(out, F, r) end.

### `bsparse`

```luau
bsparse(rows: number, cols: number, block: number, ii: { number }, jj: { number }, values: buffer): BlockSparse
```

A rows by cols block sparse matrix with block by block blocks from block triplets: block (ii[k], jj[k]) (1-based block indices) gets the block^2 entries of values from element block^2 (k - 1) + 1 on, column-major; repeated positions are summed.

### `bsparse_from_sparse`

```luau
bsparse_from_sparse(S: Sparse, block: number): BlockSparse
```

S in blocks of block by block (every block holding a stored entry of S is stored whole); S's symmetric and hermitian marks carry over.

### `bsparse_sparse`

```luau
bsparse_sparse(B: BlockSparse): Sparse
```

B as a Sparse matrix, every entry of its stored blocks stored (zeros included).

### `bsparse_nnz`

```luau
bsparse_nnz(B: BlockSparse): number
```

Number of stored entries (block^2 per stored block).

### `bsparse_block`

```luau
bsparse_block(B: BlockSparse, I: number, J: number): number?
```

The number p (1-based) of block (I, J) (1-based block indices), or nil if it is not stored: its entries are values[block^2 (p - 1) + 1 .. block^2 p], column-major. For updating values in place (a Jacobian refilled each step on a fixed pattern).

### `bsparse_get`

```luau
bsparse_get(B: BlockSparse, i: number, j: number): number
-- complex modules:
bsparse_get(B: BlockSparse, i: number, j: number): (number, number)
```

Entry (i, j), 1-based (0 if not stored).

### `bsparse_diag`

```luau
bsparse_diag(B: BlockSparse): buffer
```

The diagonal, as an array of min(rows, cols) entries.

### `bsparse_matvec`

```luau
bsparse_matvec(out: buffer?, B: BlockSparse, x: buffer): buffer
```

B x, accumulated in double precision (generated kernels for blocks of 2, 3 and 4).

### `bsparse_matvec_adjoint`

```luau
bsparse_matvec_adjoint(out: buffer?, B: BlockSparse, x: buffer): buffer
```

B^T x (B x when B is marked symmetric).

### `bsparse_jacobi`

```luau
bsparse_jacobi(B: BlockSparse): (out: buffer, r: buffer) -> buffer
```

The block Jacobi preconditioner of a square B, for the iterative solvers: out := blockdiag(B)^-1 r, the diagonal blocks inverted once (in double precision, by Gauss-Jordan elimination with partial pivoting). It keeps each node's coupling between its own unknowns, which scalar Jacobi drops, and is unchanged by rotating a node's coordinates. Every diagonal block must be stored and nonsingular; for a symmetric positive definite B it is symmetric positive definite (usable with CG).

### `bsparse_amg`

```luau
bsparse_amg(B: BlockSparse, options: AMGOptions?): BlockAMG
```

Algebraic multigrid preconditioner for a symmetric positive definite block sparse B (blocks of the size of a node's unknowns: 3 for positions in 3D). See types.BlockAMG.
