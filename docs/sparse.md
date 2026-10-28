# Sparse matrices and iterative solvers

Reference: [sparse matrices](reference/sparse.md), [iterative solvers](reference/krylov.md).

## Sparse matrices

`sparse(rows, cols, ii, jj, values)` builds a compressed sparse column matrix from
triplets (duplicates summed). Products with vectors and dense matrices, adjoint products,
transpose, add and diagonal are provided, with the preconditioners Jacobi
(`sparse_jacobi`), SSOR, ILU(0) and IC(0).

Mark a symmetric matrix with `S.symmetric = true` (or `hermitian` for complex ones): its
product then becomes a gather down the columns, which is faster when columns are long. The
mark is a promise and is not checked.

A symmetric sparse matrix with a narrow band, once reordered, is often fastest solved
directly: `graph.rcm` orders it, `sparse_symband(S, perm)` copies the reordered band into
symmetric band storage and `band_chol` factors it (see [graphs](graphs.md)).
`sparse_permute(S, perm)` gives the reordered matrix itself.

## Sparse Cholesky

```luau
local F = f64.sparse_chol(A)              -- A symmetric positive definite (Hermitian: c32, c64)
local x = f64.sparse_chol_solve(nil, F, b)
-- new values on the same pattern (the same colptr and rowind buffers): only the numbers
f64.sparse_chol(A2, { reuse = F })        -- nothing allocated
```

`sparse_chol` factors P A P^T = L L^T (L L^H for complex A) in four steps. The first two
look only at the pattern, so `reuse` skips them:

1. Order: `order = "amd"` (approximate minimum degree, the default), `"rcm"`, `"natural"`
   or a permutation. The order decides the fill-in, the entries of L where A has zeros.
2. Analyze: the elimination tree (column j's parent is the first row below the diagonal in
   column j of L), where L's entries will be, and its supernodes: runs of columns with the
   same rows below their diagonal block, stored as dense blocks (`graph.chol_analysis`).
3. Factor: supernode by supernode, each receiving the updates of the earlier ones that
   reach it. Within and between blocks the work is dense: each update is a block of dot
   products computed four rows by two columns at a time, so a value read from memory feeds
   several multiply-adds.
4. Solve: two triangular solves, 2 nnz(L) multiply-adds, also by blocks: the forward solve
   updates four rows below a block at a time, the backward one sums four of its columns at a
   time.

`F.definite` is false if a pivot was not positive; `F.nnz` counts the entries stored and
`F.madds` the multiply-adds of a factorization.

Supernodes are merged with their parents when that adds few explicit zeros (`relax`, on by
default): wider blocks run faster per multiply-add, but zeros cost multiply-adds too. The
limits (up to 4 columns while under 50% zeros, 16 under 30%, 48 under 10%, any width under
5%) measured best of those tried; CHOLMOD's looser ones doubled the drum's multiply-adds and
made its factorization 25% slower.

Measured, native (refactoring with `reuse`; the first call adds the ordering and analysis):

| matrix | unknowns | sparse: multiply-adds, refactor, first call | banded after reverse Cuthill-McKee: half bandwidth, factor |
| --- | --- | --- | --- |
| drum (`demos/drum`) | 1697 | 2.1e5, 0.49 ms, 2.7 ms | 34, 3.1 ms |
| drum | 6977 | 2.4e6, 3.6 ms, 12 ms | 66, 45 ms |
| cloth (30 by 30, 3 per node) | 2700 | 2.2e7, 25 ms, 33 ms | 182, 118 ms |
| 5-point Laplacian, 100 by 100 | 10000 | 6.3e6, 7.9 ms, 20 ms | |

The block kernels reach about 1 ns per multiply-add on the cloth's wide supernodes and 1.5
on the drum's narrower ones, against 3 for banded Cholesky. On the Laplacian, refactoring
took as long as SciPy's SuperLU factorization (7.9 ms, with a symmetric minimum degree
ordering; SciPy has no sparse Cholesky), and a solve 0.87 ms against SuperLU's 0.22
(1.8 ns per multiply-add; 1.5 ms before the solves were blocked). The
first call takes 20 ms on the Laplacian (SuperLU: 8): about 7 ordering, 4 analyzing and 8
factoring, with 3.7 MB of workspace.

Factor once and solve many times: shift-invert eigenproblems (the drum's modes), Poisson
problems on a fixed mesh, implicit steps with a fixed matrix. A factorization every frame
of the cloth (20 ms) still costs more than its multigrid-preconditioned CG.

## Sparse LU

```luau
local F = f64.sparse_lu(A)               -- any square A; F.singular if it is singular
local x = f64.sparse_lu_solve(nil, F, b)
f64.sparse_lu(A2, { reuse = F })         -- same pattern: the pivots are kept
```

`sparse_lu` factors P A Q = L U with partial pivoting, column by column (Gilbert and
Peierls): each column of L and U comes from a sparse triangular solve whose pattern a
depth-first search finds first, so the work is what the arithmetic needs. Q is a
fill-reducing order of A + A^T (`order`, as `sparse_chol`'s: AMD by default). Rows are
pivoted by threshold partial pivoting: the diagonal entry stays the pivot while it is at
least `tol` (default 0.1) times the largest in its column, which keeps the order's low fill
for the nearly symmetric patterns of PDEs; `tol = 1` is plain partial pivoting. With
`reuse`, the pivots of the earlier factorization are kept, unless one has become too small
(`F.repivoted` then says it pivoted afresh). For symmetric positive definite matrices
`sparse_chol` is several times faster.

On convection-diffusion on a 100 by 100 grid (10,000 unknowns, nonsymmetric), L and U hold
396,000 entries against SuperLU's 646,000 (SciPy's `splu`), but the factorization took 100
ms natively (463 interpreted) against SuperLU's 12, and a solve 1.5 ms against 0.38: about 8
ns per multiply-add, one column at a time. Refactoring with `reuse` took 89 ms: it repeats
each column's pattern search.

## Iterative solvers

The solvers in `krylov` take a `Space`, an operator `(out, x) -> out` and a right-hand side,
so A can be sparse, dense, structured or matrix-free. `cg` and `minres` work on any `Space`
(including `vector`); `gmres` and `bicgstab` need the complex extension the array modules
provide. `eigsh` (thick-restart Lanczos) and `eigs` (Krylov-Schur) find a few eigenvalues of
large or matrix-free operators; `demos/drum` uses `eigsh` for the vibration modes of
two isospectral drums.

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

## Block sparse matrices

When the unknowns come in groups that couple as a whole, such as the x, y and z of each
particle, store the matrix in dense b by b blocks: with 3 by 3 blocks a product costs about
a third of the plain format's, since one index serves nine entries and each entry of x is
read once per block (see [Performance](performance.md)).

```luau
local B = f64.bsparse_from_sparse(S, 3)   -- or f64.bsparse(n, n, 3, I, J, blocks)
B.symmetric = true
krylov.cg(f64.space, function(out, v) return f64.bsparse_matvec(out, B, v) end, b,
	{ precond = f64.bsparse_jacobi(B) })   -- inverts each diagonal block
```

`bsparse_block(B, I, J)` locates a block, so a Jacobian with a fixed pattern can be refilled
in place each step.

## Multigrid

For larger symmetric positive definite systems, algebraic multigrid makes CG's iteration
count nearly independent of the problem size. `bsparse_amg(B, options)` aggregates
strongly coupled nodes into coarse nodes, recursively, and returns `{ apply, update,
sizes }`: `apply` is one V-cycle (pass it as `precond`), and `update(B)` refills the
hierarchy from new values on the same pattern, as a simulation does each step. Smoothed
aggregation (the default) needs the fewest iterations, while plain aggregation
(`smooth = false`) is cheaper to set up and refill. Components held fixed go in
`options.fixed`. Measured with CG to 1e-8 (`bench/amg.luau`):

| problem (unknowns) | block Jacobi | plain aggregation | smoothed aggregation |
|---|---|---|---|
| Laplacian 100x100 (10000) | 251 its, 55 ms | 57 its, 47 ms | 14 its, 15 ms |
| anisotropic Laplacian (10000) | 520 its, 113 ms | 46 its, 45 ms | 11 its, 22 ms |
| 2D spring lattice (7200) | 487 its, 82 ms | 75 its, 41 ms | 31 its, 19 ms |
| 3D soft body (5184) | 135 its, 25 ms | 48 its, 26 ms | 30 its, 18 ms |
| cloth 60x60 (10800) | 224 its, 67 ms | 36 its, 34 ms | 14 its, 17 ms |

Setup is extra: a few ms to tens of ms for smoothed aggregation, more where coarse
operators fill in (310 ms for the anisotropic Laplacian), and a refill costs a fraction of
that. A hierarchy left stale while the matrix changes is still a valid preconditioner but
can lose its advantage quickly; on swinging cloth, refilling every 20th step cost more
iterations than block Jacobi.
