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
