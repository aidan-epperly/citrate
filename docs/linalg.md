# Linear algebra: start here

Reference: [linalg](reference/linalg.md). For full control and speed, see
[dense linear algebra](linear-algebra.md) and [sparse matrices](sparse.md).

`citrate.linalg` solves equations and finds eigenvalues without asking you to choose a
method. Give it plain Lua tables; it checks them, picks a method, and tells you what it did
and how accurate the answer is. Real numbers only.

```luau
local linalg = citrate.linalg

-- solve A x = b
local x, info = linalg.solve({ { 4, 1 }, { 1, 3 } }, { 1, 2 })
print(x[1], x[2])          -- 0.0909..., 0.6363...
print(info.method)         -- "Cholesky" (A is symmetric positive definite)
print(info.residual)       -- how far off the answer is (0 here: exact to rounding)
```

Every function returns its answer in the form you gave it: tables for tables, and the
library's arrays for its arrays.

## Matrices

A matrix is a table of rows, `{ { 1, 2 }, { 3, 4 } }`. For a large matrix that is mostly
zeros (from a grid or a mesh), list only the entries that are not zero:

```luau
-- a 1000 by 1000 matrix with 2 on the diagonal and -1 beside it
local entries = {}
for i = 1, 1000 do
	table.insert(entries, { i, i, 2 })          -- { row, column, value }
	if i < 1000 then
		table.insert(entries, { i, i + 1, -1 })
		table.insert(entries, { i + 1, i, -1 })
	end
end
local A = linalg.sparse(1000, 1000, entries)
local x = linalg.solve(A, table.create(1000, 1))
```

`linalg.matrix(rows)` builds a matrix once, if you will use it many times, and
`linalg.totable(A)` turns any matrix or vector back into tables.

## Functions

| function | what it gives |
| --- | --- |
| `solve(A, b)` | x with A x = b (A square) |
| `lstsq(A, b)` | the best fit: x making A x as close to b as possible (any shape) |
| `eig(A, options)` | eigenvalues (smallest first) and eigenvectors of a symmetric matrix |
| `svd(A)` | singular values (largest first) and vectors |
| `det(A)`, `inv(A)` | determinant, inverse |
| `norm(x)`, `cond(A)` | length of a vector (or size of a matrix), condition number |

## What `solve` does for you

- It checks the input: the shapes match, every entry is a finite number.
- It looks at A. If A is symmetric (equal to its transpose, to 12 digits), it tries
  Cholesky factorization, the fastest method. If that fails (A is not positive definite), or
  A is not symmetric, it uses LU factorization with pivoting. Dense and sparse matrices
  each have their own versions.
- It checks the answer: `info.residual` is |A x - b| / |b|. If the answer can be
  improved, it refines it (up to twice; `info.refinements` counts the steps).
- It warns you: `info.warning` says when A is nearly singular, so the answer may be
  inaccurate (`info.rcond`, for a dense A, measures this: 1 is ideal, near 0 is
  trouble). If A is exactly singular, it stops with an error.

## Eigenvalues

```luau
local values, vectors = linalg.eig({ { 2, 1 }, { 1, 2 } })   -- 1 and 3
local smallest = linalg.eig(A, { count = 5 })                 -- the 5 smallest of a big A
local largest = linalg.eig(A, { count = 5, largest = true })
```

`eig` needs a symmetric matrix: other matrices can have complex eigenvalues, which this
module does not handle (use `citrate.f64.eig`). For a sparse matrix, ask for a few
eigenvalues with `count` (6 if you do not say): it finds the smallest (or largest) quickly
without computing all of them.

## When to use the library's own functions

These functions copy their input and check their output, which costs time. When you solve
the same kind of problem many times (every frame, say), call the function `linalg` would
have chosen directly. Each function's reference entry names it: `f64.chol` and
`f64.chol_solve`, `f64.sparse_chol`, `f64.sparse_lu` and so on, which also let you reuse a
factorization.
