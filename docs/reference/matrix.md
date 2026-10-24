# Matrices

Column-major matrices: creation and access, transposes, matrix-vector products, submatrices and utilities. Parts `matrix_f32`, `matrix_f64`, `matrix_c32` and `matrix_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `matfrom`

```luau
matfrom(t: { { number } }): Matrix
-- complex modules:
matfrom(re: { { number } }, im: { { number } }?): Matrix
```

Builds a matrix from a table of rows.

### `mattotable`

```luau
mattotable(A: Matrix): { { number } }
-- complex modules:
mattotable(A: Matrix): ({ { number } }, { { number } })
```

Returns a table of rows.

### `identity`

```luau
identity(n: number): Matrix
```

The n by n identity matrix.

### `matget`

```luau
matget(A: Matrix, i: number, j: number): number
-- complex modules:
matget(A: Matrix, i: number, j: number): (number, number)
```

Entry (i, j).

### `matset`

```luau
matset(A: Matrix, i: number, j: number, v: number)
-- complex modules:
matset(A: Matrix, i: number, j: number, re: number, im: number)
```

Sets entry (i, j) to v.

### `matcopy`

```luau
matcopy(out: Matrix?, A: Matrix): Matrix
```

out := A (a new matrix if out is nil).

### `transpose`

```luau
transpose(out: Matrix?, A: Matrix): Matrix
```

out := A^T (a new matrix if out is nil).

### `matvec`

```luau
matvec(out: buffer?, A: Matrix, x: buffer): buffer
```

A x

### `trace`

```luau
trace(A: Matrix): number
-- complex modules:
trace(A: Matrix): (number, number)
```

The sum of the diagonal.

### `diag`

```luau
diag(out: buffer?, A: Matrix): buffer
```

The diagonal (min(rows, cols) entries), as an array.

### `diagm`

```luau
diagm(x: buffer): Matrix
```

The square matrix with the array x on its diagonal.

### `kron`

```luau
kron(A: Matrix, B: Matrix): Matrix
```

The Kronecker product: the block matrix [a_ij B].

### `outer`

```luau
outer(out: Matrix?, x: buffer, y: buffer): Matrix
```

The outer product x y^T.

### `ger`

```luau
ger(A: Matrix, a: number, x: buffer, y: buffer): Matrix
```

The rank-one update A := A + a x y^T, in place (BLAS ger); returns A.

### `matvec_adjoint`

```luau
matvec_adjoint(out: buffer?, A: Matrix, x: buffer): buffer
```

A^T x, without forming the transpose.

### `submatrix`

```luau
submatrix(out: Matrix?, A: Matrix, i0: number, i1: number, j0: number, j1: number): Matrix
```

Rows i0..i1 and columns j0..j1 of A (1-based, inclusive) as a new matrix (or out).

### `setsubmatrix`

```luau
setsubmatrix(A: Matrix, i0: number, j0: number, B: Matrix): Matrix
```

Copies B into A with B's first entry at (i0, j0) (1-based); returns A.

### `hstack`

```luau
hstack(list: { Matrix }): Matrix
```

[A1 A2 ...]: the matrices side by side (equal row counts).

### `vstack`

```luau
vstack(list: { Matrix }): Matrix
```

[A1; A2; ...]: the matrices stacked vertically (equal column counts).

### `adjoint`

```luau
adjoint(out: Matrix?, A: Matrix): Matrix
```

Conjugate transpose.
