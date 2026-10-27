# Matrix functions

Exponential, square root, logarithm; Sylvester and Lyapunov equations. Parts `matfun_f32`, `matfun_f64`, `matfun_c32` and `matfun_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `expm`

```luau
expm(A: Matrix): Matrix
```

### `sqrtm`

```luau
sqrtm(A: Matrix): Matrix
```

The principal square root of A (X^2 = A, eigenvalues in the right half plane). It is real unless A has a negative eigenvalue, which is an error here (use the complex module). A is not modified.

### `logm`

```luau
logm(A: Matrix): Matrix
```

The principal logarithm of A. It is real unless A has an eigenvalue on the closed negative real axis (a singular A has none at all), which is an error here (use the complex module for negative eigenvalues). A is not modified.

### `sylvester`

```luau
sylvester(A: Matrix, B: Matrix, C: Matrix): Matrix
```

The solution X of A X + X B = C (see the complex module's sylvester; the solution of a real equation is real). Nothing is modified.

### `lyapunov`

```luau
lyapunov(A: Matrix, Q: Matrix): Matrix
```

The solution X of the continuous Lyapunov equation A X + X A^T = Q.

### `dlyapunov`

```luau
dlyapunov(A: Matrix, Q: Matrix): Matrix
```

The solution X of the discrete Lyapunov equation A X A^T - X + Q = 0 (see the complex module's dlyapunov; real for real A and Q). Nothing is modified.

### `sqrtm_triangular`

```luau
sqrtm_triangular(T: Matrix, out: Matrix?): Matrix
```

The principal square root R (R^2 = T) of an upper triangular T, column by column (Björck and Hammarling): r_ii = sqrt(t_ii), and for i < j r_ij = (t_ij - sum_(i<k<j) r_ik r_kj) / (r_ii + r_jj). Undefined (infinite entries) when r_ii + r_jj = 0, which needs a zero eigenvalue with a nontrivial Jordan block or two eigenvalues on opposite branches.

### `logm_triangular`

```luau
logm_triangular(T: Matrix): Matrix
```

The principal logarithm of an upper triangular T by inverse scaling and squaring: square roots until |T^(1/2^s) - I|_1 <= 1/4, then the quadrature rule above (a diagonal Padé approximant of log(1 + x)), times 2^s. An error if T is singular to working precision.
