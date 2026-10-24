# Iterative solvers

Krylov solvers for linear systems and eigenproblems, generic over the `Space` and the operator.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `cg`

```luau
cg<T>(S: Space<T>, A: Operator<T>, b: T, options: Options<T>?): (T, Info)
```

Conjugate gradients, for symmetric (Hermitian) positive definite A. The optional preconditioner must be symmetric positive definite too. Stops early if A turns out not to be positive definite (p^H A p <= 0).

### `minres`

```luau
minres<T>(S: Space<T>, A: Operator<T>, b: T, options: Options<T>?): (T, Info)
```

MINRES (Paige and Saunders; a translation of SciPy's minres), for symmetric (Hermitian) A, which may be indefinite. It minimizes the residual over the Krylov subspace. The optional preconditioner must be symmetric (Hermitian) positive definite; MINRES then minimizes the residual in the M^-1 norm, so when that estimate meets tol the true residual is checked as well and the iteration continues if it is not yet small enough.

### `gmres`

```luau
gmres<T>(S: ComplexSpace<T>, A: Operator<T>, b: T, options: Options<T>?): (T, Info)
```

Restarted GMRES(m), m = options.restart (default 30), for general A, with optional right preconditioning. Arnoldi uses modified Gram-Schmidt run twice, so the basis stays orthonormal to working precision. maxiter counts inner iterations (matrix-vector products), over all restarts.

### `bicgstab`

```luau
bicgstab<T>(S: ComplexSpace<T>, A: Operator<T>, b: T, options: Options<T>?): (T, Info)
```

BiCGSTAB (van der Vorst) for general A, with optional right preconditioning. Stops early on a breakdown (a zero inner product).

### `eigsh`

```luau
eigsh<T>(S: Space<T>, A: Operator<T>, v0: T, k: number, options: EigOptions?): (buffer, { T }, EigInfo)
```

A few eigenvalues (as an f64 array, in the order `which` prefers) and eigenvectors of a Hermitian (symmetric) operator A, by thick-restart Lanczos: options.which is "LM" (largest magnitude, the default), "SM", "LA" (largest algebraic) or "SA"; options.ncv (default max(2k + 1, 20)) is the basis size, options.tol (default 1000 eps of the space) the relative residual |A x - lambda x| / |lambda| to reach, options.maxiter (default 300) the number of restarts. Works on any Space (complex ones need cdot and caxpy); for interior eigenvalues use a shift-invert operator. Returns values, vectors (unit, as a table) and info.

### `eigs`

```luau
eigs<T>(S: ComplexSpace<T>, A: Operator<T>, v0: T, k: number, options: EigOptions?): (buffer, { any }, EigInfo)
```

A few eigenvalues (a c64 array, in the order options.which prefers: "LM" largest magnitude (the default), "SM", "LR"/"SR" (real part), "LI"/"SI" (imaginary part)) and unit eigenvectors of a general operator A, by the Krylov-Schur method; the options are as for eigsh. On a complex space the vectors are elements of it; on a real space (Space.real) the work runs on (re, im) pairs, since eigenvectors of real operators can be complex, and each eigenvector is returned as { re = T, im = T }.
