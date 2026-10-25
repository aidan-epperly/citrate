--!strict

-- Dense column-major matrix: entry (i, j) is element (j - 1) * rows + (i - 1) of `data`.
-- The element type (f32, f64, c32, c64) is set by the module that operates on it.
export type Matrix = { rows: number, cols: number, data: buffer }

-- LU factorization with partial pivoting, P A = L U. `lu` stores L below the diagonal
-- (unit diagonal implied) and U on and above it; row k was swapped with row piv[k].
export type LU = { lu: Matrix, piv: { number }, sign: number, singular: boolean }

-- Householder QR factorization A = Q R in LAPACK's layout. `qr` holds R on and above the
-- diagonal and, below it, the vectors v_k of the reflectors H_k = I - tau_k v_k v_k^H
-- (v_k has a 1 in row k, not stored, and zeros above it). Q = H_1 H_2 ... H_min(rows, cols).
-- `tau` is an array of the same element type as `qr`.
export type QR = { qr: Matrix, tau: buffer }

-- Column-pivoted QR, A P = Q R: as QR, and column j of A P is column perm[j] of A.
-- Functions taking a QR also accept a QRP (they act on Q and R).
export type QRP = { qr: Matrix, tau: buffer, perm: { number } }

-- Cholesky factorization A = L L^H: `chol` holds L in its lower triangle (the strictly
-- upper triangle is whatever A had there). `definite` is false if A was not positive
-- definite, in which case `chol` is only partly factored.
export type Cholesky = { chol: Matrix, definite: boolean }

-- Upper Hessenberg reduction A = Q H Q^H: `hess` holds H on and above the subdiagonal and,
-- below it, the vectors v_k of the reflectors H_k = I - tau_k v_k v_k^H (v_k has a 1 in row
-- k+1, not stored, and zeros above it). Q = H_0 H_1 ... H_n-3.
export type Hessenberg = { hess: Matrix, tau: buffer }

-- Eigendecomposition of a symmetric or Hermitian matrix: `values` is a real array (same
-- precision as the matrix) in ascending order, and column j of `vectors`, if computed, is
-- a unit eigenvector for values[j].
export type Eigh = { values: buffer, vectors: Matrix? }

-- Schur factorization A = Z T Z^H: `schur` holds T (upper triangular for complex
-- matrices; quasi-triangular with standardized 2 by 2 blocks for real ones), `z` the
-- Schur vectors if computed, and `values` the eigenvalues as a complex array.
export type Schur = { schur: Matrix, z: Matrix?, values: buffer }

-- Eigendecomposition of a general matrix: `values` is a complex array and column k of
-- the complex matrix `vectors`, if computed, is a unit eigenvector for values[k].
export type Eig = { values: buffer, vectors: Matrix? }

-- Singular value decomposition A = U diag(values) V^H: `values` is a real array (same
-- precision as the matrix) in descending order, and `u` (rows by k) and `v` (cols by k),
-- if computed, have orthonormal columns, k = min(rows, cols).
export type Svd = { values: buffer, u: Matrix?, v: Matrix? }

-- Polar decomposition A = U H: U with orthonormal columns (rows, for wide A), H Hermitian
-- positive semidefinite; iterations counts the QDWH steps.
export type Polar = { u: Matrix, h: Matrix, iterations: number }

-- Bunch-Kaufman factorization P A P^T = L D L^H of a symmetric (Hermitian) matrix, in
-- LAPACK's layout; piv as LAPACK's ipiv (1-based; negative for 2 by 2 blocks).
export type LDL = { ldl: Matrix, piv: { number }, singular: boolean }

-- Generalized eigenvalues values = alpha / beta of A x = lambda B x (complex arrays;
-- beta = 0 marks an infinite eigenvalue) and, optionally, unit eigenvectors.
export type GenEig = { values: buffer, alpha: buffer, beta: buffer, vectors: Matrix? }

-- An explicit full QR factorization A = Q R (Q square unitary, R upper triangular with
-- A's shape), the form qr_update, qr_insert_col and qr_delete_col work on.
export type QRFull = { q: Matrix, r: Matrix }

-- A randomly pivoted Cholesky approximation A ~ F F^H: factor is F (n by rank), pivots
-- the chosen columns (1-based) and residual the trace of A - F F^H.
export type RPCholesky = { factor: Matrix, pivots: { number }, rank: number, residual: number }

-- A randomly pivoted QR approximation A ~ Q R (Q orthonormal, R = Q^H A): pivots are the
-- chosen columns (1-based) and residual is |A - Q R|_F^2.
export type RPQR = { q: Matrix, r: Matrix, pivots: { number }, rank: number, residual: number }

-- A plan for FFTs of length n (twiddle factors and workspace). `impl` is internal. A plan
-- belongs to the module (precision) that made it.
export type FFTPlan = { n: number, impl: any }

-- Scaling of the discrete cosine transforms: nil for SciPy's unnormalized definitions,
-- "ortho" for the orthonormal ones.
export type DctNorm = "ortho"

-- Which triangle of a matrix to use.
export type Uplo = "upper" | "lower"

-- Tridiagonal n by n matrix: the subdiagonal (n-1 entries), diagonal (n) and
-- superdiagonal (n-1) as arrays of the module's element type.
export type Tridiagonal = { lower: buffer, diag: buffer, upper: buffer }

-- LU factorization of a tridiagonal matrix with partial pivoting (LAPACK's gttrf):
-- `lu` holds the multipliers and the first two superdiagonals of U, `du2` the third
-- (fill-in from row swaps), and row k was swapped with row piv[k].
export type TridiagonalLU = { lu: Tridiagonal, du2: buffer, piv: { number }, singular: boolean }

-- LDL^H factorization of a symmetric or Hermitian positive definite tridiagonal matrix
-- (LAPACK's pttrf): `ldl.diag` holds D and `ldl.lower` the subdiagonal of the unit L.
export type TridiagonalLDL = { ldl: Tridiagonal, definite: boolean }

-- Banded n by n matrix with kl subdiagonals and ku superdiagonals in LAPACK's band storage,
-- with kl extra rows so that LU can factor in place: entry (i, j) (0-based) is row
-- kl + ku + i - j of column j of `data`, which has ld = 2 kl + ku + 1 rows.
export type Banded = { n: number, kl: number, ku: number, ld: number, data: buffer }

-- Banded LU with partial pivoting (LAPACK's gbtrf): `lu` holds L's multipliers below the
-- diagonal and U (with up to kl + ku superdiagonals) above; row k was swapped with piv[k].
export type BandedLU = { lu: Banded, piv: { number }, singular: boolean }

-- Banded Cholesky A = L L^H (LAPACK's pbtrf, lower): `chol` holds L in its lower band.
export type BandedCholesky = { chol: Banded, definite: boolean }

-- Circulant matrix (first column c), stored as its eigenvalues fft(c): all n of them
-- (complex modules) or the first n // 2 + 1 (real modules, as rfft gives them).
export type Circulant = { n: number, eig: buffer }

-- Toeplitz matrix with first column `col` and first row `row` (row[1] is ignored, as in
-- SciPy), plus a circulant embedding of power-of-two size for fast products.
export type Toeplitz = { n: number, col: buffer, row: buffer, embed: Circulant }

-- Sparse rows by cols matrix in compressed sparse column form: the entries of column j
-- (0-based) are values[p] at rows rowind[p] for colptr[j] <= p < colptr[j + 1], rows
-- strictly increasing. colptr and rowind are buffers of u32 (0-based); values is an
-- array of the module's element type. symmetric (S^T = S) and hermitian (S^H = S) are the
-- caller's promise, not checked: with one set, sparse_matvec computes S x as a product with
-- the transpose (a gather down the columns, faster than the scatter a general S needs).
export type Sparse = { rows: number, cols: number, colptr: buffer, rowind: buffer, values: buffer, symmetric: boolean?, hermitian: boolean? }

-- Block sparse matrix (block compressed sparse row) with dense block by block blocks;
-- rows and cols are multiples of block. Block row I (0-based) holds blocks at block columns
-- colind[p] for rowptr[I] <= p < rowptr[I + 1], strictly increasing; block p's block^2
-- entries are values[p * block^2 ..], column-major within the block. rowptr and colind
-- are buffers of u32 (0-based). symmetric and hermitian are promises as for Sparse (they
-- make the adjoint product a plain product). Products cost a fraction of a Sparse's with
-- the same entries when the blocks are full (one index per block, each x entry read once
-- per block): suited to systems with several unknowns per node (positions in 2D or 3D).
-- Options for bsparse_amg (all optional): fixed lists components (1-based) held in place
-- (their rows and columns act as the identity, as in ode.implicit_mechanics); smooth
-- (default true) smooths the prolongation by one damped Jacobi step (smoothed aggregation;
-- false gives plain aggregation); theta (default 0.08) is the strength threshold for
-- aggregating two nodes, |A_IJ|_F^2 >= theta^2 |A_II|_F |A_JJ|_F; alpha (default 1) scales
-- the coarse correction; coarsest (default 64) is the size solved densely (Cholesky costs
-- its cube; another level costs less).
export type AMGOptions = { fixed: { number }?, smooth: boolean?, theta: number?, alpha: number?, coarsest: number? }

-- An algebraic multigrid preconditioner (bsparse_amg): apply(out, r) is one symmetric
-- V-cycle (out := M^-1 r, symmetric positive definite, for CG); update(B) refills the
-- hierarchy from new values on the same pattern (the aggregates stay); sizes lists each
-- level's number of unknowns.
export type BlockAMG = { apply: (out: buffer, r: buffer) -> buffer, update: (B: BlockSparse) -> (), sizes: { number } }

export type BlockSparse = { rows: number, cols: number, block: number, rowptr: buffer, colind: buffer, values: buffer, symmetric: boolean?, hermitian: boolean? }

-- Incomplete LU factorization with no fill-in, ILU(0): `lu` holds L (unit diagonal, not
-- stored) below the diagonal and U on and above it, in the sparsity pattern of the
-- original matrix; diag[j] is the position of the diagonal entry of column j.
export type SparseILU = { lu: Sparse, diag: { number }, singular: boolean }

-- Incomplete Cholesky factorization with no fill-in, IC(0): `ic` holds L on and below the
-- diagonal (the strictly upper part is untouched); diag[j] is the position of the
-- diagonal entry of column j. definite is false if it broke down.
export type SparseIC = { ic: Sparse, diag: { number }, definite: boolean }

-- A Chebyshev approximation (see cheb): the Chebyshev coefficients of a polynomial on
-- [a, b], in the variable t = (2x - a - b) / (b - a), as an f64 array; happy says whether
-- construction resolved the function to the requested tolerance.
export type Chebfun = { a: number, b: number, coeffs: buffer, happy: boolean }

-- A piecewise Bézier curve (see curve.luau). Segment i covers [breaks[i], breaks[i + 1]]
-- and has the degree + 1 control points points[(i - 1) * (degree + 1) + 1 ..
-- i * (degree + 1)]; weights, if present, makes it rational (one positive weight per
-- control point). vector says the points are `vector`s (the native fast path).
export type Curve<T> = {
	space: Space<T>,
	vector: boolean,
	degree: number,
	breaks: { number },
	points: { T },
	weights: { number }?,
	closed: boolean,
}

-- Options for the curve constructors; each reads only the fields its comment lists.
export type CurveOptions<T> = {
	space: Space<T>?,
	breaks: { number }?,
	weights: { number }?,
	degree: number?,
	knots: { number }?,
	closed: boolean?,
	alpha: number?,
	ends: ("natural" | "clamped" | "not-a-knot")?,
	start: T?,
	finish: T?,
	tension: (number | { number })?,
	continuity: (number | { number })?,
	bias: (number | { number })?,
}

-- What a matrix argument holds: the matrix itself or one of its factorizations.
export type Form = "matrix" | "lu" | "qr" | "qrp" | "chol" | "ldl" | "svd"

-- The operations a representation-generic algorithm may use on a state of type T.
-- Every operation that produces a state takes an `out` slot and returns the result:
-- mutable representations (buffers) write into `out` and return it, immutable ones
-- (vector) ignore it. Algorithms must always use the returned value. `dot` is the real
-- inner product Re<x, y>: the ordinary dot product for real states, and the real part
-- of the Hermitian one for complex states (the inner product of the underlying real
-- vector space), so it is always a single number.
export type Space<T> = {
	-- Optional: the unit roundoff of the representation (for default tolerances), and
	-- whether it is real (complex coefficients are then rejected).
	eps: number?,
	real: boolean?,
	-- Optional: the weighted RMS norm of an ODE error estimate e (see ode), per component:
	-- sqrt(mean((|e_i| / (atol + rtol max(|y_i|, |ynew_i|)))^2)).
	errnorm: ((e: T, y: T, ynew: T, atol: number, rtol: number) -> number)?,
	similar: (x: T) -> T,
	copy: (out: T, x: T) -> T,
	add: (out: T, x: T, y: T) -> T,
	sub: (out: T, x: T, y: T) -> T,
	scale: (out: T, a: number, x: T) -> T,
	axpy: (out: T, a: number, x: T, y: T) -> T,
	dot: (x: T, y: T) -> number,
	norm: (x: T) -> number,
	-- Optional: x := x + a p and r := r - a q in place, returning dot(r, r) of the new r
	-- (one pass instead of three; conjugate gradients uses it when present).
	cg_update: ((x: T, r: T, p: T, q: T, a: number) -> number)?,
}

-- A Space with complex coefficients, for algorithms whose scalars are genuinely complex
-- (GMRES, BiCGSTAB). Complex numbers are (re, im) pairs. cdot is the Hermitian inner
-- product sum conj(x_i) y_i. Real spaces implement it with zero imaginary parts and
-- reject a nonzero imaginary coefficient (real data never produces one).
export type ComplexSpace<T> = Space<T> & {
	cdot: (x: T, y: T) -> (number, number),
	cscale: (out: T, ar: number, ai: number, x: T) -> T,
	caxpy: (out: T, ar: number, ai: number, x: T, y: T) -> T,
}

-- A linear operator for the iterative solvers: writes A x into out (or, for immutable
-- states, returns it) and returns the result.
export type Operator<T> = (out: T, x: T) -> T

-- Options for the iterative solvers (all optional). x0 is the initial guess and is
-- overwritten with the solution. precond applies M^-1 (CG: symmetric positive definite
-- M; GMRES and BiCGSTAB: right preconditioning). tol is the relative residual target
-- |b - A x| <= tol |b| (default 1e-5, as in SciPy), maxiter the iteration limit (default
-- 1000), restart the GMRES restart length (default 30). MINRES and CG need a symmetric
-- (Hermitian) positive definite preconditioner.
export type KrylovOptions<T> = {
	x0: T?,
	precond: Operator<T>?,
	tol: number?,
	maxiter: number?,
	restart: number?,
}

-- What an iterative solver reports: whether it met tol, the iterations used, and the
-- true relative residual |b - A x| / |b| of the returned x.
export type KrylovInfo = { converged: boolean, iterations: number, residual: number }

-- The right-hand side of an ODE y' = f(t, y): it may write into out (buffers) or return a
-- new value (vectors), and returns f(t, y).
export type OdeFunction<T> = (t: number, y: T, out: T) -> T

-- An event for ode.solve: a zero of g(t, y). direction 1 (-1) counts only increasing
-- (decreasing) crossings; a terminal event stops the integration.
export type OdeEvent<T> = { g: (t: number, y: T) -> number, terminal: boolean?, direction: number? }

-- Options for ode.solve and ode.integrator (all optional; see ode.luau).
export type OdeOptions<T> = {
	method: string?,
	rtol: number?,
	atol: number?,
	h: number?,
	max_step: number?,
	max_steps: number?,
	dense: boolean?,
	t_eval: { number }?,
	events: { OdeEvent<T> }?,
	jac: any?,
	jac_band: { number }?,
	mass: any?,
	linear: string?,
	jac_sparsity: any?,
	precond: any?,
	krylov: { restart: number?, maxiter: number? }?,
	autonomous: boolean?,
	stiff_method: string?,
}

-- The result of ode.solve: the steps, the status, statistics, events found, the solution
-- at t_eval, and at(t), the dense output.
export type OdeSolution<T> = {
	t: { number },
	y: { T },
	status: string,
	message: string,
	stats: { nfev: number, steps: number, rejected: number, njev: number?, nlu: number?, nlinear: number?, switches: number? },
	events: { { t: number, y: T, index: number } },
	t_eval: { number }?,
	y_eval: { T }?,
	at: (t: number, out: T?) -> T,
}

-- Options for the Krylov eigensolvers (all optional): which eigenvalues ("LM", "SM", and
-- "LA"/"SA" for eigsh or "LR"/"SR"/"LI"/"SI" for eigs), the relative residual tolerance,
-- the maximum number of restarts, and the basis size ncv.
export type EigOptions = { which: string?, tol: number?, maxiter: number?, ncv: number? }

-- What a Krylov eigensolver did: whether all requested pairs converged, the number of
-- restarts, and how many converged.
export type EigInfo = { converged: boolean, restarts: number, nconv: number }


-- Lightning Laplace solver (lightning.laplace).

-- A boundary condition on one side: "dirichlet" gives u = value(x, y), "neumann" gives the
-- outward normal derivative du/dn = value(x, y).
export type LightningCondition = { kind: string, value: (x: number, y: number) -> number }

export type LightningOptions = {
	-- Poles per vertex (default 20) and their clustering (default 4).
	poles: number?,
	sigma: number?,
	-- The polynomial degree (default the number of poles per vertex).
	degree: number?,
	-- A constant source f: solves Laplacian(u) = f.
	source: number?,
}

export type LightningSolution = {
	-- u at (x, y), and its gradient.
	eval: (x: number, y: number) -> number,
	gradient: (x: number, y: number) -> (number, number),
	-- The largest boundary residual on a finer set of points than the fit used, relative
	-- to the largest boundary datum (or absolute if that is zero), both weighted as in the
	-- fit (see weight below).
	error: number,
	-- The number of real unknowns of the fit.
	unknowns: number,
}

return {}
