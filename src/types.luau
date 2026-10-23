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

-- What a matrix argument holds: the matrix itself or one of its factorizations.
export type Form = "matrix" | "lu" | "qr" | "qrp" | "chol"

-- The operations a representation-generic algorithm may use on a state of type T.
-- Every operation that produces a state takes an `out` slot and returns the result:
-- mutable representations (buffers) write into `out` and return it, immutable ones
-- (vector) ignore it. Algorithms must always use the returned value.
export type Space<T> = {
	similar: (x: T) -> T,
	copy: (out: T, x: T) -> T,
	add: (out: T, x: T, y: T) -> T,
	sub: (out: T, x: T, y: T) -> T,
	scale: (out: T, a: number, x: T) -> T,
	axpy: (out: T, a: number, x: T, y: T) -> T,
	norm: (x: T) -> number,
}

return {}
