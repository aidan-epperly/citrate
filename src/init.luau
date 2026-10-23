--!strict

local types = require("@self/types")

export type Matrix = types.Matrix
export type LU = types.LU
export type QR = types.QR
export type QRP = types.QRP
export type Cholesky = types.Cholesky
export type Hessenberg = types.Hessenberg
export type Eigh = types.Eigh
export type Schur = types.Schur
export type Eig = types.Eig
export type Svd = types.Svd
export type FFTPlan = types.FFTPlan
export type DctNorm = types.DctNorm
export type Uplo = types.Uplo
export type Tridiagonal = types.Tridiagonal
export type TridiagonalLU = types.TridiagonalLU
export type TridiagonalLDL = types.TridiagonalLDL
export type Banded = types.Banded
export type BandedLU = types.BandedLU
export type BandedCholesky = types.BandedCholesky
export type Circulant = types.Circulant
export type Toeplitz = types.Toeplitz
export type Sparse = types.Sparse
export type SparseILU = types.SparseILU
export type SparseIC = types.SparseIC
export type ComplexSpace<T> = types.ComplexSpace<T>
export type Operator<T> = types.Operator<T>
export type KrylovOptions<T> = types.KrylovOptions<T>
export type KrylovInfo = types.KrylovInfo
export type Form = types.Form
export type Space<T> = types.Space<T>

return {
	f32 = require("@self/f32"),
	f64 = require("@self/f64"),
	c32 = require("@self/c32"),
	c64 = require("@self/c64"),
	complex32 = require("@self/complex32"),
	complex64 = require("@self/complex64"),
	vec = require("@self/vec"),
	krylov = require("@self/krylov"),
}
