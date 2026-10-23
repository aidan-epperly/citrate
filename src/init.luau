--!strict

local types = require("@self/types")

export type Matrix = types.Matrix
export type LU = types.LU
export type QR = types.QR
export type QRP = types.QRP
export type Cholesky = types.Cholesky
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
}
