# Matrix products

`matmul` and `gemm`, by register-blocked kernels. Parts `gemm_f32`, `gemm_f64`, `gemm_c32` and `gemm_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `matmul`

```luau
matmul(out: Matrix?, A: Matrix, B: Matrix): Matrix
```

out := A B (a new matrix if out is nil); see gemm for scaled and transposed products.

### `gemm`

```luau
gemm(C: Matrix?, alpha: number, A: Matrix, B: Matrix, beta: number?, opa: string?, opb: string?): Matrix
```

C := alpha op(A) op(B) + beta C (BLAS gemm), where op is "n" (as is), "t" or "c" (transposed; the same for real matrices). C is new (or out) when beta is 0 or nil. Sums run in double precision and each entry is rounded once.
