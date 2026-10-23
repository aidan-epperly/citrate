# citrate

A numerical math library for Luau that runs in Roblox and in standalone Luau.

Algorithms are written once and work in whatever representation you hand them: a
`vector` (computed with native single-precision arithmetic), or buffers of single- or
double-precision real or complex numbers. The data stays in that representation for the
whole computation.

## Contents

| Module | What it is |
| --- | --- |
| `f32`, `f64` | Real arrays (buffers) and column-major matrices: elementwise kernels, `axpy`, `dot`, `norm`, `matvec`, `matmul`, `transpose`, LU with partial pivoting, Cholesky, `solve`, `det`, Householder QR, column-pivoted QR with numerical rank, least squares |
| `c32`, `c64` | The same for complex arrays (interleaved real and imaginary parts), plus `conj`, `adjoint`, `dotc`, complex `cscale`/`caxpy` |
| `complex64` | Double-precision complex scalars as `(re, im)` pairs of numbers |
| `complex32` | Single-precision complex scalars stored in a `vector` |
| `vec` | `vector` as a real state space of dimension at most 3 |

```luau
local citrate = require("./citrate/src")
local f64 = citrate.f64

local A = f64.matfrom({ { 0, 1 }, { 2, 3 } })
local x = f64.solve(nil, A, f64.from({ 1, 8 }))
print(f64.totable(x)) --> { 2.5, 1 }
```

## Conventions

- Indices are 1-based. Matrices are `{ rows, cols, data }`, stored column-major.
- Every operation that produces an array or matrix takes an `out` argument first. Pass a
  buffer to write into it (it may alias an input for elementwise operations) or `nil` to
  allocate. Always use the returned value.
- Scalars such as step sizes and tolerances are plain (double-precision) numbers. Complex
  scalars are `(re, im)` pairs for `c64` and `vector`s for `c32`.
- Buffer kernels compute in double precision and round once per stored value.
- Decompositions work in place: `lu(A)`, `chol(A)`, `qr(A)` and `qrp(A)` overwrite `A` with their factors
  (LAPACK's layouts) and return it inside the factorization. `lu_copy`, `chol_copy`,
  `qr_copy` and `qrp_copy` factor a copy and leave `A` alone. `det` never modifies `A`.
  `qr_apply_q` and `qr_apply_qh` work in place on their vector.
- `solve(out, A, b, form)` takes a matrix or a factorization you already have; `form`
  says which (`"matrix"`, the default, `"lu"`, `"chol"`, `"qr"` or `"qrp"`). A square matrix is solved by
  LU, a tall one by QR least squares and a wide one by column-pivoted QR (the basic
  solution). Nothing passed to `solve` is modified.

## Writing generic algorithms

An algorithm that should run on every representation takes a `Space<T>` (see
`src/types.luau`) and touches its state only through it:

```luau
local function combination<T>(S: citrate.Space<T>, x: T, y: T): T
	local acc = S.copy(S.similar(x), x)
	return S.axpy(acc, -2, y, acc)
end

combination(citrate.vec.space, vector.create(1, 2, 3), vector.create(0, 1, 0))
combination(citrate.f32.space, citrate.f32.from({ 1, 2, 3 }), citrate.f32.from({ 0, 1, 0 }))
```

## Development

`src/f32.luau`, `src/f64.luau`, `src/c32.luau` and `src/c64.luau` are generated from
`templates/` by `python tools/generate.py`; edit the templates. One module per element
type is needed because Luau only optimizes `buffer.readf64` and friends into fast calls
when they are written out literally.

Run the tests and the type checker with the [Luau](https://github.com/luau-lang/luau/releases)
command-line tools:

```sh
python tools/generate.py --check
luau-analyze src/*.luau tests/*.luau bench/*.luau
luau tests/run.luau
```

## AI acknowledgement

citrate was written with Claude, Anthropic's AI model, working in Claude Code. Aidan
Epperly set the goals and made the design decisions: algorithms that are generic over how
data is represented, the precision and storage conventions, in-place decompositions with
copying versions, the `solve` interface and which algorithms to build. He also chose
between alternatives using the benchmarks in `bench/`. Claude wrote the code, the tests,
the benchmarks and this documentation.

**The code has not yet been verified by a human.** The test suite passes, but until that
review is done, check results that matter independently.

## License

citrate is licensed under the [Mozilla Public License 2.0](LICENSE).
