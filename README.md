# citrate

A numerical math library for Luau that runs in Roblox and in standalone Luau.

Algorithms are written once and work in whatever representation you hand them: a `vector`
(computed with native single-precision arithmetic), or buffers of single- or
double-precision real or complex numbers.

- Dense linear algebra: LU, Cholesky, QR, eigenvalues, the SVD, matrix functions
- Structured and sparse matrices, preconditioners, algebraic multigrid
- Iterative solvers: CG, MINRES, GMRES, BiCGSTAB, Lanczos, Krylov-Schur
- Differential equations: explicit, stiff, second-order and symplectic integrators,
  boundary value problems
- FFTs, Chebyshev approximation, curves and splines

```luau
local citrate = require("./citrate/src")
local f64 = citrate.f64

local A = f64.matfrom({ { 0, 1 }, { 2, 3 } })
local x = f64.solve(nil, A, f64.from({ 1, 8 }))
print(f64.totable(x)) --> { 2.5, 1 }
```

In Roblox, sync `default.project.json` with [Rojo](https://rojo.space).

**Documentation:** [guides and reference](docs/README.md).

## Testing

```sh
python tools/generate.py --check
luau-analyze src/*.luau tests/*.luau bench/*.luau
luau tests/run.luau
```

See [Development](docs/development.md) for running the suite in Roblox.

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

citrate is licensed under the [Mozilla Public License 2.0](LICENSE). Some method
coefficients come from SciPy (BSD-3-Clause) and OrdinaryDiffEq.jl (MIT); see
[third-party coefficients](docs/development.md#third-party-coefficients) and `licenses/`.
