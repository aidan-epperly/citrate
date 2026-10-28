# HODLR kernel matrices

Kernel matrices on many points compressed as HODLR matrices: factor, solve and log-determinant in about n k^2 log n.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `new`

```luau
new(points: { any }, kernel: (number) -> number, options: HodlrOptions?): Hodlr
```

A compressed kernel matrix on the points (vectors or { x, y, z } tables), for a kernel given as a function of distance. options: tol (default 1e-10, relative, for each block's decomposition), leaf (points in a leaf, default 64), proxies (per block, default 400).
