# Integration

Adaptive integration over finite or infinite intervals: Gauss-Kronrod, or tanh-sinh for singularities at the ends.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `integrate`

```luau
integrate(f: (number) -> number, a: number, b: number, options: QuadOptions?): (number, QuadInfo)
```

The integral of f over [a, b] (a or b may be infinite: -math.huge, math.huge), to relative tolerance options.tol (default 1e-10). options.method: "gauss-kronrod" (the default) or "tanh-sinh" (for singularities at the ends). Returns the value and info: error (an estimate), evaluations, converged (false if the tolerance was not met within options.maxintervals, default 1000, or 12 tanh-sinh levels), the method and, for Gauss-Kronrod, the number of intervals.

### `XGK`

```luau
XGK -- (a value)
```

### `WGK`

```luau
WGK -- (a value)
```

### `WG`

```luau
WG -- (a value)
```
