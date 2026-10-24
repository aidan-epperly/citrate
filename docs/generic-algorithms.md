# Writing generic algorithms

An algorithm that should run on every representation takes a `Space<T>` (see
`src/types.luau`) and touches its state only through it: `similar`, `copy`, `add`, `sub`,
`scale`, `axpy`, `dot` and `norm`, with optional fields such as `eps` (the unit roundoff),
`errnorm` (for ODE error control) and `cg_update` (a fused conjugate gradient update).

```luau
local function combination<T>(S: citrate.Space<T>, x: T, y: T): T
	local acc = S.copy(S.similar(x), x)
	return S.axpy(acc, -2, y, acc)
end

combination(citrate.vec.space, vector.create(1, 2, 3), vector.create(0, 1, 0))
combination(citrate.f32.space, citrate.f32.from({ 1, 2, 3 }), citrate.f32.from({ 0, 1, 0 }))
```

Operations take an `out` first and return the result: with buffers the result is written
into `out`, while `vector`s are values, so the returned one is the result and `out` is
ignored. Always use the returned value.

`ComplexSpace<T>` adds complex coefficients (`cdot`, `cscale`, `caxpy`) for algorithms
whose scalars are complex (GMRES, BiCGSTAB); the array modules' spaces provide it, and the
real ones reject a nonzero imaginary coefficient.
