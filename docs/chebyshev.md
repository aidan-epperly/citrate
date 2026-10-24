# Chebyshev approximation

Reference: [cheb](reference/cheb.md).

`cheb` approximates functions on an interval in the spirit of Chebfun: Chebyshev points and
barycentric interpolation, coefficient transforms, Clenshaw evaluation, adaptive
construction to machine precision (`fun`, with Aurentz and Trefethen's `standard_chop`),
arithmetic, derivatives, integrals, norms, roots (colleague matrices with balancing and
recursive splitting), extrema, the Chebyshev differentiation matrix, and Clenshaw-Curtis
and Gauss-Legendre quadrature.

```luau
local cheb = citrate.cheb
local F = cheb.fun(function(x) return math.exp(x) * math.sin(5 * x) end, { 0, 2 })
print(cheb.length(F), F.happy)   --> the degree + 1 needed for machine precision, true
print(cheb.eval(F, 1.3))         --> F(1.3)
print(cheb.sum(F))               --> the integral over [0, 2]
local r = cheb.roots(F)          --> its roots, as an f64 array
local hi, where = cheb.max(F)
```

Chebfuns are real-valued and live on a single interval; the boundary value solver in
`ode.bvp` returns its solutions as Chebfuns.
