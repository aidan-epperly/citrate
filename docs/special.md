# Special functions and integration

Reference: [special](reference/special.md), [quad](reference/quad.md),
[stats](reference/stats.md) (the normal distribution).

## Special functions

```luau
local special = citrate.special
special.gamma(4.5), special.lgamma(200)       -- Gamma, log |Gamma|
special.beta(2, 3)
special.erf(1), special.erfc(10)              -- erfc keeps its accuracy in the tail
special.gamma_inc(a, x), special.beta_inc(a, b, x)   -- regularized incomplete forms
special.besselj(n, x), special.bessely(n, x)  -- integer order n >= 0
special.besseli(n, x), special.besselk(n, x)
```

Against SciPy's values (`tests/special_data.luau`, generated from `scipy.special`):

| function | method | accuracy | time (native) |
| --- | --- | --- | --- |
| `gamma`, `lgamma` | Lanczos (g = 7, 9 terms), reflection below 1/2 | 1e-15 relative (1e-13 near overflow, x = 170) | 43 ns |
| `erf`, `erfc` | from the incomplete gamma function | 2e-15; erfc 2e-14 out to x = 20 | 105-143 ns |
| `gamma_inc`, `beta_inc` | series or continued fraction (Lentz) | 5e-15 | |
| `besselj`, `besseli` | power series for small x; their integrals over [0, pi] by Gauss-Legendre; Hankel's expansion for large x | 1e-14 times max(1, value) (relative for small values) | 0.1-1.2 us |
| `bessely`, `besselk` | integrals for orders 0 and 1, then upward recurrence (stable for these) | 1e-14 times max(1, value) | 3-11 us |

Oscillating functions (`besselj`, `bessely`) are accurate to about 1e-15 absolutely, so
not relatively near their zeros.

## Integration

```luau
local quad = citrate.quad
local v, info = quad.integrate(function(x) return math.exp(-x * x) end, -math.huge, math.huge)
-- v = sqrt(pi); info.error is an estimate, info.evaluations the calls of f
local w = quad.integrate(function(x) return 1 / math.sqrt(x) end, 0, 1, { method = "tanh-sinh" })
```

`method = "gauss-kronrod"` (the default) splits the interval wherever the 7-point Gauss and
15-point Kronrod estimates disagree most (QUADPACK's strategy and error estimate). It
handles smooth integrands, kinks and steep spots inside the interval, and infinite limits
(mapped to finite ones). `method = "tanh-sinh"` substitutes x = tanh(pi/2 sinh t), which
crowds points at the ends so fast that singularities there converge as quickly as smooth
integrands:

| integrand | Gauss-Kronrod | tanh-sinh |
| --- | --- | --- |
| exp(-x^2) on the whole line | 2e-16 error, 435 evaluations | 1e-16, 409 |
| 1 / sqrt(x) on [0, 1] | 4e-12, 1965 | 6e-13, 54 |
| log x on [0, 1] | 2e-13, 1005 | 1e-16, 54 |
| \|x - 1/3\| on [0, 1] | 1e-13, 525 | fails: a kink inside |
| cos(100 x) on [0, 1] | 5e-15, 945 | 5e-14, 428 |

So: Gauss-Kronrod unless the trouble is at an end. For a smooth function integrated many
times on one interval, a fixed rule is cheaper: `cheb.legpts` (Gauss-Legendre) or
`cheb.sum` (Clenshaw-Curtis on a Chebyshev interpolant).

## The normal distribution

```luau
stats.normal_cdf(1.96)          -- 0.975: the chance a standard normal number is below 1.96
stats.normal_quantile(0.975)    -- 1.96, its inverse
stats.normal_pdf(x, mean, sd)
```

`normal_cdf` is `erfc`-based, accurate far into the tails; `normal_quantile` refines
Abramowitz and Stegun's approximation (26.2.23) by Newton's method to full accuracy, down
to p = 1e-300.
