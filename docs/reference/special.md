# Special functions

Gamma and beta, their incomplete forms, erf and erfc, Bessel functions of integer order.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `gamma`

```luau
gamma(x: number): number
```

The gamma function: Gamma(n) = (n - 1)! for whole n; poles (infinity) at 0, -1, -2, ...

### `lgamma`

```luau
lgamma(x: number): number
```

The natural logarithm of |Gamma(x)|.

### `beta`

```luau
beta(a: number, b: number): number
```

The beta function B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b), for a, b > 0.

### `gamma_inc`

```luau
gamma_inc(a: number, x: number, upper: boolean?): number
```

The regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a), for a > 0 and x >= 0: from 0 at x = 0 to 1 as x grows (the chi-square and Poisson distributions' cumulative probabilities). upper = true gives Q(a, x) = 1 - P(a, x), accurately even where P is near 1.

### `beta_inc`

```luau
beta_inc(a: number, b: number, x: number): number
```

The regularized incomplete beta function I_x(a, b), for a, b > 0 and x in [0, 1]: from 0 at x = 0 to 1 at x = 1 (the beta, binomial, Student's t and F distributions' cumulative probabilities).

### `erf`

```luau
erf(x: number): number
```

The error function erf(x) = 2 / sqrt(pi) times the integral of e^-t^2 from 0 to x.

### `erfc`

```luau
erfc(x: number): number
```

The complementary error function erfc(x) = 1 - erf(x), accurate (relatively) even where it is tiny: erfc(10) = 2.1e-45.

### `besselj`

```luau
besselj(n: number, x: number): number
```

The Bessel function of the first kind J_n(x), for whole n >= 0: the shape of a drum's modes along a radius, and of waves in a cylinder. From its integral (1 / pi) times the integral of cos(n t - x sin t) over [0, pi] by Gauss-Legendre (exponentially accurate), or Hankel's expansion for large x.

### `bessely`

```luau
bessely(n: number, x: number): number
```

The Bessel function of the second kind Y_n(x), for whole n >= 0 and x > 0 (infinite at 0). Y_0 and Y_1 from their integrals (or Hankel's expansion), the rest by the upward recurrence Y_(k+1) = (2 k / x) Y_k - Y_(k-1), which is stable for Y.

### `besseli`

```luau
besseli(n: number, x: number): number
```

The modified Bessel function of the first kind I_n(x), for whole n >= 0: (1 / pi) times the integral of e^(x cos t) cos(n t) over [0, pi]. It grows like e^x / sqrt(2 pi x).

### `besselk`

```luau
besselk(n: number, x: number): number
```

The modified Bessel function of the second kind K_n(x), for whole n >= 0 and x > 0 (infinite at 0): decaying like e^-x. K_0 and K_1 from their integrals, the rest by the upward recurrence K_(k+1) = K_(k-1) + (2 k / x) K_k (stable for K).
