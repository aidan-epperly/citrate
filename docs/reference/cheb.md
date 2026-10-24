# Chebyshev approximation

Chebfun-style approximation on an interval.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `chebpts`

```luau
chebpts(n: number, kind: number?, a: number?, b: number?): buffer
```

n Chebyshev points on [a, b] (default [-1, 1]) in ascending order: of the second kind (the extrema of T_n-1, including the endpoints; the default) or of the first kind (the roots of T_n). The sine formulas make the points exactly symmetric.

### `chebweights`

```luau
chebweights(n: number, kind: number?): buffer
```

Barycentric weights for n Chebyshev points of the given kind (closed forms, scaled to have largest magnitude 1).

### `bary_weights`

```luau
bary_weights(nodes: buffer): buffer
```

Barycentric weights w_j = 1 / prod_(k ~= j) (x_j - x_k) for arbitrary distinct nodes, with each factor scaled by 4 / (max - min) against overflow and underflow, and the result scaled to have largest magnitude 1 (the formula is invariant under scaling).

### `bary`

```luau
bary(x: number, nodes: buffer, values: buffer, weights: buffer?): number
```

The barycentric interpolation formula (second, "true" form): the value at x of the polynomial through (nodes[j], values[j]), given barycentric weights (default: bary_weights(nodes); pass chebweights(n, kind) for Chebyshev points). Exact at the nodes, and numerically stable for Chebyshev points (Higham 2004).

### `vals2coeffs`

```luau
vals2coeffs(values: buffer): buffer
```

Chebyshev coefficients of the polynomial taking the given values at the n Chebyshev points of the second kind (ascending), by a type 1 DCT.

### `coeffs2vals`

```luau
coeffs2vals(coeffs: buffer): buffer
```

Values at the n Chebyshev points of the second kind (ascending) of the Chebyshev series with the given n coefficients.

### `clenshaw`

```luau
clenshaw(coeffs: buffer, t: number): number
```

Value at t in [-1, 1] of sum c_k T_k(t), by Clenshaw's recurrence.

### `standard_chop`

```luau
standard_chop(coeffs: buffer, tol: number?): number
```

How many leading Chebyshev coefficients to keep: Aurentz and Trefethen's standardChop ("Chopping a Chebyshev series", 2017), as used by Chebfun. It looks for a plateau in the monotone envelope of the coefficients below tol (relative to the largest one) and cuts where the envelope plus a gentle linear tilt is smallest. Series shorter than 17 are never chopped.

### `fun`

```luau
fun(f: (number) -> number, domain: { number }?, options: FunOptions?): Chebfun
```

A Chebyshev interpolant of f on domain = { a, b } (default { -1, 1 }). By default it is adaptive, as in Chebfun: f is sampled at 17, 33, 65, ... Chebyshev points until the coefficients reach a plateau at relative size options.tol (default machine epsilon), the series is chopped there by standard_chop, and a check at a few extra points guards against being fooled. If options.maxlength (default 65537) samples do not suffice, the result keeps them all and has happy = false. options.n instead fixes the number of points (the interpolant then has degree n - 1).

### `from_coeffs`

```luau
from_coeffs(coeffs: buffer, domain: { number }?): Chebfun
```

The Chebfun with the given Chebyshev coefficients on domain (default { -1, 1 }).

### `from_values`

```luau
from_values(values: buffer, domain: { number }?): Chebfun
```

The polynomial interpolant of the given values at the Chebyshev points of the second kind on domain (default { -1, 1 }).

### `length`

```luau
length(F: Chebfun): number
```

Number of coefficients (degree + 1).

### `eval`

```luau
eval(F: Chebfun, x: number): number
```

F(x), by Clenshaw's recurrence.

### `evalmany`

```luau
evalmany(F: Chebfun, xs: buffer): buffer
```

F at each entry of xs.

### `points`

```luau
points(F: Chebfun): buffer
```

The Chebyshev points of F (second kind, on its domain) and F's values there.

### `values`

```luau
values(F: Chebfun): buffer
```

F's values at its Chebyshev points (second kind, as chebpts gives them), as an f64 array.

### `add`

```luau
add(F: Chebfun, G: Chebfun): Chebfun
```

F + G (same domain).

### `sub`

```luau
sub(F: Chebfun, G: Chebfun): Chebfun
```

F - G (same domain).

### `scale`

```luau
scale(F: Chebfun, s: number): Chebfun
```

s F.

### `shift`

```luau
shift(F: Chebfun, s: number): Chebfun
```

F + s (adds a constant).

### `mul`

```luau
mul(F: Chebfun, G: Chebfun): Chebfun
```

F G (same domain): exact product on length(F) + length(G) - 1 points, then chopped.

### `diff`

```luau
diff(F: Chebfun, k: number?): Chebfun
```

The k-th derivative of F (default k = 1), by the Chebyshev derivative recurrence.

### `cumsum`

```luau
cumsum(F: Chebfun): Chebfun
```

The indefinite integral of F that vanishes at the left endpoint a.

### `sum`

```luau
sum(F: Chebfun): number
```

The definite integral of F over its domain (Clenshaw-Curtis, from the coefficients).

### `inner`

```luau
inner(F: Chebfun, G: Chebfun): number
```

The L2 inner product of F and G (same domain).

### `norm`

```luau
norm(F: Chebfun): number
```

The L2 norm of F.

### `roots`

```luau
roots(F: Chebfun): buffer
```

The real roots of F in its domain, in ascending order, polished by Newton's method. Roots closer together than 1e-8 (relative to the domain) are merged, keeping the one where |F| is smaller: the recursion can find a root twice near its split points.

### `minmax`

```luau
minmax(F: Chebfun): (number, number, number, number)
```

The global minimum and maximum of F on its domain, with their locations: min value, where, max value, where. The candidates are the endpoints and the roots of F'.

### `max`

```luau
max(F: Chebfun): (number, number)
```

The maximum of F and where it is attained.

### `min`

```luau
min(F: Chebfun): (number, number)
```

The minimum of F and where it is attained.

### `diffmat`

```luau
diffmat(n: number, a: number?, b: number?): types.Matrix
```

The n by n Chebyshev differentiation matrix on [a, b] (default [-1, 1]): D v holds the derivative, at the n Chebyshev points of the second kind, of the interpolant of v there. Off-diagonal entries from the barycentric weights; the diagonal makes each row sum to zero (exact for constants, and more accurate than the closed form).

### `ccquad`

```luau
ccquad(n: number, a: number?, b: number?): (buffer, buffer)
```

Clenshaw-Curtis quadrature on [a, b] (default [-1, 1]): the n Chebyshev points of the second kind and weights that integrate polynomials of degree n - 1 exactly (Waldvogel's formula, as in Trefethen's clencurt).

### `legpts`

```luau
legpts(n: number, a: number?, b: number?): (buffer, buffer)
```

Gauss-Legendre quadrature on [a, b] (default [-1, 1]): n nodes (ascending) and weights integrating polynomials of degree 2n - 1 exactly, by Golub and Welsch (eigenvalues and first eigenvector components of the Jacobi matrix). O(n^3); fine for n up to a few hundred.
