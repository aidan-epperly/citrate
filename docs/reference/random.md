# Random numbers

A seeded generator (xoshiro128**), distributions, sampling, weighted choice, random points, and Halton and Sobol quasi-random sequences. Generators' methods are listed in the guide.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `new`

```luau
new(seed: number?): Rng
```

A generator seeded with any number (the same seed, the same numbers); without a seed, one from the clock.

### `weighted`

```luau
weighted(weights: { number }): Weighted
```

A sampler choosing index i with probability weights[i] / (their sum): weighted:sample(rng) costs the same however many weights there are (Vose's alias method, built in O(n)). Weights must be nonnegative, not all zero.

### `halton`

```luau
halton(dim: number): Sequence
```

The Halton sequence in dim dimensions (at most 32): point k has coordinate j the radical inverse of k in the j-th prime base. halton:next() gives the next point (a table), starting from point 1. Simple and good in a few dimensions; its coordinates become correlated in high ones (prefer sobol there).

### `sobol`

```luau
sobol(dim: number): Sequence
```

The Sobol sequence in dim dimensions (at most 21), with Joe and Kuo's direction numbers: sobol:next() gives the next point (a table), starting from point 1 (the point 0, all zeros, is skipped). Any 2^k consecutive points from the start fill each of the 2^k equal slices of every coordinate exactly once, and pairs of coordinates nearly as evenly: integrals over the cube converge faster than with random points.

### `SOBOL_TABLE`

```luau
SOBOL_TABLE: { { any } }
```

Sobol direction numbers (Joe and Kuo, 2008, new-joe-kuo-6.21201) for dimensions 2..21: the degree s and coefficients a of each primitive polynomial x^s + ... + 1 over GF(2), and the initial odd numbers m (m_k < 2^k).
