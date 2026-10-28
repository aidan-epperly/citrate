# Statistics

Mean, variance, quantiles, histograms, correlation, a running summary, and the normal distribution.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `sum`

```luau
sum(x: any): number
```

The sum of the numbers (compensated: accurate even for many of mixed size).

### `mean`

```luau
mean(x: any): number
```

The mean (average).

### `variance`

```luau
variance(x: any, population: boolean?): number
```

The sample variance: the mean squared distance from the mean, dividing by n - 1 (pass population = true to divide by n).

### `std`

```luau
std(x: any, population: boolean?): number
```

The sample standard deviation (the square root of the variance; population as there).

### `range`

```luau
range(x: any): (number, number)
```

The smallest and largest numbers.

### `quantile`

```luau
quantile(x: any, q: number | { number }): any
```

The q-quantile (q in [0, 1]): the value below which a fraction q of the numbers lie, by linear interpolation between the sorted numbers (the common definition, "type 7"; q = 0.5 is the median). q may be a table of fractions: then a table of quantiles.

### `median`

```luau
median(x: any): number
```

The median (the middle value: half the numbers lie below it).

### `histogram`

```luau
histogram(x: any, bins: number, lo: number?, hi: number?): ({ number }, { number })
```

A histogram: how many numbers fall in each of `bins` equal intervals from lo to hi (default: the numbers' range). Returns the counts and the bins' edges (bins + 1 of them); numbers outside [lo, hi] are not counted, and hi itself counts in the last bin.

### `covariance`

```luau
covariance(x: any, y: any): number
```

The sample covariance of two lists of the same length.

### `correlation`

```luau
correlation(x: any, y: any): number
```

The (Pearson) correlation of two lists of the same length, in [-1, 1]: 1 when one is an increasing linear function of the other. NaN if either list is constant.

### `running`

```luau
running(): Running
```

A summary of numbers added one at a time (r:add(x)), in constant memory: r.count, r.mean, r.min, r.max, r:variance() and r:std() (sample), by Welford's update.

### `normal_pdf`

```luau
normal_pdf(x: number, mean: number?, sd: number?): number
```

The normal (Gaussian) density at x, for the given mean (default 0) and standard deviation (default 1).

### `normal_cdf`

```luau
normal_cdf(x: number, mean: number?, sd: number?): number
```

The normal cumulative probability: the chance that a normal number (mean, sd: default 0 and 1) is at most x. Accurate far into the tails (special.erfc).

### `normal_quantile`

```luau
normal_quantile(p: number, mean: number?, sd: number?): number
```

The normal quantile: the x with normal_cdf(x) = p, for p in (0, 1) (the inverse of normal_cdf; p = 0.975 gives 1.96 standard deviations above the mean). Abramowitz and Stegun's approximation 26.2.23, refined by Newton's method on the cdf to full accuracy.
