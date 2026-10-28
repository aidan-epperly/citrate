# Statistics

Mean, variance, quantiles, histograms, correlation, and a running summary.

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
