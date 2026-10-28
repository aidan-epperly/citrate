# Random numbers and statistics

Reference: [random](reference/random.md), [stats](reference/stats.md).

```luau
local random, stats = citrate.random, citrate.stats

local rng = random.new(42)        -- the same seed gives the same numbers everywhere
rng:uniform()                     -- in [0, 1); rng:uniform(a, b) in [a, b)
rng:integer(1, 6)                 -- a fair die
rng:normal(100, 15)               -- mean 100, standard deviation 15
rng:pick(list), rng:shuffle(list), rng:sample(list, 3)
rng:direction(), rng:in_ball(5)   -- a unit vector; a point in a ball of radius 5

local loot = random.weighted({ 70, 25, 5 })   -- common, rare, epic
local tier = loot:sample(rng)                 -- 1 70% of the time, 2 25%, 3 5%
```

## The generator

`random.new(seed)` makes a generator: xoshiro128** (Blackman and Vigna), with four 32-bit
words of state and period 2^128 - 1, passing the standard statistical test batteries.
Each generator is independent, so a world seed, a loot seed and a weather seed can each
give the same results every time, on every machine. Without a seed it takes one from the
clock.

A uniform number uses two 32-bit outputs, 53 random bits, so every double in [0, 1) that
is a multiple of 2^-53 is equally likely. It costs about 60 ns natively (177 interpreted),
against 14 for Luau's built-in `math.random`, which shares one global state and has no
distributions.

| method | gives | method used |
| --- | --- | --- |
| `uniform(a, b)` | a number in [a, b), default [0, 1) | 53 bits |
| `integer(lo, hi)` | a whole number, each equally likely | rejection (no bias) |
| `normal(mean, sd)` | bell-curve numbers | Box-Muller |
| `exponential(rate)` | waiting times (mean 1 / rate) | inversion |
| `gamma(shape, scale)`, `beta(a, b)` | skewed amounts; fractions | Marsaglia-Tsang |
| `poisson(mean)` | counts of events | inversion; Hormann's PTRS for means of 10 or more |
| `shuffle`, `pick`, `sample(list, k)` | orders and picks | Fisher-Yates |
| `direction(flat)`, `in_ball(r, flat)` | unit vectors, points in a ball (disk) | uniform by area or volume |
| `fill(x, "normal")` | an f64 array of numbers | |

`random.weighted(weights)` builds Vose's alias table once; each `:sample(rng)` then
costs the same however many weights there are.

## Quasi-random points

Random points clump and leave gaps. Quasi-random sequences spread points evenly, which
suits sampling a region (spawn points, light probes) and numerical integration:

```luau
local q = random.sobol(2)
for k = 1, 64 do
	local p = q:next()   -- { x, y } in the unit square
end
```

`random.sobol(dim)` (up to 21 dimensions, Joe and Kuo's direction numbers) fills every
coordinate's 2^k equal slices exactly once with its first 2^k points. `random.halton(dim)`
(up to 32) is simpler and good in a few dimensions. Averaging a smooth function over the
6-dimensional cube with 4095 Sobol points gave an error under 1e-3, smaller than with
random points.

## Statistics

```luau
stats.mean(x), stats.median(x), stats.std(x), stats.quantile(x, 0.95)
local counts, edges = stats.histogram(x, 20)
stats.correlation(x, y)

local frame = stats.running()     -- constant memory, for numbers arriving one at a time
frame:add(dt)                     -- each frame
print(frame.mean, frame:std(), frame.max)
```

Lists can be tables or f64 arrays. Sums are compensated (Kahan-Neumaier) and the variance
is computed in two passes (Welford's update for `running`), so long lists of similar
numbers stay accurate: the mean of 100,000 numbers near 1e8 is exact to the last digit.
