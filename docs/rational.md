# Rational approximation

Reference: [rational](reference/rational.md).

`rational.aaa(Z, F, options)` computes a rational approximation of the values F at the
sample points Z (both c64 arrays) by the AAA algorithm of Nakatsukasa, Sète and Trefethen.
The points can lie anywhere in the complex plane (an interval, a circle, a region's
boundary, scattered points), and the result is a rational function in barycentric form

  r(z) = Σ w_j f_j / (z − z_j) / Σ w_j / (z − z_j),

which interpolates F at its support points z_j. Each step adds the sample where the error
is largest as a support point and recomputes the weights by an SVD; it stops when the
error on the samples is within the tolerance (default 1e-13, relative).

```luau
local c64, rational = citrate.c64, citrate.rational
local x = {}
for i = 1, 1000 do x[i] = -1.5 + 3 * (i - 1) / 999 end
local Z = c64.complexify(citrate.f64.from(x))
local F = c64.complexify(citrate.f64.from((function()
	local t = {}
	for i, xi in x do t[i] = math.tan(xi) end
	return t
end)()))
local r = rational.aaa(Z, F)
print(c64.length(r.support), r.error)   --> 9, 9e-16
print(r.eval(0.3))                      --> tan(0.3) (as re, im)
-- r.poles, r.residues and r.zeros (c64 arrays): here the poles include ±pi/2 to 15
-- digits, with residues -1.
```

What it gives:

- Smooth functions in few terms: `exp` on [-1, 1] to machine precision with 7 support
  points (a type (6, 6) rational function).
- The poles show where a function's singularities are, even outside the sample region,
  and the residues measure them.
- Branch points and corners are approximated by poles accumulating toward them: `|x|` on
  [-1, 1] reaches 4e-14 with 47 support points, a root-exponential rate that polynomials
  (error O(1 / n)) cannot match.

Fitting below the noise or rounding level creates Froissart doublets, spurious poles with
tiny residues paired with zeros. With `cleanup` (the default) poles whose residues are
below 1e-13 times the largest |F| are removed with their nearest support points, and the
weights recomputed; `r.removed` counts them. For noisy data, set `tol` near the noise
level.

Cost: each step is an SVD of the Loewner matrix (samples by support points), so m steps on
M samples cost about M m³ / 3 operations; 2000 samples and 47 support points take about a
second.
