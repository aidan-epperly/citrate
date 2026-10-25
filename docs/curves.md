# Curves

Reference: [curve](reference/curve.md).

Curves have points in any `Space` (`vector`s natively, buffers of any dimension): Bézier
curves of any degree, polynomial or rational, and composite paths; cubic Hermite curves;
Catmull-Rom (uniform, centripetal, chordal) and Kochanek-Bartels splines; interpolating
cubic splines (natural, clamped, not-a-knot, periodic); B-splines and NURBS. Each family is
converted to one piecewise Bézier form, so everything below works the same way on every
curve: evaluation and derivatives, speed, curvature, torsion, Frenet and
rotation-minimizing frames, arclength reparameterization, even sampling by arclength,
closest points, tight bounding boxes and splitting.

```luau
local curve = citrate.curve
local path = curve.catmull_rom({ vector.create(0, 0, 0), vector.create(4, 2, 0), vector.create(8, 0, 3) })
local L = curve.length(path)
local p = curve.at_length(path, 0.25 * L)    -- the point a quarter of the way along
local points, params = curve.sample(path, 20) -- 20 points evenly spaced by arclength
local frames = curve.frames(path, params)     -- twist-free frames there, for a tube or a road
local t, q, distance = curve.closest(path, vector.create(3, 3, 3))
-- Tracking something that moves a little each frame: start from last frame's answer.
-- (closest_near is local and about 25 times faster; closest is global.)
t = curve.closest_near(path, vector.create(3.1, 3, 3), t)
```

With `vector`s, points are computed in double precision and rounded once. Other
representations pass their space: `curve.cubic_spline(points, { space = citrate.f64.space })`.

Arclength is built on first use: on each segment the speed is a Chebyshev interpolant
(split at cusps), its integral is the arclength, and the inverse t(s) is a Chebyshev
interpolant too, so `at_length` costs a binary search and a short Clenshaw sum.
Polynomial segments are first tried at a fixed 17, 25 or 33 points (the speed sampled,
the inverse solved for by Newton's method, both transformed by a direct cosine sum); only
where those do not resolve it, or the speed nearly vanishes, is the adaptive construction
used. For game-sized tolerances that is 2 to 4 times cheaper: on a 10-segment track, per
cubic segment, 17 us natively and 51 interpreted at `tol = 1e-6`, 29 and 89 at `1e-8`
(adaptive: 48 and 183, 57 and 204). `curve.arclength(C, { tol = 1e-8 })` builds the
tables to that tolerance, enough for points stored as `vector`s.
