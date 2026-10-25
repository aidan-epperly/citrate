# Curves

Bézier, Hermite, Catmull-Rom, Kochanek-Bartels, cubic spline, B-spline and NURBS curves in any `Space`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `bezier`

```luau
bezier<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
```

A Bézier curve of degree #points - 1 on options.breaks (default { 0, 1 }), rational if options.weights (positive, one per point) is given. Reads options.space and .breaks.

### `bezier_path`

```luau
bezier_path<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
```

A composite Bézier curve of the given degree (options.degree, default 3): segment i uses points (i - 1) * degree + 1 .. i * degree + 1, so neighbors share their end points. Reads options.space, .degree, .breaks (default 0 .. number of segments), .weights (one per point, making it rational) and .closed (marks the curve closed; the last point should equal the first).

### `hermite`

```luau
hermite<T>(points: { T }, tangents: { T }, options: CurveOptions<T>?): Curve<T>
```

The cubic Hermite curve through points[i] with derivatives tangents[i] (with respect to t) at breaks[i]. Reads options.space and .breaks (default 0 .. #points - 1).

### `catmull_rom`

```luau
catmull_rom<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
```

The Catmull-Rom spline through the points: a C^1 cubic with knots spaced by chord length to the power alpha (options.alpha: 0 uniform, 0.5 centripetal (the default, which never forms cusps or self-intersections within a segment), 1 chordal). The tangent at P_i is that of the Barry-Goldman construction: (P_i - P_(i-1)) / h0 - (P_(i+1) - P_(i-1)) / (h0 + h1) + (P_(i+1) - P_i) / h1. Open curves use the chord direction at the ends. Consecutive repeated points are dropped. The parameter runs over the knots. Reads options.space, .alpha and .closed.

### `kochanek_bartels`

```luau
kochanek_bartels<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
```

The Kochanek-Bartels (TCB) spline through the points, with uniform parameter spacing: options.tension, .continuity and .bias (numbers, or tables with one per point; all 0 gives the uniform Catmull-Rom spline). The tangent leaving P_i is (1-T)(1+B)(1-C)/2 (P_i - P_(i-1)) + (1-T)(1-B)(1+C)/2 (P_(i+1) - P_i) and the one arriving swaps (1-C) and (1+C). Open ends reflect the neighboring chord. Reads options.space and .closed as well.

### `cubic_spline`

```luau
cubic_spline<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
```

The C^2 cubic spline through the points. Parameters: options.breaks, or knots spaced by chord length to the power options.alpha (default 0.5, centripetal; consecutive repeated points are then dropped). End conditions, options.ends: "natural" (zero second derivative; the default), "clamped" (first derivatives options.start and .finish) or "not-a-knot" (the third derivative is continuous at the second and second-to-last points; with 3 points that is the parabola through them). options.closed gives the periodic spline, C^2 everywhere including where it closes. Reads options.space as well.

### `bspline`

```luau
bspline<T>(points: { T }, options: CurveOptions<T>?): Curve<T>
```

A B-spline (NURBS if options.weights is given, one positive weight per point) of degree options.degree (default 3) with control points `points`. Knots: options.knots (#points + degree + 1 nondecreasing values; the domain is [knots[degree + 1], knots[#points + 1]]), or by default clamped and uniform, so the curve starts at the first point, ends at the last, and has domain [0, #points - degree]. options.closed instead makes a periodic curve (uniform knots, domain [0, #points]). The curve is converted to Bézier segments by blossoming. Reads options.space as well.

### `domain`

```luau
domain<T>(C: Curve<T>): (number, number)
```

The domain [a, b] of C.

### `eval`

```luau
eval<T>(C: Curve<T>, t: number, out: T?): T
```

### `derivative`

```luau
derivative<T>(C: Curve<T>, t: number, k: number, out: T?): T
```

The k-th derivative C^(k)(t) (k >= 0; into out for buffer spaces).

### `speed`

```luau
speed<T>(C: Curve<T>, t: number): number
```

|C'(t)|.

### `tangent`

```luau
tangent<T>(C: Curve<T>, t: number, out: T?): T
```

The unit tangent C'(t) / |C'(t)| (NaN where the speed is zero).

### `curvature`

```luau
curvature<T>(C: Curve<T>, t: number): number
```

Curvature |C' x C''| / |C'|^3, in any dimension (from the Gram determinant of C' and C''). NaN where the speed is zero.

### `torsion`

```luau
torsion(C: Curve<vector>, t: number): number
```

Torsion (C' x C'') . C''' / |C' x C''|^2 of a curve of vectors. NaN where the curvature is zero.

### `frenet`

```luau
frenet<T>(C: Curve<T>, t: number): (T, T?, vector?)
```

The Frenet frame at t: unit tangent, unit normal and (vector curves only) binormal. The normal and binormal are nil where the curvature is zero.

### `frames`

```luau
frames<T>(C: Curve<T>, ts: { number }, r0: T?): { tangents: { T }, normals: { T }, binormals: { vector }? }
```

Rotation-minimizing frames at the parameters ts (ascending), by the double reflection method of Wang, Jüttler, Zheng and Liu (2008): returns the unit tangents, the reference vectors (normals) and, for vector curves, the binormals tangent x normal. The first normal is r0 made perpendicular to the first tangent; by default some perpendicular vector (vector curves) or the Frenet normal. Accuracy improves with denser ts (the error is fourth order in the spacing).

### `framing`

```luau
framing(C: Curve<vector>, options: FramingOptions?): Framing
```

### `arclength`

```luau
arclength<T>(C: Curve<T>, options: { tol: number? }?): number
```

Builds the arclength tables now with tolerance options.tol (default machine precision; larger is faster to build) and returns the length. Later arclength calls use them.

### `length`

```luau
length<T>(C: Curve<T>): number
```

The length of the curve over its domain.

### `length_at`

```luau
length_at<T>(C: Curve<T>, t: number): number
```

The arclength s(t) from the start of the domain to t (t clamped to the domain).

### `param_at`

```luau
param_at<T>(C: Curve<T>, s: number): number
```

The parameter t with s(t) = s (s clamped to [0, length]).

### `at_length`

```luau
at_length<T>(C: Curve<T>, s: number, out: T?): T
```

The point at arclength s from the start (s clamped to [0, length]).

### `sample`

```luau
sample<T>(C: Curve<T>, n: number): ({ T }, { number })
```

n >= 2 points equally spaced in arclength, from the start to the end of the curve, and their parameters.

### `closest`

```luau
closest<T>(C: Curve<T>, q: T): (number, T, number)
```

The point of the curve (over its domain) closest to q: returns the parameter, the point and the distance. Each segment's candidates are its ends and the roots of (C(t) - q) . C'(t), a polynomial of degree 2d - 1 (a Chebfun on 2d points; adaptive for rational segments); segments whose bounding sphere is farther than the best distance so far are skipped.

### `closest_near`

```luau
closest_near<T>(C: Curve<T>, q: T, t0: number): (number, T, number)
```

The locally closest point to q, by Newton's method on g(t) = (C(t) - q) . C'(t) = 0 starting from t0: returns the parameter, the point and the distance. It costs a few evaluations, so it suits tracking something that moves a little between calls (pass the previous answer as t0). It finds a local minimum of the distance, which need not be the global one; closest finds that. Safeguards: where the squared distance is not locally convex a Gauss-Newton step (dropping the curvature term) replaces Newton's; steps are halved until the distance does not increase; t is clamped to the domain, or wraps around it on a closed curve.

### `bounds`

```luau
bounds(C: Curve<vector>): (vector, vector)
```

The tight axis-aligned bounding box of a vector curve over its domain: (min, max), rounded outward to single precision. Each coordinate's extremes are at the segment ends or at roots of its derivative x'(t), found from x'(t) D(t)^2, a polynomial of degree 2d - 2 (d - 1 for polynomial segments).

### `split`

```luau
split<T>(C: Curve<T>, t: number): (Curve<T>, Curve<T>)
```

Splits C at t (strictly inside the domain) into the curves over [start, t] and [t, end], by de Casteljau subdivision of the segment containing t (in homogeneous coordinates for rational curves).
