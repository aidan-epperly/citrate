# Interpolation

Values on regular grids (bilinear, bicubic) and at scattered points (radial basis functions).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `grid`

```luau
grid(values: any, options: GridOptions?): Grid
```

An interpolant of values on a regular grid: values[i][j] (or a Matrix's entry (i, j)) is the value at (x0 + (i - 1) dx, y0 + (j - 1) dy) (x0, y0 default 0; dx, dy default 1). options.method: "cubic" (the default)  Keys' cubic convolution (a = -1/2, as Catmull-Rom): it passes through the values, its slope is continuous, and its error falls as the cube of the spacing; it can overshoot a little near sharp steps; "linear"               bilinear: never overshoots, but the slope jumps at grid lines. Points outside the grid are clamped to its edge. Needs at least 2 points each way (3 for cubic; with 2, linear is used).

### `rbf`

```luau
rbf(points: { any }, values: { number }, options: RbfOptions?): Rbf
```

An interpolant of values at scattered points (vectors, or tables { x, y } / { x, y, z }) by radial basis functions: f(p) = sum of w_k phi(|p - p_k|) plus a linear polynomial, with the weights solving a dense linear system (so it suits hundreds of points, not many thousands: the solve costs n^3 / 3). options.kernel: "thin_plate" (the default)  r^2 log r: the smoothest surface through the points (it bends least); exact for linear data; "cubic"                     r^3; "gaussian", "multiquadric"  exp(-(epsilon r)^2), sqrt(1 + (epsilon r)^2), with options.epsilon (default 1) setting the scale: smaller is smoother. options.smoothing > 0 fits the data approximately instead (larger: smoother), for noisy values. Errors if two points coincide.
