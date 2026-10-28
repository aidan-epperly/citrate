# Surfaces

Bezier, B-spline and interpolating spline surfaces through grids of points: points, derivatives, normals, meshes.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `new`

```luau
new(grid: { { any } }, options: SurfaceOptions?): Surface
```

A surface from a grid of control points (grid[i][j]: rows i = 1..m, columns j = 1..n; vectors by default). options.kind: "bspline" (the default)  a uniform clamped B-spline surface of options.degree (default 3, lowered to fit the grid): smooth, passing through the four corners and following the grid loosely; domain [0, m - p] by [0, n - p]; "bezier"                 a single Bezier patch of degree (m - 1, n - 1); domain [0, 1]^2; "spline"                 the natural cubic spline surface through every grid point, with u and v counting rows and columns from 0.

### `eval`

```luau
eval(S: Surface, u: number, v: number): any
```

The point at parameters (u, v) (clamped to the domain; see surface.domain). Building the curve across the rows dominates a single point's cost; it is kept for the last v, so points along u at a fixed v (a scanline) cost much less.

### `derivatives`

```luau
derivatives(S: Surface, u: number, v: number): (any, any)
```

The partial derivatives dS/du and dS/dv at (u, v).

### `normal`

```luau
normal(S: Surface, u: number, v: number): vector
```

The unit normal at (u, v): dS/du x dS/dv, normalized (zero where the surface is degenerate). Vectors only.

### `domain`

```luau
domain(S: Surface): (number, number, number, number)
```

The parameter domain: u in [u0, u1], v in [v0, v1].

### `mesh`

```luau
mesh(S: Surface, nu: number, nv: number): ({ vector }, { { number } }, { vector })
```

A triangle mesh of the surface: (nu + 1) by (nv + 1) points evenly spaced in (u, v), the triangles (each { a, b, c }, indices of points, two per cell, wound so their normals agree with surface.normal), and the unit normal at each point. Vectors only. For drawing (an EditableMesh) or for a heightfield's collision.
