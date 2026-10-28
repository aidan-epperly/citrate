# Surfaces and interpolation

Reference: [surface](reference/surface.md), [interp](reference/interp.md).

## Surfaces through grids of points

```luau
local surface = citrate.surface
-- grid[i][j]: a vector; rows i = 1..m, columns j = 1..n
local S = surface.new(grid, { kind = "spline" })
local p = surface.eval(S, u, v)
local n = surface.normal(S, u, v)
local points, triangles, normals = surface.mesh(S, 32, 32)   -- for an EditableMesh
```

| kind | the surface | parameters |
| --- | --- | --- |
| `"bspline"` (default) | smooth, following the grid loosely, through its four corners | u in [0, m - p], v in [0, n - p] (degree p = 3 by default) |
| `"bezier"` | one patch of degree (m - 1, n - 1) | [0, 1] by [0, 1] |
| `"spline"` | the cubic spline surface through every grid point | u = i - 1 and v = j - 1 at point (i, j) |

A tensor-product surface is a curve of curves: each grid row is a curve in v, and the
surface at (u, v) is the curve through the rows' points at v, evaluated at u. So these
surfaces are built from [curves](curves.md), with their accuracy, and work in any space
the curves do (`options.space`); normals and meshes need `vector`s.

The curve across the rows is the costly part of a point: about 29 us for a 10 by 10
B-spline. It is kept for the last v, so points along u at a fixed v cost 0.2 us, and
`surface.mesh` builds it once per row of the mesh: 2.7 ms for a 33 by 33 mesh with normals
(about 2.5 us a point).

## Values on a grid

```luau
local interp = citrate.interp
-- heights[i][j] at (x0 + (i - 1) dx, y0 + (j - 1) dy)
local H = interp.grid(heights, { dx = 4, dy = 4 })
local z = H:at(x, y)
local zx, zy = H:gradient(x, y)       -- the slope, for normals or movement
local z, zx, zy = H:sample(x, y)      -- both, for the price of one
```

`method = "cubic"` (the default) is Keys' cubic convolution, the Catmull-Rom rule in two
dimensions: it passes through the values, its slope is continuous, it reproduces quadratic
data exactly (Keys' edge rule included), and its error falls as the cube of the spacing.
`method = "linear"` is bilinear: it never overshoots, but its slope jumps at grid lines.
Outside the grid the coordinates are clamped to its edge. On a 100 by 100 grid a cubic
value costs 0.12 us natively (0.42 interpreted) and a bilinear one 0.05 (0.13).

The difference in slope matters for anything moving on the surface. In `demos/terrain`
marbles roll without friction over the same grid of heights, both ways: on the bilinear
surface each grid line they cross is a small bump that costs them speed, whatever the time
step, and they stop within seconds (81% of their energy lost in 10 s); on the cubic one
they keep rolling (8% lost in 10 s at 4 substeps a frame, halving as the substeps double).

## Values at scattered points

```luau
local R = interp.rbf(points, values)   -- points: vectors or { x, y } / { x, y, z }
local v = R:at(vector.create(1, 2, 0))
```

Radial basis function interpolation: f(p) is a weighted sum of a function of the distance
to each point, plus a linear polynomial, with weights from a dense linear system (n^3 / 3
work: 1.3 ms for 100 points, 62 ms for 400, natively). `kernel = "thin_plate"` (the
default) gives the smoothest surface through the points and reproduces linear data
exactly; `"cubic"`, `"gaussian"` and `"multiquadric"` (with `epsilon` setting their scale)
are the alternatives. `smoothing > 0` fits noisy values approximately instead of
exactly. Distances are computed in double precision even when the points are `vector`s.
