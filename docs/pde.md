# PDEs: start here

Reference: [pde](reference/pde.md).

`citrate.pde` solves three classic equations: Poisson's equation, heat flow and vibration.
You describe the region with a mesh of segments, triangles, tetrahedra or simplices of
any dimension, in a space of any dimension: a region of the plane, a solid, or a curve or
surface in space (heat flowing over a mesh's surface, a shell's vibration). The module
does the rest. Real numbers only.

```luau
local pde = citrate.pde

-- a 1 by 1 square cut into 20 by 20 cells (two triangles each)
local D = pde.fem(pde.rect_mesh(20, 20))

-- the shape of a rubber sheet held at height 0 at its edges and pushed up by a force of 1
local u = pde.poisson(D, 1, 0)
print(u[221])        -- the height at point 221, the center: 0.0735 (exactly, 0.0737)
```

## Meshes

A mesh is a list of points and a list of simplices (here triangles):

```luau
local points = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0.5, 0.5 } }
local triangles = { { 1, 2, 5 }, { 2, 3, 5 }, { 3, 4, 5 }, { 4, 1, 5 } }
local D = pde.fem(points, triangles)
```

Each simplex lists its points by their positions in `points`. Points are tables of numbers,
all the same length d (or `vector`s, d = 3); simplices are segments (2 points), triangles
(3), tetrahedra (4) and so on, all the same size, k + 1 points with k at most d.

| mesh | k | d | the equations hold |
| --- | --- | --- | --- |
| segments on a line | 1 | 1 | on an interval: a string |
| triangles in the plane | 2 | 2 | in a region: a drum, a plate |
| tetrahedra | 3 | 3 | in a solid |
| segments in space | 1 | 2 or 3 | along a wire, whatever its shape |
| triangles in space | 2 | 3 | on a surface: a mesh's skin, a shell, a sphere |

When k is less than d, the Laplacian is the curve's or surface's own (the Laplace-Beltrami
operator): heat spreads over the surface, not through the space around it.

`pde.mesh(polygon, { size, angle, holes })` meshes a polygon (with holes) with
well-shaped triangles: no angle below `angle` (degrees, default 20) and no edge longer than
`size` (see [geometry](geometry.md)):

```luau
local L = { { 0, 0 }, { 2, 0 }, { 2, 1 }, { 1, 1 }, { 1, 2 }, { 0, 2 } }
local points, triangles, info = pde.mesh(L, { size = 0.05 })   -- 3180 points, 20 ms
local D = pde.fem(points, triangles)
```

`pde.grid_mesh(counts, lengths)` makes a box's mesh in any dimension: `pde.grid_mesh({ 10,
10, 10 })` is a unit cube in 10 by 10 by 10 cells, each cut into 6 tetrahedra (a cell in d
dimensions into d!). `pde.rect_mesh(nx, ny, width, height)` is the 2-D case. `pde.fem`
checks the mesh (no flat simplices, no points left out) and finds its boundary: the points
on faces that only one simplex has (a closed surface, like a sphere, has none). `D.n` is
the number of points, `D.coords[a]` their a-th coordinates (`D.x`, `D.y`, `D.z` the first
three), `D.volume` the mesh's total length, area or volume, and `D.boundary` and
`D.interior` which points are which.

A value at each point is a field, given as:
- a number, the same everywhere: `1`;
- a function of the position: `function(x, y) return x * y end` (one argument per
  coordinate);
- a table with one number per point.

## The equations

| function | the equation | what you get |
| --- | --- | --- |
| `poisson(D, f, g)` | -(the Laplacian of u) = f inside, u = g on the boundary | u at every point |
| `heat(D, u0, { dt = ... })` | u_t = kappa (the Laplacian of u), starting from u0 | a stepper: `H:step(10)` moves 10 steps of dt; `H.u` and `H.t` |
| `modes(D, count)` | -(the Laplacian of u) = lambda u, u = 0 on the boundary | the lowest `count` values of lambda and their shapes |

```luau
-- heat: a plate starting at 0, its edges held at 100
local H = pde.heat(D, 0, { dt = 0.01, boundary = 100 })
H:step(50)
print(H.t, H.u[221])         -- 0.5, 99.98: nearly filled
```

```luau
-- vibration: the lowest 4 modes of the square (a drum): lambda is 2, 5, 5, 8 times pi^2
local lambdas, shapes = pde.modes(D, 4)
```

A mode's frequency is proportional to the square root of its lambda. `heat` is stable
for any time step; very large steps smooth more than the true flow would. On a closed
surface there is no boundary to hold: `modes` gives the free modes (the first is a
constant, lambda = 0) and `poisson` is refused, since its answer would be fixed only up to
a constant.

## Grids

On a regular grid, `pde.grid_laplacian(nx, ny, nz)` gives the matrix of -(u_xx + u_yy +
u_zz) as a sparse matrix, for `citrate.linalg` (or the library's sparse solvers);
`pde.grid_laplacian({ n1, n2, n3, n4 })` takes any number of dimensions:

```luau
local A = pde.grid_laplacian(100, 100)            -- 10,000 points, spacing 1
local u = citrate.linalg.solve(A, table.create(10000, 1))
```

Past the grid's edges the values are zero (`boundary = "dirichlet"`, the default), or no
heat flows out (`"neumann"`), or the grid wraps around (`"periodic"`). The spacing is
`options.h`.

## Accuracy

The meshes use linear finite elements: errors shrink like the square of the simplices'
size (halving the cells divides them by about 4), in any dimension. Linear solutions are
exact. Measured:

| mesh | points | result | error |
| --- | --- | --- | --- |
| the unit square, 20 by 20 cells | 441 | the four lowest modes | within 1% |
| the unit square, 20 by 20 | 441 | Poisson's equation, sin(pi x) sin(pi y) | 0.2% of its peak |
| the unit cube, 12 by 12 by 12 cells (10,368 tetrahedra) | 2,197 | Poisson's equation, sin(pi x) sin(pi y) sin(pi z) | 0.6% (54 ms in all) |
| the unit cube, 20 by 20 by 20 (48,000 tetrahedra) | 9,261 | the same | 0.2% (0.5 s) |
| the unit sphere, an icosahedron's triangles split 4 times | 2,562 | the 9 lowest modes (exactly l (l + 1): 0, 2 three times, 6 five times) | 0.15% |
| an L-shape, `pde.mesh` with size 0.1 and angle 25 | about 700 | Poisson's equation, sin(x) e^y | second order (tested) |

Finer meshes cost more: a 100 by 100 square has 10,000 points (its grid Laplacian took
33 ms to solve).
