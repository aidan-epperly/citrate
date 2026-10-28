# PDEs: start here

Reference: [pde](reference/pde.md).

`citrate.pde` solves three classic equations on a region of the plane: Poisson's
equation, heat flow and vibration. You describe the region with a triangle mesh; the module
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

A mesh is a list of points and a list of triangles:

```luau
local points = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0.5, 0.5 } }
local triangles = { { 1, 2, 5 }, { 2, 3, 5 }, { 3, 4, 5 }, { 4, 1, 5 } }
local D = pde.fem(points, triangles)
```

Each triangle lists three points by their positions in `points`. `pde.rect_mesh(nx, ny,
width, height)` makes a rectangle's mesh for you. `pde.fem` checks the mesh (no flat
triangles, no points left out) and finds its boundary: the points on edges that only one
triangle has. `D.n` is the number of points, `D.x` and `D.y` their coordinates, `D.boundary`
and `D.interior` which points are which.

A value at each point is a field, given as:
- a number, the same everywhere: `1`;
- a function of the position: `function(x, y) return x * y end`;
- a table with one number per point.

## The equations

| function | the equation | what you get |
| --- | --- | --- |
| `poisson(D, f, g)` | -(u_xx + u_yy) = f inside, u = g on the boundary | u at every point |
| `heat(D, u0, { dt = ... })` | u_t = kappa (u_xx + u_yy), starting from u0 | a stepper: `H:step(10)` moves 10 steps of dt; `H.u` and `H.t` |
| `modes(D, count)` | -(u_xx + u_yy) = lambda u, u = 0 on the boundary | the lowest `count` values of lambda and their shapes |

```luau
-- heat: a plate starting at 0, its edges held at 100
local H = pde.heat(D, 0, { dt = 0.01, boundary = 100 })
H:step(50)
print(H.t, H.u[221])         -- 0.5, 99.98: nearly filled

-- vibration: the lowest 4 modes of the square (a drum): lambda is 2, 5, 5, 8 times pi^2
local lambdas, shapes = pde.modes(D, 4)
```

A mode's frequency is proportional to the square root of its lambda. `heat` is stable
for any time step; very large steps smooth more than the true flow would.

## Grids

On a regular grid, `pde.grid_laplacian(nx, ny, nz)` gives the matrix of -(u_xx + u_yy +
u_zz) as a sparse matrix, for `citrate.linalg` (or the library's sparse solvers):

```luau
local A = pde.grid_laplacian(100, 100)            -- 10,000 points, spacing 1
local u = citrate.linalg.solve(A, table.create(10000, 1))
```

Past the grid's edges the values are zero (`boundary = "dirichlet"`, the default), or no
heat flows out (`"neumann"`), or the grid wraps around (`"periodic"`). The spacing is
`options.h`.

## Accuracy

The meshes use linear finite elements: errors shrink like the square of the triangle size
(halving the cells divides them by about 4). On the 20 by 20 square, the four lowest modes'
eigenvalues are within 1% of the exact ones, and Poisson's solution for sin(pi x) sin(pi y)
within 0.2% of its peak. Finer meshes cost more: a 100 by 100 square has 10,000 points (its
grid Laplacian took 33 ms to solve).
