# PDEs, simple

Grid Laplacians; Poisson's equation, heat flow and vibration modes on triangle meshes (real numbers).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `grid_laplacian`

```luau
grid_laplacian(n1: number | { number }, ny: (number | GridOptions)?, nz: number?, options: GridOptions?): Sparse
```

The matrix of -(the Laplacian) on a grid of points spaced h apart, in any number of dimensions: grid_laplacian({ n1, n2, ..., nd }, options), or grid_laplacian(nx, ny, nz, options) for up to three (ny and nz default to 1). Row k is (2 d u_k - the sum of its 2 d neighbors) / h^2, counting only the axes with more than one point: the 3-, 5-, 7-, ... point stencil. Points are numbered with the first axis fastest: point (i1, ..., id) is 1 + (i1 - 1) + n1 (i2 - 1) + n1 n2 (i3 - 1) + .... options.boundary says what lies past the edges: "dirichlet" (the default)  zero: the matrix is symmetric positive definite; "neumann"                  no flow out: each row counts only the neighbors it has (the matrix is singular: adding a constant changes nothing); "periodic"                 the grid wraps around (singular too). Returns a sparse matrix marked symmetric.

### `grid_mesh`

```luau
grid_mesh(counts: { number }, lengths: { number }?): ({ { number } }, { { number } })
```

A box lengths[1] by lengths[2] by ... (default 1 on each axis) in any number of dimensions, cut into counts[1] by counts[2] by ... cells, each cut into d! simplices (the Kuhn triangulation: one per order of the axes, from the cell's lowest corner to its highest by unit steps along the axes in that order; in 2-D, two triangles, in 3-D six tetrahedra). Returns the points (tables of d numbers, the first axis fastest) and the simplices (tables of d + 1 indices of points), for pde.fem.

### `rect_mesh`

```luau
rect_mesh(nx: number, ny: number, width: number?, height: number?): ({ { number } }, { { number } })
```

A rectangle width by height (default 1 by 1) cut into nx by ny cells, each cut into two triangles: grid_mesh({ nx, ny }, { width, height }).

### `fem`

```luau
fem(points: { any }, simplices: { { number } }): Domain
```

A domain for pde.poisson, pde.heat and pde.modes from a mesh of simplices of any dimension k (segments, triangles, tetrahedra, ...) whose points have any number d >= k of coordinates: points are tables of d numbers or vectors (d = 3), and simplices tables of k + 1 indices of points. With k < d the mesh is a curve or surface in a larger space, and the Laplacian is the surface's own (Laplace-Beltrami): triangles in 3-D give heat flow and vibration on a surface. Assembled with linear finite elements: on a simplex with vertices p0, ..., pk, E = [p1 - p0, ..., pk - p0] and G = E^T E, its volume is sqrt(det G) / k!, its stiffness is volume * B^T G^-1 B (B maps the vertex values to their differences from p0's), and each vertex gets volume / (k + 1) of the (lumped) mass. The boundary is the points on faces (k - 1 of a simplex's vertices) that belong to only one simplex; a closed surface has none. Errors on a flat simplex, an index outside the points, or a point no simplex uses.

### `poisson`

```luau
poisson(D: Domain, f: any, g: any): ({ number }, linalg.Info)
```

Poisson's equation -(the Laplacian of u) = f on a mesh domain, with u = g on the boundary (f and g: numbers, functions of the coordinates or tables per point; both default to 0). Returns u (a table, one value per point) and the solver's info (linalg.solve). A mesh without a boundary (a closed surface) is refused: u would be fixed only up to a constant.

### `heat`

```luau
heat(D: Domain, u0: any, options: HeatOptions): Heat
```

Heat flow du/dt = kappa (the Laplacian of u) on a mesh domain from u0 (a number, a function of the coordinates or a table per point), with u held at options.boundary on the boundary (default: u0's boundary values) and diffusivity options.kappa (default 1). Returns a stepper: heat:step(count) advances count steps of options.dt (default 1 step); heat.u is the field and heat.t the time. Backward Euler, stable for any dt (large steps smooth more than the true flow would); its matrix M + dt kappa K is factored once (sparse Cholesky) and reused every step.

### `modes`

```luau
modes(D: Domain, count: number?): ({ number }, { { number } }, linalg.Info)
```

The count (default 6) lowest vibration modes of a mesh domain held at zero on its boundary (free, on a closed surface: the first eigenvalue is then 0, a constant): -(the Laplacian of u) = lambda u, the drum's equation. Returns the eigenvalues lambda (ascending; a mode's frequency is proportional to sqrt(lambda)), the modes (tables of one value per point, zero on the boundary, scaled so the integral of u^2 is 1) and the solver's info (linalg.eig on M^-1/2 K M^-1/2).
