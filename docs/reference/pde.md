# PDEs, simple

Grid Laplacians; Poisson's equation, heat flow and vibration modes on triangle meshes (real numbers).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `grid_laplacian`

```luau
grid_laplacian(nx: number, ny: number?, nz: number?, options: GridOptions?): Sparse
```

The matrix of -(the Laplacian) on an nx by ny by nz grid of points spaced h apart (ny and nz default to 1: a line or a plane): row k is (2 d u_k - the sum of its 2 d neighbors) / h^2 in d dimensions, the 3-, 5- or 7-point stencil. Points are numbered with x fastest: point (i, j, k) is i + nx (j - 1) + nx ny (k - 1). options.boundary says what lies past the edges: "dirichlet" (the default)  zero: the matrix is symmetric positive definite; "neumann"                  no flow out: each row counts only the neighbors it has (the matrix is singular: adding a constant changes nothing); "periodic"                 the grid wraps around (singular too). Returns a sparse matrix marked symmetric.

### `rect_mesh`

```luau
rect_mesh(nx: number, ny: number, width: number?, height: number?): ({ { number } }, { { number } })
```

A rectangle width by height (default 1 by 1) cut into nx by ny cells, each cut into two triangles: the points ({ x, y }, row by row from the corner (0, 0)) and the triangles ({ a, b, c }, indices of points), for pde.fem.

### `fem`

```luau
fem(points: { any }, triangles: { { number } }): Domain
```

A domain for pde.poisson, pde.heat and pde.modes from a triangle mesh: points (each { x, y } or a vector) and triangles (each { a, b, c }, indices of points). Assembled with linear finite elements: on a triangle of area A with corners p1, p2, p3, the stiffness is (b_s b_t + c_s c_t) / (4 A) with b_s = y_u - y_w and c_s = x_w - x_u ((s, u, w) cyclic), and each corner gets A / 3 of the (lumped) mass. Errors on a triangle of zero area, an index outside the points, or a point no triangle uses.

### `poisson`

```luau
poisson(D: Domain, f: any, g: any): ({ number }, linalg.Info)
```

Poisson's equation -(the Laplacian of u) = f on a mesh domain, with u = g on the boundary (f and g: numbers, functions of (x, y) or tables per point; both default to 0). Returns u (a table, one value per point) and the solver's info (linalg.solve).

### `heat`

```luau
heat(D: Domain, u0: any, options: HeatOptions): Heat
```

Heat flow du/dt = kappa (the Laplacian of u) on a mesh domain from u0 (a number, a function of (x, y) or a table per point), with u held at options.boundary on the boundary (default: u0's boundary values) and diffusivity options.kappa (default 1). Returns a stepper: heat:step(count) advances count steps of options.dt (default 1 step); heat.u is the field and heat.t the time. Backward Euler, stable for any dt (large steps smooth more than the true flow would); its matrix M + dt kappa K is factored once (sparse Cholesky) and reused every step.

### `modes`

```luau
modes(D: Domain, count: number?): ({ number }, { { number } }, linalg.Info)
```

The count (default 6) lowest vibration modes of a mesh domain held at zero on its boundary: -(the Laplacian of u) = lambda u, the drum's equation. Returns the eigenvalues lambda (ascending; a mode's frequency is proportional to sqrt(lambda)), the modes (tables of one value per point, zero on the boundary, scaled so the integral of u^2 is 1) and the solver's info (linalg.eig on M^-1/2 K M^-1/2).
