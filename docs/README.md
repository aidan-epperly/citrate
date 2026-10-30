# citrate documentation

Guides:

- [Getting started](getting-started.md): installing, the modules, conventions
- [Demos](demos.md): the Roblox demos, with videos
- Start here, without choosing methods: [linear algebra](linalg.md) (solve equations, find
  eigenvalues) and [PDEs](pde.md) (Poisson's equation, heat flow and vibration on meshes of any dimension, surfaces included)
- [Dense linear algebra](linear-algebra.md): solving, factorizations, eigenvalues, the SVD,
  matrix functions, structured matrices, Fourier transforms
- [Sparse matrices and iterative solvers](sparse.md): sparse and block sparse storage,
  preconditioners, multigrid, Krylov solvers
- [Differential equations](ode.md): explicit, stiff, second-order and symplectic
  integrators, implicit mechanics, boundary value problems
- [Root finding and optimization](optimize.md): scalar roots and minima, systems, least
  squares, minimization, and how Chebyshev and AAA alternatives compare
- [Random numbers and statistics](random.md): seeded generators, distributions, weighted
  choice, quasi-random points; means, quantiles, histograms
- [Special functions and integration](special.md): gamma, erf, Bessel functions; adaptive
  integration; the normal distribution
- [Graphs](graphs.md): sparsity patterns as graphs, orderings for banded solvers, shortest
  paths
- [Spatial queries](spatial.md): uniform grids and k-d trees, which to use, measured
  against SciPy and Roblox's own queries
- [Geometry](geometry.md): exact predicates, Delaunay triangulations and convex hulls in any
  dimension, polygon triangulation and quality meshes, point in polygon, closest points
- [Rational approximation](rational.md): the AAA algorithm
- [Laplace problems on polygons](lightning.md): the lightning solver
- [Chebyshev approximation](chebyshev.md)
- [Curves](curves.md)
- [Rotations](rotations.md): quaternions and CFrames, rotation splines, averaging, fitting,
  integrating angular velocity
- [Surfaces and interpolation](surfaces.md): surfaces through grids of points, heightfields,
  scattered data (thousands of points through HODLR compression)
- [Writing generic algorithms](generic-algorithms.md): the `Space` interface
- [Performance](performance.md): what costs time in Luau, and what was measured
- [Development](development.md): how the code is generated, testing (also in Roblox),
  third-party code

Reference: [every public function](reference/README.md), generated from the source
comments.
