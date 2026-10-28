# citrate documentation

Guides:

- [Getting started](getting-started.md): installing, the modules, conventions
- Start here, without choosing methods: [linear algebra](linalg.md) (solve equations, find
  eigenvalues) and [PDEs](pde.md) (Poisson's equation, heat flow and vibration on meshes)
- [Dense linear algebra](linear-algebra.md): solving, factorizations, eigenvalues, the SVD,
  matrix functions, structured matrices, Fourier transforms
- [Sparse matrices and iterative solvers](sparse.md): sparse and block sparse storage,
  preconditioners, multigrid, Krylov solvers
- [Differential equations](ode.md): explicit, stiff, second-order and symplectic
  integrators, implicit mechanics, boundary value problems
- [Root finding and optimization](optimize.md): scalar roots and minima, systems, least
  squares, minimization, and how Chebyshev and AAA alternatives compare
- [Graphs](graphs.md): sparsity patterns as graphs, orderings for banded solvers, shortest
  paths
- [Spatial queries](spatial.md): uniform grids and k-d trees, which to use, measured
  against SciPy and Roblox's own queries
- [Rational approximation](rational.md): the AAA algorithm
- [Laplace problems on polygons](lightning.md): the lightning solver
- [Chebyshev approximation](chebyshev.md)
- [Curves](curves.md)
- [Writing generic algorithms](generic-algorithms.md): the `Space` interface
- [Performance](performance.md): what costs time in Luau, and what was measured
- [Development](development.md): how the code is generated, testing (also in Roblox),
  third-party code

Reference: [every public function](reference/README.md), generated from the source
comments.
