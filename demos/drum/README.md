# Drum: can you hear the shape of a drum?

Kac asked in 1966 whether a drum's frequencies determine its shape. Gordon, Webb and
Wolpert answered no in 1992 with two different polygons, each made of seven half-squares,
that have the same frequencies. This demo shows that pair side by side:

- The table lists both drums' lowest 12 eigenvalues (frequency squared) and Driscoll's
  values, which are accurate to 12 digits (SIAM Review 1997, scaled to this size).
- Click a drum to strike it at that point. It rings as the sum of its modes, each at its
  own frequency, with light damping.
- The buttons play one mode on both drums. The shapes differ, but the drums oscillate in
  step.

How it is computed (`drum.luau`):

- Each drum's Laplacian (clamped edges) is discretized by linear finite elements on a
  criss-cross mesh: each grid square is cut into 16 by 16 subsquares, and each subsquare
  into four triangles by both diagonals. That gives 1697 unknowns per drum. The mass
  matrix is lumped, so the eigenvalues are those of D^-1/2 K D^-1/2 with K sparse.
- The library's `krylov.eigsh` (thick-restart Lanczos) finds the 12 smallest. This takes
  about 0.1 s per drum, yielding to Roblox while it runs.
- Each mode then evolves exactly, as a damped rotation of its amplitude and velocity, so
  each frame only adds up the modes at each node.

The mesh is unchanged by the reflections and quarter turns of the grid. The proof of
isospectrality builds each eigenfunction of one drum from signed sums of an
eigenfunction of the other, triangle by triangle, and it works for this discrete problem
too. So the two discrete drums agree to rounding, not just to the discretization error
(`bench.luau`):

| subdivisions | unknowns | drums A and B differ by | largest error against Driscoll |
| --- | --- | --- | --- |
| 2 | 17 | 2e-15 | 38% |
| 4 | 89 | 2e-14 | 13% |
| 8 | 401 | 1e-13 | 3.2% |
| 16 | 1697 | 4e-13 | 0.8% |
| 32 | 6977 | 2e-12 | 0.2% |

The error against the exact values falls about 4 times per halving of the mesh, and more
slowly for the lowest mode, which is limited by the reentrant corners.

For larger meshes, shift-invert Lanczos would pay off: each step solves with K by
multigrid-preconditioned CG. That takes a fixed number of steps, while plain Lanczos needs
more steps as the mesh refines. On the L-shaped drum the crossover was near 12,000
unknowns.

Files:

- `drum.luau`: the model (mesh, assembly, modes, interpolation), usable outside Roblox.
- `client.client.luau`: the Roblox client script (EditableMesh rendering, strikes, the
  table and buttons).
- `drum.project.json`: builds a playable place with `rojo build
  demos/drum/drum.project.json -o drum.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the agreement and accuracy table above (`luau -O2 --codegen
  demos/drum/bench.luau`).
