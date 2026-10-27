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
- The library's `krylov.eigsh` (thick-restart Lanczos) finds the 12 smallest, as the
  largest of the inverse (shift-invert). Each product with the inverse is a solve with a
  banded Cholesky factorization (`band_chol`), after `graph.rcm` (reverse Cuthill-McKee)
  reorders the mesh: as numbered the band is 872 wide, reordered it is 34. This takes about
  0.03 s per drum natively, yielding to Roblox while it runs.
- Each mode then evolves exactly, as a damped rotation of its amplitude and velocity, so
  each frame only adds up the modes at each node.

The mesh is unchanged by the reflections and quarter turns of the grid. The proof of
isospectrality builds each eigenfunction of one drum from signed sums of an
eigenfunction of the other, triangle by triangle, and it works for this discrete problem
too. So the two discrete drums agree to rounding, not just to the discretization error
(`bench.luau`):

| subdivisions | unknowns | time, shift-invert / Lanczos on the matrix (native) | interpreted | drums A and B differ by | largest error against Driscoll |
| --- | --- | --- | --- | --- | --- |
| 2 | 17 | 0 / 0 ms | 0 / 0 ms | 3e-15 | 38% |
| 4 | 89 | 1 / 3 ms | 9 / 23 ms | 8e-15 | 13% |
| 8 | 401 | 6 / 16 ms | 45 / 159 ms | 7e-15 | 3.2% |
| 16 | 1697 | 27 / 113 ms | 230 / 1392 ms | 6e-15 | 0.8% |
| 32 | 6977 | 199 / 1107 ms | 1365 / 14196 ms | 5e-14 | 0.2% |

The error against the exact values falls about 4 times per halving of the mesh, and more
slowly for the lowest mode, which is limited by the reentrant corners.

Lanczos on the matrix itself (`method = "lanczos"`) needs more steps as the mesh refines,
since the lowest eigenvalues crowd together relative to the spread of the spectrum; on the
inverse they are the largest and well separated, so it takes about 50 solves at every size.
The cost is the factorization: n kd^2 / 2 multiply-adds for half bandwidth kd, about 3 ns
each natively (2.9 ms at 1697 unknowns, kd = 34; 44 ms at 6977, kd = 66). Without the
reordering (kd = 872) it would take about 2 s, slower than plain Lanczos. Shift-invert
also resolves the small eigenvalues more accurately, so the two drums agree more closely
than they did by plain Lanczos (4e-13 at 1697 unknowns).

Files:

- `drum.luau`: the model (mesh, assembly, modes, interpolation), usable outside Roblox.
- `client.client.luau`: the Roblox client script (EditableMesh rendering, strikes, the
  table and buttons).
- `drum.project.json`: builds a playable place with `rojo build
  demos/drum/drum.project.json -o drum.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the agreement and accuracy table above (`luau -O2 --codegen
  demos/drum/bench.luau`).
