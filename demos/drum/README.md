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
  sparse Cholesky factorization (`sparse_chol`, ordered by approximate minimum degree).
  This takes about 0.025 s per drum natively, yielding to Roblox while it runs.
- Each mode then evolves exactly, as a damped rotation of its amplitude and velocity, so
  each frame only adds up the modes at each node.

The mesh is unchanged by the reflections and quarter turns of the grid. The proof of
isospectrality builds each eigenfunction of one drum from signed sums of an
eigenfunction of the other, triangle by triangle, and it works for this discrete problem
too. So the two discrete drums agree to rounding, not just to the discretization error
(`bench.luau`):

| subdivisions | unknowns | time: shift-invert, sparse / banded / Lanczos on the matrix (native) | interpreted | drums A and B differ by | largest error against Driscoll |
| --- | --- | --- | --- | --- | --- |
| 2 | 17 | 0.2 / 0.2 / 0 ms | 0.5 / 0.4 / 0 ms | 2e-15 | 38% |
| 4 | 89 | 1.5 / 1.4 / 3 ms | 9.7 / 10 / 26 ms | 4e-15 | 13% |
| 8 | 401 | 5.1 / 5.8 / 16 ms | 46 / 44 / 167 ms | 1e-14 | 3.2% |
| 16 | 1697 | 24 / 29 / 116 ms | 180 / 222 / 1404 ms | 3e-15 | 0.8% |
| 32 | 6977 | 93 / 197 / 1118 ms | 823 / 1449 / 14941 ms | 5e-15 | 0.2% |

The error against the exact values falls about 4 times per halving of the mesh, and more
slowly for the lowest mode, which is limited by the reentrant corners.

Lanczos on the matrix itself (`method = "lanczos"`) needs more steps as the mesh refines,
since the lowest eigenvalues crowd together relative to the spread of the spectrum; on the
inverse they are the largest and well separated, so it takes about 50 solves at every size.
Shift-invert also resolves the small eigenvalues more accurately, so the two drums agree
more closely than they did by plain Lanczos (4e-13 at 1697 unknowns).

The factorization is the main cost. `method = "banded"` makes it banded, for comparison:
after reverse Cuthill-McKee the band is 34 wide at 1697 unknowns (872 as numbered) and 66
at 6977, and banded Cholesky's n kd^2 / 2 multiply-adds took 2.9 and 44 ms. The sparse
factorization, ordered by approximate minimum degree, has a sixth of the multiply-adds or
fewer and took 0.5 and 3.6 ms; at 6977 unknowns it also stores a third of the band's
entries, so each solve costs less. At small sizes the ordering itself (a few ms) is most
of its time.

Files:

- `drum.luau`: the model (mesh, assembly, modes, interpolation), usable outside Roblox.
- `client.client.luau`: the Roblox client script (EditableMesh rendering, strikes, the
  table and buttons).
- `drum.project.json`: builds a playable place with `rojo build
  demos/drum/drum.project.json -o drum.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the agreement and accuracy table above (`luau -O2 --codegen
  demos/drum/bench.luau`).
