# Graphs

Reference: [graph](reference/graph.md).

```luau
local graph = citrate.graph
-- from edges (both ways unless directed), with lengths
local G, w = graph.from_edges(n, ii, jj, { weights = lengths })
-- or the pattern of a square sparse matrix, without a copy
local P = graph.of(S)

local order, level = graph.bfs(G, 1)          -- visit order, distance in edges
local labels, count = graph.components(G)
local dist, prev = graph.dijkstra(G, w, 1)    -- shortest paths from vertex 1
local route = graph.path(prev, 42)            -- vertices from 1 to 42
local route, info = graph.astar(G, w, 1, 42, function(v) return straight_line(v, 42) end)
local perm = graph.rcm(P)                     -- an ordering that narrows the band
```

A graph is a sparsity pattern: `{ n, ptr, ind }`, where vertex v's neighbors (0-based)
are `ind[ptr[v]]` to `ind[ptr[v + 1] - 1]`, u32 buffers. That is how a `Sparse` matrix
stores its columns and a `BlockSparse` its block rows, so `graph.of` views either without
copying, and a weighted graph is a pattern plus an f64 array of lengths aligned with
`ind` (a `Sparse`'s own values, for instance). Vertices are 1-based in every argument and
result. Orders and permutations are buffers of 32-bit integers, as pivots are: a
permutation `perm` puts vertex `perm[k]` k-th.

## Orderings for banded solvers

A banded factorization costs n kd^2 / 2 multiply-adds for half bandwidth kd, and the
bandwidth depends on how the unknowns are numbered. A mesh numbered as it was generated can
have a band nearly n wide; reverse Cuthill-McKee (`graph.rcm`) renumbers it breadth-first
from a vertex at the edge of the graph, so neighbors get nearby numbers and the band
shrinks to about the width of the mesh, around sqrt(n) for a 2-D mesh.

```luau
local perm = graph.rcm(graph.of(A))               -- A symmetric
local F = f64.band_chol(f64.sparse_symband(A, perm))
-- solve A x = b: permute b, solve, permute back
for k = 1, n do f64.set(pb, k, f64.get(b, buffer.readi32(perm, 4 * (k - 1)))) end
f64.band_chol_solve(pb, F, pb)
for k = 1, n do f64.set(x, buffer.readi32(perm, 4 * (k - 1)), f64.get(pb, k)) end
```

`sparse_permute(S, perm)` gives P S P^T as a sparse matrix, and `sparse_symband(S, perm)`
its band in symmetric band storage, ready for `band_chol`. `graph.bandwidth(G, perm)` and
`graph.profile(G, perm)` measure an ordering.

Measured on the drum demo's finite-element meshes (`demos/drum`):

| unknowns | half bandwidth, as numbered / reordered | Cholesky factorization, reordered | the 12 lowest modes: shift-invert / Lanczos on the matrix |
| --- | --- | --- | --- |
| 1697 | 872 / 34 | 2.9 ms | 27 / 113 ms |
| 6977 | 3536 / 66 | 44 ms | 199 / 1107 ms |

At about 3 ns per multiply-add the factorization's time follows n kd^2 / 2; as numbered
it would take about 2 s at 1697 unknowns. On a 200 by 200 grid numbered at random,
`graph.rcm` finds the same bandwidth as SciPy's `reverse_cuthill_mckee` (39712 to 200) in
7.6 ms against SciPy's 1.6 ms (`bench/scipy`).

A grid numbered row by row already has about the narrowest band a grid can have, so
reordering does not help it: the cloth demo's band is the same before and after.
`ode.implicit_mechanics` reorders on its own: with `solver = "auto"` it computes the
ordering once per pattern and uses it when the band narrows, so a rope or chain numbered
in any order is still factored directly rather than iterated.

## Orderings for sparse Cholesky

`graph.amd` (approximate minimum degree) orders a symmetric pattern for the least fill in
its Cholesky factor, and `graph.etree` gives the factor's elimination tree and a postorder;
`graph.chol_analysis` builds the whole symbolic factorization (`sparse_chol` uses all three;
see [sparse matrices](sparse.md)). On the drum's mesh at 1697 unknowns, L has 17,478
entries ordered by AMD against 39,822 by reverse Cuthill-McKee; on the cloth, 281,277
against 336,195. AMD keeps all its lists in one integer workspace, pruned in place; it
ordered the 100 by 100 grid (10,000 vertices) in 7 ms natively and 30 interpreted, with 1.5
MB of workspace.

## Shortest paths

Radius graphs of point sets come from [spatial queries](spatial.md): `grid:graph(r)` and
`kdtree:graph(r)` join the points within r, with their distances as weights.

`graph.dijkstra(G, w, source)` gives every vertex's distance and predecessor, from one
source or a list of them (the distance to the nearest), and can stop at a target or past a
distance limit. `graph.astar` searches for one target guided by a heuristic h(v), which
must not exceed v's true distance to the target (a straight-line distance between
waypoints joined by straight edges): the closer h is to the true distance, the fewer
vertices it expands. On a 200 by 200 grid with random edge lengths, Dijkstra's search
from one vertex took 9.1 ms against SciPy's 3.0 ms.
