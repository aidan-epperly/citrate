# Graphs

Graphs as sparsity patterns: construction, breadth-first search, components, orderings (reverse Cuthill-McKee, approximate minimum degree), symbolic Cholesky analysis and shortest paths (Dijkstra, A*).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `of`

```luau
of(S: Sparse | BlockSparse): Graph
```

The graph of a square Sparse matrix's pattern (vertex j's neighbors are the rows stored in column j) or of a BlockSparse's block pattern, sharing its buffers.

### `from_edges`

```luau
from_edges(n: number, ii: { number }, jj: { number }, options: EdgeOptions?): (Graph, buffer?)
```

The graph on vertices 1..n with an edge between ii[k] and jj[k] (both ways unless options.directed), and its weights (an f64 array aligned with the graph's ind) when options.weights gives one per edge. Repeated edges keep the smallest weight; self loops are dropped.

### `from_triangles`

```luau
from_triangles(n: number, tris: { number }): Graph
```

The edges of a triangle mesh on vertices 1..n: tris lists 3 vertices per triangle.

### `symmetrize`

```luau
symmetrize(G: Graph): Graph
```

The undirected graph with G's edges both ways (the pattern of A + A^T), self loops dropped.

### `degrees`

```luau
degrees(G: Graph): buffer
```

Number of neighbors of each vertex: a buffer of 32-bit integers.

### `bfs`

```luau
bfs(G: Graph, source: number): (buffer, buffer)
```

Breadth-first search from source (1-based) through G's edges. Returns the vertices reached in the order visited (a buffer of 32-bit integers, 1-based) and each vertex's level (its distance in edges: 0 at the source, -1 if not reached).

### `components`

```luau
components(G: Graph): (buffer, number)
```

Connected components of an undirected graph (symmetrize a directed one first): each vertex's component (1-based, numbered in order of their lowest vertex; a buffer of 32-bit integers) and the number of components.

### `inverse`

```luau
inverse(perm: buffer): buffer
```

The positions of a permutation: inv[perm[k]] = k.

### `bandwidth`

```luau
bandwidth(G: Graph, perm: buffer?): number
```

The half bandwidth of G's adjacency matrix with vertices ordered by perm (default: as numbered): the largest |position(v) - position(w)| over edges v - w.

### `profile`

```luau
profile(G: Graph, perm: buffer?): number
```

The profile (envelope size) of a symmetric pattern ordered by perm: the sum over rows i of i minus the first column stored in row i. A skyline or banded factorization's work grows with it.

### `rcm`

```luau
rcm(G: Graph): buffer
```

Reverse Cuthill-McKee ordering of an undirected graph (symmetrize a directed one first): breadth-first from a pseudo-peripheral vertex of each component (George and Liu's search: repeated searches from a lowest-degree vertex of the last level while the depth grows), visiting each vertex's unvisited neighbors in increasing degree, then reversed. Neighbors in the graph end up near each other in the order, so the matrix's band narrows: on a mesh of n vertices numbered arbitrarily, from up to n to about sqrt(n). Cost O(m log d) for m edges and degree d. Returns the permutation (buffer of 32-bit integers): vertex perm[k] goes k-th.

### `dijkstra`

```luau
dijkstra(G: Graph, w: buffer?, source: number | { number }, options: PathOptions?): (buffer, buffer)
```

Dijkstra's shortest paths from source (a vertex, or a list of them, 1-based) along G's edges, of lengths w (an f64 array aligned with G's ind, nonnegative, such as a Sparse matrix's values; nil for 1 each). options.target stops once that vertex is settled and options.limit once the distance exceeds it. Returns each vertex's distance (an f64 array; math.huge if not reached) and predecessor on a shortest path (a buffer of 32-bit integers, 1-based; 0 for sources and unreached vertices): graph.path(prev, v) is the path. O((n + m) log n).

### `path`

```luau
path(prev: buffer, v: number, dist: buffer?): { number }?
```

The path to v (1-based) from predecessors prev (as dijkstra's): its vertices from the source to v, or nil if v was not reached (and is not a source).

### `astar`

```luau
astar(G: Graph, w: buffer?, source: number, target: number, h: (number) -> number): ({ number }?, AstarInfo)
```

A* search for a shortest path from source to target (1-based) with edge lengths w (as dijkstra's) and a heuristic h(v) (1-based) never above v's true distance to target (a straight-line distance, for points joined by straight edges): the closer h is to the true distance, the fewer vertices it expands; h = 0 is Dijkstra's search. Returns the path (vertices from source to target) or nil, and info: the path's cost and the number of vertices expanded.

### `amd`

```luau
amd(G: Graph): buffer
```

Approximate minimum degree ordering of an undirected graph (a symmetric pattern; symmetrize a directed one first), after Amestoy, Davis and Duff (SIAM J. Matrix Anal. Appl., 1996). Eliminating a vertex joins its neighbors into a clique: the ordering eliminates, at each step, a vertex of least degree, which keeps the Cholesky factor L of a matrix with this pattern sparse. The graph of eliminated vertices is kept implicitly (each eliminated vertex an "element": the clique of its remaining neighbors), degrees are bounded from above rather than counted, vertices with the same neighbors are merged and eliminated together, and elements whose cliques lie inside a newer one are absorbed, so the work grows about as the number of edges. All lists live in one integer workspace and are pruned in place (each step removes at least one entry from every list it touches, so the new element fits); the workspace is compacted when a new element does not fit at its end. Returns the permutation (a buffer of 32-bit integers): vertex perm[k] goes k-th.

### `etree`

```luau
etree(G: Graph, perm: buffer?): (buffer, buffer)
```

The elimination tree of a symmetric pattern with its vertices ordered by perm (default as numbered), by Liu's algorithm with path compression: column j's parent is the first row below the diagonal in column j of the Cholesky factor. Returns the parents (a buffer of 32-bit integers, positions in the order, 1-based; 0 at a root) and a postorder (a permutation of the positions, children before parents).

### `chol_analysis`

```luau
chol_analysis(G: Graph, perm: buffer, options: CholOptions?): CholSymbolic
```

The symbolic Cholesky factorization of a symmetric pattern with vertices ordered by perm: the order is postordered (its elimination tree's children before parents, which keeps L the same and makes its supernodes contiguous), then the columns of L are grouped into supernodes, runs of columns with the same rows below their diagonal block, stored as dense blocks. With options.relax (default true), a supernode is merged into its parent when that adds few explicit zeros (up to 4 columns while under 50% of the entries are zeros, up to 16 under 30%, up to 48 under 10%, any width under 5%): wider blocks for the dense kernels, at the cost of multiply-adds on zeros. These limits measured best on the drum and cloth matrices (CHOLMOD's looser 80% for up to 16 columns doubled the drum's multiply-adds and cost 25% in time). Also maps each stored entry of the matrix to its place in L.
