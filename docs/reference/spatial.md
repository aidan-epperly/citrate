# Spatial queries

Uniform grids and k-d trees on point sets: radius, nearest and k nearest queries, all pairs within a radius, radius graphs.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `grid`

```luau
grid(points: { vector }, cell: number): Grid
```

A uniform grid of the points with cells `cell` wide (a query radius, typically: queries of radius up to cell scan 27 cells). grid:rebuild(points) refills it for moved points (or another array), reusing its storage.

### `kdtree`

```luau
kdtree(points: { vector }, options: KdOptions?): KdTree
```

A k-d tree of the points (options.leaf: points per leaf, default 8). tree:rebuild(points) builds it again, reusing its storage; tree:refit(points) keeps its splits.
