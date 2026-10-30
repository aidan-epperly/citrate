# Spatial queries

Uniform grids and k-d trees on point sets: radius, nearest and k nearest queries, all pairs within a radius, radius graphs.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `grid`

```luau
grid(points: { any }, cell: number): any
```

The constructors: points as vectors use the vector structures, anything else (tables or f64 arrays of d numbers) the ones for any dimension.

### `kdtree`

```luau
kdtree(points: { any }, options: KdOptions?): any
```
