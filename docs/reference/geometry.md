# Geometry

Computational geometry in any dimension: exact orientation and in-sphere tests, Delaunay triangulations, convex hulls.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `orient`

```luau
orient(...: any): number
```

The orientation of d + 1 points in d dimensions (as separate arguments): 1, -1 or 0 exactly. In 2-D, 1 when a, b, c turn counterclockwise; in 3-D, 1 when d lies on the side of the plane through a, b, c from which they turn clockwise (the sign of det[b - a; c - a; d - a]); in general the sign of det[p_2 - p_1; ...; p_(d+1) - p_1].

### `insphere`

```luau
insphere(...: any): number
```

Whether the last point lies inside (1), on (0) or outside (-1) the sphere through the other d + 1 (in 2-D, the circle through three points), exactly. The d + 1 points must not lie in a hyperplane (three on a line, in 2-D).

### `delaunay`

```luau
delaunay(points: { any }): Triangulation
```

The Delaunay triangulation of points in d >= 2 dimensions (vectors are 3-D; tables or f64 arrays of d numbers): no point lies strictly inside any simplex's circumsphere. Returns a table: simplices (each a table of the d + 1 indices of its vertices, positively oriented: counterclockwise triangles in 2-D), hull (the convex hull's facets: d indices each, in 2-D its edges, counterclockwise around the hull), vertices (the points used: all but duplicates) and duplicates (index of a point -> the index of the point it repeats). Points on a common circle (sphere) are triangulated in one of the valid ways. Errors if all the points lie in a hyperplane (a line, in 2-D).

### `hull`

```luau
hull(points: { any }): Triangulation
```

The convex hull of points in d >= 2 dimensions: a table with hull (the facets: d indices each, oriented so that each facet with a point inside the hull is positively oriented; in 2-D the edges, counterclockwise), vertices (the indices of its corners) and simplices (a triangulation of its inside). Facets may be coplanar (a square face of a cube comes as two triangles), and may have points on the hull's boundary that are not corners among their vertices (a point inserted before the hull grew past it).

### `_predicates`

```luau
_predicates(d: number): Predicates
```

### `_triangulate`

```luau
_triangulate(points: { any }, options: TriangulationOptions?, who: string): Triangulation
```

### `_read_points`

```luau
_read_points(points: { any }, who: string): ({ number }, number)
```

The coordinates of a list of points, d numbers each, into one table; returns it and d.
