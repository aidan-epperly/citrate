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

### `triangulate`

```luau
triangulate(boundary: { any }, options: PolygonOptions?): ({ any }, { { number } })
```

The triangles filling a polygon: boundary is its points in order around it (either way; tables or f64 arrays of 2 numbers), options.holes a list of polygons inside it. Returns the points (the boundary's, then each hole's: the indices the triangles use) and the triangles (counterclockwise, 3 indices each): the constrained Delaunay triangulation (the polygon's edges kept; otherwise as near Delaunay as they allow), with no new points. Errors if the polygon's edges cross or a point repeats.

### `mesh`

```luau
mesh(boundary: { any }, options: MeshOptions?): ({ { number } }, { { number } }, MeshInfo)
```

A quality triangle mesh of a polygon (with holes): boundary and options.holes as for triangulate; options.size the longest edge wanted (default: no limit), options.angle the smallest angle wanted (degrees, default 20; refinement may not end above about 33), options.max_points a limit on the points (default 200000; reaching it is reported in info.warning). Returns the points (tables { x, y }: the polygon's, in order, then the new ones), the triangles (counterclockwise) and info: method, points, triangles, min_angle (degrees, over the mesh).

### `inside`

```luau
inside(p: any, polygon: { any }): number
```

Whether point p lies inside polygon (its points in order, either way; 2-D): 1 inside, 0 on its boundary, -1 outside, exactly. Inside means a nonzero winding number (Hormann and Agathos's crossing rule, with exact orientations), so a polygon that winds around twice holds its middle. O(n) a query; for many queries of one polygon, triangulate it.

### `closest_on_segment`

```luau
closest_on_segment<T>(p: T, a: T, b: T, space: Space<T>?): (T, number)
```

The point of segment a, b nearest p, and its parameter t in [0, 1] (the point is a + t (b - a)).

### `closest_on_triangle`

```luau
closest_on_triangle<T>(p: T, a: T, b: T, c: T, space: Space<T>?): (T, number, number, number)
```

The point of triangle a, b, c nearest p, and its barycentric coordinates u, v, w (the point is u a + v b + w c).

### `closest_between_segments`

```luau
closest_between_segments<T>(p1: T, q1: T, p2: T, q2: T, space: Space<T>?): (T, T, number, number)
```

The nearest points of segments p1, q1 and p2, q2: c1 = p1 + s (q1 - p1) and c2 = p2 + t (q2 - p2), with s and t in [0, 1]; returns c1, c2, s, t (the distance is |c1 - c2|).

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
