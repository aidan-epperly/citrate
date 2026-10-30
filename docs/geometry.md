# Geometry

Reference: [geometry](reference/geometry.md).

```luau
local geometry = citrate.geometry

geometry.orient(a, b, c)                 -- 1 counterclockwise, -1 clockwise, 0 collinear: exactly
geometry.insphere(a, b, c, p)            -- 1 if p is inside the circle through a, b, c

local T = geometry.delaunay(points)      -- any dimension
local k, weights = T:locate(p)           -- the simplex holding p, and p's barycentric weights
local G, lengths = T:graph()             -- natural neighbors, for citrate.graph

local H = geometry.hull(points)          -- H.vertices: the corners; H.hull: the facets

local pts, tris = geometry.triangulate(polygon, { holes = { hole } })   -- 2-D
local pts, tris, info = geometry.mesh(polygon, { size = 0.1, angle = 25 })
geometry.inside(p, polygon)              -- 1 inside, 0 on the boundary, -1 outside

local q, t = geometry.closest_on_segment(p, a, b)
local q, u, v, w = geometry.closest_on_triangle(p, a, b, c)
local c1, c2 = geometry.closest_between_segments(a, b, c, d)
```

Points are vectors (3-D), or tables or f64 arrays of d numbers, in any dimension d. The
triangulation, hull and predicates work in any dimension, with the determinants written
out for 2-D and 3-D; polygons and meshes are 2-D (below); closest points work in any
dimension, through the points' own arithmetic (a `Space`: vectors, f64 arrays, or one of
your own as the last argument).

## Exact signs

Everything here branches on the sign of a determinant: which side of a line a point is on,
whether it is inside a circle. Computed in floating point, a sign can come out wrong for
nearly degenerate points, and a triangulation built on a wrong sign can overlap itself or
loop forever. Nearly degenerate is common: a grid of points has four points on every
square's circle, and a row of points on a line. Kettner et al.'s classic example (points
near the line through (12, 12) and (24, 24)) gets 40 of 256 orientations wrong in floating
point; `geometry.orient` gets them all right (tested).

So signs are exact: each determinant is computed in floating point with a bound on its
rounding error (Shewchuk's bounds in 2-D and 3-D, a general one otherwise), and only when
the bound leaves the sign in doubt is it computed again exactly, in expansion arithmetic
(a number as a sum of doubles; Shewchuk 1997). Two cheaper exact paths come first: grids
of integer (or 1/16) coordinates, where the floating-point determinant is itself exact,
cost 0.3 us; anything else 1 to 8 us. Measured natively:

| | random points | an integer grid |
| --- | --- | --- |
| 2-D Delaunay, 10,000 points | 31 ms (3.1 us a point) | 36 ms |
| 3-D Delaunay, 10,000 points | 175 ms (18 us a point) | 153 ms (8,000 points) |
| 4-D Delaunay, 2,000 points | 0.75 s | |
| 2-D hull, 10,000 points | 10 ms | 13 ms |
| 3-D hull, 10,000 points | 54 ms | 33 ms (8,000 points) |

Coordinates must stay below about 1e150 in size, so that products do not overflow.

## Delaunay triangulations and hulls

`geometry.delaunay(points)` returns the Delaunay triangulation: simplices (triangles in
2-D, tetrahedra in 3-D, each d + 1 point indices, positively oriented: counterclockwise
triangles) whose circumspheres hold no other point. Points on a common circle are
triangulated in one of the valid ways; repeated points are listed in `T.duplicates`. Its
convex hull's facets come too (`T.hull`, counterclockwise edges in 2-D).

It is Bowyer and Watson's algorithm: each point removes the simplices whose circumsphere
holds it and joins itself to the hole's boundary. The triangulation is closed by "ghost"
simplices to a point at infinity, so points outside the current hull, and collinear or
coplanar ones, need no special cases. Points go in a randomized spatial order (BRIO,
sorted along a k-d tree in each round) and are located by walking from the last one, so
each insertion costs about a constant: n log n for the sort, then linear.

`T:locate(p)` finds the simplex holding p and p's barycentric weights in it, which
interpolate values given at the points linearly (piecewise linear interpolation of
scattered data; for smooth interpolation, see [radial basis functions](surfaces.md)).
`T:graph()` gives the natural-neighbor graph, with edge lengths, for
[`citrate.graph`](graphs.md)'s routes.

`geometry.hull(points)` is the convex hull in any dimension: its corners (`H.vertices`),
its facets (`H.hull`, oriented outward: each facet followed by an inside point is
positively oriented) and a triangulation of its inside. It is the same insertion, with
only points outside the hull going in; in 2-D, Andrew's monotone chain instead (2.6 times
faster).

## Polygons and meshes (2-D)

These are two-dimensional by nature: a triangulation that keeps given edges need not
exist in three dimensions (Schonhardt's twisted prism cannot be cut into tetrahedra without
new points), so meshing a solid is a separate, larger problem. For solids, triangulate
points with `geometry.delaunay`, or make a box's mesh with `pde.grid_mesh`.

`geometry.triangulate(polygon, { holes })` fills a polygon (its points in order, either
way, with holes inside it) with triangles using only its own points: the constrained
Delaunay triangulation, which keeps the polygon's edges and is otherwise as close to
Delaunay as they allow (each edge forced in by removing the triangles it crosses and
filling each side again by Anglada's algorithm). Polygons whose edges cross, or with a
repeated point, are refused.

`geometry.mesh(polygon, { holes, size, angle })` adds points until every triangle has no
angle below `angle` (degrees, default 20) and no edge longer than `size`: Ruppert's Delaunay
refinement, with Shewchuk's improvements (splits near the polygon's corners on power-of-2
shells, and no refining of the triangles a sharp corner forces), so it ends even when the
polygon has sharp corners, which keep their angle. Up to about 33 degrees refinement ends in
practice (20.7 is proved). It costs about 6 us a point natively (12,500 points in 77 ms).
`pde.mesh` is the same, for [PDEs](pde.md).

`geometry.inside(p, polygon)` tells whether p is inside (the winding number is not zero),
on the boundary, or outside, exactly.

## Closest points

`closest_on_segment`, `closest_on_triangle` and `closest_between_segments` give nearest
points (and their parameters: t along the segment, barycentric u, v, w on the triangle),
by Ericson's algorithms, which need only differences and dot products: so any dimension,
and any representation with a `Space`. Vectors compute in single precision, as vectors do.
