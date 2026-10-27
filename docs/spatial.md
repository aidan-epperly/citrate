# Spatial queries

Reference: [spatial](reference/spatial.md).

```luau
local spatial = citrate.spatial
-- a uniform grid: the fastest for many queries of one radius on moving points
local G = spatial.grid(points, 4)        -- points: { vector }; cells 4 studs wide
G:rebuild(points)                        -- each frame, after the points move
local near, count = G:radius(p, 4, out)  -- indices within 4 of p, into out
local I, J, m = G:pairs(4)               -- every pair within 4

-- a k-d tree: nearest points, varying radii, sets queried many times
local T = spatial.kdtree(points)
local i, d = T:nearest(p)
local idx, dist = T:knn(p, 8)            -- the 8 nearest, nearest first
local near, count = T:radius(p, r)
T:refit(points)                          -- points moved a little: keep the splits

-- either one's radius graph, for the graph module
local Gr, w = T:graph(18)                -- edges within 18, w their lengths
local route = graph.astar(Gr, w, s, t, heuristic)
```

Points are a `{ vector }` array in 3-D (leave z at 0 for 2-D). Results are 1-based
indices into that array. Queries that return a list take an optional table to fill, so a
query made every frame need not allocate (`bench/alloc.luau`: 0 bytes per query, rebuild
and refit).

## Grid or tree

| | uniform grid | k-d tree |
| --- | --- | --- |
| build | O(n), by counting sort into hashed cells | O(n log n), median splits |
| radius query | the cells within r: 27 when r is at most the cell size | O(log n + k) |
| nearest, k nearest | not provided | yes |
| moving points | rebuild each frame | refit while they move little, else rebuild |
| best for | many queries of one radius on points that all move (flocks, particles) | nearest points, varying radii, sets queried many times (waypoints, spawn points) |

The grid's cells are hashed into about two slots per point, so its memory follows the
number of points, not the size of the world.

Measured, every point's neighbors within r with about 12 each (`bench/spatial.luau`;
native, interpreted in brackets):

| points | layout | every pair | grid: rebuild + queries | grid: pairs | k-d tree: rebuild + queries |
| --- | --- | --- | --- | --- | --- |
| 2000 | uniform | 5.7 (17) ms | 2.1 (5.5) ms | 1.3 (4.4) ms | 3.1 (6.8) ms |
| 2000 | 20 clusters | 5.9 (17) ms | 2.5 (6.3) ms | 1.4 (4.7) ms | 3.3 (7.4) ms |
| 10000 | uniform | | 12 (29) ms | 6.6 (23) ms | 19 (44) ms |
| 10000 | 20 clusters | | 14 (34) ms | 7.8 (26) ms | 23 (49) ms |

Testing every pair grows as n^2, at about 2.9 ns per pair natively. Clustered points slow
both structures by about the same amount, 7 to 18%.

A refit tree keeps its splits, so its queries slow down as the points drift (2000 points,
queries for every point):

| drift since the build (in units of r) | 0 | 0.25 | 1 | 4 |
| --- | --- | --- | --- | --- |
| radius queries after refit | 2.5 ms | 2.7 ms | 3.9 ms | 9.9 ms |
| after rebuild | 2.6 ms | 2.5 ms | 2.4 ms | 1.8 ms |

Refitting takes 0.02 ms and rebuilding 0.5 ms. Refit while the points have moved less
than about r/2 since the last rebuild, and rebuild after that.

Against SciPy's `cKDTree` on 10000 points (`bench/scipy`):

| operation | citrate, native | interpreted | SciPy |
| --- | --- | --- | --- |
| build | 3.9 ms | 9.4 ms | 1.6 ms |
| the 8 nearest of every point | 24 ms | 56 ms | 12 ms |
| every point's neighbors within r | 15 ms | 33 ms | 25 ms |

The radius queries beat SciPy's because SciPy returns a Python list for every point.

In Roblox (`demos/flock`), finding 1000 boids' neighbors took 0.84 ms per frame with the
grid and 4.1 ms with one `workspace:GetPartBoundsInRadius` per boid, which also needs a
part for each point.
