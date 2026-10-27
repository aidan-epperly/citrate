# Flock: spatial queries

Two scenes in one place, switched by the panel's buttons:

- **Flock.** Boids, each steering by its neighbors within 4 studs: keeping apart, matching
  their heading, moving toward their center, and turning back from the walls of a box.
  Every frame, each boid's neighbors are found by one of four methods, chosen with the
  buttons:
  - citrate's uniform grid (`spatial.grid`, rebuilt each frame);
  - its k-d tree (`spatial.kdtree`, rebuilt each frame);
  - testing every pair;
  - Roblox's `workspace:GetPartBoundsInRadius`, one query per boid.

  The panel shows the time each method takes per frame. The count buttons set 500, 1000
  or 2000 boids.
- **Navigation.** About 400 waypoints scattered among obstacles. Each waypoint is joined to
  the waypoints within 18 studs (`kdtree:graph`) when the straight way between them clears
  every obstacle. Click the floor to send the walkers there. Each walker routes by A*
  (`graph.astar`, with the straight-line distance as heuristic) from the waypoint nearest
  it to the one nearest the click. The route of the first walker is drawn.

Measured in Roblox Studio (time per frame spent finding neighbors):

| boids | grid | k-d tree | every pair | `GetPartBoundsInRadius` |
| --- | --- | --- | --- | --- |
| 1000 | 0.84 ms | 1.00 ms | 3.2 ms | 4.1 ms |
| 2000 | 1.8 ms | | | 8.9 ms |

Roblox's query costs about 4 us per call. The grid finds all of a frame's neighbors for
less than the cost of 250 such calls.

Outside Roblox (`bench.luau`, the flock after a second of flight; native / interpreted):

| boids | grid | k-d tree | every pair |
| --- | --- | --- | --- |
| 500 | 0.30 / 1.1 ms | 0.37 / 0.90 ms | 0.80 / 2.4 ms |
| 1000 | 0.70 / 2.3 ms | 0.90 / 2.2 ms | 3.1 / 9.5 ms |
| 2000 | 1.6 / 4.7 ms | 2.3 / 6.0 ms | 13 / 38 ms |

Testing every pair grows as n^2, and the grid and tree grow as n.

Routes on the navigation scene (`bench.luau`, 20 routes; native):

| waypoints | A* | Dijkstra's search to the same target |
| --- | --- | --- |
| 300 | 0.017 ms, 17 waypoints expanded | 0.038 ms |
| 1000 | 0.046 ms, 40 waypoints expanded | 0.22 ms |

The heuristic steers A* toward the target, and its advantage grows with the number of
waypoints.

Files:

- `flock.luau`: both scenes, usable outside Roblox.
- `client.client.luau`: the Roblox client script. Its folder `workspace.FlockDemo` also
  takes the attributes `Method`, `Scene`, `Count` and `Goal`, for driving it from a script.
- `flock.project.json`: builds a playable place with `rojo build
  demos/flock/flock.project.json -o flock.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the tables above (`luau -O2 [--codegen] demos/flock/bench.luau`).
