# Terrain: marbles on a heightfield

Two copies of a terrain known only at grid points (the dark dots, every 5 studs), with
marbles rolling on them from the same starting places. The marbles are stepped the same
way on both and there is no friction, so ideally they would roll forever. Only the surface
between the grid points differs:

- Left: bilinear interpolation, the usual approach. Its slope jumps at every grid line, as
  a triangle mesh's does at every edge.
- Right: citrate's `interp.grid` (Keys' cubic convolution), whose slope is continuous.

Each frame a marble is pushed by gravity's component along the surface, moves, and has its
velocity made tangent to the surface where it lands, which is how games usually roll a
ball on terrain. Where the slope jumps, the velocity that was tangent on one side of a
grid line points into the surface on the other, and removing that part is a small
collision: each crossing costs a fraction of the marble's speed that no time step makes
smaller. On the cubic surface the only loss is the time step's, and it halves each time
the substeps double.

Energy lost (as a fraction of what the marbles could lose), 200 marbles, measured outside
Roblox (`bench.luau`):

| substeps per 60 Hz frame | bilinear, after 10 s | after 30 s | cubic, after 10 s | after 30 s |
| --- | --- | --- | --- | --- |
| 1 | 81% | 99% | 26% | 61% |
| 2 | 81% | 100% | 15% | 39% |
| 4 (default) | 81% | 99% | 8% | 22% |
| 8 | 81% | 99% | 5% | 12% |

So on the bilinear terrain the marbles stop within seconds whatever the time step, while on
the cubic one they keep rolling. A step costs 0.12 us per marble on the bilinear surface
and 0.25 on the cubic one natively (0.31 and 0.86 interpreted); in Studio, 200 marbles at 4
substeps took 0.10 and 0.21 ms per frame.

The panel shows each side's energy lost, how many marbles are still rolling, and the time
per frame; the buttons set the number of marbles (50, 200, 1000) and substeps (1 to 8),
and restart.

Files:

- `terrain.luau`: the heights, both surfaces and the marbles' step, usable outside Roblox.
- `client.client.luau`: the Roblox client script (the terrain meshes are EditableMeshes,
  4 vertices per grid spacing, each sampled from its own surface).
- `terrain.project.json`: builds a playable place with `rojo build
  demos/terrain/terrain.project.json -o terrain.rbxl` (open it in Roblox Studio and press
  Play).
- `bench.luau`: the table above (`luau -O2 [--codegen] demos/terrain/bench.luau`).
