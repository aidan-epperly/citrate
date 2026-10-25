# Track: moving along a curve at constant speed

Carts move along a closed track at a set speed, so each frame needs the point at a given
distance along the curve. The demo shows two copies of the track:

- Left: a lookup table, the way popular Roblox Bézier modules parameterize by arclength
  (bstummer's Bezier, for one): the chord lengths at evenly spaced parameters (101 per
  curve by default), a linear scan for the entry below the wanted distance, and linear
  interpolation in the parameter.
- Right: citrate's `curve.at_length`, with the arclength tables built to a tolerance of
  1e-8.

The plots show the lead cart's speed relative to the set speed, and the panels the time
per frame to place all the carts. The buttons change the table size, the number of carts
and the speed.

Measured outside Roblox (`bench.luau`, on this track of 10 cubic segments):

| method | speed, relative to the set speed | largest position error | time per point, interpreted / native |
| --- | --- | --- | --- |
| table, 101 samples per segment | 0.9951 to 1.0045 | 2.4e-3 studs | 0.63 / 0.34 us |
| table, 1001 samples | 0.9999 to 1.0001 | 2.8e-5 studs | 4.6 / 2.8 us |
| table, 4701 samples | 0.9999 to 1.0001 | 4.8e-5 studs | 20 / 13 us |
| citrate, tol 1e-8 | 1.0000 to 1.0000 | 7.6e-6 studs | 0.74 / 0.32 us |

With the default table the speed wobbles by about half a percent, at every table entry.
Making the table fine enough to remove that costs time in proportion to its size, because
each lookup scans the table. citrate costs about what the coarse table does and is
accurate to what a Vector3 holds. Building citrate's tables costs more: per cubic segment 29 us natively and 89 interpreted
(at tol 1e-8), once per track. In Studio, with 1000 carts: 0.36 ms per frame for
the 101-sample table, 14 ms for the 4701-sample one, and 0.33 ms for citrate.

On a single sharply bent cubic (the curve in bstummer's documentation examples is of that
kind), the 101-sample table's speed varied by 2% and its positions by 4.4e-3 studs.

Files:

- `track.luau`: the track, both methods, usable outside Roblox.
- `client.client.luau`: the Roblox client script.
- `track.project.json`: builds a playable place with `rojo build
  demos/track/track.project.json -o track.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the table above (`luau -O2 [--codegen] demos/track/bench.luau`).
