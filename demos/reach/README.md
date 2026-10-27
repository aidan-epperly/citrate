# Reach: an arm reaching for a moving target

Two copies of an arm of 5 segments, each joint turning about two axes within limits, reach
for the same target as it moves through and beyond their reach:

- Left: CCD (cyclic coordinate descent), the usual approach: each joint axis in turn, from
  the hand back to the base, turned to swing the hand toward the target and clamped to its
  limits, 16 sweeps a frame.
- Right: citrate's `optimize.least_squares` (Levenberg-Marquardt) on the hand's distance
  from the target plus a pull toward a rest pose, from last frame's angles. The limits are
  kept smoothly by solving for phi with angle = limit tanh(phi).

The panels show each one's time per frame, its hand's distance from the target, and its
largest angle change in a frame: CCD's jumps (when a clamped joint makes the next one swing
the other way) show as twitching.

Measured outside Roblox (`bench.luau`: 10 s of the target's path at 60 Hz):

| method | time per frame, native / interpreted | largest distance while in reach | largest angle change in a frame | distance from the rest pose |
| --- | --- | --- | --- | --- |
| citrate, Levenberg-Marquardt, Jacobian reused ("lazy"), up to 4 iterations a frame (the demo) | 0.070 / 0.25 ms | 0.016 studs | 0.08 rad | 1.02 |
| citrate, the same with Broyden updates | 0.081 / 0.33 ms | 0.016 studs | 0.08 rad | 1.02 |
| citrate, the same with fresh Jacobians | 0.11 / 0.38 ms | 0.016 studs | 0.12 rad | 1.02 |
| citrate, L-BFGS (Wolfe line search) | 1.0 / 1.8 ms | 0.016 studs | 0.06 rad | 1.02 |
| citrate, L-BFGS (Chebyshev line search) | 0.64 / 2.6 ms | 0.017 studs | 0.09 rad | 1.02 |
| CCD, 4 sweeps a frame | 0.025 / 0.047 ms | 0.33 studs | 2.8 rad | 2.25 |
| CCD, 16 sweeps a frame | 0.096 / 0.16 ms | 0.083 studs | 2.7 rad | 2.20 |

Without the cap on iterations, Levenberg-Marquardt took 6.5 iterations a frame and 0.20 ms
(0.31 ms with the SVD-based solver it replaced): the target moves little between frames,
so polishing each frame's answer bought nothing visible.

The rest-pose pull keeps the hand up to 0.016 studs from targets it could reach exactly;
without it the arm's redundancy (10 angles for a 3-D position) lets the solution drift
between nearly equal poses from frame to frame. The derivatives are by differences with
relative steps of 1e-4 (`diff_step`), since the arm is computed with `vector`s (single
precision).

Files:

- `reach.luau`: the arm, both solvers and the target's path, usable outside Roblox.
- `client.client.luau`: the Roblox client script.
- `reach.project.json`: builds a playable place with `rojo build demos/reach/reach.project.json
  -o reach.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the table above (`luau -O2 [--codegen] demos/reach/bench.luau`).
