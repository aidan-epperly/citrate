# Demos

Each demo is a Roblox place built from `demos/<name>` (`rojo build
demos/<name>/<name>.project.json -o <name>.rbxl`, then open it in Studio and press Play),
and each sets the library beside the usual approach. The clips were recorded in Roblox
Studio; click a preview for the full video.

## Terrain: marbles on a heightfield

[![terrain](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/terrain.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/terrain.mp4)

The same heights, known every 5 studs, interpolated bilinearly (left) and by
`interp.grid` (right), with frictionless marbles on both. The bilinear surface's slope
jumps at every grid line, and its marbles lose their energy there and stop.
[More](../demos/terrain/README.md).

## Cloth: implicit against explicit

[![cloth](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/cloth.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/cloth.mp4)

Two cloths holding a 1000 kg bar with the same CPU time per frame: one implicit step
(`ode.implicit_mechanics`, multigrid-preconditioned) against explicit substeps, which can
only afford soft springs. [Same springs instead](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/cloth_stiffness.mp4).
[More](../demos/cloth/README.md).

## Flow: a vortex street

[![flow](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/flow.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/flow.mp4)

Incompressible Navier-Stokes flow past an obstacle on a 192 by 96 grid, solved every
frame; the wake sheds vortices from each side in turn. [More](../demos/flow/README.md).

## Drum: can you hear the shape of a drum?

[![drum](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/drum.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/drum.mp4)

Gordon, Webb and Wolpert's two differently shaped drums with the same frequencies,
computed by finite elements and shift-invert Lanczos. Each button plays one mode on both,
and they oscillate in step; clicking a drum strikes it. [More](../demos/drum/README.md).

## Flock: spatial queries

[![flock](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/flock.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/flock.mp4)

A thousand boids finding their neighbors each frame through `spatial` grids or k-d trees,
then walkers routed by A* (`graph.astar`) among obstacles. [More](../demos/flock/README.md).

## Track: constant speed along a curve

[![track](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/track.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/track.mp4)

Carts placed by arclength with a lookup table, as popular Bezier modules do (left; its
speed wobbles), and with `curve.at_length` (right). [More](../demos/track/README.md).

## Path: a roller coaster

[![path](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/path.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/path.mp4)

The path component turns a folder of Parts into a smooth track and moves trains along it by
distance, with twist-free frames. The clip rides the coaster on a Catmull-Rom curve through
the Parts (left), whose curvature jumps at each one, and on the component's C2 cubic spline
(right), whose curvature is continuous. [More](../demos/path/README.md).

## Reach: an arm reaching for a target

[![reach](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/reach.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/reach.mp4)

A 5-segment arm within joint limits: cyclic coordinate descent, the usual approach (left),
against `optimize.least_squares` from last frame's angles (right), whose joints never jump. [More](../demos/reach/README.md).

## Tails: the same motion at any update rate

[![tails](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/tails.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/tails.mp4)

Tails stepped at 60, 30 and 15 Hz: SmartBone 2's rule (top) moves differently at each
rate, while citrate's XPBD (middle, fitted to SmartBone at 60 Hz) and implicit (bottom)
tails move alike at every rate. [More](../demos/tails/README.md).

## Bones: the dynamic bones component

[![bones](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/bones.gif)](https://raw.githubusercontent.com/wiki/aidan-epperly/citrate/media/bones.mp4)

Creatures with tails and antennae of real `Bone` instances, updated every frame, every
second frame and every third, as the component throttles distant rigs.
[More](../demos/bones/README.md).
