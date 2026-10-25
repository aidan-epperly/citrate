# Tails: the same motion at any update rate

Six identical tails hang off roots that shake, rest and shake again. The top row moves by
SmartBone 2's rule (a popular Roblox module for dynamic bones), the bottom row is the
same tail as a physical system advanced by citrate. The columns step at 60, 30 and 15 Hz:
SmartBone steps an object at 60 Hz up to 15 studs from the camera and lowers the rate
linearly to 0 at 45 studs, so these are its rates at 15, 30 and 37.5 studs.

- SmartBone's rule (`tails.smartbone`, from its `Bone:StepPhysics` and "Spring" constraint
  with the default settings): a Verlet step that keeps `1 - Damping` of the velocity per
  step, gravity added as `g dt^2`, then each bone pulled `Elasticity * dt` of the way back
  to its rest position and set at its length from its parent. Damping per step means
  damping per second grows with the rate, and gravity per step (dt^2) against the pull
  back (dt) means the droop at rest grows as the rate falls.
- citrate (`tails.physical`): stiff rods with dashpots, springs toward the rest shape, drag
  and gravity, advanced by `ode.implicit_mechanics` with the rational step (`rho_inf =
  0.5`): one linear solve per step, stable however stiff the rods, and nearly free of
  numerical damping at these rates. Its springs and drag were chosen so that at 60 Hz it
  droops and settles like SmartBone's tail.

Measured outside Roblox (`bench.luau`; one shake of 4/3 s, then rest):

| | rate | tip droop at rest | moving after the shake | time per step, native / interpreted |
| --- | --- | --- | --- | --- |
| SmartBone rule | 60 Hz | 0.78 studs | 2.9 s | 0.5 / 0.7 us |
| | 30 Hz | 1.47 studs | 4.4 s | |
| | 15 Hz | 2.54 studs | 5.1 s | |
| citrate | 60 Hz | 0.78 studs | 2.7 s | 35 / 160 us |
| | 30 Hz | 0.78 studs | 2.7 s | |
| | 15 Hz | 0.78 studs | 2.7 s | |

So a SmartBone tail looks different at each distance from the camera, three times as
droopy at 37.5 studs as up close, while citrate's looks the same at every rate. It costs
far more: about 35 us per tail step natively and 160 us interpreted (Roblox clients run
scripts interpreted), against under 1 us. In Studio the three citrate tails took about 0.1
ms per frame together.

SmartBone 2 is Copyright (c) 2023 SmartBone contributors, MIT license; `tails.smartbone`
reproduces its update rule.

Files:

- `tails.luau`: both models, usable outside Roblox.
- `client.client.luau`: the Roblox client script.
- `tails.project.json`: builds a playable place with `rojo build
  demos/tails/tails.project.json -o tails.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the table above (`luau -O2 [--codegen] demos/tails/bench.luau`).
