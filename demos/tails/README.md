# Tails: the same motion at any update rate

Nine identical tails hang off roots that shake, rest and shake again. The top row moves by
SmartBone 2's rule (a popular Roblox module for dynamic bones); the other two are the same
tail as a physical system, advanced by two of citrate's solvers. The columns step at 60,
30 and 15 Hz: SmartBone steps an object at 60 Hz up to 15 studs from the camera and lowers
the rate linearly to 0 at 45 studs, so these are its rates at 15, 30 and 37.5 studs.

- SmartBone's rule (`tails.smartbone`, from its `Bone:StepPhysics` and "Spring" constraint
  with the default settings): a Verlet step that keeps `1 - Damping` of the velocity per
  step, gravity added as `g dt^2`, then each bone pulled `Elasticity * dt` of the way back
  to its rest position and set at its length from its parent. Damping per step means
  damping per second grows with the rate, and gravity per step (dt^2) against the pull
  back (dt) means the droop at rest grows as the rate falls.
- citrate implicit (`tails.physical`): stiff rods with dashpots, springs toward the rest
  shape, drag and gravity, advanced by `ode.implicit_mechanics` with the rational step
  (`rho_inf = 0.5`): one direct (banded) solve per update, stable however stiff the rods
  and nearly free of numerical damping at these rates. Its springs and drag were chosen so
  that at 60 Hz it droops and settles like SmartBone's tail.
- citrate XPBD (`tails.xpbd`): the same rods, springs, drag and gravity as compliant
  constraints for `citrate.xpbd`, in substeps of 1/240 s whatever the update rate.

Measured outside Roblox (`bench.luau`; one shake of 4/3 s, then rest; time per update):

| | rate | tip droop at rest | moving after the shake | native | interpreted |
| --- | --- | --- | --- | --- | --- |
| SmartBone rule | 60 Hz | 0.78 studs | 2.9 s | 0.5 us | 0.7 us |
| | 30 Hz | 1.47 studs | 4.4 s | 0.6 us | 0.7 us |
| | 15 Hz | 2.54 studs | 5.1 s | 0.6 us | 0.8 us |
| citrate implicit | 60, 30, 15 Hz | 0.78 studs | 2.7 s | 14 us | 66 us |
| citrate XPBD, 1/240 s substeps | 60 Hz | 0.79 studs | 2.6 s | 1.9 us | 14 us |
| | 30 Hz | 0.79 studs | 2.6 s | 3.7 us | 31 us |
| | 15 Hz | 0.79 studs | 2.6 s | 7.6 us | 56 us |
| citrate XPBD, 2 substeps per update | 60 Hz | 0.81 studs | 2.2 s | 1.1 us | 6.9 us |
| | 30 Hz | 0.88 studs | 1.8 s | 1.1 us | 7.4 us |
| | 15 Hz | 1.18 studs | 1.5 s | 1.3 us | 7.2 us |

So a SmartBone tail looks different at each distance from the camera, three times as
droopy at 37.5 studs as up close, while both of citrate's look the same at every rate.
XPBD gets there for a few times SmartBone's cost per second of motion (14 us per 60 Hz
update interpreted, as Roblox clients run scripts, against 0.7); its cost follows the
time simulated, so lowering the rate saves nothing unless the substeps per update are
fixed too, which brings back some dependence on the rate. The implicit solver takes any
step size stably, so its cost per update stays the same as the rate falls. In Studio a
row of three tails took about 0.003 ms per frame (SmartBone), 0.009 ms (XPBD) and 0.05
ms (implicit).

SmartBone 2 is Copyright (c) 2023 SmartBone contributors, MIT license; `tails.smartbone`
reproduces its update rule.

Files:

- `tails.luau`: the three models, usable outside Roblox.
- `client.client.luau`: the Roblox client script.
- `tails.project.json`: builds a playable place with `rojo build
  demos/tails/tails.project.json -o tails.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the table above (`luau -O2 [--codegen] demos/tails/bench.luau`).
