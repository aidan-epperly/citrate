# Cloth: implicit versus explicit

Two 30 by 30 mass-spring cloths (9 kg; structural, shear and bending springs), pinned
along their top edge, each holding a 1000 kg steel bar along the bottom, and given the
same CPU time per frame:

- **Implicit**: one linearized backward Euler step per 60 Hz frame with
  `ode.implicit_mechanics` (the cloth's own spring-block operator for M - h D - h^2 K, block
  Jacobi preconditioned conjugate gradients), with stiff springs (1,000,000).
- **Explicit**: symplectic Euler, with as many substeps as fit in the implicit step's time
  (measured at startup), and therefore the stiffest springs stable at that many substeps
  (about 12 s^2 for s substeps; around 35,000 at 54 substeps). Under the bar's weight it
  stretches about a quarter of its length and bounces like rubber, while the implicit
  cloth holds its length.

Explicit methods are cheaper per frame on soft cloth (a few substeps suffice); the
implicit step wins once the springs are stiff relative to the time step, as here. The
explicit cloth hides itself and restarts if it ever comes apart.

Workspace attributes set the scenario: `Stiffness` (the implicit springs, default
1,000,000), `Bar` (kg, default 1000), `Wind` (true to start with wind) and `Record` (true
for a clean view: a fixed camera and only the readouts on screen; also set
`Players.CharacterAutoLoads` to false to leave out the avatar).

Files:

- `cloth.luau`: the cloth model (forces, sparse Jacobians, the spring-block system for
  the implicit step, and the explicit step), usable outside Roblox.
- `client.client.luau`: the Roblox client script (calibration, EditableMesh rendering,
  wind and reset buttons).
- `cloth.project.json`: builds a playable place with `rojo build
  demos/cloth/cloth.project.json -o cloth.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: timings outside Roblox (`luau -O2 --codegen demos/cloth/bench.luau`).

Measured (30 by 30, native code): the implicit step costs about 3 to 6 ms per frame at a
CG tolerance of 1e-2 and 5 to 12 ms at 1e-3 (the demo's setting), rising with stiffness as
CG needs more iterations; in a Roblox Studio playtest, about 5 ms at stiffness 2000 and 12
ms at 18000.
