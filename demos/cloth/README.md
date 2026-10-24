# Cloth: implicit versus explicit

Two 30 by 30 mass-spring cloths (9 kg; structural, shear and bending springs), pinned
along their top edge, each holding a 1000 kg steel bar along the bottom, and given the
same CPU time per frame:

- **Implicit**: one linearized backward Euler step per 60 Hz frame with
  `ode.implicit_mechanics` (the cloth's own spring-block operator for M - h D - h^2 K and
  preconditioned conjugate gradients), with stiff springs (1,000,000).
- **Explicit**: symplectic Euler, with as many substeps as fit in the implicit step's time
  (measured at startup), and therefore the stiffest springs stable at that many substeps
  (measured limit 26 to 30 s^2 for s substeps; the demo uses 22 s^2, about 66,000 at 55
  substeps). Under the bar's weight it stretches by 10 to 20 percent and bounces, while
  the implicit cloth holds its length.

With `Match = "stiffness"` both cloths get the same springs and the explicit one the
substeps they need: at 66,550 both stretch to about 1.11 m, the implicit step costs less
(about 2.4 ms against 6.6 in Studio), and the remaining difference is damping: backward Euler damps the
bounce away (numerically, more than a real cloth would), while the explicit cloth keeps
bouncing.

Explicit methods are cheaper per frame on soft cloth (a few substeps suffice); the
implicit step wins once the springs are stiff relative to the time step, as here. The
explicit cloth hides itself and restarts if it ever comes apart.

The implicit step is the library's: `ode.implicit_mechanics` with the cloth's 3 by 3 block
sparse Jacobians (`block_jacobian`), preconditioned by the library's algebraic multigrid
(`amg = { smooth = false }`: plain aggregation, refilled every frame) or block Jacobi.
One-level preconditioners need more iterations as the cloth grows; multigrid keeps them
nearly flat (stiffness 1e6, moving in the wind, desktop):

| cloth | block Jacobi | multigrid |
| --- | --- | --- |
| 30 by 30 | 3.6 ms, 31 iterations | 3.0 ms, 5 iterations |
| 60 by 60 | 22.8 ms, 60 iterations | 13.9 ms, 8 iterations |

With few iterations block Jacobi can be faster (softer springs, or a settled cloth), so the
demo times both at startup and keeps the faster. An earlier version had its own multigrid
written against the springs; the library's is within about 10% of it at 30 by 30 and
faster at 60 by 60.

Workspace attributes set the scenario: `Stiffness` (the implicit springs, default
1,000,000), `Match` (`"time"`, the default, or `"stiffness"`), `Preconditioner`
(`"auto"`, the default, `"multigrid"` or `"jacobi"`), `Bar` (kg, default 1000), `Wind` (true to start with wind) and `Record` (true
for a clean view: a fixed camera and only the readouts on screen; also set
`Players.CharacterAutoLoads` to false to leave out the avatar).

Files:

- `cloth.luau`: the cloth model (forces, sparse and 3 by 3 block sparse Jacobians, and the
  explicit step), usable outside Roblox.
- `client.client.luau`: the Roblox client script (calibration, EditableMesh rendering,
  wind and reset buttons).
- `cloth.project.json`: builds a playable place with `rojo build
  demos/cloth/cloth.project.json -o cloth.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: timings outside Roblox (`luau -O2 --codegen demos/cloth/bench.luau`).

In a Roblox Studio playtest (30 by 30, equal springs of stiffness 66,550, multigrid) the
implicit step took 2.6 ms per frame and the explicit cloth's 55 substeps 5.5 ms.
