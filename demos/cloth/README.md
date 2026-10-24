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

Preconditioners (`multigrid.luau`, `cloth.luau`): block Jacobi, block symmetric
Gauss-Seidel, and an aggregation multigrid V-cycle for spring networks (greedy aggregation
along the structural springs, Galerkin coarse levels built from summed spring blocks,
damped block Jacobi smoothing, a dense Cholesky solve at the coarsest level). One-level
preconditioners need twice the iterations when the cloth doubles in size; multigrid keeps
them nearly flat:

| cloth, stiffness 1e6 (while moving) | block Jacobi | multigrid |
| --- | --- | --- |
| 30 by 30 | 7.2 ms, 59 iterations | 4.9 ms, 12 iterations |
| 60 by 60 | 54 ms, 120 iterations | 29 ms, 20 iterations |

Settled at stiffness 1e6 on 30 by 30, block Jacobi is a little faster (its iterations are
cheap and few); at 66,550 multigrid is (2.4 against 3.9 ms in Studio). The demo times
both at startup and keeps the faster.

Workspace attributes set the scenario: `Stiffness` (the implicit springs, default
1,000,000), `Match` (`"time"`, the default, or `"stiffness"`), `Preconditioner`
(`"auto"`, the default, `"multigrid"`, `"sgs"` or `"jacobi"`), `Bar` (kg, default 1000), `Wind` (true to start with wind) and `Record` (true
for a clean view: a fixed camera and only the readouts on screen; also set
`Players.CharacterAutoLoads` to false to leave out the avatar).

Files:

- `cloth.luau`: the cloth model (forces, sparse Jacobians, the spring-block system and
  preconditioners for the implicit step, and the explicit step), usable outside Roblox.
- `multigrid.luau`: the aggregation multigrid preconditioner for spring networks.
- `client.client.luau`: the Roblox client script (calibration, EditableMesh rendering,
  wind and reset buttons).
- `cloth.project.json`: builds a playable place with `rojo build
  demos/cloth/cloth.project.json -o cloth.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: timings outside Roblox (`luau -O2 --codegen demos/cloth/bench.luau`).

Measured (30 by 30, native code): the implicit step costs about 3 to 6 ms per frame at a
CG tolerance of 1e-2 and 5 to 12 ms at 1e-3 (the demo's setting), rising with stiffness as
CG needs more iterations; in a Roblox Studio playtest, about 5 ms at stiffness 2000 and 12
ms at 18000.
