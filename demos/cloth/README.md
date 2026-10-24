# Cloth: implicit versus explicit

Two 30 by 30 mass-spring cloths (structural, shear and bending springs, pinned at two
corners) side by side at the same spring stiffness:

- **Implicit**: one linearized backward Euler step per 60 Hz frame with
  `ode.implicit_mechanics`, using the cloth's own spring-block operator for
  M - h D - h^2 K and a block-Jacobi preconditioned conjugate gradient solve.
- **Explicit**: symplectic Euler with 10 substeps per frame, as games usually step cloth.

Press **Stiffer**: around a stiffness of 6000 the explicit cloth explodes (it needs more
substeps for each increase), while the implicit one stays stable at any stiffness for a
fixed cost per frame.

Files:

- `cloth.luau`: the cloth model (forces, sparse Jacobians, the spring-block system for
  the implicit step, and the explicit step), usable outside Roblox.
- `client.client.luau`: the Roblox client script (EditableMesh rendering, controls).
- `cloth.project.json`: builds a playable place with `rojo build
  demos/cloth/cloth.project.json -o cloth.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: timings outside Roblox (`luau -O2 --codegen demos/cloth/bench.luau`).

Measured (30 by 30, native code): the implicit step costs about 3 to 6 ms per frame at a
CG tolerance of 1e-2 and 5 to 12 ms at 1e-3 (the demo's setting), rising with stiffness as
CG needs more iterations; in a Roblox Studio playtest, about 5 ms at stiffness 2000 and 12
ms at 18000.
