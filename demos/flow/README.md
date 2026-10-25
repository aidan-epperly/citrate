# Flow: a vortex street in real time

Incompressible 2D Navier-Stokes flow past an obstacle, solved every frame on a 192 by 96
grid. Behind a cylinder, above a Reynolds number of about 47, the wake becomes unstable
and sheds vortices alternately from each side: a Karman vortex street.

- Drag the obstacle with the mouse.
- Switch between a cylinder, a flat plate and an airfoil. The streamlined airfoil keeps a
  thin, steady wake.
- Show the dye, injected as streaks upstream, or the vorticity (red counterclockwise,
  blue clockwise).
- Change the Reynolds number (20 to 500). At 20 the wake stays steady, and from about 50
  it sheds.

## The solver (`flow.luau`)

A projection method on a periodic staggered (MAC) grid. Each step:

1. **Advection.** The velocity and the dye are advected by MacCormack's scheme (Selle et
   al. 2008): a semi-Lagrangian step back along the flow, one forward again, half the round
   trip's error corrected, and the result clamped to the values interpolated. Within two
   cells of the solid it falls back to the first-order step: there the flow nearly
   stagnates, and the correction, which is dispersive like Lax-Wendroff at small Courant
   numbers, left grid-scale wiggles in the dye.
2. **Inflow.** In the last tenth of the box (the fringe, not drawn), the flow relaxes to
   the uniform inflow and the dye to its streaks, so the periodic box behaves like a
   channel with inflow and outflow.
3. **Obstacle.** The velocity is zeroed on faces inside the solid (Brinkman penalization in
   the limit), which is first-order accurate at the boundary.
4. **Diffusion and projection, by one FFT pair.** u and v are real, so z = u + iv carries
   both through the library's `fft2`. In Fourier space, viscosity multiplies each mode by
   exp(nu lambda dt). That is exact for the 5-point Laplacian and stable for any time
   step. The projection solves L p = div u with the exact symbols of the staggered
   differences. The discrete divergence after each step is about 5e-14.

Advection matters because shedding depends on the effective viscosity. First-order
semi-Lagrangian advection adds a numerical viscosity of a(1 - a) h^2 / (2 dt), where a is
the fractional part of the number of cells moved per step. That shows in the shedding
(`bench.luau`; Strouhal number St = f D / U measured two diameters behind the cylinder;
for comparison, Williamson's fit for an unconfined cylinder is 0.166 at Re 100 and 0.192
at Re 200):

| advection | Re | shedding | ms per step |
| --- | --- | --- | --- |
| MacCormack | 30 | none | 7.2 |
| MacCormack | 100 | St 0.159, amplitude 0.43 U | 7.1 |
| MacCormack | 200 | St 0.173, amplitude 0.72 U | 7.0 |
| first order | 30 | none | 4.4 |
| first order | 100 | St 0.137, amplitude 0.12 U | 4.4 |

First-order advection makes Re 100 behave like Re of about 60.

Cost per step (desktop, native): advection about 5 ms (3 fields, 2 passes, about 40 ns
per point and pass, with the bilinear interpolation written out inline, since as a call
it cost three times as much) and diffusion with projection about 2 ms (1.6 ms of it the FFT
pair). In Studio the solver takes about 7.5 ms per frame and drawing about 0.1 ms.

## Files

- `flow.luau`: the solver, usable outside Roblox.
- `client.client.luau`: the Roblox client script (an EditableImage, dragging, buttons).
- `flow.project.json`: builds a playable place with `rojo build
  demos/flow/flow.project.json -o flow.rbxl` (open it in Roblox Studio and press Play).
- `bench.luau`: the table above (`luau -O2 --codegen demos/flow/bench.luau`).
