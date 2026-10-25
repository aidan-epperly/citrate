# Laplace problems on polygons

Reference: [lightning](reference/lightning.md).

`lightning.laplace` solves Laplace's equation, or Poisson's with a constant source, on a
polygon with Dirichlet or Neumann data on each side, by Gopal and Trefethen's lightning
method. The solution is a sum of harmonic functions (the real and imaginary parts of a
polynomial and of simple poles d / (z - p)) fitted to the boundary data by weighted least
squares. Each term solves the equation exactly, so only the boundary is sampled, and the
poles cluster exponentially close to each vertex, outside the polygon. That is where
solutions are singular: at reentrant corners, and where the boundary condition changes
(like r^(1/2) when a Dirichlet side meets a Neumann side in a straight line). The error
decreases like exp(-C sqrt(N)) in the number of unknowns N, where grids and finite elements
converge only algebraically.

A condition that changes partway along a side is given by putting a vertex there.

## Example: a membrane hung from a rod

A unit square of cloth under tension, pinned along the middle half of its top edge and free
elsewhere, sags under its own weight; for small sag the deflection u (in units of the
weight over the tension) solves Laplacian(u) = -1 with u = 0 on the rod and du/dn = 0 on the
free edges:

```luau
local lightning = citrate.lightning
local function zero() return 0 end
local free = { kind = "neumann", value = zero }
local S = lightning.laplace(
	{ { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0.75, 1 }, { 0.25, 1 }, { 0, 1 } }, -- counterclockwise
	{ free, free, free, { kind = "dirichlet", value = zero }, free, free },
	{ poles = 24, source = -1 }
)
print(S.eval(0.5, 0))   --> 0.6097241866 (the sag at the middle of the bottom edge;
                        --    0.6097242064 to 10 digits, from more poles)
print(S.error)          --> 1.8e-8 (the weighted boundary residual)
```

The ends of the rod are r^(1/2) singularities. Against a finite-volume grid solved by CG
with multigrid (`bench/lightning.luau`):

| method | unknowns | error in the sag | time |
| --- | --- | --- | --- |
| lightning, 8 poles per vertex | 113 | 1.2e-4 | 15 ms |
| lightning, 16 poles per vertex | 225 | 6.8e-7 | 96 ms |
| lightning, 32 poles per vertex | 449 | 8.2e-10 | 730 ms |
| finite volumes, 64 by 64 | 4,192 | 5.4e-3 | 41 ms |
| finite volumes, 256 by 256 | 65,920 | 1.4e-3 | 670 ms |

The grid's error halves with each refinement (first order, because of the singularity), so
it would need about 10^8 points per side to match the lightning solver at 450 unknowns.

## Compression

The solution is u = Re f (plus the particular solution for a source), with f analytic: the
polynomial and poles of the fit. `S.compress(tol?)` runs [AAA](rational.md) on f's values
at the boundary points and returns a shorter rational representation of the same solution,
`{ eval, rational, terms, error }`; by the maximum modulus principle its error inside is
no larger than on the boundary. AAA places poles only where f needs them, where the fit put
a fixed cluster at every vertex. On the hanging membrane:

| poles per vertex | fit's complex terms | after AAA | max difference | evaluation |
| --- | --- | --- | --- | --- |
| 16 | 113 | 47 | 1.4e-7 | 3.2 to 0.7 us |
| 24 | 169 | 60 | 3.2e-8 | 5.4 to 0.9 us |
| 32 | 225 | 75 | 3.4e-9 | 8.0 to 1.0 us |

Compressing costs more than the fit (0.5 s, 1.8 s and 5 s here): AAA computes an SVD at
every step. Only Re f is fitted, so f's imaginary part carries an arbitrary constant (13065
in this example); compress removes it and measures its tolerance against Re f.

## Notes

- The least-squares rows are weighted by sqrt(r), and Neumann rows by r sqrt(r), with r
  the distance to the nearest vertex. The power 1/2 matches how strongly the solution
  depends on boundary values near a change of condition; with Gopal and Trefethen's weight
  r (made for Dirichlet problems) the example above stalled near 1e-4.
- `S.error` is the boundary residual in that weighted norm, relative to the data, on twice
  as many points as the fit used; in the example it tracks the error of the solution.
- Options: `poles` per vertex (default 20), `sigma` (their clustering, default 4),
  `degree` of the polynomial part (default the number of poles), `source`.
