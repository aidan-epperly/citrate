# Performance

Luau compiled to native code (`--!native`, which every citrate module uses) is limited
here mostly by instructions, not by memory bandwidth: measured on a desktop CPU, and the
same in Roblox Studio, a loop iteration costs about 1.6 ns, a buffer read about 0.25 ns and
a Lua table lookup about 1 ns. A kernel that handles E entries per iteration with R reads
costs roughly (1.6 + 0.25 R) / E ns per entry, so short loop bodies are dominated by the
loop itself. The kernels are designed around that:

- **Unrolling.** The elementwise kernels (`axpy`, `dot`, ...) handle 4 entries per
  iteration, which made them 3-4 times faster. Sparse products are unrolled by 4.
- **Register blocking.** Matrix products keep an 8 by 16 block of sums (8 by 8 for
  complex) in locals, so each value read feeds many multiply-adds; this made `matmul` about
  9 times faster at n = 256 than a plain triple loop. Luau allows at most 200 locals per
  function, which bounds the block size.
- **Blocks for sparse matrices.** With 3 by 3 blocks one index serves nine entries and each
  entry of x is read once per block: 0.54 ns per entry against 1.5 for the unrolled scalar
  format and 4.2 before either change (100,000 entries of a cloth Jacobian).
- **Fused operations.** Conjugate gradients updates x and r and computes |r|^2 in one pass
  (`Space.cg_update`) instead of three.
- **Inlining.** Luau inlines small local functions within a module but not across modules:
  a one-line helper called from another module cost 14 ns against 2.3 ns, while calls that
  do a whole operation (an `axpy` over thousands of entries) cost the same either way. The
  array modules are split into parts that call each other only for whole operations; small
  helpers are copied into each part. Interpreted, even an inlined helper costs something
  (the copies of its arguments and results: 25% in an LU update loop, 65% with complex
  values), so the templates' element-access helpers are expanded by the generator instead
  (`--@inline`; see [development](development.md)).
- **Offsets per column, typed.** Banded routines compute each column's base offset once
  and add the row's (interpreted, the full index arithmetic per entry was most of the inner
  loop: the factorizations got 10-45% faster). Natively that only pays if the base is
  annotated `number`: computed from table fields (`B.ld`), it is untyped to the native
  compiler, which then splits the inner loop around a generic fallback and was 5-30%
  slower than recomputing the index.

Measured on a desktop CPU (`bench/table.luau`), native as on Roblox servers and in Studio,
and interpreted as on players' devices:

| operation | native | interpreted |
| --- | --- | --- |
| `matmul`, f64, n = 256 | 5.4 ms | 41 ms |
| `matmul`, c64, n = 256 | 18 ms | 170 ms |
| LU solve, n = 120 | 1.7 ms | 12 ms |
| `eigh`, n = 120 (with vectors) | 10.5 ms | 89 ms |
| `eig`, n = 120 (with vectors) | 35 ms | 270 ms |
| `svd`, n = 120 (with vectors) | 21 ms | 170 ms |
| `fft`, 4096 points | 0.073 ms | 0.81 ms |
| `fft2`, 128 by 128 | 0.57 ms | 5.2 ms |
| CG iteration, 200x200 Laplacian (matrix-free) | 0.51 ms | 4.2 ms |

Interpreted code is 7 to 11 times slower here. Small per-frame work, where a game spends
its client time, is listed with each demo: a curve point (`demos/track`) 0.32 us native
and 0.74 interpreted; a 6-bone tail's update by XPBD (`demos/tails`) 1.2 and 5.7 us; by
the implicit integrator 11 and 61 us.

## In Roblox

- Roblox compiles a module to native code only if it starts with `--!native` (the
  command-line `luau --codegen` compiles everything, so it cannot show the difference): in
  Studio, evaluating a curve by arclength took 0.33 us natively and 0.77 us without it.
- Roblox compiles `--!native` scripts to native code only on servers and in Studio; on
  players' devices, client scripts are interpreted. Interpreted code is about 7 times
  slower here: an LU update loop costs 18 ns per entry against 2.5, and a step of the flow
  demo 48 ms against 7.0. Timings in Studio are those of native code.
- Roblox servers flush subnormal numbers to zero, and clients flush single-precision ones,
  so results at subnormal magnitudes differ there.
- Studio slows a script down after it has run for a long time without yielding (a frame of
  the cloth demo went from 6.7 to 81 ms after a 25 s busy loop); run long benchmarks in
  several calls.
- In Roblox, `typeof(v)` of a vector is `"Vector3"`; citrate tests `type(v) == "vector"`.
