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
and 0.74 interpreted; a 6-bone tail's update by XPBD (`demos/tails`) 1.2 and 5.4 us; by
the implicit integrator 11 and 61 us.

## Against SciPy

`bench/scipy/run.py` times the same problems in citrate and in SciPy (1.18, NumPy 2.5,
OpenBLAS 0.3.34, on Python 3.12) on a 16-thread desktop CPU, and measures each call's
memory: the peak private memory of a process that builds the inputs and makes the call
once (or ten times), less that of one that only builds the inputs. SciPy's times are with
one BLAS thread, as citrate has; its first BLAS or LAPACK call in a process also allocates
about 32 MB of OpenBLAS buffers, once, which its memory figures include.

| problem | citrate, native | citrate, interpreted | SciPy | memory, citrate (1 / 10 calls) | memory, SciPy |
| --- | --- | --- | --- | --- | --- |
| 8 by 8 solve | 1.7 us | 8.7 us | 12 us | under 0.5 MB | 32 MB |
| a cubic spline at one point | 0.46 us | 0.82 us | 3.1 us | under 0.5 MB | under 0.5 MB |
| `matmul`, n = 256 | 5.2 ms | 40 ms | 0.30 ms | 0.4 / 5.1 MB | 32 MB |
| solve, n = 500 | 113 ms | 856 ms | 2.4 ms | 1.4 / 10.8 MB | 37 MB |
| `eigh` / `eig` / `svd`, n = 200 | 46 / 123 / 93 ms | 394 / 1150 / 733 ms | 2.1 / 9.0 / 3.0 ms | up to 0.7 / 5.3 MB | 32-34 MB |
| FFT, 65536 points | 1.5 ms | 17 ms | 0.73 ms | 3.8 MB | under 0.5 MB |
| banded Cholesky solve, n = 100,000, kd = 10 (`symbanded`) | 25 ms | 158 ms | 8.1 ms | 9.5 / 36 MB | 41 MB |
| sparse product, 1.25 million entries | 4.3 ms | 27 ms | 0.70 ms | under 0.5 MB | under 0.5 MB |
| 100 CG iterations, 200 by 200 grid | 82 ms | 656 ms | 12 ms | 1.0 / 5.4 MB | under 0.5 MB |
| Lorenz to t = 100, dp5 / RK45 (rtol 1e-8) | 56 ms | 98 ms | 259 ms | 4.1 / 8.5 MB | 4.3 MB |
| Lorenz, DOP853 | 43 ms | 75 ms | 137 ms | 0.8 / 4.3 MB | 1.7 MB |
| Van der Pol (mu = 1000), Radau | 11 ms | 22 ms | 96 ms | 0.1 MB | 65 MB |
| Van der Pol, BDF | 7.0 ms | 13 ms | 73 ms | 0.0 / 0.9 MB | 33 MB |
| AAA, 1000 points | 20 ms | 150 ms | 1.9 ms | 1.2 / 2.4 MB | 80 MB |
| reverse Cuthill-McKee, 200 by 200 grid numbered at random (same bandwidth: 200) | 7.6 ms | 25 ms | 1.6 ms | 0.7 / 4.4 MB | 0.2 MB |
| Dijkstra from one vertex, the same grid | 9.1 ms | 32 ms | 3.0 ms | 1.1 / 5.3 MB | 0.8 MB |
| k-d tree of 10,000 points: build / 8 nearest of each / neighbors within r of each (`cKDTree`) | 3.9 / 24 / 15 ms | 9.4 / 56 / 33 ms | 1.6 / 12 / 25 ms | up to 0.7 MB | up to 8 MB |
| sparse Cholesky, 5-point Laplacian 100 by 100: refactor / first call / solve (SciPy: SuperLU) | 7.9 / 20 / 1.5 ms | 58 / 111 / 10 ms | 7.9 / 8.0 / 0.22 ms | 0 / 3.7 MB | 70 MB |

Large dense problems are SciPy's: OpenBLAS's blocked vector kernels reach about 110
GFLOP/s on one thread where Luau, one floating-point operation at a time, reaches about 6
(`matmul`). Where memory traffic weighs more (FFT, banded, sparse, graphs) the gap shrinks
to 2 to 7 times. Many small calls are citrate's: the stiff and nonstiff solvers take the same
steps as SciPy's (identical counts for Radau and BDF) but SciPy spends about 20 us a step
in Python, so citrate is 3 to 9 times faster natively and still faster interpreted, as are
single small calls. Runtime footprint: 22 MB for Luau with citrate, 54 MB for Python with
NumPy and SciPy. Repeated calls (the second memory figure) hold garbage until Luau's
collector runs; factorizations given `reuse` and the integrators make none.

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
