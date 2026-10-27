"""Runs the citrate/SciPy comparison: time per call and memory, case by case.

    python bench/scipy/run.py [--luau PATH] [cases...]

Time: each side times its call in-process (best of 5 batches of at least 0.2 s, after a
warm-up call): citrate natively (luau --codegen, as Roblox servers and Studio run it) and
interpreted (as players' devices do); SciPy with one BLAS thread (as citrate has) and with
its default thread count.

Memory: the peak private memory (Windows' PeakPagefileUsage) of a process that builds the
inputs and makes the call once (and ten times), minus that of one that only builds the
inputs, the median of three runs each. Both sides build their inputs without large
temporaries (freed memory would be reused by the call and hide it). On SciPy's side the
first BLAS or LAPACK call also allocates OpenBLAS's buffers, once per process: see
solve_8. It includes everything the call allocates,
temporaries too; citrate's garbage counts until Luau's collector frees it. Results go to
bench/scipy/results.json and a Markdown table on stdout.
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = [
    "solve_8", "spline_eval_one",
    "matmul_256", "solve_120", "solve_500", "eigh_200", "eig_200", "svd_200",
    "fft_4096", "fft_65536", "banded_chol_100000", "sparse_matvec_500", "cg_200",
    "ode_dp5_lorenz", "ode_dop853_lorenz", "ode_radau_vdp", "ode_bdf_vdp",
    "spline_build_1000", "spline_eval_100000", "aaa_1000",
]
BLAS = {"matmul_256", "solve_120", "solve_500", "eigh_200", "eig_200", "svd_200", "banded_chol_100000"}


class Counters(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_private(cmd, env=None):
    """Runs cmd to completion; its peak private memory in bytes."""
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = p.communicate()
    assert p.returncode == 0, err.decode(errors="ignore")
    c = Counters()
    c.cb = ctypes.sizeof(c)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(wt.HANDLE(int(p._handle)), ctypes.byref(c), c.cb)
    assert ok
    return c.PeakPagefileUsage


def timed(cmd, env=None):
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout.split()
    return float(out[0]), " ".join(out[1:])


def main():
    args = sys.argv[1:]
    luau = "luau"
    if args[:1] == ["--luau"]:
        luau, args = args[1], args[2:]
    cases = args or CASES
    lua = lambda native, *a: [luau, "-O2"] + (["--codegen"] if native else []) + [str(HERE / "cases.luau"), "-a", *a]
    py = lambda *a: [sys.executable, str(HERE / "cases.py"), *a]
    one = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    results = {}
    results["footprint"] = {
        "citrate_mb": statistics.median(peak_private(lua(True, "none", "mem", "0")) for _ in range(3)) / 2**20,
        "scipy_mb": statistics.median(peak_private(py("none", "mem", "0"), one) for _ in range(3)) / 2**20,
    }
    print("footprint", json.dumps(results["footprint"]), file=sys.stderr, flush=True)
    for name in cases:
        r = {}
        r["citrate_native"], r["citrate_stats"] = timed(lua(True, name, "time"))
        r["citrate_interpreted"], _ = timed(lua(False, name, "time"))
        r["scipy_1thread"], r["scipy_stats"] = timed(py(name, "time"), one)
        if name in BLAS:
            r["scipy_threads"], _ = timed(py(name, "time"))
        for side, cmd, env in (("citrate", lambda *a: lua(True, *a), None), ("scipy", py, one)):
            base = statistics.median(peak_private(cmd(name, "mem", "0"), env) for _ in range(3))
            r[f"{side}_base_mb"] = base / 2**20
            for reps in (1, 10):
                peak = statistics.median(peak_private(cmd(name, "mem", str(reps)), env) for _ in range(3))
                r[f"{side}_mem{reps}_mb"] = max(peak - base, 0) / 2**20
        results[name] = r
        print(name, json.dumps(r), file=sys.stderr, flush=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=1))

    def t(s):
        return f"{s * 1e6:.3g} us" if s < 1e-3 else f"{s * 1e3:.3g} ms"

    print("| case | citrate native | citrate interpreted | SciPy, 1 thread | SciPy, all threads | native / SciPy 1 thread | memory, citrate | memory, SciPy |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        if name == "footprint":
            continue
        ratio = r["citrate_native"] / r["scipy_1thread"]
        print(f"| {name} | {t(r['citrate_native'])} | {t(r['citrate_interpreted'])} | {t(r['scipy_1thread'])} | "
              f"{t(r['scipy_threads']) if 'scipy_threads' in r else ''} | {ratio:.3g} | "
              f"{r['citrate_mem1_mb']:.2f} / {r['citrate_mem10_mb']:.2f} MB | {r['scipy_mem1_mb']:.2f} / {r['scipy_mem10_mb']:.2f} MB |")


if __name__ == "__main__":
    main()
