"""The SciPy side of the comparison (bench/scipy/run.py runs it):
    python cases.py <case> time      -- prints seconds per call
    python cases.py <case> mem <r>   -- the inputs, r calls (r = 0: the inputs only)
The inputs are the same as cases.luau's: the same generator, sizes and problems."""
import sys
import time

import numpy as np
import scipy.fft
import scipy.integrate
import scipy.interpolate
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

seed = 12345


def rnd():
    """Park-Miller, as in cases.luau; values in [-0.5, 0.5)."""
    global seed
    seed = (seed * 16807) % 2147483647
    return seed / 2147483647 - 0.5


def random_matrix(n):
    # column-major fill, as citrate's
    # (fromiter: no temporary list, which would leave freed memory that masks a call's)
    return np.fromiter((rnd() for _ in range(n * n)), float, n * n).reshape((n, n), order="F")


def random_vector(n):
    return np.fromiter((rnd() for _ in range(n)), float, n)


def laplacian(m):
    T = scipy.sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(m, m))
    I = scipy.sparse.identity(m)
    E = scipy.sparse.diags([-1, -1], [-1, 1], shape=(m, m))
    return (scipy.sparse.kron(I, T) + scipy.sparse.kron(E, I)).tocsr()


setups = {}
STATS = [""]


def case(name):
    def register(f):
        setups[name] = f
        return f
    return register


@case("matmul_256")
def _():
    A, B = random_matrix(256), random_matrix(256)
    return lambda: A @ B


for n in (8, 120, 500):
    def make(n=n):
        A, b = random_matrix(n), random_vector(n)
        return lambda: scipy.linalg.solve(A, b)
    setups[f"solve_{n}"] = make


@case("eigh_200")
def _():
    A = random_matrix(200)
    H = A + A.T
    return lambda: scipy.linalg.eigh(H)


@case("eig_200")
def _():
    A = random_matrix(200)
    return lambda: scipy.linalg.eig(A)


@case("svd_200")
def _():
    A = random_matrix(200)
    return lambda: scipy.linalg.svd(A)


for n in (4096, 65536):
    def make(n=n):
        v = random_vector(2 * n)
        x = v[0::2] + 1j * v[1::2]
        return lambda: scipy.fft.fft(x)
    setups[f"fft_{n}"] = make


@case("banded_chol_100000")
def _():
    n, kd = 100000, 10
    ab = np.zeros((kd + 1, n))  # lower form: ab[i - j, j] = A[i, j]
    for j in range(n):
        for i in range(j, min(n, j + kd + 1)):
            ab[i - j, j] = 2 * kd + 1 if i == j else rnd()
    b = random_vector(n)
    return lambda: scipy.linalg.solveh_banded(ab, b, lower=True)


@case("sparse_matvec_500")
def _():
    S = laplacian(500)
    x = random_vector(500 * 500)
    return lambda: S @ x


@case("cg_200")
def _():
    S = laplacian(200)
    b = random_vector(200 * 200)
    count = [0]

    def run():
        count[0] = 0

        def cb(_):
            count[0] += 1
        scipy.sparse.linalg.cg(S, b, rtol=1e-30, atol=0, maxiter=100, callback=cb)
        assert count[0] == 100
    return run


def lorenz(t, y):
    return [10 * (y[1] - y[0]), y[0] * (28 - y[2]) - y[1], y[0] * y[1] - 8 / 3 * y[2]]


def vdp(t, y):
    return [y[1], 1000 * ((1 - y[0] * y[0]) * y[1]) - y[0]]


for method, ours, problem in (("RK45", "dp5", "lorenz"), ("DOP853", "dop853", "lorenz"), ("Radau", "radau", "vdp"), ("BDF", "bdf", "vdp")):
    def make(method=method, problem=problem):
        f = lorenz if problem == "lorenz" else vdp
        y0 = [1.0, 1.0, 1.0] if problem == "lorenz" else [2.0, 0.0]
        tspan = (0, 100) if problem == "lorenz" else (0, 3000)
        rtol, atol = (1e-8, 1e-10) if problem == "lorenz" else (1e-6, 1e-6)

        def run():
            sol = scipy.integrate.solve_ivp(f, tspan, y0, method=method, rtol=rtol, atol=atol)
            assert sol.success, sol.message
            STATS[0] = f"steps {len(sol.t) - 1} nfev {sol.nfev}"
        return run
    setups[f"ode_{ours}_{problem}"] = make


def spline_points():
    n = 1000
    xs = np.array([(i - 1) / (n - 1) * 10 for i in range(1, n + 1)])
    ys = np.array([np.sin(x) + 0.1 * rnd() for x in xs])
    return xs, ys


@case("spline_build_1000")
def _():
    xs, ys = spline_points()
    return lambda: scipy.interpolate.CubicSpline(xs, ys)  # not-a-knot ends by default


@case("spline_eval_100000")
def _():
    xs, ys = spline_points()
    C = scipy.interpolate.CubicSpline(xs, ys)
    t = np.arange(100000) / 99999 * 10
    return lambda: C(t)


@case("spline_eval_one")
def _():
    xs, ys = spline_points()
    C = scipy.interpolate.CubicSpline(xs, ys)
    return lambda: C(3.7)


@case("aaa_1000")
def _():
    x = -1 + 2 * np.arange(1000) / 999
    f = np.exp(x) / (1.05 - x * x) + np.tanh(5 * x)
    return lambda: scipy.interpolate.AAA(x, f, rtol=1e-13, max_terms=100)


@case("none")
def _():
    return lambda: None


def main():
    name, mode = sys.argv[1], sys.argv[2]
    op = setups[name]()
    if mode == "time":
        op()  # warm up
        best = float("inf")
        for _ in range(5):
            n, t0 = 0, time.perf_counter()
            while True:
                op()
                n += 1
                if time.perf_counter() - t0 > 0.2:
                    break
            best = min(best, (time.perf_counter() - t0) / n)
        print(f"{best:.9g} {STATS[0]}")
    else:
        for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 1):
            op()


if __name__ == "__main__":
    main()
