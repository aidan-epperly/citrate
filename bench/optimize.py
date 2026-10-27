"""SciPy's counts on the problems of bench/optimize.luau (python bench/optimize.py):
brentq and minimize_scalar (bounded) on the scalar functions; least_squares ('lm'),
minimize (L-BFGS-B, Nelder-Mead) and root ('hybr') on the Moré-Garbow-Hillstrom problems of
tests/mgh.luau (the same definitions), with exact Jacobians and gradients, and L-BFGS-B with
finite differences by size. Prints Markdown tables."""
import math

import numpy as np

np.seterr(all="ignore")
from scipy import optimize


def solved(v, fstar):
    return v <= fstar + 1e-5 * max(1e-3, fstar) + 1e-9


# scalar functions (as in bench/optimize.luau)
ROOTS = [
    ("cos x - x on [0, 1]", lambda x: math.cos(x) - x, 0, 1),
    ("x^3 - 2x - 5 on [2, 3]", lambda x: x ** 3 - 2 * x - 5, 2, 3),
    ("pole nearby: 1/(x - 1.01) + 1.5 on [-1, 0.9]", lambda x: 1 / (x - 1.01) + 1.5, -1, 0.9),
    ("kink: |x - 0.3| - 0.1 on [0, 0.25]", lambda x: abs(x - 0.3) - 0.1, 0, 0.25),
    ("front: tanh(50 (x - 0.2)) on [-1, 1]", lambda x: math.tanh(50 * (x - 0.2)), -1, 1),
    ("exp(x) - 1000 on [0, 10]", lambda x: np.exp(x) - 1000, 0, 10),
]
MINS = [
    ("(x - 0.3)^2 on [-1, 1]", lambda x: (x - 0.3) ** 2, -1, 1),
    ("x^4 - x^2 + 0.3 x on [-1.5, 1.5] (two minima)", lambda x: x ** 4 - x ** 2 + 0.3 * x, -1.5, 1.5),
    ("x^4 - x^2 - 0.3 x on [-1.5, 1.5] (global on the right)", lambda x: x ** 4 - x ** 2 - 0.3 * x, -1.5, 1.5),
    ("sin(5 x) + 0.5 x on [0, 3]", lambda x: math.sin(5 * x) + 0.5 * x, 0, 3),
]

S5, S10, S90 = math.sqrt(5), math.sqrt(10), math.sqrt(90)
BARD_Y = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]
KO_Y = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
KO_U = [4, 2, 1, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]


def theta(x1, x2):
    t = math.atan(x2 / x1) / (2 * math.pi)
    return t + 0.5 if x1 < 0 else t


def helical_jac(x):
    q = x[0] ** 2 + x[1] ** 2
    s = math.sqrt(q)
    return [[100 * x[1] / (2 * math.pi * q), -100 * x[0] / (2 * math.pi * q), 10], [10 * x[0] / s, 10 * x[1] / s, 0], [0, 0, 1]]


def bard_r(x):
    return [BARD_Y[i - 1] - (x[0] + i / ((16 - i) * x[1] + min(i, 16 - i) * x[2])) for i in range(1, 16)]


def bard_j(x):
    out = []
    for i in range(1, 16):
        u, v, w = i, 16 - i, min(i, 16 - i)
        d = (v * x[1] + w * x[2]) ** 2
        out.append([-1, u * v / d, u * w / d])
    return out


def ko_r(x):
    return [KO_Y[i] - x[0] * (KO_U[i] ** 2 + KO_U[i] * x[1]) / (KO_U[i] ** 2 + KO_U[i] * x[2] + x[3]) for i in range(11)]


def ko_j(x):
    out = []
    for i in range(11):
        u = KO_U[i]
        num, den = u * u + u * x[1], u * u + u * x[2] + x[3]
        out.append([-num / den, -x[0] * u / den, x[0] * num * u / den ** 2, x[0] * num / den ** 2])
    return out


def ext_rosenbrock(n):
    def r(x):
        out = []
        for i in range(0, n, 2):
            out += [10 * (x[i + 1] - x[i] ** 2), 1 - x[i]]
        return out

    def j(x):
        J = np.zeros((n, n))
        for i in range(0, n, 2):
            J[i, i], J[i, i + 1], J[i + 1, i] = -20 * x[i], 10, -1
        return J
    return (f"Extended Rosenbrock {n}", n, n, [-1.2, 1] * (n // 2), 0, True, r, j)


def trig(n):
    def r(x):
        s = sum(math.cos(v) for v in x)
        return [n - s + (i + 1) * (1 - math.cos(x[i])) - math.sin(x[i]) for i in range(n)]

    def j(x):
        J = np.tile(np.sin(x), (n, 1))
        for i in range(n):
            J[i, i] += (i + 1) * math.sin(x[i]) - math.cos(x[i])
        return J
    return (f"Trigonometric {n}", n, n, [1 / n] * n, 2.79506e-5 / 2, False, r, j)


def dbv(n):
    h = 1 / (n + 1)
    x0 = [(i * h) * (i * h - 1) for i in range(1, n + 1)]

    def r(x):
        out = []
        for i in range(n):
            t = (i + 1) * h
            out.append(2 * x[i] - (x[i - 1] if i > 0 else 0) - (x[i + 1] if i < n - 1 else 0) + h * h * (x[i] + t + 1) ** 3 / 2)
        return out

    def j(x):
        J = np.zeros((n, n))
        for i in range(n):
            t = (i + 1) * h
            J[i, i] = 2 + 1.5 * h * h * (x[i] + t + 1) ** 2
            if i > 0:
                J[i, i - 1] = -1
            if i < n - 1:
                J[i, i + 1] = -1
        return J
    return (f"Discrete boundary value {n}", n, n, x0, 0, True, r, j)


def broyden(n):
    def r(x):
        return [(3 - 2 * x[i]) * x[i] - (x[i - 1] if i > 0 else 0) - 2 * (x[i + 1] if i < n - 1 else 0) + 1 for i in range(n)]

    def j(x):
        J = np.zeros((n, n))
        for i in range(n):
            J[i, i] = 3 - 4 * x[i]
            if i > 0:
                J[i, i - 1] = -1
            if i < n - 1:
                J[i, i + 1] = -2
        return J
    return (f"Broyden tridiagonal {n}", n, n, [-1] * n, 0, True, r, j)


PROBLEMS = [
    ("Rosenbrock", 2, 2, [-1.2, 1], 0, True, lambda x: [10 * (x[1] - x[0] ** 2), 1 - x[0]], lambda x: [[-20 * x[0], 10], [-1, 0]]),
    ("Freudenstein-Roth", 2, 2, [0.5, -2], 24.4921, False,
     lambda x: [-13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1], -29 + x[0] + ((x[1] + 1) * x[1] - 14) * x[1]],
     lambda x: [[1, 10 * x[1] - 3 * x[1] ** 2 - 2], [1, 3 * x[1] ** 2 + 2 * x[1] - 14]]),
    ("Powell badly scaled", 2, 2, [0, 1], 0, True,
     lambda x: [1e4 * x[0] * x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - 1.0001],
     lambda x: [[1e4 * x[1], 1e4 * x[0]], [-np.exp(-x[0]), -np.exp(-x[1])]]),
    ("Brown badly scaled", 2, 3, [1, 1], 0, False, lambda x: [x[0] - 1e6, x[1] - 2e-6, x[0] * x[1] - 2], lambda x: [[1, 0], [0, 1], [x[1], x[0]]]),
    ("Beale", 2, 3, [1, 1], 0, False,
     lambda x: [[1.5, 2.25, 2.625][i - 1] - x[0] * (1 - x[1] ** i) for i in (1, 2, 3)],
     lambda x: [[-(1 - x[1] ** i), x[0] * i * x[1] ** (i - 1)] for i in (1, 2, 3)]),
    ("Jennrich-Sampson", 2, 10, [0.3, 0.4], 124.362 / 2, False,
     lambda x: [2 + 2 * i - (np.exp(i * x[0]) + np.exp(i * x[1])) for i in range(1, 11)],
     lambda x: [[-i * np.exp(i * x[0]), -i * np.exp(i * x[1])] for i in range(1, 11)]),
    ("Helical valley", 3, 3, [-1, 0, 0], 0, True,
     lambda x: [10 * (x[2] - 10 * theta(x[0], x[1])), 10 * (math.sqrt(x[0] ** 2 + x[1] ** 2) - 1), x[2]], helical_jac),
    ("Bard", 3, 15, [1, 1, 1], 8.21487e-3 / 2, False, bard_r, bard_j),
    ("Box 3D", 3, 10, [0, 10, 20], 0, False,
     lambda x: [np.exp(-0.1 * i * x[0]) - np.exp(-0.1 * i * x[1]) - x[2] * (np.exp(-0.1 * i) - np.exp(-i)) for i in range(1, 11)],
     lambda x: [[-0.1 * i * np.exp(-0.1 * i * x[0]), 0.1 * i * np.exp(-0.1 * i * x[1]), -(np.exp(-0.1 * i) - np.exp(-i))] for i in range(1, 11)]),
    ("Powell singular", 4, 4, [3, -1, 0, 1], 0, True,
     lambda x: [x[0] + 10 * x[1], S5 * (x[2] - x[3]), (x[1] - 2 * x[2]) ** 2, S10 * (x[0] - x[3]) ** 2],
     lambda x: [[1, 10, 0, 0], [0, 0, S5, -S5], [0, 2 * (x[1] - 2 * x[2]), -4 * (x[1] - 2 * x[2]), 0],
                [2 * S10 * (x[0] - x[3]), 0, 0, -2 * S10 * (x[0] - x[3])]]),
    ("Wood", 4, 6, [-3, -1, -3, -1], 0, False,
     lambda x: [10 * (x[1] - x[0] ** 2), 1 - x[0], S90 * (x[3] - x[2] ** 2), 1 - x[2], S10 * (x[1] + x[3] - 2), (x[1] - x[3]) / S10],
     lambda x: [[-20 * x[0], 10, 0, 0], [-1, 0, 0, 0], [0, 0, -2 * S90 * x[2], S90], [0, 0, -1, 0], [0, S10, 0, S10], [0, 1 / S10, 0, -1 / S10]]),
    ("Kowalik-Osborne", 4, 11, [0.25, 0.39, 0.415, 0.39], 3.07505e-4 / 2, False, ko_r, ko_j),
    ext_rosenbrock(10),
    trig(10),
    dbv(10),
    broyden(10),
]


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def main():
    print("## SciPy: scalar roots (brentq) and minima (minimize_scalar, bounded)\n")
    rows = []
    for name, f, a, b in ROOTS:
        _, r = optimize.brentq(f, a, b, xtol=1e-12 * (b - a), full_output=True)
        rows.append((name, r.function_calls))
    table(["function", "brentq evaluations"], rows)
    rows = []
    for name, f, a, b in MINS:
        res = optimize.minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-10 * (b - a)})
        rows.append((name, f"{res.nfev}, x = {res.x:.6f}"))
    table(["function", "evaluations, x"], rows)

    print("## SciPy on the MGH problems (exact derivatives)\n")
    rows = []
    for name, n, m, x0, fstar, is_root, r, j in PROBLEMS:
        R = lambda x, r=r: np.array(r(x), dtype=float)
        Jf = lambda x, j=j: np.array(j(x), dtype=float)
        f = lambda x, R=R: 0.5 * float(R(x) @ R(x))
        g = lambda x, R=R, Jf=Jf: Jf(x).T @ R(x)
        ls = optimize.least_squares(R, x0, jac=Jf, method="lm")
        lb = optimize.minimize(f, x0, jac=g, method="L-BFGS-B")
        nm = optimize.minimize(f, x0, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000})
        cell_root = ""
        if is_root:
            rt = optimize.root(R, x0, jac=Jf, method="hybr")
            cell_root = f"{rt.nfev} / {rt.njev}" + ("" if rt.success and np.max(np.abs(rt.fun)) < 1e-8 else " (not solved)")
        mark = lambda v: "" if solved(v, fstar) else f" (f = {v:.2g})"
        rows.append((name, f"{ls.nfev} / {ls.njev}" + mark(ls.cost), f"{lb.nfev} / {lb.njev}" + mark(lb.fun), f"{nm.nfev}" + mark(nm.fun), cell_root))
    table(["problem", "least_squares lm (evals / jacs)", "L-BFGS-B (f / g)", "Nelder-Mead", "root hybr (evals / jacs)"], rows)

    print("## SciPy L-BFGS-B with finite-difference gradients\n")
    rows = []
    for gen in (ext_rosenbrock, dbv):
        for n in (10, 30, 100):
            name, n, m, x0, fstar, is_root, r, j = gen(n)
            R = lambda x, r=r: np.array(r(x), dtype=float)
            f = lambda x, R=R: 0.5 * float(R(x) @ R(x))
            res = optimize.minimize(f, x0, method="L-BFGS-B")
            rows.append((name, f"{res.nfev}, {res.nit}" + ("" if solved(res.fun, 1e-9) else f" (f = {res.fun:.2g})")))
    table(["problem", "f evaluations, iterations"], rows)


if __name__ == "__main__":
    main()
