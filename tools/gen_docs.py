"""Generate the API reference in docs/reference/ from the doc comments in src/.

Each public function is listed with its signature and the comment above its definition,
in source order, so the reference cannot drift from the code. The array and matrix pages
come from the double-precision parts (f64, c64); the single-precision modules have the same
functions (complex scalars are vectors in c32 instead of (re, im) pairs).

Usage: python tools/gen_docs.py [--check] (generate.py runs it).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "docs" / "reference"

PARTS = [
    ("array", "Arrays", "Creation and access, elementwise arithmetic, reductions and norms, and each module's `space`."),
    ("matrix", "Matrices", "Column-major matrices: creation and access, transposes, matrix-vector products, submatrices and utilities."),
    ("gemm", "Matrix products", "`matmul` and `gemm`, by register-blocked kernels."),
    ("factor", "Factorizations and solves", "LU, Cholesky, QR, column-pivoted QR, LDL^T, `solve`, determinants, factorization updates and iterative refinement."),
    ("eigen", "Eigenvalues and singular values", "Hessenberg and Schur forms, Hermitian and general eigenproblems, the SVD, the polar decomposition, generalized eigenproblems, rank and condition numbers, randomized low-rank approximations."),
    ("matfun", "Matrix functions", "Exponential, square root, logarithm; Sylvester and Lyapunov equations."),
    ("fft", "Fourier and cosine transforms", "FFTs of any length (complex modules), real FFTs and discrete cosine transforms (real modules)."),
    ("structured", "Structured matrices", "Triangular, tridiagonal, banded, circulant and Toeplitz matrices."),
    ("sparse", "Sparse matrices", "Compressed sparse column and block sparse storage, products, preconditioners and algebraic multigrid."),
]
OTHER = [
    ("krylov", "Iterative solvers", "Krylov solvers for linear systems and eigenproblems, generic over the `Space` and the operator."),
    ("ode", "Differential equations", "Initial value problems (explicit, stiff, second-order, symplectic), implicit mechanics and boundary value problems."),
    ("optimize", "Root finding and optimization", "Scalar roots and minima (Brent, Newton, and global ones through Chebyshev and AAA interpolants), systems (Newton), least squares (Levenberg-Marquardt) and minimization (L-BFGS, Nelder-Mead)."),
    ("cheb", "Chebyshev approximation", "Chebfun-style approximation on an interval."),
    ("rational", "Rational approximation", "The AAA algorithm: rational approximation from samples anywhere in the complex plane, with poles, residues and zeros."),
    ("lightning", "Lightning Laplace solver", "Laplace's and Poisson's equations on polygons by rational functions (Gopal and Trefethen)."),
    ("curve", "Curves", "Bézier, Hermite, Catmull-Rom, Kochanek-Bartels, cubic spline, B-spline and NURBS curves in any `Space`."),
    ("xpbd", "Position-based dynamics", "Particles and compliant constraints by extended position-based dynamics (XPBD) with substeps."),
    ("complex64", "complex64", "Double-precision complex scalars as `(re, im)` pairs."),
    ("complex32", "complex32", "Single-precision complex scalars stored in a `vector`."),
    ("vec", "vec", "`vector` as a real `Space` of dimension at most 3."),
]

DEF = re.compile(r"^(?:local )?function (?:\w+\.)?(\w+)(.*)$")
ALIAS = re.compile(r"^local (\w+)(?:\s*:[^=]+)?\s*=\s*([\w.]+)$")
REQ = re.compile(r'^local (\w+) = require\("\./(\w+)"\)')


def parse(name):
    """(definitions: name -> (signature, comment lines), aliases, requires, exports)."""
    lines = (SRC / f"{name}.luau").read_text(encoding="utf-8").split("\n")
    defs, aliases, reqs = {}, {}, {}
    for i, line in enumerate(lines):
        m = REQ.match(line)
        if m:
            reqs[m.group(1)] = m.group(2)
            continue
        m = DEF.match(line)
        a = ALIAS.match(line)
        t = re.match(r"^local (\w+)\s*:\s*([^=]+?)\s*=\s*\{", line)  # a typed table value
        if not m and not a and not t:
            continue
        j = i
        comment = []
        while j > 0 and lines[j - 1].startswith("--") and not lines[j - 1].startswith("--!"):
            comment.insert(0, lines[j - 1][2:].strip())
            j -= 1
        if m:
            defs.setdefault(m.group(1), (m.group(1) + m.group(2), comment, i))
        elif t:
            defs.setdefault(t.group(1), (f"{t.group(1)}: {t.group(2)}", comment, i))
        else:
            aliases.setdefault(a.group(1), (a.group(2), comment, i))
    exports = []
    start = max(i for i, l in enumerate(lines) if l == "return {")
    for line in lines[start + 1:]:
        if line == "}":
            break
        m = re.match(r"^\t(\w+) = ([\w.]+),", line)
        if m:
            exports.append((m.group(1), m.group(2)))
    return defs, aliases, reqs, exports


CACHE = {}


def lookup(module, expr, depth=0):
    """The signature, comment and source position of expr in module (following aliases and
    requires)."""
    if module not in CACHE:
        CACHE[module] = parse(module)
    defs, aliases, reqs, _ = CACHE[module]
    if "." in expr:
        mod, field = expr.split(".", 1)
        if mod in reqs:
            return lookup(reqs[mod], field, depth + 1)
        return None
    if expr in defs:
        return defs[expr] + (module,)
    if expr in aliases and depth < 5:
        target, comment, pos = aliases[expr]
        found = lookup(module, target, depth + 1)
        if found and comment:
            return (found[0], comment, pos, module)
        return found
    return None


def clean_signature(sig, public):
    # the public name, without the local's own name (aliases)
    sig = re.sub(r"^\w+", public, sig)
    return sig.rstrip()


def page(title, intro, entries):
    out = [f"# {title}", "", intro, "", "_Generated from the source comments by `tools/gen_docs.py`; do not edit._", ""]
    for public, sig, comment in entries:
        out.append(f"### `{public}`")
        out.append("")
        out.append("```luau")
        out.append(sig)
        out.append("```")
        out.append("")
        if comment:
            out.append(" ".join(comment))
            out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def part_entries(part):
    """Public functions of a part, real and complex, in source order."""
    entries = {}
    for tag in ("f64", "c64"):
        _, _, _, exports = parse(tag)
        for public, expr in exports:
            mod, field = expr.split(".", 1) if "." in expr else (None, expr)
            if mod != part:
                continue
            found = lookup(f"{part}_{tag}", field)
            if not found:
                continue
            sig, comment, pos, _ = found
            sig = clean_signature(sig, public)
            key = public
            if key in entries:
                old = entries[key]
                if old[1] != sig:
                    old[1] = old[1] + "\n-- complex modules:\n" + sig
                continue
            entries[key] = [pos if tag == "f64" else 10 ** 6 + pos, sig, comment]
    return [(k, v[1], v[2]) for k, v in sorted(entries.items(), key=lambda kv: kv[1][0])]


def module_entries(module):
    _, _, _, exports = parse(module)
    entries = []
    for public, expr in exports:
        found = lookup(module, expr)
        if not found:
            entries.append((public, f"{public} -- (a value)", []))
            continue
        sig, comment, pos, _ = found
        entries.append((public, clean_signature(sig, public), comment))
    return entries


def outputs():
    files = {}
    index = ["# API reference", "", "_Generated by `tools/gen_docs.py`._", "",
             "The array modules `f32`, `f64`, `c32` and `c64` share these pages: each is the union of its parts",
             "(for example `f64` combines `array_f64`, `matrix_f64`, ...). Signatures are shown for `f64` and,",
             "where they differ, `c64`; `c32` takes complex scalars as `vector`s where `c64` takes `(re, im)` pairs.", ""]
    for part, title, intro in PARTS:
        files[f"{part}.md"] = page(title, intro + " Parts `" + part + "_f32`, `" + part + "_f64`, `" + part + "_c32` and `" + part + "_c64`.", part_entries(part))
        index.append(f"- [{title}]({part}.md)")
    index.append("")
    for module, title, intro in OTHER:
        files[f"{module}.md"] = page(title, intro, module_entries(module))
        index.append(f"- [{title}]({module}.md)")
    files["README.md"] = "\n".join(index) + "\n"
    return files


def main(check=False):
    stale = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in outputs().items():
        path = OUT / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            stale.append(f"docs/reference/{name}")
            if not check:
                path.write_text(text, encoding="utf-8", newline="\n")
    return stale


if __name__ == "__main__":
    print(main("--check" in sys.argv[1:]) or "docs up to date")
