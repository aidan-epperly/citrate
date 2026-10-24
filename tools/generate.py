"""Generate the per-type array modules in src/ from templates/.

Each element type gets its own module because Luau only turns `buffer.readf64` and
friends into fast calls when they are written out literally; passing them in as
parameters made the kernels 3-10x slower.

Template syntax:
  @W@, @E@, @EPS@, @READ@, @WRITE@, @NAME@, @SCALAR@  substituted per type
  --@f32 / --@f64 / --@c32 / --@c64 <line>       line kept (marker removed) only for that type
  --@include <file>                             replaced by templates/<file>

Usage: python tools/generate.py [--check]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    ("real.luau.in", "f32.luau", {"NAME": "Single-precision real", "W": "4", "EPS": "2 ^ -24", "CMOD": "c32",
                                  "READ": "buffer.readf32", "WRITE": "buffer.writef32", "TAG": "f32"}),
    ("real.luau.in", "f64.luau", {"NAME": "Double-precision real", "W": "8", "EPS": "2 ^ -53", "CMOD": "c64",
                                  "READ": "buffer.readf64", "WRITE": "buffer.writef64", "TAG": "f64"}),
    ("complex.luau.in", "c32.luau", {"NAME": "Single-precision complex", "W": "4", "E": "8", "EPS": "2 ^ -24",
                                     "READ": "buffer.readf32", "WRITE": "buffer.writef32",
                                     "SCALAR": "vectors (x = re, y = im), as in complex32",
                                     "TAG": "c32"}),
    ("complex.luau.in", "c64.luau", {"NAME": "Double-precision complex", "W": "8", "E": "16", "EPS": "2 ^ -53",
                                     "READ": "buffer.readf64", "WRITE": "buffer.writef64",
                                     "SCALAR": "(re, im) pairs of numbers, as in complex64",
                                     "TAG": "c64"}),
]

MARKER = re.compile(r"^--@(f32|f64|c32|c64) ?(.*)$")
INCLUDE = re.compile(r"^--@include (\S+)$")
PLACEHOLDER = re.compile(r"@([A-Z]+)@")


def expand_includes(template: str) -> list[str]:
    lines = []
    for line in template.splitlines():
        m = INCLUDE.match(line)
        if m:
            lines.extend(expand_includes((ROOT / "templates" / m.group(1)).read_text(encoding="utf-8")))
        else:
            lines.append(line)
    return lines


def render(template: str, subs: dict) -> str:
    lines = []
    for line in expand_includes(template):
        m = MARKER.match(line)
        if m:
            if m.group(1) != subs.get("TAG"):
                continue
            line = m.group(2)
        lines.append(line)
    text = "\n".join(lines) + "\n"

    def sub(m):
        key = m.group(1)
        if key not in subs:
            raise KeyError(f"no value for @{key}@")
        return subs[key]

    return PLACEHOLDER.sub(sub, text)


def main() -> int:
    check = "--check" in sys.argv[1:]
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_kernels

    written = gen_kernels.main(check)
    stale = [f"templates/{name}" for name in written] if check else []
    for name in [] if check else written:
        print(f"wrote templates/{name}")
    for template, target, subs in TARGETS:
        text = render((ROOT / "templates" / template).read_text(encoding="utf-8"), subs)
        path = ROOT / "src" / target
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if check:
            stale.append(target)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote src/{target}")
    if stale:
        print("out of date (run python tools/generate.py): " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
