#!/usr/bin/env python3
"""Compare two outputs of check_ceilings_numpy.py (h5 step H5-0b).

    python h5/scripts/compare_ceilings_npz.py A.npz B.npz [--md]

Rules: the key sets must be equal; integer and boolean arrays must be identical; float arrays
are compared by max abs diff (expected 0); ``timing__`` keys and the numpy version are excluded.
Exit status 0 only if everything is identical.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def group_of(key: str) -> str:
    """Group for the report: the representation or the field kind, without the shape and axis."""
    # keys are <shape>__x<axis>__<target>__<field...>
    parts = key.split("__")
    if len(parts) < 4:  # meta__shapes and similar
        return parts[0]
    return f"{parts[2]} / {parts[3]}"


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--md"]
    if len(args) != 2:
        raise SystemExit("usage: compare_ceilings_npz.py A.npz B.npz [--md]")
    a = np.load(args[0], allow_pickle=False)
    b = np.load(args[1], allow_pickle=False)
    ka = {k for k in a.files if not k.startswith("timing__") and k != "meta__numpy_version"}
    kb = {k for k in b.files if not k.startswith("timing__") and k != "meta__numpy_version"}
    ok = True
    if ka != kb:
        ok = False
        print(f"key sets differ: only in A {sorted(ka - kb)[:5]}, only in B {sorted(kb - ka)[:5]}")

    stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "identical": 0, "max_abs": 0.0, "kind": ""})
    for k in sorted(ka & kb):
        x, y = a[k], b[k]
        g = stats[group_of(k)]
        g["n"] += 1
        if x.shape != y.shape or x.dtype != y.dtype:
            ok = False
            g["kind"] = "SHAPE/DTYPE MISMATCH"
            print(f"{k}: shape/dtype differ {x.shape} {x.dtype} vs {y.shape} {y.dtype}")
            continue
        if x.dtype.kind in "biu" or x.dtype.kind in "US":
            same = bool(np.array_equal(x, y))
            g["kind"] = "exact"
            if same:
                g["identical"] += 1
            else:
                ok = False
                print(f"{k}: integer/bool/string arrays differ")
        else:
            g["kind"] = "float"
            if x.size == 0:
                g["identical"] += 1
                continue
            diff = np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))
            g["max_abs"] = max(g["max_abs"], float(diff))
            if diff == 0.0 or (np.isnan(x).all() and np.isnan(y).all()):
                g["identical"] += 1
            else:
                ok = False
                print(f"{k}: float arrays differ, max abs diff {diff:.3e}")

    print(f"\n| group | arrays | kind | identical | max abs diff |")
    print("|---|---|---|---|---|")
    for name in sorted(stats):
        g = stats[name]
        print(f"| {name} | {g['n']} | {g['kind']} | {g['identical']}/{g['n']} | {g['max_abs']:.3e} |")
    print("\nVERDICT:", "IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
