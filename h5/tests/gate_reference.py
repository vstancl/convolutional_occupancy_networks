"""Reference numbers from csgbox's own gate code. Run with the csgbox venv (it needs manifold3d):

    /home/wish/PhD/csgbox/.venv/bin/python h5/tests/gate_reference.py CSGBOX_ROOT OUT_PREFIX SHAPE_DIR...

Writes ``OUT_PREFIX.json`` (``process_shape`` per shape, ``gate_ceilings.metrics`` of the merge, and the
``evaluate_axis`` dicts of the gate decoder per rep, shape and axis) and ``OUT_PREFIX.npz`` (the decoded
hits of the gate decoder: offsets, t, enter per rep, shape and axis). The decoder is ``csgbox.gate._decode``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def main(argv: list[str]) -> int:
    csg_root = Path(argv[0])
    out_prefix = Path(argv[1])
    shapes = [Path(s) for s in argv[2:]]
    sys.path.insert(0, str(csg_root))
    sys.path.insert(0, str(csg_root / "scripts"))

    import gate_ceilings as gc
    from csgbox.constants import GATE_HIT_DELTA, GEN_CHECK_N
    from csgbox.ceilings import evaluate_axis, gt_hits
    from csgbox.gate import _decode, process_shape
    from csgbox.grid import cell_size
    from csgbox.tree import from_json

    reps = ("hits_8", "hits_16", "occfrac_128", "tsdf_128", "binary_128")
    h = cell_size(GEN_CHECK_N)
    deltas = (GATE_HIT_DELTA, h / 2.0)
    results = [process_shape(str(s)) for s in shapes]
    met = gc.metrics(gc.merge(results))

    axis_eval: dict = {rep: {} for rep in reps}
    npz: dict[str, np.ndarray] = {}
    for s in shapes:
        tree = from_json((s / "tree.json").read_text(encoding="utf-8"))
        for axis in range(3):
            gt = gt_hits(tree, axis, GEN_CHECK_N)
            for rep in reps:
                dec = _decode(rep, tree, gt, axis)
                axis_eval[rep].setdefault(s.name, []).append(_plain(evaluate_axis(gt, dec, axis, deltas, h)))
                key = f"{rep}|{s.name}|{axis}"
                npz[key + "|offsets"] = np.asarray(dec.offsets, dtype=np.int64)
                npz[key + "|t"] = np.asarray(dec.t, dtype=np.float64)
                npz[key + "|enter"] = np.asarray(dec.enter, dtype=bool)
                npz[key + "|gt_offsets"] = np.asarray(gt.offsets, dtype=np.int64)

    doc = {"process": _plain(results), "metrics": _plain(met), "axis_eval": axis_eval, "deltas": [str(d) for d in deltas]}
    out_prefix.with_suffix(".json").write_text(json.dumps(doc), encoding="utf-8")
    np.savez(out_prefix.with_suffix(".npz"), **npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
