#!/usr/bin/env python3
"""Numerical check of csgbox.ceilings under one numpy version (h5 step H5-0b).

Run once per environment and compare the two output files with compare_ceilings_npz.py:

    ~/PhD/convonet/.venv/bin/python h5/scripts/check_ceilings_numpy.py OUT.npz
    uv run --project ~/PhD/csgbox python h5/scripts/check_ceilings_numpy.py OUT.npz

For each of SAMPLE_SHAPES test shapes (numpy Generator seed 0, from csg/test.lst) and each
ray axis, the script computes the float64 GT hits from tree.json, then:

- hits_8: ``decode_hits`` (CSR);
- tsdf_128: ``tsdf_values`` (float32), decoded with ``decode_tsdf``;
- binary_128: ``binary_occupancy`` (bool), decoded with ``decode_binary``;
- occfrac_128: ``occ_fraction`` (float64, stored as float32), decoded with ``decode_occfrac``;
- ``evaluate_axis`` metrics of each decoded representation against the GT, at delta 0.002.

Also times the full-shape encoding of the four targets (median of TIMING_RUNS runs per shape,
all three axes, GT already computed). Timings are stored under ``timing__`` keys only.

Keys are written in sorted order. Everything is pure numpy: no mesh library is loaded.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

H5_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H5_DIR))
from _paths import add_external_paths  # noqa: E402

SAMPLE_SHAPES: int = 20  # test shapes checked
SELECTION_SEED: int = 0  # seed of the shape selection (numpy Generator)
TIMING_RUNS: int = 5  # repetitions per shape for the median timing
DELTA: float = 0.002  # matching tolerance of the metrics (GATE_HIT_DELTA)
L_HITS: int = 8  # hits_L of the hit representation
M_CELLS: int = 128  # samples of the column representations
DATA_ROOT: Path = Path.home() / "PhD" / "data" / "csgbox" / "v1"  # csg/ and csg/test.lst live here
TARGETS: tuple[str, ...] = ("hits_8", "tsdf_128", "binary_128", "occfrac_128")


def select_test_shapes(data_root: Path) -> list[str]:
    ids = (data_root / "csg" / "test.lst").read_text(encoding="utf-8").split()
    rng = np.random.Generator(np.random.PCG64(SELECTION_SEED))
    picked = rng.choice(len(ids), size=SAMPLE_SHAPES, replace=False)
    return sorted(ids[i] for i in picked)


def encode_all(gt, ceil) -> dict:
    """Encodings of one axis: the four targets as they are stored (before decoding)."""
    vals, z, h = ceil.tsdf_values(gt, M_CELLS)
    occ = ceil.binary_occupancy(gt, M_CELLS)
    f = ceil.occ_fraction(gt, M_CELLS).astype(np.float32)
    return {"hits_8": ceil.decode_hits(gt, L_HITS), "tsdf_128": (vals, z, h), "binary_128": occ, "occfrac_128": f}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_ceilings_numpy.py OUT.npz")
    out_path = Path(sys.argv[1])
    np_version = add_external_paths()

    import csgbox.ceilings as ceil
    from csgbox.grid import cell_size
    from csgbox.tree import from_json

    h = cell_size(M_CELLS)
    shapes = select_test_shapes(DATA_ROOT)
    arrays: dict[str, np.ndarray] = {}
    timings: dict[str, list[float]] = {t: [] for t in TARGETS}

    for shape in shapes:
        tree = from_json((DATA_ROOT / "csg" / shape / "tree.json").read_text(encoding="utf-8"))
        gts = [ceil.gt_hits(tree, axis, 128) for axis in range(3)]

        # timing: median of TIMING_RUNS full-shape encodings of each target, over the three axes
        per_target: dict[str, list[float]] = {t: [] for t in TARGETS}
        for _ in range(TIMING_RUNS):
            for target in TARGETS:
                t0 = time.perf_counter()
                for gt in gts:
                    _ = _encode_one(ceil, gt, target)
                per_target[target].append(time.perf_counter() - t0)
        for target in TARGETS:
            timings[target].append(statistics.median(per_target[target]))

        for axis, gt in enumerate(gts):
            enc = encode_all(gt, ceil)
            decoded = {
                "hits_8": enc["hits_8"],
                "tsdf_128": ceil.decode_tsdf(enc["tsdf_128"][0], enc["tsdf_128"][1], enc["tsdf_128"][2]),
                "binary_128": ceil.decode_binary(gt, M_CELLS),
                "occfrac_128": ceil.decode_occfrac(enc["occfrac_128"]),
            }
            prefix = f"{shape}__x{axis}"
            # encodings (what a network would be trained to produce or read)
            arrays[f"{prefix}__hits_8__offsets"] = enc["hits_8"].offsets
            arrays[f"{prefix}__hits_8__t"] = enc["hits_8"].t
            arrays[f"{prefix}__hits_8__enter"] = enc["hits_8"].enter
            arrays[f"{prefix}__tsdf_128__values"] = enc["tsdf_128"][0]
            arrays[f"{prefix}__binary_128__occupancy"] = enc["binary_128"]
            arrays[f"{prefix}__occfrac_128__fraction"] = enc["occfrac_128"]
            # hits decoded back from each encoding
            for target, dec in decoded.items():
                arrays[f"{prefix}__{target}__dec_offsets"] = dec.offsets
                arrays[f"{prefix}__{target}__dec_t"] = dec.t
                arrays[f"{prefix}__{target}__dec_enter"] = dec.enter
                # metrics against the GT
                res = ceil.evaluate_axis(gt, dec, axis, (DELTA,), h)
                for key, val in _flatten_metrics(res).items():
                    arrays[f"{prefix}__{target}__m__{key}"] = val

    for target in TARGETS:
        arrays[f"timing__{target}__median_per_shape_s"] = np.asarray(timings[target], dtype=np.float64)
    arrays["meta__numpy_version"] = np.asarray(np_version)
    arrays["meta__shapes"] = np.asarray(shapes)

    out = {k: arrays[k] for k in sorted(arrays)}
    np.savez(out_path, **out)
    print(f"numpy {np_version}: {len(shapes)} shapes, {len(out)} arrays -> {out_path}")
    for target in TARGETS:
        print(f"{target}: median over shapes of the 3-axis encode = {statistics.median(timings[target]) * 1e3:.2f} ms")


def _encode_one(ceil, gt, target: str):
    if target == "hits_8":
        return ceil.decode_hits(gt, L_HITS)
    if target == "tsdf_128":
        return ceil.tsdf_values(gt, M_CELLS)
    if target == "binary_128":
        return ceil.binary_occupancy(gt, M_CELLS)
    return ceil.occ_fraction(gt, M_CELLS).astype(np.float32)


def _flatten_metrics(res: dict) -> dict[str, np.ndarray]:
    """Flatten the nested dict of evaluate_axis into key -> array (keys contain no '__')."""
    out: dict[str, np.ndarray] = {}
    for k, v in res.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"d{k}__{kk}"] = np.asarray(vv)
        else:
            out[k] = np.asarray(v)
    return out


if __name__ == "__main__":
    main()
