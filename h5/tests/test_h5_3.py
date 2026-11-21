"""Tests of h5 step H5-3: ray-space evaluation (decoder outputs to RayHits, matching, aggregation).

Run from the ConvONet venv (use -s to see the tables and timings):

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_h5_3.py -q -s

Test a and b compare with csgbox's gate code, which needs manifold3d. It is run in a subprocess with the
csgbox venv (``h5/tests/gate_reference.py``); those two tests are skipped if that venv is missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.ceilings import gt_hits  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402
from csgbox.tree import from_json  # noqa: E402

import data as D  # noqa: E402
from data import DATA_ROOT, read_shape_axes, ray_fields, shape_ids  # noqa: E402
from eval_rays import (  # noqa: E402
    EVAL_DELTAS,
    OCC_SNAP_RECOMMENDED,
    aggregate,
    evaluate_outputs,
    evaluate_shape,
    outputs_to_rayhits,
    val_metric,
)
from models import H5RayModel  # noqa: E402

# Tolerances of this file. Hit positions: float32 storage of t (about 3e-8 in world units) and the
# 1e-6 clip of the tsdf logits. Rates: two computations of the same ratio.
ATOL_DT: float = 1e-6
ATOL_RATE: float = 1e-9
CSG_ROOT = Path("/home/wish/PhD/csgbox")
CSG_PY = CSG_ROOT / ".venv" / "bin" / "python"
GATE_REF = H5_DIR / "tests" / "gate_reference.py"
TARGETS = ("hits_8", "hits_16", "occfrac_128", "tsdf_128", "binary_128")
OCC_SNAP_CHECK: float = OCC_SNAP_RECOMMENDED  # snapping that reproduces the gate on the test shapes (docs/h5-3.md)
N_ALL = 3 * D.N_AXIS
H = cell_size(128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMING_SHAPES: int = 200  # task spec
TIMING_RAYS: int = 4096  # task spec


def _ten_test_shapes() -> list[str]:
    ids = shape_ids("test")
    rng = np.random.Generator(np.random.PCG64(0))
    return [ids[i] for i in rng.choice(len(ids), size=10, replace=False)]


SHAPES = _ten_test_shapes()


@pytest.fixture(scope="session")
def reference(tmp_path_factory):
    """Gate reference (csgbox venv) for the 10 test shapes: process_shape, metrics, and decoded hits."""
    if not CSG_PY.exists():
        pytest.skip(f"csgbox venv not found at {CSG_PY}")
    out = tmp_path_factory.mktemp("gate") / "ref"
    dirs = [str(DATA_ROOT / "csg" / s) for s in SHAPES]
    subprocess.run([str(CSG_PY), str(GATE_REF), str(CSG_ROOT), str(out), *dirs], check=True)
    doc = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    npz = np.load(out.with_suffix(".npz"))
    return doc, npz


def _logit(p: np.ndarray) -> np.ndarray:
    """The network-format logit of a probability: clipped to [1e-6, 1 - 1e-6] (task spec)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def perfect_outputs(target: str, axes, ray_axis: np.ndarray, ray_index: np.ndarray) -> dict:
    """Network-format outputs that encode the GT of the given rays exactly (task spec, test a)."""
    if target in D.HITS_K:
        s, mask = D.hits_targets(axes, ray_axis, ray_index, D.HITS_K[target])
        return {"s_pred": s.astype(np.float64), "mask_logits": np.where(mask, 20.0, -20.0)}
    y = D.dense_values(axes, ray_axis, ray_index, target)
    if target in ("occfrac_128", "binary_128"):
        return {"logits": _logit(y.astype(np.float64))}
    yy = np.clip(y.astype(np.float64) / (2.5 * H), -1 + 1e-6, 1 - 1e-6)
    return {"logits": np.arctanh(yy)}


def _gate_axis_hits(reference, rep: str, shape: str, axis: int):
    doc, npz = reference
    key = f"{rep}|{shape}|{axis}"
    return npz[key + "|offsets"], npz[key + "|t"], npz[key + "|enter"]


# ---------------------------------------------------------------- a. perfect predictions vs the gate

@pytest.mark.parametrize("target", TARGETS)
def test_a_perfect_predictions_match_gate(reference, target):
    doc, _npz = reference
    g = np.arange(N_ALL, dtype=np.int64)
    ray_axis, ray_index, _uv = ray_fields(g)
    n_hit_diff = {}
    rays_differ_total = 0
    for shape in SHAPES:
        axes = read_shape_axes(DATA_ROOT / "csg" / shape)
        outs = perfect_outputs(target, axes, ray_axis, ray_index)
        dec = outputs_to_rayhits(target, outs, ray_axis)
        for a in range(3):
            off, t, _enter = _gate_axis_hits(reference, target, shape, a)
            ours = dec[a]
            diff = np.flatnonzero(np.diff(off) != np.diff(ours.offsets))
            rays_differ_total += int(diff.size)
            n_hit_diff[(shape, a)] = int(diff.size)
            if diff.size == 0 and t.size:
                assert np.max(np.abs(t - ours.t)) < ATOL_DT
    print(f"\na {target}: rays whose hit count differs from the gate (10 shapes, 3 axes, all rays): "
          f"{rays_differ_total} of {len(SHAPES) * 3 * D.N_AXIS}")
    if target == "occfrac_128":
        # The literal rule (sigmoid of the clipped logit, then decode_occfrac) creates spurious partial
        # cells: a fraction of 1e-6 is 'partial' for decode_occfrac. Documented in docs/h5-3.md.
        assert rays_differ_total > 0
        # With the fractions snapped within OCC_SNAP_CHECK of 0 and 1, the gate is reproduced exactly.
        for shape in SHAPES:
            axes = read_shape_axes(DATA_ROOT / "csg" / shape)
            outs = perfect_outputs(target, axes, ray_axis, ray_index)
            dec = outputs_to_rayhits(target, outs, ray_axis, occ_snap=OCC_SNAP_CHECK)
            for a in range(3):
                off, _t, _e = _gate_axis_hits(reference, target, shape, a)
                assert np.array_equal(np.diff(off), np.diff(dec[a].offsets)), f"{shape} axis {a}"
    else:
        assert rays_differ_total == 0


@pytest.mark.parametrize("target", TARGETS)
def test_a_counts_and_matches_equal_gate(reference, target):
    """evaluate_outputs (our decode of perfect outputs against gt_hits) equals the gate's evaluate_axis."""
    doc, _npz = reference
    g = np.arange(N_ALL, dtype=np.int64)
    ray_axis, ray_index, _uv = ray_fields(g)
    mismatches = []
    for shape in SHAPES:
        axes = read_shape_axes(DATA_ROOT / "csg" / shape)
        outs = perfect_outputs(target, axes, ray_axis, ray_index)
        tree = from_json((DATA_ROOT / "csg" / shape / "tree.json").read_text(encoding="utf-8"))
        gt = [gt_hits(tree, a) for a in range(3)]
        occ_snap = OCC_SNAP_CHECK if target == "occfrac_128" else None
        ours = evaluate_outputs(target, outs, gt, ray_axis, ray_index, occ_snap=occ_snap)
        for a in range(3):
            ref = doc["axis_eval"][target][shape][a]
            for key in [str(d) for d in EVAL_DELTAS]:
                for field in ("matched", "dt_n"):
                    if ours[a][key][field] != ref[key][field]:
                        mismatches.append((shape, a, key, field, ours[a][key][field], ref[key][field]))
                if ours[a][key]["thin_recalled"] != ref[key]["thin_recalled"]:
                    mismatches.append((shape, a, key, "thin_recalled"))
            for field in ("gt_hits", "dec_hits", "rays"):
                if ours[a][field] != ref[field]:
                    mismatches.append((shape, a, field))
    print(f"\na {target}: evaluate_axis fields compared with the gate: {len(mismatches)} mismatches")
    assert not mismatches, mismatches[:10]


# ---------------------------------------------------------------- b. aggregation equals gate_ceilings

def test_b_aggregation_equals_gate_metrics(reference):
    doc, _npz = reference
    per_axis = [r for shape in SHAPES for r in doc["axis_eval"]["hits_8"][shape]]
    summary, md = aggregate(per_axis)
    assert "# Ray-space evaluation" in md
    gate = doc["metrics"]["hits_8"]["deltas"]
    for key in [str(d) for d in EVAL_DELTAS]:
        m = summary["deltas"][key]
        g = gate[key]
        for name in ("precision", "recall", "f1", "dt_mean", "dt_p95"):
            assert np.isclose(m[name], g[name], rtol=ATOL_RATE, atol=ATOL_RATE, equal_nan=True), (key, name)
        assert np.allclose(np.asarray(m["thin_recall"], float), np.asarray(g["thin_recall"], float),
                           atol=ATOL_RATE, equal_nan=True), key
        assert np.isclose(summary["count_err_mean"], g["count_err_mean"], rtol=ATOL_RATE, atol=ATOL_RATE)
    print("\nb aggregate vs gate (hits_8, 10 shapes): precision, recall, F1, mean/p95 |dt|, thin recall equal")


def test_b_thin_walls_gaps_pooled_is_population_weighted(reference):
    doc, _npz = reference
    per_axis = [r for shape in SHAPES for r in doc["axis_eval"]["hits_8"][shape]]
    summary, _ = aggregate(per_axis)
    tt = np.asarray(summary["thin_total"], dtype=float)
    key = str(EVAL_DELTAS[0])
    rec = np.asarray(summary["deltas"][key]["thin_recalled"], dtype=float)
    small = slice(0, 4)  # bins with L/h < 1
    expect = (rec[0, small].sum() + rec[1, small].sum()) / (tt[0, small].sum() + tt[1, small].sum())
    assert summary["deltas"][key]["walls_gaps_pooled_L_lt_1"] == pytest.approx(expect, abs=ATOL_RATE)


# ---------------------------------------------------------------- c. untrained models run end to end

@pytest.mark.parametrize("target", TARGETS + ("hits_8_plane", "tsdf_128"))
def test_c_untrained_model_runs(target):
    torch.manual_seed(0)
    model = H5RayModel(target).to(DEVICE)
    rng = np.random.default_rng(0)
    ids = shape_ids("val")
    ray_axis = rng.integers(0, 3, 300)
    ray_index = rng.integers(0, D.N_AXIS, 300)
    res = evaluate_shape(model, ids[0], "val", rays=(ray_axis, ray_index), device=DEVICE)
    assert len(res) == 3
    summary = val_metric(model, n_shapes=2, n_rays=256, device=DEVICE)
    assert summary["n_shapes"] == 2
    assert set(summary) >= {"deltas", "thin_total", "gt_hits", "dec_hits", "n_shapes"}


def test_c_untrained_all_rays_runs():
    torch.manual_seed(0)
    model = H5RayModel("hits_8").to(DEVICE)
    res = evaluate_shape(model, shape_ids("val")[0], "val", rays="all", device=DEVICE)
    assert sum(r["rays"] for r in res) == N_ALL


# ---------------------------------------------------------------- d. timing of val_metric

@pytest.mark.parametrize("target", ["hits_8", "occfrac_128"])
def test_d_val_metric_timing(target):
    torch.manual_seed(0)
    model = H5RayModel(target).to(DEVICE)
    t0 = time.perf_counter()
    summary = val_metric(model, n_shapes=TIMING_SHAPES, n_rays=TIMING_RAYS, device=DEVICE)
    dt = time.perf_counter() - t0
    print(f"\nd val_metric untrained {target}: {TIMING_SHAPES} shapes x {TIMING_RAYS} rays: {dt:.1f} s wall "
          f"({dt / TIMING_SHAPES * 1e3:.0f} ms per shape); F1 at 0.002 {summary['deltas']['0.002']['f1']:.4g}")
    assert summary["n_shapes"] == TIMING_SHAPES
