"""Sanity tests of the H5-6 diagnostics (per-hit distances, thin bins, recall, decoded/GT ratio, M comparability).

Five test shapes (``Generator(PCG64(0))`` over ``csg/test.lst``), all rays, the decoded hits are made from the GT:

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_h5_6.py -q -s

a. decoded = GT: recalls 1, within 1, medians below 1e-6, decoded/GT 1.
b. every t shifted by +0.001: ordinary within(0.002) 1, median within 1 % of 0.001.
c. every t shifted by +0.003: ordinary within(0.002) 0, within(h/2) 1 (h/2 = 0.0043).
d. every 10th material interval (enter + exit) removed: decoded/GT about 0.9, the removed GT hits have distance
   > 0 (or inf), M comparable.
e. 30 spurious intervals added per ray: decoded/GT > 1.25, M not comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.ceilings import RayHits  # noqa: E402

from data import DATA_ROOT, shape_ids  # noqa: E402
from eval_rays import (  # noqa: E402
    EVAL_DELTAS,
    H,
    aggregate,
    evaluate_axis_diag,
    m_comparable,
    nearest_same_type,
    prepare_all,
)

DK, DH = str(EVAL_DELTAS[0]), str(EVAL_DELTAS[1])
N_SHAPES = 5


def _test_shapes() -> list[str]:
    ids = shape_ids("test", DATA_ROOT)
    rng = np.random.Generator(np.random.PCG64(0))
    return [ids[i] for i in rng.choice(len(ids), size=N_SHAPES, replace=False)]


@pytest.fixture(scope="module")
def preps():
    return [prepare_all(s, DATA_ROOT) for s in _test_shapes()]


def _copy(gt: RayHits, t: np.ndarray | None = None) -> RayHits:
    return RayHits(gt.offsets.copy(), gt.t.copy() if t is None else t, gt.enter.copy(), None)


def _from_hits(ray: np.ndarray, t: np.ndarray, enter: np.ndarray, n_rays: int) -> RayHits:
    """RayHits from per-hit ray indices (sorted by ray, then t)."""
    order = np.lexsort((t, ray))
    ray, t, enter = ray[order], t[order], enter[order]
    counts = np.bincount(ray, minlength=n_rays).astype(np.int64)
    offsets = np.zeros(n_rays + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return RayHits(offsets, t, enter.astype(bool), None)


def _drop_every_10th_material(gt: RayHits) -> tuple[RayHits, np.ndarray]:
    """Removes both hits of every 10th material interval (enter then exit) of the GT, over all its rays."""
    ray = gt.ray_of()
    k = np.flatnonzero(ray[1:] == ray[:-1])  # interval k joins hits k and k + 1
    material = gt.enter[k] & ~gt.enter[k + 1]
    chosen = k[material][::10]
    drop = np.zeros(gt.t.size, dtype=bool)
    drop[chosen] = True
    drop[chosen + 1] = True
    keep = ~drop
    return _from_hits(ray[keep], gt.t[keep], gt.enter[keep], gt.n_rays), drop


def _add_spurious(gt: RayHits, rng: np.random.Generator, per_ray: int = 30) -> RayHits:
    """Adds ``per_ray`` random material intervals to every ray of the GT."""
    R = gt.n_rays
    lo, hi = float(gt.t.min()), float(gt.t.max())
    ray_new = np.repeat(np.arange(R), per_ray)
    a = rng.uniform(lo, hi, size=ray_new.size)
    b = a + rng.uniform(0.001, 0.05, size=ray_new.size)
    ray = np.concatenate([gt.ray_of(), ray_new, ray_new])
    t = np.concatenate([gt.t, a, b])
    enter = np.concatenate([gt.enter, np.ones(ray_new.size, bool), np.zeros(ray_new.size, bool)])
    return _from_hits(ray, t, enter, R)


def _run(preps, make_dec) -> dict:
    """Aggregate summary over all shapes and axes; ``make_dec(gt, axis)`` gives the decoded hits."""
    results = []
    for prep in preps:
        for a in range(3):
            gt = prep.gt_sub[a]
            results.append(evaluate_axis_diag(gt, make_dec(gt, a), a, tuple(EVAL_DELTAS), H))
    return aggregate(results)[0]


def _finite_bins(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[~np.isnan(v)]


def test_a_decoded_equals_gt(preps):
    s = _run(preps, lambda gt, a: _copy(gt))
    dg = s["diag"]
    assert dg["ord_n"] > 0 and sum(dg["thin_n"]) > 0, "the test shapes must have ordinary and thin hits"
    for d in (DK, DH):
        assert dg["ord_recall"][d] == 1.0
        assert dg["ord_within"][d] == 1.0
        assert np.all(_finite_bins(dg["thin_recall"][d]) == 1.0)
        assert np.all(_finite_bins(dg["thin_within"][d]) == 1.0)
    assert dg["ord_median"] < 1e-6
    assert np.all(_finite_bins(dg["thin_median"]) < 1e-6)
    assert s["dec_gt_ratio"] == 1.0
    print(f"\na: ordinary {dg['ord_n']}, thin {sum(dg['thin_n'])}; decoded/GT {s['dec_gt_ratio']}")


def test_b_shift_0_001(preps):
    s = _run(preps, lambda gt, a: _copy(gt, gt.t + 0.001))
    dg = s["diag"]
    assert dg["ord_within"][DK] == 1.0
    assert dg["ord_median"] == pytest.approx(0.001, rel=0.01)
    print(f"\nb: ordinary within(0.002) {dg['ord_within'][DK]}, median {dg['ord_median']:.6g}")


def test_c_shift_0_003(preps):
    assert float(DH) == pytest.approx(0.0043, abs=1e-4)
    s = _run(preps, lambda gt, a: _copy(gt, gt.t + 0.003))
    dg = s["diag"]
    assert dg["ord_within"][DK] == 0.0
    assert dg["ord_within"][DH] == 1.0
    print(f"\nc: ordinary within(0.002) {dg['ord_within'][DK]}, within(h/2) {dg['ord_within'][DH]}")


def test_d_removed_intervals(preps):
    removed_dist = []

    def make(gt, a):
        dec, drop = _drop_every_10th_material(gt)
        removed_dist.append(nearest_same_type(gt, dec)[drop])
        return dec

    s = _run(preps, make)
    assert s["dec_gt_ratio"] == pytest.approx(0.9, abs=0.02)
    dist = np.concatenate(removed_dist)
    assert dist.size > 0
    assert np.all((dist > 0) | np.isinf(dist))
    assert m_comparable(s)
    print(f"\nd: decoded/GT {s['dec_gt_ratio']:.4f}; removed GT hits {dist.size}, "
          f"min distance {dist[np.isfinite(dist)].min() if np.isfinite(dist).any() else 'inf'}")


def test_e_spurious_intervals(preps):
    rng = np.random.Generator(np.random.PCG64(1))
    s = _run(preps, lambda gt, a: _add_spurious(gt, rng))
    assert s["dec_gt_ratio"] > 1.25
    assert not m_comparable(s)
    print(f"\ne: decoded/GT {s['dec_gt_ratio']:.3f}, M comparable {m_comparable(s)}")
