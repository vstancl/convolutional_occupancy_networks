"""Tests of the H5-G1b distance histograms of ``eval_rays`` (stored for the error CDF).

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_dist_hist.py -q

a. ``dist_hist_counts``: the layout [underflow, 200 bins, overflow, inf], edges included on the right bin.
b. On three test shapes with decoded = GT shifted by +0.001: every ordinary and thin wall/gap count lies in a bin
   entirely below 0.002; shifted by +0.003: every ordinary count lies above 0.002 (thin hits not: inside a thin
   feature another shifted hit of the same type can be nearer). The totals equal ord_n and the number of thin
   wall/gap endpoint hits (counted independently from ``thin_pairs``).
c. The histograms change no other field: the diagnostics with and without ``axis`` agree apart from ``dist_hist``.
d. A decoded axis without hits puts every GT hit in the inf entry.
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

from csgbox.ceilings import RayHits, thin_bins, thin_pairs  # noqa: E402

from data import DATA_ROOT, shape_ids  # noqa: E402
from eval_rays import (  # noqa: E402
    DIST_HIST_EDGES,
    DIST_HIST_NB,
    EVAL_DELTAS,
    H,
    aggregate,
    dist_hist_counts,
    evaluate_axis_diag,
    hit_diagnostics,
    prepare_all,
)

N_SHAPES = 3


@pytest.fixture(scope="module")
def preps():
    ids = shape_ids("test", DATA_ROOT)
    rng = np.random.Generator(np.random.PCG64(1))
    return [prepare_all(ids[i], DATA_ROOT) for i in rng.choice(len(ids), size=N_SHAPES, replace=False)]


def _shift(gt: RayHits, dt: float) -> RayHits:
    return RayHits(gt.offsets.copy(), gt.t + dt, gt.enter.copy(), None)


def _summary(preps, make_dec) -> dict:
    res = [evaluate_axis_diag(p.gt_sub[a], make_dec(p.gt_sub[a]), a, tuple(EVAL_DELTAS), H)
           for p in preps for a in range(3)]
    return aggregate(res)[0]


def test_a_layout():
    d = np.array([1e-6, 1e-5, 0.5, 1.0, 2.0, np.inf, np.nan])
    c = dist_hist_counts(d[:-1])
    assert len(c) == DIST_HIST_NB + 3 == 203
    assert c[0] == 1  # 1e-6: underflow
    assert c[1] == 1  # 1e-5: the first bin (left edge included)
    assert c[DIST_HIST_NB + 1] == 2  # 1.0 and 2.0: overflow
    assert c[DIST_HIST_NB + 2] == 1  # inf
    k = int(np.searchsorted(DIST_HIST_EDGES, 0.5, side="right"))
    assert c[k] == 1 and DIST_HIST_EDGES[k - 1] <= 0.5 < DIST_HIST_EDGES[k]
    assert sum(c) == 6


def _wg_endpoints(preps) -> int:
    """Independent count: distinct GT hits ending a thin wall or gap with L/h < 1."""
    n = 0
    for p in preps:
        for a in range(3):
            gt = p.gt_sub[a]
            k, length, cls = thin_pairs(gt, a)
            sel = (cls.astype(int) <= 1) & (thin_bins(length, H) < 4)
            n += np.union1d(k[sel], k[sel] + 1).size
    return n


@pytest.mark.parametrize("dt,below", [(0.001, True), (0.003, False)])
def test_b_shifted(preps, dt, below):
    s = _summary(preps, lambda gt: _shift(gt, dt))
    dh = s["diag"]["dist_hist"]
    assert dh["edges"] == DIST_HIST_EDGES.tolist()
    ord_c, thin_c = np.array(dh["ord"]), np.array(dh["thin_wg_lt1"])
    assert ord_c.sum() == s["diag"]["ord_n"] > 0
    assert thin_c.sum() == _wg_endpoints(preps) > 0
    lo = np.concatenate([[0.0], DIST_HIST_EDGES[:-1], [DIST_HIST_EDGES[-1], np.inf]])  # lower bound of each entry
    hi = np.concatenate([[DIST_HIST_EDGES[0]], DIST_HIST_EDGES[1:], [np.inf, np.inf]])  # upper bound
    if below:  # +0.001: the nearest same-type decoded hit is at most 0.001 away, for ordinary and thin hits
        for c in (ord_c, thin_c):
            assert np.all(hi[c > 0] <= 0.002)
    else:  # +0.003: ordinary hits only (as test_h5_6 c); inside a thin feature another shifted hit can be nearer
        assert np.all(lo[ord_c > 0] >= 0.002)


def test_c_no_other_field_changes(preps):
    p = preps[0]
    for a in range(3):
        gt = p.gt_sub[a]
        dec = _shift(gt, 0.001)
        with_h = hit_diagnostics(gt, dec, H, EVAL_DELTAS, axis=a)
        without = hit_diagnostics(gt, dec, H, EVAL_DELTAS)
        assert "dist_hist" not in without
        with_h.pop("dist_hist")
        assert with_h == without


def test_d_empty_decoded(preps):
    p = preps[0]
    gt = p.gt_sub[0]
    empty = RayHits(np.zeros(gt.n_rays + 1, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=bool), None)
    g = hit_diagnostics(gt, empty, H, EVAL_DELTAS, axis=0)
    assert g["dist_hist"]["ord"][-1] == g["ord"]["n"] == sum(g["dist_hist"]["ord"])
