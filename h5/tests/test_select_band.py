"""Tests of the decoding-setting selection with the decoded/GT band (H5-K1-1a, ``train._select``).

Synthetic summaries: only ``deltas['0.002']['f1']``, ``dec_hits`` and ``gt_hits`` are read (the ratio is
``dec_hits / gt_hits``).

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_select_band.py -q

a. best F1 out of the band, second best F1 in the band -> the second.
b. no setting in the band -> the one whose ratio is closest to the band (distance to the nearer edge).
c. all settings in the band -> the plain best F1.
Also: ties of the distance go to the higher F1; a NaN ratio is never in the band; an all-in-band tie of F1
goes to the first setting.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from eval_rays import RATIO_BAND  # noqa: E402
from train import _select  # noqa: E402


def summary(ratio: float, f1: float) -> dict:
    """A summary whose decoded/GT ratio is ``ratio`` (GT hits 1000) and whose F1 at 0.002 is ``f1``."""
    return {"deltas": {"0.002": {"f1": f1}}, "gt_hits": 1000, "dec_hits": round(ratio * 1000)}


def test_a_best_out_of_band_second_in_band():
    s = [summary(1.5, 0.90), summary(1.1, 0.80), summary(1.0, 0.50)]
    choice = _select(s)
    assert choice["index"] == 1
    assert choice["in_band"] is True
    assert choice["ratio"] == 1.1


def test_b_none_in_band_closest_ratio():
    # distances to the band: 0.3 (0.5), 0.15 (1.4), 0.75 (2.0): the closest is 1.4 despite the lower F1
    s = [summary(0.5, 0.90), summary(1.4, 0.20), summary(2.0, 0.99)]
    choice = _select(s)
    assert choice["index"] == 1
    assert choice["in_band"] is False
    assert choice["ratio"] == 1.4


def test_b_tie_of_distance_goes_to_higher_f1():
    # both ratios are 0.05 outside the upper edge: the higher F1 wins
    s = [summary(1.3, 0.50), summary(1.3, 0.70)]
    assert _select(s)["index"] == 1


def test_b_lower_side_uses_lower_edge():
    # 0.7 is 0.1 below 0.8; 1.4 is 0.15 above 1.25: the lower-side ratio is closer
    s = [summary(1.4, 0.90), summary(0.7, 0.10)]
    choice = _select(s)
    assert choice["index"] == 1
    assert math.isclose(choice["ratio"], 0.7)


def test_c_all_in_band_plain_best_f1():
    s = [summary(0.9, 0.30), summary(1.2, 0.60), summary(1.0, 0.55)]
    choice = _select(s)
    assert choice["index"] == 1
    assert choice["in_band"] is True


def test_c_band_edges_are_inside():
    lo, hi = RATIO_BAND
    s = [summary(lo, 0.40), summary(hi, 0.45)]
    choice = _select(s)
    assert choice["index"] == 1
    assert choice["in_band"] is True


def test_c_equal_f1_goes_to_first():
    s = [summary(1.0, 0.60), summary(1.1, 0.60)]
    assert _select(s)["index"] == 0


def test_nan_f1_never_wins_in_band():
    s = [summary(1.0, float("nan")), summary(1.1, 0.10)]
    assert _select(s)["index"] == 1


def test_nan_ratio_is_never_in_band():
    # no GT hits: the ratio is NaN, the setting cannot be in the band; the other one is chosen
    nan_ratio = {"deltas": {"0.002": {"f1": 0.99}}, "gt_hits": 0, "dec_hits": 5}
    s = [nan_ratio, summary(2.0, 0.10)]
    choice = _select(s)
    assert choice["index"] == 1
    assert choice["in_band"] is False
