"""Tests of h5 step H5-1a: ray utilities, slot encoding, slot and binary decoders.

Run from the ConvONet venv (pytest 9.2 is available there):

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests -q

Data: 20 test shapes chosen with numpy Generator(PCG64(0)) from csg/test.lst, all three axes.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

import csgbox.ceilings as ceil  # noqa: E402
import csgbox.constants as cc  # noqa: E402
from csgbox.tree import from_json  # noqa: E402

from rays import load_axis_rayhits, subset_rays  # noqa: E402
from slots import decode_slots, encode_slots  # noqa: E402
from to_hits import BOX_HI, BOX_LO, decode_binary_array  # noqa: E402

DATA_ROOT = Path.home() / "PhD" / "data" / "csgbox" / "v1"
SAMPLE_SHAPES = 20
SELECTION_SEED = 0
M = 128
# Tolerance of the dexel t against the float64 GT: the file stores float32, and half an ulp of
# float32 at |t| <= 0.55 is about 3e-8. One full ulp (6e-8) is the allowed bound.
TOL_T_FLOAT32 = 6e-8
# Tolerance for the hand-computed synthetic cases (exact arithmetic up to rounding).
TOL_SYNTH = 1e-12


@lru_cache(maxsize=1)
def shape_ids() -> tuple[str, ...]:
    ids = (DATA_ROOT / "csg" / "test.lst").read_text(encoding="utf-8").split()
    rng = np.random.Generator(np.random.PCG64(SELECTION_SEED))
    picked = rng.choice(len(ids), size=SAMPLE_SHAPES, replace=False)
    return tuple(sorted(ids[i] for i in picked))


def dexel_path(shape: str) -> Path:
    return DATA_ROOT / "csg" / shape / "dexels_128.npz"


def tree_path(shape: str) -> Path:
    return DATA_ROOT / "csg" / shape / "tree.json"


def assert_hits_equal(a, b, msg: str = "") -> None:
    assert np.array_equal(a.offsets, b.offsets), f"offsets differ {msg}"
    assert np.array_equal(a.enter, b.enter), f"enter differs {msg}"
    assert np.array_equal(a.t, b.t), f"t differs {msg}: max abs {np.max(np.abs(a.t - b.t)) if a.t.size else 0}"


# ---------------------------------------------------------------------------
# a. load_axis_rayhits vs csgbox gt_hits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_load_matches_analytic_gt(axis):
    worst = 0.0
    for shape in shape_ids():
        gt = ceil.gt_hits(from_json(tree_path(shape).read_text(encoding="utf-8")), axis, M)
        ld = load_axis_rayhits(dexel_path(shape), axis)
        assert np.array_equal(ld.offsets, gt.offsets), f"{shape} axis {axis}: offsets"
        assert np.array_equal(ld.enter, gt.enter), f"{shape} axis {axis}: enter"
        if gt.t.size:
            worst = max(worst, float(np.max(np.abs(ld.t - gt.t))))
    assert worst <= TOL_T_FLOAT32, f"axis {axis}: max |dt| {worst}"
    print(f"\naxis {axis}: max |dt| dexel vs analytic = {worst:.3e}")


# ---------------------------------------------------------------------------
# b. subset_rays then the encoders equal the rows of the full encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_b_subset_encoders_match_full(axis):
    rng = np.random.default_rng(SELECTION_SEED + axis)
    for shape in shape_ids():
        gt = load_axis_rayhits(dexel_path(shape), axis)
        full_tsdf, _, _ = ceil.tsdf_values(gt, M)
        full_bin = ceil.binary_occupancy(gt, M)
        full_occ = ceil.occ_fraction(gt, M)
        # a random subset in random order, with 500 rays of the 16384 and no repeats
        rays = rng.choice(gt.n_rays, size=500, replace=False)
        sub = subset_rays(gt, rays)
        assert sub.n_rays == rays.size
        vals, _, _ = ceil.tsdf_values(sub, M)
        assert np.array_equal(vals, full_tsdf[rays]), f"{shape} axis {axis}: tsdf"
        assert np.array_equal(ceil.binary_occupancy(sub, M), full_bin[rays]), f"{shape} axis {axis}: binary"
        assert np.array_equal(ceil.occ_fraction(sub, M), full_occ[rays]), f"{shape} axis {axis}: occfrac"


def test_b_subset_rays_keeps_hits_of_repeated_rays():
    gt = load_axis_rayhits(dexel_path(shape_ids()[0]), 2)
    counts = np.diff(gt.offsets)
    ray = int(np.flatnonzero(counts >= 2)[0])
    sub = subset_rays(gt, np.array([ray, ray, 0]))
    seg = gt.t[gt.offsets[ray]:gt.offsets[ray + 1]]
    assert np.array_equal(sub.t[sub.offsets[0]:sub.offsets[1]], seg)
    assert np.array_equal(sub.t[sub.offsets[1]:sub.offsets[2]], seg)


# ---------------------------------------------------------------------------
# c. decode_binary_array equals csgbox decode_binary bit for bit
# ---------------------------------------------------------------------------


def test_c_box_constants_agree():
    assert BOX_LO == cc.BOX_LO and BOX_HI == cc.BOX_HI


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_c_decode_binary_array_equals_csgbox(axis):
    for shape in shape_ids():
        gt = load_axis_rayhits(dexel_path(shape), axis)
        occ = ceil.binary_occupancy(gt, M)
        assert_hits_equal(decode_binary_array(occ), ceil.decode_binary(gt, M), f"{shape} axis {axis}")


# ---------------------------------------------------------------------------
# d. decode_slots(encode_slots(gt, K)) equals decode_hits(gt, 2K)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_d_slots_round_trip_equals_decode_hits(K, axis):
    for shape in shape_ids():
        gt = load_axis_rayhits(dexel_path(shape), axis)
        t, mask = encode_slots(gt, K)
        dec = decode_slots(t, mask.astype(np.float64), K)
        assert_hits_equal(dec, ceil.decode_hits(gt, 4 * K), f"{shape} axis {axis} K={K}")


def test_d_encode_shapes_and_missing_slots():
    gt = load_axis_rayhits(dexel_path(shape_ids()[0]), 0)
    t, mask = encode_slots(gt, 2)
    assert t.shape == (gt.n_rays, 8) and mask.shape == (gt.n_rays, 4)
    # a ray with no hits has all slots missing
    counts = np.diff(gt.offsets)
    empty = int(np.flatnonzero(counts == 0)[0])
    assert not mask[empty].any() and np.all(t[empty] == 0)


# ---------------------------------------------------------------------------
# e. synthetic cases of decode_slots (K = 2)
# ---------------------------------------------------------------------------


def _row(front: list, back: list, probs: tuple[float, float, float, float]):
    """One row for K=2: front [(e,x),(e,x)] and back [B1 (e,x), B0 (e,x)] with mask probabilities.

    Columns of t: F0, F1, B1, B0 (enter, exit each). Mask order: F0, F1, B1, B0.
    """
    t = np.zeros(8)
    for slot, (e, x) in enumerate(front):
        if e is not None:
            t[2 * slot], t[2 * slot + 1] = e, x
    for slot, (e, x) in enumerate(back):  # back list is [B1, B0]
        if e is not None:
            t[4 + 2 * slot], t[4 + 2 * slot + 1] = e, x
    return t, np.array(probs, dtype=np.float64)


def test_e_synthetic_slot_decoding():
    none = (None, None)
    rows = [
        # 1. one interval predicted twice with small offsets: front F0 and back B0 -> one average
        _row([(0.10, 0.30), none], [none, (0.1001, 0.3003)], (0.9, 0.0, 0.0, 0.8)),
        # 2. two disjoint intervals on both sides, not merged
        _row([(0.0, 0.1), (0.2, 0.3)], [none, none], (0.9, 0.9, 0.0, 0.0)),
        # 3. overlapping front/back intervals with IoU < 0.5: union
        _row([(0.0, 1.0), none], [none, (0.8, 2.0)], (0.9, 0.0, 0.0, 0.9)),
        # 4. enter >= exit is dropped (both front intervals, probability above thr)
        _row([(0.5, 0.5), (0.6, 0.4)], [none, none], (0.9, 0.9, 0.0, 0.0)),
        # 5. probability exactly thr is not kept (strict >); the interval is kept at 0.6
        _row([none, none], [none, (0.1, 0.2)], (0.0, 0.0, 0.0, 0.5)),
        _row([none, none], [none, (0.1, 0.2)], (0.0, 0.0, 0.0, 0.6)),
        # 6. nothing above threshold: no hits
        _row([(0.0, 0.1), none], [none, none], (0.2, 0.0, 0.0, 0.0)),
    ]
    t = np.stack([r[0] for r in rows])
    p = np.stack([r[1] for r in rows])
    dec = decode_slots(t, p, 2)
    got = [dec.t[dec.offsets[r]:dec.offsets[r + 1]] for r in range(len(rows))]
    avg_e = 0.5 * (0.10 + 0.1001)
    avg_x = 0.5 * (0.30 + 0.3003)
    expected = [
        [avg_e, avg_x],
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 2.0],
        [],  # case 4: dropped
        [],  # case 5, probability 0.5
        [0.1, 0.2],  # case 5, probability 0.6
        [],  # case 6
    ]
    for r, (g, e) in enumerate(zip(got, expected)):
        assert g.size == len(e), f"row {r}: {g} vs {e}"
        assert np.allclose(g, e, rtol=0, atol=TOL_SYNTH), f"row {r}: {g} vs {e}"
    # merged row carries one interval; union row carries one interval
    assert dec.enter[dec.offsets[0]] and not dec.enter[dec.offsets[0] + 1]


# ---------------------------------------------------------------------------
# f. (report only) per-slot positive rate: scripts/pos_weight.py
# ---------------------------------------------------------------------------


def test_f_pos_weight_script_runs_on_one_shape():
    sys.path.insert(0, str(H5_DIR / "scripts"))
    from pos_weight import per_slot_counts  # noqa: E402

    counts = per_slot_counts([shape_ids()[0]], K=2)
    assert counts["rays_all"] == 3 * M * M
