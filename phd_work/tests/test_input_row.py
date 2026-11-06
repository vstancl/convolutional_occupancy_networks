"""[phd] Tests for phd_work/input_row.py (superset of occupancy from dexel slots, CPU)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHD_WORK = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(PHD_WORK, '..'))
sys.path.insert(0, REPO)

from phd_work.dexels import RES, voxels_to_dexel_slots  # noqa: E402
from phd_work.input_row import plane_superset, superset_occupancy  # noqa: E402


def column_superset(vox, L, plane=1, u=1, v=2):
    """Superset along the ray of plane `plane` at image index (u, v) (ray axis 2 for plane 1)."""
    slots = voxels_to_dexel_slots(vox, L)
    return plane_superset(vox, slots, plane, L)[u, v, :]


def expected_true(ks):
    out = np.zeros(RES, dtype=bool)
    out[list(ks)] = True
    return out


def make_column(ks):
    vox = np.zeros((RES, RES, RES), dtype=bool)
    for k in ks:
        vox[1, 2, k] = True
    return vox


def test_alternating_L32_exact():
    vox = make_column(range(0, 32, 2))
    assert np.array_equal(column_superset(vox, 32), vox[1, 2, :])


def test_alternating_L4_fills_middle():
    # 16 single-voxel runs, n = 32 > 4: kept e0,x0,e15,x15 -> fill k = 0..30 (gap x0..e15)
    vox = make_column(range(0, 32, 2))
    got = column_superset(vox, 4)
    assert np.array_equal(got, expected_true(range(0, 31)))


def test_alternating_L1_fills_to_far_end():
    vox = make_column(range(0, 32, 2))
    assert np.array_equal(column_superset(vox, 1), expected_true(range(0, 32)))


def test_two_runs_L2_fills_gap():
    # runs 2..4 and 10, n = 4 > 2: entry of first run to exit of last run
    vox = make_column(list(range(2, 5)) + [10])
    assert np.array_equal(column_superset(vox, 2), expected_true(range(2, 11)))


def test_two_runs_L4_exact():
    # n = 4 <= 4: exact union, the empty gap 5..9 stays empty
    vox = make_column(list(range(2, 5)) + [10])
    assert np.array_equal(column_superset(vox, 4), vox[1, 2, :])


def test_parity_of_dropped_pairs_L8():
    # runs at 0, 4, 8, 12, 16 (n = 10 > 8). Kept first half: pairs (0), (4); gap 5..11 is
    # dropped middle; kept second half: pairs (12), (16). Empty space 1..3 and 13..15 stays empty.
    vox = make_column([0, 4, 8, 12, 16])
    got = column_superset(vox, 8)
    expected = expected_true([0] + list(range(4, 13)) + [16])
    assert np.array_equal(got, expected)


def test_superset_contains_original_random():
    rng = np.random.default_rng(0)
    for L in (1, 2, 4, 8, 32):
        for density in (0.05, 0.3):
            vox = rng.random((RES, RES, RES)) < density
            occ = superset_occupancy(vox, L)
            assert occ[vox].all(), (L, density)


def test_L32_exact_random():
    rng = np.random.default_rng(1)
    for density in (0.05, 0.3, 0.6):
        vox = rng.random((RES, RES, RES)) < density
        assert np.array_equal(superset_occupancy(vox, 32), vox)


def test_single_plane_L32_exact_random():
    # one plane, L = 32: the exact union of runs on that axis is the occupied set itself
    rng = np.random.default_rng(2)
    for plane in (0, 1, 2):
        for density in (0.05, 0.3, 0.6):
            vox = rng.random((RES, RES, RES)) < density
            assert np.array_equal(superset_occupancy(vox, 32, planes=(plane,)), vox), (plane, density)


def test_single_plane_superset_contains_original():
    rng = np.random.default_rng(3)
    for plane in (0, 1, 2):
        for L in (1, 2, 4):
            vox = rng.random((RES, RES, RES)) < 0.2
            occ = superset_occupancy(vox, L, planes=(plane,))
            assert occ[vox].all(), (plane, L)


def test_single_plane_is_its_own_plane_only():
    # one voxel: its run on the plane's rays is that voxel alone (n = 2 <= L, exact)
    vox = np.zeros((RES, RES, RES), dtype=bool)
    vox[3, 7, 11] = True
    for plane in (0, 1, 2):
        assert np.array_equal(superset_occupancy(vox, 2, planes=(plane,)), vox), plane
