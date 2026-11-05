"""[phd] Tests for phd_work/dexels.py (voxel -> dexel slot conversion)."""
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PHD_WORK = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(PHD_WORK, '..'))
sys.path.insert(0, PHD_WORK)
sys.path.insert(0, REPO)

import dexels  # noqa: E402
from dexels import RES, H, CENTRES, voxels_to_dexel_slots, dexel_slots_to_voxels  # noqa: E402

C3 = ['03001627', '04379243', '03636649']  # chair, table, lamp (3-class config)
DATA = os.path.join(REPO, 'data', 'ShapeNet')


def empty():
    return np.zeros((RES, RES, RES), dtype=bool)


def entry(i):
    return CENTRES[i] - H / 2


def exit_(i):
    return CENTRES[i] + H / 2


def test_shape_and_dtype():
    for L in (1, 2, 4, 32):
        out = voxels_to_dexel_slots(empty(), L)
        assert out.shape == (3, 2 * L, RES, RES)
        assert out.dtype == np.float32


def test_empty_grid():
    for L in (1, 4, 32):
        out = voxels_to_dexel_slots(empty(), L)
        assert not out.any()
        if L % 2 == 0:  # reconstruction needs even L
            assert dexel_slots_to_voxels(out, 0).sum() == 0


def test_single_voxel_axis_mapping_asymmetric():
    # Voxel (i, j, k) = (3, 7, 11) on an asymmetric position, L = 2 (entry and exit kept).
    vox = empty()
    vox[3, 7, 11] = True
    out = voxels_to_dexel_slots(vox, 2)
    # plane 0 'xz': rays along axis 1 (j=7); image indices (i, k) = (3, 11)
    assert out[0, 0, 3, 11] == pytest.approx(entry(7), abs=1e-7)
    assert out[0, 1, 3, 11] == pytest.approx(exit_(7), abs=1e-7)
    # plane 1 'xy': rays along axis 2 (k=11); image indices (i, j) = (3, 7)
    assert out[1, 0, 3, 7] == pytest.approx(entry(11), abs=1e-7)
    assert out[1, 1, 3, 7] == pytest.approx(exit_(11), abs=1e-7)
    # plane 2 'yz': rays along axis 0 (i=3); image indices (j, k) = (7, 11)
    assert out[2, 0, 7, 11] == pytest.approx(entry(3), abs=1e-7)
    assert out[2, 1, 7, 11] == pytest.approx(exit_(3), abs=1e-7)
    # masks set only at the hit slots, nothing else
    for p, (u, v) in enumerate([(3, 11), (3, 7), (7, 11)]):
        mask = out[p, 2:]
        assert mask[0, u, v] == 1 and mask[1, u, v] == 1
        assert mask.sum() == 2
        assert np.count_nonzero(out[p, :2]) == 2
    # the three planes reconstruct the same voxel
    for p in range(3):
        assert np.array_equal(dexel_slots_to_voxels(out, p), vox)


def test_single_voxel_depths_are_centre_minus_and_plus_half_cell():
    vox = empty()
    vox[0, 0, 0] = True
    out = voxels_to_dexel_slots(vox, 2)
    # first voxel: entry is below the lowest centre by h/2
    assert out[1, 0, 0, 0] == pytest.approx(-0.5 - H / 2, abs=1e-7)
    assert out[1, 1, 0, 0] == pytest.approx(-0.5 + H / 2, abs=1e-7)


def test_single_voxel_L1_keeps_entry_only():
    vox = empty()
    vox[5, 5, 5] = True
    out = voxels_to_dexel_slots(vox, 1)
    assert out.shape == (3, 2, RES, RES)
    assert out[0, 0, 5, 5] == pytest.approx(entry(5), abs=1e-7)
    assert out[0, 1, 5, 5] == 1  # mask of the single slot
    assert out[1, 0, 5, 5] == pytest.approx(entry(5), abs=1e-7)


def test_full_column():
    vox = empty()
    vox[2, 5, :] = True  # full column along voxel axis 2
    out = voxels_to_dexel_slots(vox, 2)
    # plane 1 'xy': one ray at (2, 5), entry at c_0 - h/2 and exit at c_31 + h/2
    assert out[1, 0, 2, 5] == pytest.approx(-0.5 - H / 2, abs=1e-7)
    assert out[1, 1, 2, 5] == pytest.approx(0.5 + H / 2, abs=1e-7)
    assert out[1, 2:, 2, 5].tolist() == [1, 1]
    # plane 0 'xz': rays along axis 1 through (2, k) each hit the single voxel j=5
    for k in range(RES):
        assert out[0, 0, 2, k] == pytest.approx(entry(5), abs=1e-7)
        assert out[0, 1, 2, k] == pytest.approx(exit_(5), abs=1e-7)
    assert out[0, 2:, 2, :].sum() == 2 * RES
    for p in range(3):
        assert np.array_equal(dexel_slots_to_voxels(out, p), vox)


def test_two_runs_L4_keeps_all():
    vox = empty()
    vox[0, 0, 2:5] = True    # run 2..4
    vox[0, 0, 10] = True     # run 10..10
    out = voxels_to_dexel_slots(vox, 4)
    assert np.allclose(out[1, :4, 0, 0], [entry(2), exit_(4), entry(10), exit_(10)], atol=1e-7)
    assert out[1, 4:, 0, 0].tolist() == [1, 1, 1, 1]


def test_two_runs_L2_keeps_first_and_last():
    vox = empty()
    vox[0, 0, 2:5] = True
    vox[0, 0, 10] = True
    out = voxels_to_dexel_slots(vox, 2)
    # first hit (entry of run 2..4) and last hit (exit of run 10)
    assert out[1, 0, 0, 0] == pytest.approx(entry(2), abs=1e-7)
    assert out[1, 1, 0, 0] == pytest.approx(exit_(10), abs=1e-7)
    assert out[1, 2:, 0, 0].tolist() == [1, 1]


def alternating_column():
    vox = empty()
    vox[1, 2, ::2] = True  # 16 single-voxel runs at k = 0, 2, ..., 30 -> 32 hits
    return vox


def test_alternating_L32_keeps_all_32_hits():
    vox = alternating_column()
    out = voxels_to_dexel_slots(vox, 32)
    expected = []
    for j in range(16):
        expected += [entry(2 * j), exit_(2 * j)]
    assert np.allclose(out[1, :32, 1, 2], expected, atol=1e-7)
    assert out[1, 32:, 1, 2].tolist() == [1] * 32
    assert np.array_equal(dexel_slots_to_voxels(out, 1), vox)


def test_alternating_L4_keeps_first_two_and_last_two():
    vox = alternating_column()
    out = voxels_to_dexel_slots(vox, 4)
    # first two hits: run 0 entry and exit; last two hits: run 15 (k=30) entry and exit
    expected = [entry(0), exit_(0), entry(30), exit_(30)]
    assert np.allclose(out[1, :4, 1, 2], expected, atol=1e-7)
    assert out[1, 4:, 1, 2].tolist() == [1, 1, 1, 1]


def test_alternating_L1_keeps_first_hit_only():
    vox = alternating_column()
    out = voxels_to_dexel_slots(vox, 1)
    assert out[1, 0, 1, 2] == pytest.approx(entry(0), abs=1e-7)
    assert out[1, 1, 1, 2] == 1


def test_invalid_L_and_shape():
    for bad in (0, 3, 5, 33, 34, -2):
        with pytest.raises(ValueError):
            voxels_to_dexel_slots(empty(), bad)
    with pytest.raises(ValueError):
        voxels_to_dexel_slots(np.zeros((8, 8, 8), dtype=bool), 2)


def random_grids():
    rng = np.random.default_rng(0)
    grids = []
    for density in (0.02, 0.1, 0.3, 0.5, 0.9):
        grids.append(rng.random((RES, RES, RES)) < density)
    # blobby grids with long runs, as in real objects
    for _ in range(5):
        g = np.zeros((RES, RES, RES), dtype=bool)
        for _ in range(20):
            lo = rng.integers(0, RES - 4, size=3)
            size = rng.integers(1, 12, size=3)
            g[lo[0]:lo[0] + size[0], lo[1]:lo[1] + size[1], lo[2]:lo[2] + size[2]] = True
        grids.append(g)
    return grids


@pytest.mark.parametrize('idx', range(10))
def test_lossless_round_trip_random(idx):
    vox = random_grids()[idx]
    slots = voxels_to_dexel_slots(vox, 32)
    for p in range(3):
        assert np.array_equal(dexel_slots_to_voxels(slots, p), vox), 'plane %d' % p


def real_model_paths(n=100, seed=0):
    paths = []
    for c in C3:
        lst = os.path.join(DATA, c, 'train.lst')
        if not os.path.exists(lst):
            return None
        with open(lst) as f:
            for m in f.read().split('\n'):
                if m:
                    paths.append(os.path.join(DATA, c, m, 'model.binvox'))
    return random.Random(seed).sample(paths, n)


def test_real_models_round_trip_L32():
    paths = real_model_paths()
    if paths is None:
        pytest.skip('ShapeNet data not present at %s' % DATA)
    from src.utils import binvox_rw
    for path in paths:
        with open(path, 'rb') as f:
            vox = binvox_rw.read_as_3d_array(f).data.astype(np.float32)
        slots = voxels_to_dexel_slots(vox, 32)
        for p in range(3):
            assert np.array_equal(dexel_slots_to_voxels(slots, p), vox > 0.5), (path, p)
