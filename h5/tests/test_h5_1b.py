"""Tests of h5 step H5-1b: the ray dataset, the mixture sampler and the on-the-fly targets.

Run from the ConvONet venv (use -s to see the timing table of test f):

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_h5_1b.py -q -s

Data: 20 test shapes chosen with numpy Generator(PCG64(0)) from csg/test.lst (as in H5-1a).
"""

from __future__ import annotations

import os
import sys
import time
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
from csgbox.constants import BOX_HI, BOX_LO  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402

from data import (  # noqa: E402
    DATA_ROOT,
    DEXELS_FILE,
    DENSE_TARGETS,
    HITS_K,
    M,
    TARGETS,
    RayShapeDataset,
    fixed_val_rays,
    read_shape_axes,
    shape_ids,
)
from rays import subset_rays  # noqa: E402
from slots import encode_slots, decode_slots  # noqa: E402

N_RAYS = 1024
N_POINTS = 10000
SAMPLE_SHAPES = 20
SELECTION_SEED = 0
# Test e: round trip of the hits targets through float32 s, against the float32 dexel decoder.
TOL_ROUNDTRIP_T = 1e-6
# Test c: sampler fraction of rays with >= 2 hits, against 0.5 + 0.5 P_all(>=2).
TOL_SAMPLER_FRACTION = 0.02
# Test f: the item budget of one DataLoader batch, and the time it must stay under.
TIMING_ITEMS = 32
TIMING_BUDGET_S = 0.15


@lru_cache(maxsize=1)
def sample_indices() -> tuple[int, ...]:
    """Positions in test.lst of the 20 sample shapes (PCG64(0), sorted)."""
    n = len(shape_ids("test"))
    rng = np.random.Generator(np.random.PCG64(SELECTION_SEED))
    return tuple(sorted(int(i) for i in rng.choice(n, size=SAMPLE_SHAPES, replace=False)))


def shape_dir(index: int) -> Path:
    return DATA_ROOT / "csg" / shape_ids("test")[index]


# ---------------------------------------------------------------------------
# a. shapes and dtypes
# ---------------------------------------------------------------------------

EXPECTED = {
    "hits_8": {"hits_s": (np.float32, (N_RAYS, 8)), "hits_mask": (np.bool_, (N_RAYS, 4))},
    "hits_16": {"hits_s": (np.float32, (N_RAYS, 16)), "hits_mask": (np.bool_, (N_RAYS, 8))},
    "occfrac_128": {"y": (np.float32, (N_RAYS, M)), "informative": (np.bool_, (N_RAYS, M))},
    "tsdf_128": {"y": (np.float32, (N_RAYS, M)), "informative": (np.bool_, (N_RAYS, M))},
    "binary_128": {"y": (np.bool_, (N_RAYS, M)), "informative": (np.bool_, (N_RAYS, M))},
}


@pytest.mark.parametrize("target", TARGETS)
def test_a_shapes_and_dtypes(target):
    ds = RayShapeDataset("test", target, n_rays=N_RAYS, n_points=N_POINTS, seed=0)
    item = ds[sample_indices()[0]]
    assert set(item) == {"inputs", "ray_axis", "ray_index", "ray_origin_uv", *EXPECTED[target]}
    assert item["inputs"].dtype == np.float32 and item["inputs"].shape == (N_POINTS, 3)
    assert item["ray_axis"].dtype == np.int64 and item["ray_axis"].shape == (N_RAYS,)
    assert item["ray_index"].dtype == np.int64 and item["ray_index"].shape == (N_RAYS,)
    assert item["ray_origin_uv"].dtype == np.float32 and item["ray_origin_uv"].shape == (N_RAYS, 2)
    for key, (dtype, shape) in EXPECTED[target].items():
        assert item[key].dtype == dtype, f"{target} {key} dtype {item[key].dtype}"
        assert item[key].shape == shape, f"{target} {key} shape {item[key].shape}"
    assert set(np.unique(item["ray_axis"])) <= {0, 1, 2}
    assert item["ray_index"].min() >= 0 and item["ray_index"].max() < M * M
    uv = item["ray_origin_uv"].astype(np.float64)
    assert uv.min() > BOX_LO and uv.max() < BOX_HI


def test_a_ray_origin_is_cell_centre():
    """ray_origin_uv is the centre of cell (col, row) = (r % N, r // N), in (y, z), (x, z) or (x, y)."""
    ds = RayShapeDataset("test", "hits_8", n_rays=N_RAYS, n_points=N_POINTS, seed=0)
    item = ds[sample_indices()[0]]
    h = cell_size(M)
    col = item["ray_index"] % M
    row = item["ray_index"] // M
    expect = np.stack([BOX_LO + (col + 0.5) * h, BOX_LO + (row + 0.5) * h], axis=1)
    assert np.allclose(item["ray_origin_uv"], expect, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# b. determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["hits_8", "tsdf_128"])
def test_b_same_seed_same_items(target):
    a = RayShapeDataset("test", target, seed=123)
    b = RayShapeDataset("test", target, seed=123)
    for index in sample_indices()[:3]:
        x, y = a[index], b[index]
        assert set(x) == set(y)
        for key in x:
            assert np.array_equal(x[key], y[key]), f"{key} differs for index {index}"
        again = a[index]  # a second call on the same dataset gives the same item too
        for key in x:
            assert np.array_equal(x[key], again[key])


def test_b_different_index_different_rays():
    ds = RayShapeDataset("test", "hits_8", seed=7)
    i0, i1 = sample_indices()[0], sample_indices()[1]
    r0 = ds[i0]["ray_index"], ds[i0]["ray_axis"]
    r1 = ds[i1]["ray_index"], ds[i1]["ray_axis"]
    assert not (np.array_equal(r0[0], r1[0]) and np.array_equal(r0[1], r1[1]))


def test_b_different_seed_different_rays():
    i = sample_indices()[0]
    a = RayShapeDataset("test", "hits_8", seed=1)[i]["ray_index"]
    b = RayShapeDataset("test", "hits_8", seed=2)[i]["ray_index"]
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# c. sampler: fraction of rays with >= 2 hits
# ---------------------------------------------------------------------------


def test_c_sampler_fraction_of_multi_hit_rays():
    ds = RayShapeDataset("test", "hits_8", n_rays=N_RAYS, n_points=N_POINTS, seed=0)
    hits_multi = 0
    total = 0
    p_all = []
    for index in sample_indices():
        axes = read_shape_axes(shape_dir(index))
        counts_all = np.concatenate([np.diff(a.offsets) for a in axes])
        p_all.append(np.mean(counts_all >= 2))
        item = ds[index]
        counts = np.empty(N_RAYS, dtype=np.int64)
        for a in range(3):
            rows = item["ray_axis"] == a
            counts[rows] = np.diff(axes[a].offsets)[item["ray_index"][rows]]
        hits_multi += int(np.sum(counts >= 2))
        total += N_RAYS
    frac = hits_multi / total
    expected = 0.5 + 0.5 * float(np.mean(p_all))
    print(f"\nsampler: fraction >= 2 hits {frac:.4f}, expected 0.5 + 0.5 P_all = {expected:.4f}")
    assert abs(frac - expected) <= TOL_SAMPLER_FRACTION


# ---------------------------------------------------------------------------
# d. targets equal the rows of the full-shape encodings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_d_targets_equal_full_encoding_rows(target):
    ds = RayShapeDataset("test", target, n_rays=N_RAYS, n_points=N_POINTS, seed=0)
    h = cell_size(M)
    for index in sample_indices()[:10]:
        item = ds[index]
        axes = read_shape_axes(shape_dir(index))
        for a in range(3):
            rows = np.flatnonzero(item["ray_axis"] == a)
            if rows.size == 0:
                continue
            r = item["ray_index"][rows]
            if target in HITS_K:
                K = HITS_K[target]
                t_full, m_full = encode_slots(axes[a], K)
                exp_s = ((t_full - BOX_LO) / h).astype(np.float32)[r]
                assert np.array_equal(item["hits_s"][rows], exp_s), f"{index} axis {a} s"
                assert np.array_equal(item["hits_mask"][rows], m_full[r]), f"{index} axis {a} mask"
            elif target == "occfrac_128":
                exp = ceil.occ_fraction(axes[a], M).astype(np.float32)[r]
                assert np.array_equal(item["y"][rows], exp), f"{index} axis {a} occfrac"
            elif target == "tsdf_128":
                exp = ceil.tsdf_values(axes[a], M)[0][r]
                assert np.array_equal(item["y"][rows], exp), f"{index} axis {a} tsdf"
            else:
                exp = ceil.binary_occupancy(axes[a], M)[r]
                assert np.array_equal(item["y"][rows], exp), f"{index} axis {a} binary"


# ---------------------------------------------------------------------------
# e. encode -> decode round trip of the hits targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", list(HITS_K))
def test_e_hits_round_trip(target):
    K = HITS_K[target]
    h = cell_size(M)
    ds = RayShapeDataset("test", target, n_rays=N_RAYS, n_points=N_POINTS, seed=0)
    worst = 0.0
    for index in sample_indices():
        item = ds[index]
        axes = read_shape_axes(shape_dir(index))
        for a in range(3):
            rows = np.flatnonzero(item["ray_axis"] == a)
            if rows.size == 0:
                continue
            s = item["hits_s"][rows].astype(np.float64)
            t_back = BOX_LO + s * h
            dec = decode_slots(t_back, item["hits_mask"][rows].astype(np.float64), K)
            ref = ceil.decode_hits(subset_rays(axes[a], item["ray_index"][rows]), 4 * K)
            assert np.array_equal(dec.offsets, ref.offsets), f"{index} axis {a}: hit counts differ"
            assert np.array_equal(dec.enter, ref.enter), f"{index} axis {a}: enter differs"
            if dec.t.size:
                worst = max(worst, float(np.max(np.abs(dec.t - ref.t))))
    print(f"\n{target}: max |dt| after round trip {worst:.3e}")
    assert worst <= TOL_ROUNDTRIP_T


# ---------------------------------------------------------------------------
# validation rays
# ---------------------------------------------------------------------------


def test_fixed_val_rays_are_fixed():
    a = fixed_val_rays(3, n=4096)
    b = fixed_val_rays(3, n=4096)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert a[1].shape == (4096,) and a[0].dtype == np.int64


# ---------------------------------------------------------------------------
# f. timing and file size (measurement; prints the table, asserts only that it ran)
# ---------------------------------------------------------------------------


def test_f_timing_and_size(capsys):
    lines = []
    sizes = []
    for index in sample_indices():
        sizes.append(os.path.getsize(shape_dir(index) / DEXELS_FILE))
    size_med = float(np.median(sizes))
    for target in TARGETS:
        ds = RayShapeDataset("test", target, n_rays=N_RAYS, n_points=N_POINTS, seed=0)
        ds[sample_indices()[0]]  # warm-up: file cache and imports
        times = []
        for index in list(sample_indices()) + list(sample_indices())[: TIMING_ITEMS - SAMPLE_SHAPES]:
            t0 = time.perf_counter()
            ds[index]
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        workers = int(np.ceil(TIMING_ITEMS * med / TIMING_BUDGET_S))
        lines.append(f"{target:12s} median {med * 1e3:7.2f} ms/item  workers for 32 items < 0.15 s: {workers}")
    lines.append(f"dexels_128.npz median size {size_med / 1e6:.2f} MB (20 test shapes)")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    assert len(lines) == len(TARGETS) + 1
