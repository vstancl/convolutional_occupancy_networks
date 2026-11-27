"""H5-K2-1a: ray-interval query sampler (``data.build_rayint_item``), on 5 train shapes, both variants.

Run with ``-s`` to see the printed counts and shares:

    .venv/bin/python -m pytest h5/tests/test_rayint.py -q -s
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.constants import BOX_HI, BOX_LO  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402
from csgbox.tree import eval_points, from_json  # noqa: E402

import data  # noqa: E402
from data import (  # noqa: E402
    DATA_ROOT,
    N_QUERY,
    RAYINT_TARGETS,
    PointShapeDataset,
    _rayint_sample,
    build_point_item,
    build_rayint_item,
    ray_fields,
    shape_dir_of,
    shape_ids,
)

N_POINTS = 10000
NOISE = 0.001
N_RAYS = 512
END_TOL = 1e-6  # points closer than this (in t) to an end of their segment are not checked against the tree

pytestmark = pytest.mark.skipif(not (DATA_ROOT / "csg" / "train.lst").exists(), reason="dataset not present")


def _shapes() -> list[int]:
    ids = shape_ids("train")
    return [int(i) for i in np.random.Generator(np.random.PCG64(0)).choice(len(ids), size=5, replace=False)]


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _sample(index: int, target: str, seed: int):
    return _rayint_sample(shape_dir_of("train", index), _rng(seed), target, N_POINTS, NOISE, N_QUERY, N_RAYS)


def _ray_t(item: dict, extra: dict) -> np.ndarray:
    """The coordinate of every query point on its ray axis (from the float32 query), float64."""
    seg, pick = extra["segments"], extra["pick"]
    axis = ray_fields(extra["g"])[0][seg["row"][pick]]
    return item["query"][np.arange(item["query"].shape[0]), axis].astype(np.float64)


@pytest.mark.parametrize("target", RAYINT_TARGETS)
def test_a_shapes_dtypes_and_box(target):
    for index in _shapes():
        item = build_rayint_item(shape_dir_of("train", index), _rng(index), target, N_POINTS, NOISE)
        assert set(item) == {"inputs", "query", "occ", "from_surface"}
        assert item["inputs"].shape == (N_POINTS, 3) and item["inputs"].dtype == np.float32
        assert item["query"].shape == (N_QUERY, 3) and item["query"].dtype == np.float32
        assert item["occ"].shape == (N_QUERY,) and item["occ"].dtype == np.float32
        assert item["from_surface"].shape == (N_QUERY,) and item["from_surface"].dtype == np.bool_
        assert not item["from_surface"].any()
        assert set(np.unique(item["occ"])) <= {0.0, 1.0}
        q = item["query"].astype(np.float64)
        assert np.all(q >= np.float32(BOX_LO)) and np.all(q <= np.float32(BOX_HI))


@pytest.mark.parametrize("target", RAYINT_TARGETS)
def test_b_occupancy_matches_the_analytic_tree(target):
    disagree, near = 0, 0
    for index in _shapes():
        item, extra = _sample(index, target, index)
        tree = from_json((shape_dir_of("train", index) / "tree.json").read_text(encoding="utf-8"))
        seg, pick = extra["segments"], extra["pick"]
        t = _ray_t(item, extra)
        far = np.minimum(t - seg["start"][pick], seg["end"][pick] - t) > END_TOL
        inside = eval_points(tree, item["query"].astype(np.float64))
        disagree += int(np.sum(far & (inside != (item["occ"] > 0.5))))
        near += int(np.sum(~far))
    print(f"\n{target}: disagreements {disagree} (5 shapes x {N_QUERY} points), within {END_TOL:g} of an end {near}")
    assert disagree == 0


def test_c_variant_a_favours_short_segments():
    h = cell_size(128)
    index = _shapes()[0]
    shares = {}
    for target in RAYINT_TARGETS:
        short = total = 0
        for draw in range(200):
            item, extra = _sample(index, target, 1000 + draw)
            seg, pick = extra["segments"], extra["pick"]
            short += int(np.sum((seg["end"][pick] - seg["start"][pick]) < h))
            total += pick.size
        shares[target] = short / total
    print(f"\nshare of points in segments shorter than h(128), shape {shape_ids('train')[index]}, 200 draws: "
          + ", ".join(f"{k} {v:.4f}" for k, v in shares.items()))
    a, b, c = shares["points_rayint_a"], shares["points_rayint_b"], shares["points_rayint_c"]
    assert a > 0 and a > 5 * b  # "clearly larger": at least five times the share of the control
    assert c > 2 * a  # H5-K2-1c: the thin-focused variant c clearly above variant a


def test_d_variant_b_mean_occupancy_matches_the_rays():
    for index in _shapes():
        item, extra = _sample(index, "points_rayint_b", index)
        seg = extra["segments"]
        length = seg["end"] - seg["start"]
        exact = float(np.sum(length * seg["occ"]) / np.sum(length))
        got = float(item["occ"].mean())
        print(f"\nshape {shape_ids('train')[index]}: mean occ of the points {got:.4f}, along the rays {exact:.4f}")
        assert abs(got - exact) <= 0.02


def _digest(item: dict) -> str:
    h = hashlib.sha256()
    for key in ("inputs", "query", "occ", "from_surface"):
        h.update(np.ascontiguousarray(item[key]).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("target", RAYINT_TARGETS)
def test_e_same_seed_same_arrays(target):
    index = _shapes()[0]
    a = build_rayint_item(shape_dir_of("train", index), _rng(7), target, N_POINTS, NOISE)
    b = build_rayint_item(shape_dir_of("train", index), _rng(7), target, N_POINTS, NOISE)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    ds = PointShapeDataset("train", target, seed=3)
    np.testing.assert_array_equal(ds[index]["query"], ds[index]["query"])


# sha256 of build_point_item(train shape 0, Generator(PCG64(123)), target, 10000, 0.001), computed before K2-1a
POINT_ITEM_DIGEST = {
    "points_std": "33fe5ca534228f7bbfffa9a65b45f875a21176edf7e4478c7b3fbd29051b79ce",
    "points_strong": "28460511332d4f88c0eb3b97b655aa2feb8af8352b726a4ba677029dd961c66b",
}


# sha256 of build_rayint_item(train shape 0, Generator(PCG64(123)), target, 10000, 0.001), computed before K2-1c
RAYINT_ITEM_DIGEST = {
    "points_rayint_a": "f11b7e378bdebed99e305b5bec33980afa7b973d28a5f42b671d928a19789fd0",
    "points_rayint_b": "6ebfe7b8f0945453778c38193f2e079a2d07ef4db1b66b371f6ee831bd06c4a9",
}


@pytest.mark.parametrize("target", sorted(RAYINT_ITEM_DIGEST))
def test_e_rayint_a_and_b_unchanged_by_variant_c(target):
    item = build_rayint_item(shape_dir_of("train", 0), _rng(123), target, N_POINTS, NOISE)
    assert _digest(item) == RAYINT_ITEM_DIGEST[target]


@pytest.mark.parametrize("target", sorted(POINT_ITEM_DIGEST))
def test_e_build_point_item_unchanged(target):
    item = build_point_item(shape_dir_of("train", 0), _rng(123), target, N_POINTS, NOISE)
    assert _digest(item) == POINT_ITEM_DIGEST[target]


def test_encoder_input_is_that_of_build_point_item():
    index = _shapes()[0]
    p = build_point_item(shape_dir_of("train", index), _rng(5), "points_strong", N_POINTS, NOISE)
    r = build_rayint_item(shape_dir_of("train", index), _rng(5), "points_rayint_a", N_POINTS, NOISE)
    np.testing.assert_array_equal(p["inputs"], r["inputs"])
    assert data.RAYINT_N_RAYS == N_RAYS


# ---------------------------------------------------------------- H5-K2-1b: the rayint targets are point decoders

def test_rayint_targets_are_point_decoders_everywhere():
    import train
    from eval_rays import decoding_grid
    from models import POINT_TARGET_NAMES, PointModel, build_model

    assert POINT_TARGET_NAMES == data.POINT_TARGETS + RAYINT_TARGETS + data.MIX_TARGETS  # K2'-1 adds the mixtures
    for target in RAYINT_TARGETS:
        assert target in train.ALL_TARGETS
        assert isinstance(build_model(target), PointModel)
        assert decoding_grid(target) == decoding_grid("points_strong")
        cfg = train.load_config(H5_DIR / "configs" / f"{target}.yaml", {"out_dir": "/nonexistent"})
        ref = train.load_config(H5_DIR / "configs" / "points_strong.yaml", {"out_dir": "/nonexistent"})
        assert {k: v for k, v in cfg.items() if k != "target"} == {k: v for k, v in ref.items() if k != "target"}


def test_train_samples_use_the_rayint_builder():
    import train

    index = _shapes()[0]
    ds = train.TrainSamples([shape_ids("train")[index]], "points_rayint_a", 64, 1000, NOISE, 0)
    item = ds[0]
    rng = np.random.default_rng(np.random.SeedSequence([0, 2, 0]))
    ref = build_rayint_item(shape_dir_of("train", index), rng, "points_rayint_a", 1000, NOISE, n_rays=64)
    for key in ref:
        np.testing.assert_array_equal(item[key], ref[key])
