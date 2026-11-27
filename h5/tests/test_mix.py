"""H5-K2'-1: mixture samplers points_mix_a / points_mix_b (``data.build_rayint_item`` with ``MIX_TARGETS``).

Half of the query points are the near-surface points of points_strong (``data.surface_half``), the rest come from
the ray-interval sampler of variant a or b. On 5 train shapes:

    .venv/bin/python -m pytest h5/tests/test_mix.py -q -s

a. shapes, dtypes, ``from_surface`` (the first ``N_QUERY // 2`` True), the ray half inside the box;
b. the surface half equals ``surface_half`` drawn from an rng in the same state (after the encoder input), and the
   surface half of points_strong is ``surface_half`` after its uniform half (the code shares the one function);
c. the ray half has 0 occupancy disagreements with ``csgbox.tree.eval_points`` (points within 1e-6 of a segment end
   not checked, as in test_rayint);
d. variant a favours short segments more than b in the ray half (as test_rayint c);
e. determinism: the same seed gives the same arrays, another seed other arrays;
f. wiring: the targets are point decoders in train, models and the configs; TrainSamples and PointShapeDataset use
   ``build_rayint_item``.
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

from csgbox.constants import BOX_HI, BOX_LO  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402
from csgbox.tree import eval_points, from_json  # noqa: E402

from data import (  # noqa: E402
    DATA_ROOT,
    MIX_TARGETS,
    N_QUERY,
    POINTCLOUD_FILE,
    POINTS_FILE,
    PointShapeDataset,
    _rayint_sample,
    build_point_item,
    build_rayint_item,
    ray_fields,
    read_occupancy_points,
    shape_dir_of,
    shape_ids,
    surface_half,
)

N_POINTS = 10000
NOISE = 0.001
N_RAYS = 512
HALF = N_QUERY // 2
END_TOL = 1e-6

pytestmark = pytest.mark.skipif(not (DATA_ROOT / "csg" / "train.lst").exists(), reason="dataset not present")


def _shapes() -> list[int]:
    ids = shape_ids("train")
    return [int(i) for i in np.random.Generator(np.random.PCG64(0)).choice(len(ids), size=5, replace=False)]


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _after_encoder_input(shape_dir: Path, rng: np.random.Generator) -> np.random.Generator:
    """Consume the encoder-input draws of build_point_item / build_rayint_item from ``rng``."""
    with np.load(shape_dir / POINTCLOUD_FILE) as pc:
        rng.choice(pc["points"].shape[0], size=N_POINTS, replace=False)
    rng.normal(0.0, NOISE, size=(N_POINTS, 3))
    return rng


@pytest.mark.parametrize("target", MIX_TARGETS)
def test_a_shapes_dtypes_and_halves(target):
    for index in _shapes():
        item = build_rayint_item(shape_dir_of("train", index), _rng(index), target, N_POINTS, NOISE)
        assert set(item) == {"inputs", "query", "occ", "from_surface"}
        assert item["inputs"].shape == (N_POINTS, 3) and item["inputs"].dtype == np.float32
        assert item["query"].shape == (N_QUERY, 3) and item["query"].dtype == np.float32
        assert item["occ"].shape == (N_QUERY,) and item["occ"].dtype == np.float32
        assert item["from_surface"].dtype == np.bool_
        assert item["from_surface"][:HALF].all() and not item["from_surface"][HALF:].any()
        assert set(np.unique(item["occ"])) <= {0.0, 1.0}
        q = item["query"][HALF:].astype(np.float64)
        assert np.all(q >= np.float32(BOX_LO)) and np.all(q <= np.float32(BOX_HI))


@pytest.mark.parametrize("target", MIX_TARGETS)
def test_b_surface_half_is_that_of_points_strong(target):
    for index in _shapes():
        shape_dir = shape_dir_of("train", index)
        item = build_rayint_item(shape_dir, _rng(index), target, N_POINTS, NOISE)
        q_ref, o_ref = surface_half(shape_dir, _after_encoder_input(shape_dir, _rng(index)), HALF)
        np.testing.assert_array_equal(item["query"][:HALF], q_ref.astype(np.float32))
        np.testing.assert_array_equal(item["occ"][:HALF], o_ref.astype(np.float32))
        # points_strong: uniform half, then surface_half from the same stream
        strong = build_point_item(shape_dir, _rng(index), "points_strong", N_POINTS, NOISE)
        rng = _after_encoder_input(shape_dir, _rng(index))
        uni_p, _ = read_occupancy_points(shape_dir / POINTS_FILE)
        rng.choice(uni_p.shape[0], size=N_QUERY - HALF, replace=False)
        q_s, o_s = surface_half(shape_dir, rng, HALF)
        np.testing.assert_array_equal(strong["query"][N_QUERY - HALF:], q_s.astype(np.float32))
        np.testing.assert_array_equal(strong["occ"][N_QUERY - HALF:], o_s.astype(np.float32))


@pytest.mark.parametrize("target", MIX_TARGETS)
def test_c_ray_half_matches_the_analytic_tree(target):
    disagree, near = 0, 0
    for index in _shapes():
        item, extra = _rayint_sample(shape_dir_of("train", index), _rng(index), target, N_POINTS, NOISE, N_QUERY,
                                     N_RAYS)
        assert extra["n_surface"] == HALF
        seg, pick = extra["segments"], extra["pick"]
        q = item["query"][HALF:]
        axis = ray_fields(extra["g"])[0][seg["row"][pick]]
        t = q[np.arange(q.shape[0]), axis].astype(np.float64)
        far = np.minimum(t - seg["start"][pick], seg["end"][pick] - t) > END_TOL
        tree = from_json((shape_dir_of("train", index) / "tree.json").read_text(encoding="utf-8"))
        inside = eval_points(tree, q.astype(np.float64))
        disagree += int(np.sum(far & (inside != (item["occ"][HALF:] > 0.5))))
        near += int(np.sum(~far))
    print(f"\n{target}: ray half disagreements {disagree} (5 shapes x {N_QUERY - HALF}), within {END_TOL:g} of an end {near}")
    assert disagree == 0


def test_d_variant_a_favours_short_segments():
    h = cell_size(128)
    share = {}
    for target in MIX_TARGETS:
        short = total = 0
        for index in _shapes():
            _item, extra = _rayint_sample(shape_dir_of("train", index), _rng(index), target, N_POINTS, NOISE,
                                          N_QUERY, N_RAYS)
            seg, pick = extra["segments"], extra["pick"]
            short += int(np.sum((seg["end"][pick] - seg["start"][pick]) < h))
            total += pick.size
        share[target] = short / total
    print(f"\nshare of ray-half points in segments shorter than h: {share}")
    assert share["points_mix_a"] > 5 * share["points_mix_b"]


@pytest.mark.parametrize("target", MIX_TARGETS)
def test_e_determinism(target):
    shape_dir = shape_dir_of("train", _shapes()[0])
    a = build_rayint_item(shape_dir, _rng(7), target, N_POINTS, NOISE)
    b = build_rayint_item(shape_dir, _rng(7), target, N_POINTS, NOISE)
    c = build_rayint_item(shape_dir, _rng(8), target, N_POINTS, NOISE)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["query"], c["query"])


def test_f_wiring():
    import train
    from eval_rays import decoding_grid
    from models import POINT_TARGET_NAMES, PointModel, build_model

    ref = train.load_config(H5_DIR / "configs" / "points_strong.yaml", {"out_dir": "/nonexistent"})
    index = _shapes()[0]
    for target in MIX_TARGETS:
        assert target in train.ALL_TARGETS and target in POINT_TARGET_NAMES
        assert isinstance(build_model(target), PointModel)
        assert decoding_grid(target) == decoding_grid("points_strong")
        cfg = train.load_config(H5_DIR / "configs" / f"{target}.yaml", {"out_dir": "/nonexistent"})
        assert {k: v for k, v in cfg.items() if k != "target"} == {k: v for k, v in ref.items() if k != "target"}
        ds = train.TrainSamples([shape_ids("train")[index]], target, 64, 1000, NOISE, 0)
        rng = np.random.default_rng(np.random.SeedSequence([0, 2, 0]))
        want = build_rayint_item(shape_dir_of("train", index), rng, target, 1000, NOISE, n_rays=64)
        got = ds[0]
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])
        pds = PointShapeDataset("train", target, n_points=1000, seed=3)
        rng = np.random.default_rng(np.random.SeedSequence([3, 0]))
        want = build_rayint_item(DATA_ROOT / "csg" / pds.ids[0], rng, target, 1000, NOISE)
        got = pds[0]
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])
