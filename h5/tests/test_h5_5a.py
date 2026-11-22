"""Tests of h5 step H5-5a: point decoders and the ray protocol.

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_h5_5a.py -q -s

Test a compares with csgbox's gate (``points_512``) through the csgbox venv (``gate_reference.py``), as in
H5-3. Tests c and d need a GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.ceilings import evaluate_axis, gt_hits  # noqa: E402
from csgbox.constants import BOX_HI, BOX_LO, GEN_CHECK_N  # noqa: E402
from csgbox.grid import cell_size, ray_grid  # noqa: E402
from csgbox.tree import eval_points, from_json  # noqa: E402

import train  # noqa: E402
from data import DATA_ROOT, POINT_TARGETS, PointShapeDataset, build_point_item, shape_ids  # noqa: E402
from eval_points import ray_geometry, ray_protocol  # noqa: E402
from eval_rays import decoding_grid, grid_summaries, prepare_fixed  # noqa: E402
from models import build_model  # noqa: E402
from rays import subset_rays  # noqa: E402

CSG_ROOT = Path("/home/wish/PhD/csgbox")
CSG_PY = CSG_ROOT / ".venv" / "bin" / "python"
GATE_REF = H5_DIR / "tests" / "gate_reference.py"
CONFIGS = H5_DIR / "configs"
H = cell_size(GEN_CHECK_N)
ORACLE_N: int = 512  # task spec: N = 512 samples per ray, iso 0.5
ORACLE_ISO: float = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_GPU = torch.cuda.is_available()


def _ten_test_shapes() -> list[str]:
    ids = shape_ids("test")
    rng = np.random.Generator(np.random.PCG64(0))
    return [ids[i] for i in rng.choice(len(ids), size=10, replace=False)]


SHAPES = _ten_test_shapes()


def _oracle(tree):
    """The analytic occupancy of ``tree.json`` as a probability (1 inside, 0 outside), float64."""

    def fn(points: torch.Tensor) -> torch.Tensor:
        occ = eval_points(tree, points[0].cpu().numpy())
        return torch.as_tensor(occ.astype(np.float64), dtype=torch.float64)[None]

    return fn


@pytest.fixture(scope="session")
def gate_points(tmp_path_factory):
    """Gate ``points_512`` (csgbox venv) for the 10 test shapes: per-axis evaluate_axis dicts and decoded hits."""
    if not CSG_PY.exists():
        pytest.skip(f"csgbox venv not found at {CSG_PY}")
    out = tmp_path_factory.mktemp("gate_pts") / "ref"
    dirs = [str(DATA_ROOT / "csg" / s) for s in SHAPES]
    subprocess.run([str(CSG_PY), str(GATE_REF), str(CSG_ROOT), str(out), *dirs], check=True)
    doc = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    return doc, np.load(out.with_suffix(".npz"))


# ---------------------------------------------------------------- a. oracle versus the gate's points_512

def test_a_ray_geometry_is_the_gate_grid():
    for a in range(3):
        origins, dirs = ray_geometry(np.full(GEN_CHECK_N ** 2, a), np.arange(GEN_CHECK_N ** 2))
        g_o, g_d = ray_grid(a, GEN_CHECK_N)
        assert np.array_equal(origins, g_o) and np.array_equal(dirs, g_d)


def test_a_oracle_matches_gate_points_512(gate_points):
    doc, npz = gate_points
    total_rays = total_diff = total_boundary = 0
    rep_fields = []
    for shape in SHAPES:
        tree = from_json((DATA_ROOT / "csg" / shape / "tree.json").read_text(encoding="utf-8"))
        for a in range(3):
            gt = gt_hits(tree, a, GEN_CHECK_N)
            n = gt.n_rays
            origins, dirs = ray_geometry(np.full(n, a), np.arange(n))
            dec = ray_protocol(_oracle(tree), origins, dirs, [ORACLE_ISO], n=ORACLE_N,
                               dtype=torch.float64)[0]
            off = npz[f"points_512|{shape}|{a}|offsets"]
            gate_counts = np.diff(off)
            ours = np.diff(dec.offsets)
            # the protocol's boundary hits: +1 enter at BOX_LO if the first sample is inside, +1 exit at BOX_HI
            # if the last sample is inside. Gate's decode_points has neither.
            z_first = BOX_LO + 0.5 * (BOX_HI - BOX_LO) / ORACLE_N
            z_last = BOX_HI - 0.5 * (BOX_HI - BOX_LO) / ORACLE_N
            p_first = origins + z_first * dirs
            p_last = origins + z_last * dirs
            first_in = eval_points(tree, p_first).astype(np.int64)
            last_in = eval_points(tree, p_last).astype(np.int64)
            assert np.array_equal(ours, gate_counts + first_in + last_in), (shape, a)
            total_rays += n
            total_diff += int(np.count_nonzero(ours != gate_counts))
            total_boundary += int(np.count_nonzero(first_in | last_in))
            # evaluate fields: gate vs ours with the boundary hits removed (only the documented rule differs)
            keep_t = (dec.t != BOX_LO) & (dec.t != BOX_HI)
            strip = _strip(dec, keep_t, n)
            res = evaluate_axis(gt, strip, a, tuple(float(d) for d in doc["deltas"]), H)
            ref = doc["axis_eval"]["points_512"][shape][a]
            for key in doc["deltas"]:
                for field in ("matched", "dt_n"):
                    if res[key][field] != ref[key][field]:
                        rep_fields.append((shape, a, key, field, res[key][field], ref[key][field]))
            for field in ("gt_hits", "rays"):
                if res[field] != ref[field]:
                    rep_fields.append((shape, a, field))
    print(f"\na oracle vs gate points_512 (10 shapes, 3 axes, N = {ORACLE_N}, iso {ORACLE_ISO}): "
          f"{total_rays} rays; rays whose hit count differs from the gate: {total_diff} "
          f"(all explained by the boundary hits: {total_boundary} rays have a sample inside at a box end); "
          f"evaluate fields with boundary hits removed: {len(rep_fields)} mismatches")
    assert total_diff == total_boundary  # a ray differs iff a box end is inside (the hit added by the protocol)
    assert not rep_fields, rep_fields[:10]


def _strip(dec, keep_t: np.ndarray, n: int):
    """``RayHits`` of ``dec`` without the hits at the box ends (the protocol's boundary convention)."""
    from csgbox.ceilings import RayHits

    ray = np.repeat(np.arange(n), np.diff(dec.offsets))
    keep = keep_t
    counts = np.bincount(ray[keep], minlength=n).astype(np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return RayHits(offsets, dec.t[keep], dec.enter[keep], None)


# ---------------------------------------------------------------- b. dataset shapes and dtypes

def test_b_point_items_shapes_dtypes_and_strong_halves():
    shape = DATA_ROOT / "csg" / shape_ids("train")[0]
    for target in POINT_TARGETS:
        it = build_point_item(shape, np.random.default_rng(0), target, 10000, 0.001)
        assert it["inputs"].shape == (10000, 3) and it["inputs"].dtype == np.float32
        assert it["query"].shape == (2048, 3) and it["query"].dtype == np.float32
        assert it["occ"].shape == (2048,) and it["occ"].dtype == np.float32
        assert set(np.unique(it["occ"])) <= {0.0, 1.0}
        if target == "points_strong":
            assert int(it["from_surface"].sum()) == 1024
            assert int((~it["from_surface"]).sum()) == 1024
        else:
            assert not it["from_surface"].any()
    ds = PointShapeDataset("train", "points_strong", seed=0)
    item = ds[0]
    assert item["query"].shape == (2048, 3) and len(ds) == len(shape_ids("train"))
    again = PointShapeDataset("train", "points_strong", seed=0)[0]
    assert np.array_equal(item["query"], again["query"]), "a fixed seed gives identical items"


# ---------------------------------------------------------------- c. short training run of points_strong

def _cfg(out: Path, **kw) -> dict:
    over = {"out_dir": str(out), "log_every": 50, "val_every": 100, "val_shapes": 4, "val_rays": 512,
            "max_it": 200}
    over.update(kw)
    return train.load_config(CONFIGS / "points_strong.yaml", over)


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.mark.skipif(not HAS_GPU, reason="GPU")
def test_c_points_strong_short_run(tmp_path):
    out = tmp_path / "ps"
    train.run(_cfg(out))
    log = _lines(out / "log.jsonl")
    assert [r["iteration"] for r in log] == [50, 100, 150, 200]
    first = np.mean([r["loss"] for r in log[:2]])
    last = np.mean([r["loss"] for r in log[2:]])
    assert last < first, f"loss did not decrease ({first:.2f} -> {last:.2f})"
    for name in ("model_last.pt", "model_best.pt", "model_it200.pt", "model_it200.json", "config.json"):
        assert (out / name).exists(), name
    val = _lines(out / "val.jsonl")
    assert [r["iteration"] for r in val] == [100, 200]
    for r in val:
        names = [s["setting_name"] for s in r["settings"]]
        assert names == ["iso=0.2", "iso=0.5"], names
    print(f"\nc points_strong 200 it: BCE sum-mean {first:.1f} (it 50-100) -> {last:.1f} (it 150-200); "
          f"val F1 at it 200: iso 0.2 {val[1]['f1'][0]:.5f}, iso 0.5 {val[1]['f1'][1]:.5f}")


# ---------------------------------------------------------------- d. evaluation is deterministic

@pytest.mark.skipif(not HAS_GPU, reason="GPU")
def test_d_two_validations_identical(tmp_path):
    out = tmp_path / "det"
    cfg = _cfg(out, max_it=100, val_every=100)
    train.run(cfg)
    blob = torch.load(out / "model_it100.pt", map_location=DEVICE, weights_only=False)
    model = build_model("points_strong").to(DEVICE)
    model.load_state_dict(blob["state_dict"])
    items = [prepare_fixed(cfg["val_split"], i, 512) for i in range(4)]
    settings = decoding_grid("points_strong")
    f_a = [s["deltas"]["0.002"]["f1"] for s in grid_summaries(model, items, settings, DEVICE)]
    f_b = [s["deltas"]["0.002"]["f1"] for s in grid_summaries(model, items, settings, DEVICE)]
    print(f"\nd two validations of the same checkpoint: F1 {f_a} and {f_b}")
    assert f_a == f_b


# ---------------------------------------------------------------- a2. the protocol's own rules (synthetic)

def test_a2_boundary_and_bisection_rules():
    n = 64
    origins, dirs = ray_geometry(np.zeros(4, dtype=np.int64), np.array([0, 1, 2, 3]))
    # always inside: one enter at BOX_LO and one exit at BOX_HI per ray (the protocol's boundary convention)
    inside = lambda p: torch.ones((1, p.shape[1]), dtype=torch.float64)  # noqa: E731
    dec = ray_protocol(inside, origins, dirs, [0.5], n=n, dtype=torch.float64)[0]
    assert np.array_equal(np.diff(dec.offsets), [2, 2, 2, 2])
    assert np.array_equal(dec.enter, [True, False] * 4)
    assert np.allclose(dec.t, [BOX_LO, BOX_HI] * 4, atol=0, rtol=0)
    # half-space x > c: one enter at c (within the bisection bracket)
    c = 0.1234
    half = lambda p: (p[..., 0] > c).double()  # noqa: E731  (x-rays: the ray coordinate is x)
    dec2 = ray_protocol(half, origins, dirs, [0.5], n=n, dtype=torch.float64)[0]
    # outside before c, inside after it up to the box end: the enter at c and the boundary exit at BOX_HI
    assert np.array_equal(np.diff(dec2.offsets), [2, 2, 2, 2])
    assert np.array_equal(dec2.enter, [True, False] * 4)
    assert np.all(dec2.t[1::2] == BOX_HI)
    dec2_t = dec2.t[0::2]
    h_n = (BOX_HI - BOX_LO) / n
    bracket = h_n / 2 ** 8  # width after 8 bisection steps
    assert np.all(np.abs(dec2_t - c) <= bracket), (dec2_t, c, bracket)
