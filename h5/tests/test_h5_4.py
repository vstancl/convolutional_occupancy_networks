"""Smoke tests of the h5 training loop (H5-4), on GPU and real data.

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_h5_4.py -q -s

Each test writes its run into a temporary directory.
"""

from __future__ import annotations

import json
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

import train  # noqa: E402
from eval_rays import decoding_grid, grid_summaries, prepare_fixed  # noqa: E402
from models import H5RayModel  # noqa: E402

CONFIGS = H5_DIR / "configs"
HAS_GPU = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not HAS_GPU, reason="the training smoke tests run on GPU")

# Tolerances. Resume (test b): the GPU is not bit-deterministic (scatter in the encoder), so the two
# trajectories may differ slightly; the difference is reported and must stay within RESUME_RTOL.
RESUME_RTOL: float = 1e-3
ROUND_TRIP_ATOL: float = 1e-6  # reloaded checkpoint vs the value logged at the same iteration
SMOKE_IT: int = 200
SMOKE_VAL_EVERY: int = 100
SMOKE_VAL_SHAPES: int = 4
SMOKE_VAL_RAYS: int = 512
SMOKE_LOG_EVERY: int = 50


def _cfg(target: str, out: Path, **kw) -> dict:
    over = {"out_dir": str(out), "log_every": SMOKE_LOG_EVERY, "val_every": SMOKE_VAL_EVERY,
            "val_shapes": SMOKE_VAL_SHAPES, "val_rays": SMOKE_VAL_RAYS, "max_it": SMOKE_IT}
    over.update(kw)
    return train.load_config(CONFIGS / f"{target}.yaml", over)


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _loss_at(log: list[dict], it: int) -> float:
    return next(r["loss"] for r in log if r["iteration"] == it)


# ---------------------------------------------------------------- a. short runs, files, val grid

@pytest.mark.parametrize("target", ["hits_8", "occfrac_128"])
def test_a_short_run(tmp_path, target):
    out = tmp_path / target
    train.run(_cfg(target, out))
    log = _lines(out / "log.jsonl")
    assert [r["iteration"] for r in log] == [50, 100, 150, 200]
    first = np.mean([r["loss"] for r in log[:2]])
    last = np.mean([r["loss"] for r in log[2:]])
    assert last < first, f"{target}: loss did not decrease ({first:.4f} -> {last:.4f})"
    for name in ("model_last.pt", "model_best.pt", "model_it100.pt", "model_it200.pt", "model_it200.json",
                 "config.json"):
        assert (out / name).exists(), name
    val = _lines(out / "val.jsonl")
    assert [r["iteration"] for r in val] == [100, 200]
    n_settings = len(decoding_grid(target))
    for r in val:
        assert len(r["settings"]) == n_settings
        assert len(r["f1"]) == n_settings
        assert r["selected"] in [s["setting_name"] for s in r["settings"]]
    if target == "occfrac_128":
        assert all("occ_snap" in s["setting"] for s in val[0]["settings"])
    if target == "hits_8":
        assert all("thr" in s["setting"] for s in val[0]["settings"])
    print(f"\na {target}: loss {first:.4f} (it 50-100) -> {last:.4f} (it 150-200); "
          f"val F1 at it 200 per setting: {[round(x, 5) for x in val[1]['f1']]}")


def test_a_log_has_parts_and_memory(tmp_path):
    out = tmp_path / "occ"
    train.run(_cfg("occfrac_128", out, max_it=50, val_every=50))
    rec = _lines(out / "log.jsonl")[0]
    assert {"excess", "entropy_floor", "informative", "other"} <= set(rec["parts"])
    assert rec["peak_gib"] > 0 and rec["s_per_it"] > 0


# ---------------------------------------------------------------- b. resume

def test_b_resume_matches_uninterrupted(tmp_path):
    full = tmp_path / "full"
    train.run(_cfg("hits_8", full))
    part = tmp_path / "part"
    train.run(_cfg("hits_8", part, max_it=100))
    res = train.run(_cfg("hits_8", part, max_it=SMOKE_IT), resume=True)
    assert res["iteration"] == SMOKE_IT
    log_full = _lines(full / "log.jsonl")
    log_part = _lines(part / "log.jsonl")
    its = [r["iteration"] for r in log_part]
    assert its == [50, 100, 150, 200], "the resumed run continues the iteration counter"
    diffs = []
    for it in (50, 100, 150, 200):
        a = _loss_at(log_full, it)
        b = _loss_at(log_part, it)
        diffs.append(abs(a - b) / abs(a))
    print(f"\nb resume: max relative loss difference at it 50/100/150/200: {max(diffs):.3g} "
          f"(per point {[f'{d:.2g}' for d in diffs]})")
    assert max(diffs) <= RESUME_RTOL


# ---------------------------------------------------------------- c. checkpoint round trip

def test_c_checkpoint_round_trip(tmp_path):
    out = tmp_path / "rt"
    cfg = _cfg("hits_8", out, max_it=SMOKE_IT)
    train.run(cfg)
    val = _lines(out / "val.jsonl")[-1]
    assert val["iteration"] == SMOKE_IT
    device = torch.device("cuda")
    blob = torch.load(out / f"model_it{SMOKE_IT}.pt", map_location=device, weights_only=False)
    model = H5RayModel(blob["target"]).to(device)
    model.load_state_dict(blob["state_dict"])
    items = [prepare_fixed(cfg["val_split"], i, SMOKE_VAL_RAYS) for i in range(SMOKE_VAL_SHAPES)]
    summaries = grid_summaries(model, items, decoding_grid("hits_8"), device)
    reloaded = [s["deltas"]["0.002"]["f1"] for s in summaries]
    logged = val["f1"]
    diff = max(abs(a - b) for a, b in zip(reloaded, logged))
    print(f"\nc round trip: max |F1 reloaded - logged| over the grid = {diff:.3g}")
    assert diff <= ROUND_TRIP_ATOL
    assert blob["decoding"] == val["settings"][[s["setting_name"] for s in val["settings"]].index(val["selected"])]["setting"]


# ---------------------------------------------------------------- d. train subset with eval on train

def test_d_train_subset_eval_on_train(tmp_path):
    out = tmp_path / "subset"
    cfg = _cfg("hits_8", out, max_it=50, val_every=50, train_subset=16, eval_on_train=True)
    train.run(cfg)
    subset = json.loads((out / "train_subset.json").read_text(encoding="utf-8"))
    assert len(subset) == 16 and len(set(subset)) == 16
    assert subset == train.choose_train_shapes(cfg), "the subset is the same from the same seed"
    val = _lines(out / "val.jsonl")
    assert len(val) == 1 and val[0]["iteration"] == 50
    for s in val[0]["settings"]:
        assert s["n_shapes"] == 16
    rays_per_setting = {s["setting_name"]: s["rays"] for s in val[0]["settings"]}
    assert set(rays_per_setting.values()) == {3 * 128 * 128 * 16}, rays_per_setting
    print(f"\nd eval_on_train: 16 shapes, all rays: {3 * 128 * 128 * 16} rays per setting, "
          f"F1 per setting {[round(x, 5) for x in val[0]['f1']]}")
