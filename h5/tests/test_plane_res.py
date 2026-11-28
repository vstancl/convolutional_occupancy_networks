"""H5-K2-4: the encoder plane resolution as a config key of the point decoders, and the K2-4 report rule.

    .venv/bin/python -m pytest h5/tests/test_plane_res.py -q -s

a. ``model: {plane_resolution: 128}`` builds 128^2 planes; the default is 64; unknown keys and resolutions the U-Net
   cannot take are errors; the parameter set does not depend on the resolution (which is why b is needed);
b. ``load_trained`` rebuilds a checkpoint with its own model config, from ``model_cfg`` in the checkpoint or from the
   ``config.json`` next to it, and gives the same logits as the trained model; a checkpoint without either is 64;
c. the resolved config of points_mix_a_p128 differs from the one of points_mix_a in ``model`` only;
d. the K2-4 rule at its thresholds (CONFIRMED / NOT EXPLAINED / UNDECIDED, the band condition) and the margins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))
if str(H5_DIR / "scripts") not in sys.path:
    sys.path.append(str(H5_DIR / "scripts"))  # after h5/: h5/eval_rays.py must win over scripts/eval_rays.py

from models import PLANE_RES, build_model, load_trained  # noqa: E402
from train import CONFIG_DIR, load_config  # noqa: E402


def _planes(model, n=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(2, n, 3, generator=g) - 0.5
    return pts, model.encode(pts)


def test_a_resolution_key():
    m64 = build_model("points_mix_a")
    m128 = build_model("points_mix_a", {"plane_resolution": 128})
    assert PLANE_RES == 64
    _, p64 = _planes(m64)
    _, p128 = _planes(m128)
    assert {k: tuple(v.shape[-2:]) for k, v in p64.items()} == {k: (64, 64) for k in ("xz", "xy", "yz")}
    assert {k: tuple(v.shape[-2:]) for k, v in p128.items()} == {k: (128, 128) for k in ("xz", "xy", "yz")}
    # same parameter names and shapes: a 128 state dict loads into a 64 model without an error
    assert {k: v.shape for k, v in m64.state_dict().items()} == {k: v.shape for k, v in m128.state_dict().items()}
    with pytest.raises(ValueError):
        build_model("points_mix_a", {"plane_res": 128})
    with pytest.raises(ValueError):
        build_model("points_mix_a", {"plane_resolution": 100})  # not a multiple of 2^(depth-1) = 8


def test_b_load_trained(tmp_path):
    torch.manual_seed(1)
    model = build_model("points_mix_a", {"plane_resolution": 128}).eval()
    inputs, _ = _planes(model)
    query = torch.rand(2, 300, 3) - 0.5
    with torch.no_grad():
        ref = model(inputs, query)["logits"]
    dev = torch.device("cpu")
    # (1) model_cfg in the checkpoint
    ck = tmp_path / "a" / "model_it10.pt"
    ck.parent.mkdir()
    torch.save({"target": "points_mix_a", "model_cfg": {"plane_resolution": 128}, "state_dict": model.state_dict()}, ck)
    m, _ = load_trained(ck, dev, "points_mix_a")
    assert m.encoder.reso_plane == 128 and not m.training
    with torch.no_grad():
        assert torch.equal(m(inputs, query)["logits"], ref)
    # (2) no model_cfg: from config.json beside it
    ck2 = tmp_path / "b" / "model_it10.pt"
    ck2.parent.mkdir()
    torch.save({"target": "points_mix_a", "state_dict": model.state_dict()}, ck2)
    (ck2.parent / "config.json").write_text(json.dumps({"target": "points_mix_a", "model": {"plane_resolution": 128}}))
    m2, _ = load_trained(ck2, dev)
    assert m2.encoder.reso_plane == 128
    with torch.no_grad():
        assert torch.equal(m2(inputs, query)["logits"], ref)
    # (3) neither: the default resolution (older runs), and loading it there silently gives other logits
    ck3 = tmp_path / "c" / "model_it10.pt"
    ck3.parent.mkdir()
    torch.save({"target": "points_mix_a", "state_dict": model.state_dict()}, ck3)
    m3, _ = load_trained(ck3, dev)
    assert m3.encoder.reso_plane == 64
    with torch.no_grad():
        assert not torch.allclose(m3(inputs, query)["logits"], ref)
    with pytest.raises(ValueError):
        load_trained(ck, dev, "points_strong")


def test_c_resolved_config_differs_in_model_only():
    a = load_config(CONFIG_DIR / "points_mix_a.yaml", {"out_dir": "/x"})
    b = load_config(CONFIG_DIR / "points_mix_a_p128.yaml", {"out_dir": "/x"})
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"model"}
    assert a["model"] is None and b["model"] == {"plane_resolution": 128}


def test_d_k2_4_rule():
    from k1_1_report import K2_4_RULE, STEPS
    from q0_report import DK

    rule = STEPS["k2_4"]["rule"]

    def row(ord_r, m, band=True):
        return {"ord_recall": {DK: ord_r}, "M": m, "in_band": band}

    assert K2_4_RULE == {"ord": 0.291, "M": 0.031, "M_low": 0.01}
    assert rule(row(0.291, 0.031))["decision"] == "CONFIRMED"  # thresholds inclusive
    assert rule(row(0.291, 0.031, band=False))["decision"] == "UNDECIDED"
    assert rule(row(0.2909, 0.05))["decision"] == "UNDECIDED"
    assert rule(row(0.30, 0.0309))["decision"] == "UNDECIDED"
    assert rule(row(0.30, 0.01))["decision"] == "UNDECIDED"  # M < 0.01 is strict
    assert rule(row(0.30, 0.0099))["decision"] == "NOT EXPLAINED"
    assert rule(row(0.20, 0.005, band=False))["decision"] == "NOT EXPLAINED"
    r = rule(row(0.30, 0.02))
    assert r["margin_ord"] == pytest.approx(0.009) and r["margin_M"] == pytest.approx(-0.011)
    assert r["margin_M_low"] == pytest.approx(0.01)
    # the K2-3 rule is unchanged
    k23 = STEPS["k2_3"]["rule"]
    assert k23(row(0.271, 0.031))["decision"] == "CONTINUE"
    assert k23(row(0.2825, 0.0047))["decision"] == "STOP"
