"""Tests of h5 step H5-2: ray trunk, heads, losses; micro-overfit; time and memory report.

Run from the ConvONet venv (use -s to see the micro-overfit losses and the timing table):

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_h5_2.py -q -s

Tests f and g need a GPU and the dataset under ~/PhD/data/csgbox/v1; the others run on CPU.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.constants import BOX_LO  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402

from src.conv_onet.models.decoder import LocalDecoder  # noqa: E402

import models  # noqa: E402
from models import (  # noqa: E402
    COL_AXIS,
    H5RayModel,
    N_SAMPLES,
    PLANES,
    PLANE_OF_AXIS,
    PLANE_INDEX_OF_AXIS,
    PlaneHitsHead,
    RayTrunk,
    n_params,
    perpendicular_plane_feature,
    ray_sample_points,
    summed_plane_feature,
)
from losses import POS_WEIGHT_CAP, POS_WEIGHT_K2, dense_loss, hits_loss  # noqa: E402

# Test tolerances. The float32 world points carry about 6e-8 relative error on values near 0.55.
ATOL_POINTS: float = 1e-6  # float32 sample coordinates vs float64 reference
ATOL_PARITY: float = 1e-6  # summed plane feature vs LocalDecoder (task spec)
ATOL_LOSS: float = 1e-10  # hand-computed float64 losses
ATOL_OVERFIT_RATIO: float = 0.5  # the loss must fall by more than this fraction
OVERFIT_STEPS: int = 300  # task spec
OVERFIT_LR: float = 1e-4  # task spec
OVERFIT_RAYS: int = 1024  # task spec
TIMING_BATCH: int = 32  # task spec: 32 shapes x 1024 rays x 10000 points
TIMING_RAYS: int = 1024
TIMING_POINTS: int = 10000
TIMING_REPEATS: int = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_GPU = torch.cuda.is_available()
M = 128
H = cell_size(M)
K_TARGETS = {"hits_8": 2, "hits_16": 4}
ALL_TARGETS = ("hits_8", "hits_16", "hits_8_plane", "occfrac_128", "tsdf_128", "binary_128")


def _random_batch(B: int, R: int, N: int, gen: torch.Generator):
    inputs = (torch.rand(B, N, 3, generator=gen) - 0.5).float().to(DEVICE)
    ray_axis = torch.randint(0, 3, (B, R), generator=gen).to(DEVICE)
    uv = ((torch.rand(B, R, 2, generator=gen) - 0.5) * 1.1).float().to(DEVICE)
    return inputs, ray_axis, uv


def _random_planes(B: int, gen: torch.Generator, C: int = 32, res: int = 64):
    return {name: torch.randn(B, C, res, res, generator=gen).to(DEVICE) for name in PLANES}


# ---------------------------------------------------------------- a. ray sample points

def test_a_ray_sample_points_axes():
    gen = torch.Generator().manual_seed(1)
    B, R = 2, 50
    ray_axis = torch.randint(0, 3, (B, R), generator=gen)
    uv = ((torch.rand(B, R, 2, generator=gen) - 0.5) * 1.1).float()
    pts = ray_sample_points(ray_axis, uv).numpy()
    assert pts.shape == (B, R, N_SAMPLES, 3)
    assert pts.dtype == np.float32
    centres = BOX_LO + (2.0 * np.arange(N_SAMPLES) + 1.0) * H  # centres of cells 2m, 2m+1
    row_axis = (2, 2, 1)
    for b in range(B):
        for r in range(R):
            a = int(ray_axis[b, r])
            col, row = COL_AXIS[a], row_axis[a]
            np.testing.assert_allclose(pts[b, r, :, a], centres, atol=ATOL_POINTS)
            np.testing.assert_allclose(pts[b, r, :, col], uv[b, r, 0].item(), atol=ATOL_POINTS)
            np.testing.assert_allclose(pts[b, r, :, row], uv[b, r, 1].item(), atol=ATOL_POINTS)


def test_a_axis_order_is_col_row():
    # x-ray: (y, z); y-ray: (x, z); z-ray: (x, y). Distinct uv values make a swap visible.
    uv = torch.tensor([[[0.1, -0.2]]], dtype=torch.float32)
    for a, expect in ((0, (0.1, -0.2)), (1, (0.1, -0.2)), (2, (0.1, -0.2))):
        pts = ray_sample_points(torch.tensor([[a]]), uv)[0, 0].numpy()
        others = [i for i in range(3) if i != a]
        assert pts[0, others[0]] == pytest.approx(expect[0], abs=ATOL_POINTS)
        assert pts[0, others[1]] == pytest.approx(expect[1], abs=ATOL_POINTS)


# ---------------------------------------------------------------- b. trunk input parity

def test_b_summed_plane_feature_matches_local_decoder():
    gen = torch.Generator().manual_seed(2)
    B, T = 2, 3000
    p = ((torch.rand(B, T, 3, generator=gen) - 0.5) * 1.2).float()  # includes points outside [-0.5, 0.5]
    planes = {name: torch.randn(B, 32, 64, 64, generator=gen) for name in PLANES}
    ld = LocalDecoder(dim=3, c_dim=32, hidden_size=32)  # ConvONet's own decoder, padding 0.1, bilinear
    c_ref = 0
    for name in ("xz", "xy", "yz"):  # LocalDecoder.forward order
        c_ref = c_ref + ld.sample_plane_feature(p.clone(), planes[name], plane=name)
    c_ref = c_ref.transpose(1, 2)
    c = summed_plane_feature(p, planes)
    assert c.shape == (B, T, 32)
    assert torch.allclose(c, c_ref, atol=ATOL_PARITY)


def test_b_trunk_uses_the_same_plane_feature():
    gen = torch.Generator().manual_seed(3)
    B, R = 1, 4
    trunk = RayTrunk()
    planes = {name: torch.randn(B, 32, 64, 64, generator=gen) for name in PLANES}
    ray_axis = torch.tensor([[0, 1, 2, 0]])
    uv = torch.zeros(B, R, 2)
    pts = ray_sample_points(ray_axis, uv)  # (B, R, 64, 3)
    got = trunk(pts, planes)
    assert got.shape == (B, R, 32, N_SAMPLES)
    # manual: x = fc_p(p) + fc_c(c) on the flattened points, then the blocks, by hand
    flat = pts.reshape(B, R * N_SAMPLES, 3)
    x = trunk.fc_p(flat) + trunk.fc_c(summed_plane_feature(flat, planes))
    x = x.reshape(B, R, N_SAMPLES, 32).permute(0, 1, 3, 2).reshape(B * R, 32, N_SAMPLES)
    for blk in trunk.blocks:
        x = blk(x)
    assert torch.allclose(got.reshape(B * R, 32, N_SAMPLES), x, atol=1e-5)


# ---------------------------------------------------------------- c. shapes, ranges, dtype

@pytest.mark.parametrize("target", ALL_TARGETS)
def test_c_head_shapes(target):
    gen = torch.Generator().manual_seed(4)
    B, R, N = 2, 16, 300
    inputs, ray_axis, uv = _random_batch(B, R, N, gen)
    model = H5RayModel(target).to(DEVICE)
    out = model(inputs, ray_axis, uv)
    if target in K_TARGETS or target == "hits_8_plane":
        K = 2 if target in ("hits_8", "hits_8_plane") else 4
        assert out["s_pred"].shape == (B, R, 4 * K)
        assert out["mask_logits"].shape == (B, R, 2 * K)
        assert out["s_pred"].dtype == torch.float32
    else:
        assert out["logits"].shape == (B, R, 128)


@pytest.mark.parametrize("target", ("hits_8", "hits_16", "hits_8_plane"))
def test_c_s_pred_range_and_dtype_under_autocast(target):
    gen = torch.Generator().manual_seed(5)
    B, R, N = 2, 64, 300
    inputs, ray_axis, uv = _random_batch(B, R, N, gen)
    model = H5RayModel(target).to(DEVICE)
    amp_dtype = torch.float16 if DEVICE.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype):
        out = model(inputs, ray_axis, uv)
    s = out["s_pred"]
    assert s.dtype == torch.float32
    lo, hi = (0.0, 128.0) if target == "hits_8_plane" else (-1.0, 129.0)
    assert float(s.detach().min()) >= lo and float(s.detach().max()) <= hi


def test_c_hits_head_formula_with_zero_features():
    # Zero trunk features and zero weights: every sample gets the same softmax weight, and the
    # residual is the bias only. Then s = mean_m(2m + 1) + 2 tanh(bias_r).
    head = models.HitsHead(K=2)
    with torch.no_grad():
        for layer in (head.attn, head.resid):
            layer.weight.zero_()
            layer.bias.zero_()
        head.resid.bias[0] = 1.0
    x = torch.zeros(1, 1, 32, N_SAMPLES)
    s, _ = head(x)
    mean_cell = float(np.mean(2.0 * np.arange(N_SAMPLES) + 1.0))
    assert float(s[0, 0, 0].detach()) == pytest.approx(mean_cell + 2.0 * np.tanh(1.0), abs=1e-4)
    assert float(s[0, 0, 1].detach()) == pytest.approx(mean_cell, abs=1e-4)


def test_c_plane_head_depends_on_perpendicular_plane_only():
    # Replacing one plane may change only the rays whose perpendicular plane it is:
    # 'yz' <-> x-rays (axis 0), 'xz' <-> y-rays (axis 1), 'xy' <-> z-rays (axis 2).
    gen = torch.Generator().manual_seed(6)
    B, R = 1, 60
    ray_axis = torch.randint(0, 3, (B, R), generator=gen)
    uv = ((torch.rand(B, R, 2, generator=gen) - 0.5) * 1.1).float()
    planes = {name: torch.randn(B, 32, 64, 64, generator=gen) for name in PLANES}
    head = PlaneHitsHead(c_dim=32, K=2).eval()
    axes = ray_axis[0].numpy()
    with torch.no_grad():
        s0, _ = head(perpendicular_plane_feature(ray_axis, uv, planes))
        for name in PLANES:
            swapped = dict(planes)
            swapped[name] = torch.randn(B, 32, 64, 64, generator=gen)
            s1, _ = head(perpendicular_plane_feature(ray_axis, uv, swapped))
            differs = (s1 != s0)[0].any(dim=-1).numpy()
            own_axis = PLANE_OF_AXIS.index(name)
            assert not differs[axes != own_axis].any(), f"plane {name} changed a ray of another axis"
            assert differs[axes == own_axis].any(), f"plane {name} had no effect on its own rays"


# ---------------------------------------------------------------- d. losses on hand-made cases

def _bce_np(x, y):
    # binary cross entropy with logits, float64, numerically stable form
    return np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))


def _hits_case():
    # K = 2, R = 3 rays, slots in t order [F0 e,x, F1 e,x, B1 e,x, B0 e,x]; mask order [F0, F1, B1, B0].
    mask_tgt = np.array([[1, 0, 0, 1],   # ray 0: F0 and B0
                         [1, 1, 0, 0],   # ray 1: F0 and F1
                         [0, 0, 0, 0]], dtype=bool)  # ray 2: no interval
    s_tgt = np.zeros((3, 8))
    s_pred = np.zeros((3, 8))
    # ray 0: F0 errors (1, 3), B0 errors (2, 4); masked hits get error 100
    s_pred[0] = [1, 3, 100, 100, 100, 100, 2, 4]
    # ray 1: F0 errors (5, 7), F1 errors (2, 0); masked hits error 100
    s_pred[1] = [5, 7, 2, 0, 100, 100, 100, 100]
    # ray 2: nothing exists; large error
    s_pred[2] = [1000] * 8
    return s_pred, s_tgt, mask_tgt


def test_d_hits_loss_per_slot_average_and_masked_slots():
    s_pred, s_tgt, mask_tgt = _hits_case()
    logits = np.zeros((3, 4))
    pw = np.array(POS_WEIGHT_K2)
    t = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt)  # noqa: E731
    out = hits_loss(t(s_pred), t(logits), t(s_tgt), torch.as_tensor(mask_tgt), t(pw))
    # present hit slots: h0 (F0 e) errors (1, 5) -> 3; h1 (F0 x) (3, 7) -> 5; h2 (F1 e) (2) -> 2;
    # h3 (F1 x) (0) -> 0; h6 (B0 e) (2) -> 2; h7 (B0 x) (4) -> 4. h4, h5 not present.
    # L_t = mean over the 6 present slots = 16 / 6.
    assert float(out["L_t"]) == pytest.approx(16.0 / 6.0, abs=ATOL_LOSS)
    # BCE at logit 0 is log 2 times (pos_weight if positive else 1); mean over 3 rays x 4 slots.
    expect = 0.0
    for r in range(3):
        for j in range(4):
            y = mask_tgt[r, j]
            expect += np.log(2.0) * (pw[j] if y else 1.0)
    expect /= 12.0
    assert float(out["L_mask"]) == pytest.approx(expect, abs=ATOL_LOSS)
    assert float(out["total"]) == pytest.approx(16.0 / 6.0 + expect, abs=ATOL_LOSS)


def test_d_hits_loss_ignores_masked_targets_and_predictions():
    s_pred, s_tgt, mask_tgt = _hits_case()
    logits = np.random.default_rng(0).normal(size=(3, 4))
    pw = np.array(POS_WEIGHT_K2)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    base = hits_loss(t(s_pred), t(logits), t(s_tgt), torch.as_tensor(mask_tgt), t(pw))
    # ray 2 has no interval: its predictions and targets must not matter
    s_pred2 = s_pred.copy()
    s_pred2[2, :] = -5.0
    s_tgt2 = s_tgt.copy()
    s_tgt2[2, :] = 77.0
    out2 = hits_loss(t(s_pred2), t(logits), t(s_tgt2), torch.as_tensor(mask_tgt), t(pw))
    assert float(out2["L_t"]) == pytest.approx(float(base["L_t"]), abs=ATOL_LOSS)
    # ray 0 has no F1 and no B1 interval: the targets and predictions of hits 2..5 must not matter
    s_pred3 = s_pred.copy()
    s_pred3[0, [2, 3, 4, 5]] = -40.0
    s_tgt3 = s_tgt.copy()
    s_tgt3[0, [2, 3, 4, 5]] = 77.0
    out3 = hits_loss(t(s_pred3), t(logits), t(s_tgt3), torch.as_tensor(mask_tgt), t(pw))
    assert float(out3["L_t"]) == pytest.approx(float(base["L_t"]), abs=ATOL_LOSS)


def test_d_hits_loss_pos_weight_cap_and_lambda():
    s_pred, s_tgt, mask_tgt = _hits_case()
    logits = np.zeros((3, 4))
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    big = hits_loss(t(s_pred), t(logits), t(s_tgt), torch.as_tensor(mask_tgt), t([30.0, 1.0, 1.0, 1.0]))
    capped = hits_loss(t(s_pred), t(logits), t(s_tgt), torch.as_tensor(mask_tgt), t([POS_WEIGHT_CAP, 1.0, 1.0, 1.0]))
    assert float(big["L_mask"]) == pytest.approx(float(capped["L_mask"]), abs=ATOL_LOSS)
    under = hits_loss(t(s_pred), t(logits), t(s_tgt), torch.as_tensor(mask_tgt), t([9.0, 1.0, 1.0, 1.0]))
    assert float(under["L_mask"]) != pytest.approx(float(capped["L_mask"]), abs=1e-6)
    half = hits_loss(t(s_pred), t(logits), t(s_tgt), torch.as_tensor(mask_tgt), t(POS_WEIGHT_K2), lam=0.5)
    full = hits_loss(t(s_pred), t(logits), t(s_tgt), torch.as_tensor(mask_tgt), t(POS_WEIGHT_K2), lam=1.0)
    assert float(half["total"]) == pytest.approx(float(half["L_t"]) + 0.5 * float(half["L_mask"]), abs=ATOL_LOSS)
    assert float(full["total"]) > float(half["total"])


def test_d_hits_loss_empty_batch_has_no_hit_term():
    mask_tgt = np.zeros((2, 4), dtype=bool)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    out = hits_loss(t(np.ones((2, 8))), t(np.zeros((2, 4))), t(np.zeros((2, 8))), torch.as_tensor(mask_tgt),
                    t(POS_WEIGHT_K2))
    assert float(out["L_t"]) == 0.0


def _dense_case():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(1, 128))
    inf = np.zeros((1, 128), dtype=bool)
    inf[0, :10] = True
    return logits, inf


def test_d_dense_loss_occfrac_formula():
    logits, inf = _dense_case()
    y = np.zeros((1, 128))
    y[0, :10] = 0.3  # soft targets in the informative cells
    y[0, 10:] = 1.0  # saturated cells elsewhere
    t = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt)  # noqa: E731
    out = dense_loss(t(logits), t(y), torch.as_tensor(inf), "occfrac_128")
    l_inf = _bce_np(logits[0, :10], 0.3).mean()
    l_oth = _bce_np(logits[0, 10:], 1.0).mean()
    assert float(out["total"]) == pytest.approx(0.5 * l_inf + 0.5 * l_oth, abs=ATOL_LOSS)
    assert float(out["informative"]) == pytest.approx(l_inf, abs=ATOL_LOSS)
    assert float(out["other"]) == pytest.approx(l_oth, abs=ATOL_LOSS)


def test_d_dense_loss_binary_formula():
    logits, inf = _dense_case()
    y = np.zeros((1, 128), dtype=bool)
    y[0, ::3] = True
    t = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt)  # noqa: E731
    out = dense_loss(t(logits), torch.as_tensor(y), torch.as_tensor(inf), "binary_128")
    yf = y.astype(float)
    l_inf = _bce_np(logits[0, :10], yf[0, :10]).mean()
    l_oth = _bce_np(logits[0, 10:], yf[0, 10:]).mean()
    assert float(out["total"]) == pytest.approx(0.5 * l_inf + 0.5 * l_oth, abs=ATOL_LOSS)


def test_d_dense_loss_tsdf_formula_in_units_of_h():
    logits, inf = _dense_case()
    rng = np.random.default_rng(4)
    y = rng.uniform(-2.0, 2.0, size=(1, 128)) * H  # stored values in world units
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    out = dense_loss(t(logits), t(y), torch.as_tensor(inf), "tsdf_128")
    pred_cells = 2.5 * np.tanh(logits[0])  # 2.5 h tanh(x) in units of h
    err = np.abs(pred_cells - y[0] / H)
    expect = 0.5 * err[:10].mean() + 0.5 * err[10:].mean()
    assert float(out["total"]) == pytest.approx(expect, abs=1e-9)


def test_d_dense_loss_empty_set_is_skipped():
    logits, _ = _dense_case()
    y = np.full((1, 128), 0.5)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    none_inf = torch.zeros((1, 128), dtype=torch.bool)
    out = dense_loss(t(logits), t(y), none_inf, "occfrac_128")
    assert float(out["informative"]) == 0.0
    assert float(out["total"]) == pytest.approx(0.5 * float(out["other"]), abs=ATOL_LOSS)
    all_inf = torch.ones((1, 128), dtype=torch.bool)
    out2 = dense_loss(t(logits), t(y), all_inf, "occfrac_128")
    assert float(out2["total"]) == pytest.approx(0.5 * float(out2["informative"]), abs=ATOL_LOSS)


# ---------------------------------------------------------------- e. gradients reach the encoder

@pytest.mark.parametrize("target", ALL_TARGETS)
def test_e_gradients_reach_encoder(target):
    gen = torch.Generator().manual_seed(7)
    B, R, N = 2, 8, 300
    inputs, ray_axis, uv = _random_batch(B, R, N, gen)
    model = H5RayModel(target).to(DEVICE)
    out = model(inputs, ray_axis, uv)
    if "s_pred" in out:
        loss = out["s_pred"].sum() + out["mask_logits"].sum()
    else:
        loss = out["logits"].sum()
    loss.backward()
    enc = list(model.encoder.named_parameters())
    assert all(p.grad is not None for _, p in enc), "every encoder parameter must get a gradient"
    nonzero_tops = {}
    for name, p in enc:
        top = name.split(".")[0]
        nonzero_tops.setdefault(top, False)
        if float(p.grad.abs().sum()) > 0:
            nonzero_tops[top] = True
    assert all(nonzero_tops.values()), f"encoder parts without gradient: {nonzero_tops}"
    if model.trunk is not None:
        assert any(float(p.grad.abs().sum()) > 0 for p in model.trunk.parameters())


# ---------------------------------------------------------------- f. micro-overfit (GPU)

def _two_train_items(target: str):
    from data import RayShapeDataset  # noqa: E402

    ds = RayShapeDataset("train", target, n_rays=OVERFIT_RAYS, n_points=10000, seed=0)
    items = [ds[0], ds[1]]
    keys = items[0].keys()
    return {k: torch.as_tensor(np.stack([it[k] for it in items])).to(DEVICE) for k in keys}


def _loss_of(model, batch, target):
    out = model(batch["inputs"], batch["ray_axis"], batch["ray_origin_uv"])
    if target in K_TARGETS:  # only hits_8 is overfitted (K = 2); the default K = 2 weights apply
        pw = torch.tensor(POS_WEIGHT_K2, device=DEVICE)
        d = hits_loss(out["s_pred"], out["mask_logits"], batch["hits_s"], batch["hits_mask"], pw)
    else:
        d = dense_loss(out["logits"], batch["y"], batch["informative"], target)
    # occfrac_128: the soft-target BCE floor is removed, the criterion is on the excess (H5-3 step 0)
    return d["excess"] if target == "occfrac_128" else d["total"]


@pytest.mark.skipif(not HAS_GPU, reason="micro-overfit runs on GPU")
@pytest.mark.parametrize("target", ["hits_8", "occfrac_128"])
def test_f_micro_overfit(target):
    torch.manual_seed(0)
    batch = _two_train_items(target)
    model = H5RayModel(target).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=OVERFIT_LR)
    model.train()
    with torch.no_grad():
        init = float(_loss_of(model, batch, target))
    losses = []
    for _ in range(OVERFIT_STEPS):
        opt.zero_grad()
        loss = _loss_of(model, batch, target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        final = float(_loss_of(model, batch, target))
    name = "excess" if target == "occfrac_128" else "loss"
    print(f"\nf {target}: initial {name} {init:.6f}, last training-step {name} {losses[-1]:.6f}, "
          f"{name} after {OVERFIT_STEPS} steps {final:.6f}, drop {(1 - final / init) * 100:.1f} %")
    assert final < (1.0 - ATOL_OVERFIT_RATIO) * init, f"{target}: {name} did not drop by more than 50 %"


# ---------------------------------------------------------------- g. time and memory (GPU)

@pytest.mark.skipif(not HAS_GPU, reason="timing and peak memory are measured on GPU")
def test_g_timing_and_memory_report():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(8)
    rows = []
    for target in ("hits_8", "hits_8_plane", "occfrac_128"):
        model = H5RayModel(target).to(DEVICE)
        K = K_TARGETS.get(target, 2)
        parts = {"encoder": n_params(model.encoder), "trunk": n_params(model.trunk), "head": n_params(model.head)}
        inputs = (torch.rand(TIMING_BATCH, TIMING_POINTS, 3, generator=gen) - 0.5).to(DEVICE)
        ray_axis = torch.randint(0, 3, (TIMING_BATCH, TIMING_RAYS), generator=gen).to(DEVICE)
        uv = ((torch.rand(TIMING_BATCH, TIMING_RAYS, 2, generator=gen) - 0.5) * 1.1).float().to(DEVICE)
        if target in ("hits_8", "hits_8_plane"):
            s_t = (torch.rand(TIMING_BATCH, TIMING_RAYS, 4 * K, generator=gen) * 128).to(DEVICE)
            m_t = (torch.rand(TIMING_BATCH, TIMING_RAYS, 2 * K, generator=gen) < 0.5).to(DEVICE)
            pw = torch.tensor(POS_WEIGHT_K2, device=DEVICE)

            def step():
                out = model(inputs, ray_axis, uv)
                return hits_loss(out["s_pred"], out["mask_logits"], s_t, m_t, pw)["total"]
        else:
            y = torch.rand(TIMING_BATCH, TIMING_RAYS, 128, generator=gen).to(DEVICE)
            inf = (y > 0) & (y < 1)

            def step():
                out = model(inputs, ray_axis, uv)
                return dense_loss(out["logits"], y, inf, target)["total"]

        def fwd_bwd():
            loss = step()
            loss.backward()
            model.zero_grad(set_to_none=True)

        fwd_bwd()  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(TIMING_REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fwd_bwd()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append((target, parts, statistics.median(times), peak))
        del model, inputs, ray_axis, uv
        torch.cuda.empty_cache()

    print("\ng  batch 32 x 1024 rays x 10000 points, fp32, forward + backward + loss, "
          f"GPU {torch.cuda.get_device_name(0)}")
    print(f"{'target':<14}{'encoder':>10}{'trunk':>9}{'head':>9}{'time (s)':>11}{'peak GiB':>10}")
    for target, parts, t, peak in rows:
        print(f"{target:<14}{parts['encoder']:>10,}{parts['trunk']:>9,}{parts['head']:>9,}{t:>11.3f}{peak:>10.2f}")
    assert all(r[3] > 0 for r in rows)
