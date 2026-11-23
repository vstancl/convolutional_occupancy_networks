"""Ray-space evaluation of the h5 ray decoders (H5-3).

A decoder output for a set of rays becomes ``RayHits`` (one per axis), which is compared with the
analytic ground truth ``csgbox.ceilings.gt_hits`` through ``csgbox.ceilings.evaluate_axis``. The matching,
the thin-feature classes and the per-axis counts are the csgbox ones, so the numbers are comparable
with the Gate B report. The aggregation (``aggregate``) is a reimplementation of the gate's ``merge`` and
``metrics``; the tests check it against ``scripts/gate_ceilings.py``.

Conventions: ``ray_axis`` (R,) with 0 = x, 1 = y, 2 = z; ``ray_index = i_col + N i_row`` (N = 128);
hit coordinates ``s = (t - BOX_LO) / h`` in cell units of the 128 grid.

Importing this module calls ``_paths.add_external_paths()``, as the other h5 modules do.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from _paths import add_external_paths

add_external_paths()

from csgbox.ceilings import (  # noqa: E402
    RayHits,
    _sample_positions,
    decode_occfrac,
    decode_tsdf,
    evaluate_axis,
    gt_hits,
    match,
    thin_bins,
)
from csgbox.constants import (  # noqa: E402
    BOX_LO,
    GATE_BIN_EDGES,
    GATE_DT_BINS,
    GATE_DT_LOG_HI,
    GATE_DT_LOG_LO,
    GATE_HIT_DELTA,
    GATE_TSDF_TAU_CELLS,
    GEN_CHECK_N,
    GEN_THIN_CELLS,
)
from csgbox.grid import cell_size  # noqa: E402
from csgbox.thin import CLASSES  # noqa: E402
from csgbox.tree import from_json  # noqa: E402

from data import DATA_ROOT, N_AXIS, POINTCLOUD_FILE, fixed_val_rays, ray_fields, shape_ids  # noqa: E402
from eval_points import ray_geometry, ray_protocol, network_prob_fn  # noqa: E402
from models import POINT_TARGET_NAMES  # noqa: E402
from rays import subset_rays  # noqa: E402
from slots import decode_slots  # noqa: E402
from to_hits import decode_binary_array  # noqa: E402

M: int = 128  # the decoded resolution of the dense targets
H: float = cell_size(M)
BIN_LABELS: tuple[str, ...] = tuple(f"[{a:g},{b:g})" for a, b in zip(GATE_BIN_EDGES[:-1], GATE_BIN_EDGES[1:]))
N_SMALL_BINS: int = sum(1 for b in GATE_BIN_EDGES[1:] if b <= 1.0)  # bins with L/h < 1 (here 4)
# Threshold on the predicted probability of an interval slot or a binary/occupancy cell (decoder rule).
EVAL_THR: float = 0.5
# Points given to the encoder at evaluation: a fixed subset of the shape's cloud with the training noise.
EVAL_N_POINTS: int = 10000
EVAL_NOISE: float = 0.001
EVAL_POINTS_SEED: int = 0  # SeedSequence([EVAL_POINTS_SEED, int(shape_id)])
# Deltas of the matching (task spec: 0.002 and h/2 at N = 128).
EVAL_DELTAS: tuple[float, float] = (GATE_HIT_DELTA, cell_size(GEN_CHECK_N) / 2.0)
EVAL_CHUNK_RAYS: int = 8192
# Snapping of occupancy fractions for decode_occfrac (see outputs_to_rayhits). 1e-5 lies above the clip
# artefact (1e-6) and below the smallest true partial fraction in the 10 test shapes (2.2e-5): with it the
# gate is reproduced exactly on those shapes; 1e-4 differs on 5 rays. Not the default (literal rule).
OCC_SNAP_RECOMMENDED: float = 1e-5
MASK_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)  # decoding grid of hits and binary targets
# decoding grid of occfrac_128. H5-6 extends the H5-5 grid (1e-3, 1e-2, 0.03, 0.1) to larger snaps; 1e-3 is kept
# so that settings selected in the pilot stay in the grid.
OCC_SNAP_GRID: tuple[float, ...] = (1e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.4, 0.45)
# Metric M is reported as "not comparable" when decoded/GT hits is outside this band (H5-6).
RATIO_BAND: tuple[float, float] = (0.8, 1.25)
POINT_ISO_GRID: tuple[float, ...] = (0.2, 0.5)  # iso of the point decoders (ray protocol, N = 512)
TARGET_K: dict[str, int] = {"hits_8": 2, "hits_16": 4, "hits_8_plane": 2}  # K per hits target
N_ALL: int = 3 * N_AXIS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (float64)."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def _empty(n_rays: int = 0) -> RayHits:
    return RayHits(np.zeros(n_rays + 1, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=bool), None)


def _decode_axis(target: str, out: dict, setting: dict) -> RayHits:
    """Decoder rule of one target for the rows of one axis under one decoding setting.

    ``setting`` may hold ``thr`` (hits and binary: the probability threshold) and ``occ_snap``
    (occfrac: the snapping, None for the literal rule). Missing keys take the defaults.
    """
    thr = float(setting.get("thr", EVAL_THR))
    occ_snap = setting.get("occ_snap")
    if target in TARGET_K:
        K = TARGET_K[target]
        t = BOX_LO + out["s_pred"] * H
        return decode_slots(t, _sigmoid(out["mask_logits"]), K, thr=thr)
    if target == "occfrac_128":
        f = _sigmoid(out["logits"])
        if occ_snap is not None:  # off by default: values within occ_snap of 0 or 1 are snapped
            f = np.where(f < occ_snap, 0.0, np.where(f > 1.0 - occ_snap, 1.0, f))
        return decode_occfrac(f)
    if target == "binary_128":
        return decode_binary_array(_sigmoid(out["logits"]) > thr)
    if target == "tsdf_128":
        z, h = _sample_positions(M)
        values = GATE_TSDF_TAU_CELLS * h * np.tanh(np.asarray(out["logits"], dtype=np.float64))
        return decode_tsdf(values, z, h)
    raise ValueError(f"unknown target {target!r}")


def decode_by_axis(target: str, outputs: dict, ray_axis: np.ndarray, setting: dict) -> list[RayHits]:
    """One ``RayHits`` per axis for the rows of ``outputs`` under one decoding setting."""
    ray_axis = np.asarray(ray_axis)
    result = []
    for a in range(3):
        rows = np.flatnonzero(ray_axis == a)
        if rows.size == 0:
            result.append(_empty())
            continue
        sub = {k: np.asarray(v)[rows] for k, v in outputs.items()}
        result.append(_decode_axis(target, sub, setting))
    return result


def outputs_to_rayhits(target: str, outputs: dict, ray_axis: np.ndarray, thr: float = EVAL_THR,
                       occ_snap: float | None = None) -> list[RayHits]:
    """Decoder outputs of R rays to one ``RayHits`` per axis (rays kept in input order within each axis).

    ``outputs`` holds numpy arrays with R rows: ``s_pred`` and ``mask_logits`` for the hits targets,
    ``logits`` for the dense ones. Rules: hits_K: ``t = BOX_LO + s h`` and ``decode_slots`` with
    ``sigmoid(mask_logits)``; occfrac: ``decode_occfrac(sigmoid(logits))``; binary:
    ``decode_binary_array(sigmoid(logits) > thr)``; tsdf: ``decode_tsdf(2.5 h tanh(logits), z, h)``
    with ``z, h`` from ``csgbox.ceilings`` (as ``tsdf_values`` uses them).

    ``occ_snap`` (default None, the literal rule): occupancy fractions below ``occ_snap`` are set to 0 and
    above ``1 - occ_snap`` to 1 before ``decode_occfrac``. Without it, a logit clipped to 1e-6 gives a
    fraction of 1e-6, which ``decode_occfrac`` reads as a partial cell, so every empty cell becomes a
    spurious thin wall (see docs/h5-3.md).
    """
    return decode_by_axis(target, outputs, ray_axis, {"thr": thr, "occ_snap": occ_snap})


def _gt_axes(tree) -> list[RayHits]:
    """Analytic float64 GT of the three axes of a shape (all N^2 rays each)."""
    return [gt_hits(tree, a, GEN_CHECK_N) for a in range(3)]


def shape_points(shape_id: str, root: Path = DATA_ROOT) -> np.ndarray:
    """``EVAL_N_POINTS`` points of the cloud, with noise, from a seeded Generator (float32)."""
    rng = np.random.default_rng(np.random.SeedSequence([EVAL_POINTS_SEED, int(shape_id)]))
    with np.load(Path(root) / "csg" / shape_id / POINTCLOUD_FILE) as pc:
        pts = pc["points"]
        idx = rng.choice(pts.shape[0], size=EVAL_N_POINTS, replace=False)
        p = pts[idx].astype(np.float64)
    return (p + rng.normal(0.0, EVAL_NOISE, size=p.shape)).astype(np.float32)


def _rays_of(rays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ray_axis, ray_index, ray_origin_uv)`` of the requested rays ("all" or ``(ray_axis, ray_index)``)."""
    if isinstance(rays, str):
        if rays != "all":
            raise ValueError("rays must be 'all' or (ray_axis, ray_index)")
        return ray_fields(np.arange(N_ALL, dtype=np.int64))
    ray_axis, ray_index = (np.asarray(x, dtype=np.int64) for x in rays)
    return ray_fields(ray_axis * N_AXIS + ray_index)


@dataclass
class Prepared:
    """One shape with its rays and their GT: ``gt_sub[a]`` holds the rays of axis a, in input order."""

    shape_id: str
    ray_axis: np.ndarray
    ray_index: np.ndarray
    uv: np.ndarray
    gt_sub: list[RayHits]


_PREPARED: dict[tuple, Prepared] = {}  # in-memory cache of prepared shapes (validation GT is computed once)


def clear_prepared_cache() -> None:
    _PREPARED.clear()


def _prepare(shape_id: str, ray_axis, ray_index, uv, root: Path) -> Prepared:
    tree = from_json((Path(root) / "csg" / shape_id / "tree.json").read_text(encoding="utf-8"))
    gt = _gt_axes(tree)
    ray_axis = np.asarray(ray_axis)
    ray_index = np.asarray(ray_index)
    gt_sub = [subset_rays(gt[a], ray_index[np.flatnonzero(ray_axis == a)]) for a in range(3)]
    return Prepared(shape_id, ray_axis, ray_index, np.asarray(uv), gt_sub)


def prepare_fixed(split: str, index: int, n_rays: int = 4096, root: Path = DATA_ROOT) -> Prepared:
    """Shape ``index`` of ``split.lst`` with its fixed rays (``fixed_val_rays``) and their GT. Cached in memory."""
    key = ("fixed", str(root), split, int(index), int(n_rays))
    if key not in _PREPARED:
        sid = shape_ids(split, Path(root))[index]
        ray_axis, ray_index, uv = fixed_val_rays(index, n=n_rays, split=split, root=Path(root))
        _PREPARED[key] = _prepare(sid, ray_axis, ray_index, uv, Path(root))
    return _PREPARED[key]


def prepare_all(shape_id: str, root: Path = DATA_ROOT) -> Prepared:
    """Every ray of a shape with the full GT (``rays="all"``). Cached in memory."""
    key = ("all", str(root), shape_id)
    if key not in _PREPARED:
        ray_axis, ray_index, uv = ray_fields(np.arange(N_ALL, dtype=np.int64))
        _PREPARED[key] = _prepare(shape_id, ray_axis, ray_index, uv, Path(root))
    return _PREPARED[key]


@contextmanager
def deterministic_kernels():
    """Deterministic CUDA kernels for the forward pass. Without them the encoder's point scatter
    (ConvONet's scatter_mean / scatter_max) gives plane features that differ by up to ~6e-4 between two runs
    on the same input; threshold decisions then flip and F1 moves by ~1e-4 (H5-4). Used for evaluation only:
    the backward of grid_sample has no deterministic CUDA kernel, so training stays non-deterministic."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def run_decoder(model, prep: Prepared, device=None, chunk_rays: int = EVAL_CHUNK_RAYS,
                root: Path = DATA_ROOT) -> dict:
    """One network pass over a shape: the encoder once, the decoder in chunks. Numpy outputs, one row per ray."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    with torch.no_grad(), deterministic_kernels():
        inputs = torch.as_tensor(shape_points(prep.shape_id, Path(root)), device=device)[None]
        planes = model.encode(inputs)
        chunks: dict[str, list] = {}
        for start in range(0, prep.ray_axis.size, chunk_rays):
            sl = slice(start, start + chunk_rays)
            ax_t = torch.as_tensor(prep.ray_axis[sl], device=device)[None]
            uv_t = torch.as_tensor(prep.uv[sl], device=device)[None]
            out = model.decode(planes, ax_t, uv_t)
            for k, v in out.items():
                chunks.setdefault(k, []).append(v[0].detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def score_grid(target: str, outputs: dict, ray_axis: np.ndarray, gt_sub: list[RayHits], settings: list[dict],
               deltas=EVAL_DELTAS) -> list[list[dict]]:
    """For every decoding setting, the three ``evaluate_axis`` dicts (one list per setting), from one set of outputs."""
    grid = []
    for st in settings:
        dec = decode_by_axis(target, outputs, ray_axis, st)
        grid.append([evaluate_axis_diag(gt_sub[a], dec[a], a, tuple(deltas), H) for a in range(3)])
    return grid


def evaluate_shape(model, shape_id: str, split: str, rays="all", thr: float = EVAL_THR, device=None,
                   chunk_rays: int = EVAL_CHUNK_RAYS, root: Path = DATA_ROOT, deltas=EVAL_DELTAS,
                   occ_snap: float | None = None, settings: list[dict] | None = None):
    """Per-axis ``evaluate_axis`` results of one shape for the rays requested.

    ``rays`` is "all" or ``(ray_axis, ray_index)``. With ``settings=None`` (default) the single setting
    ``{thr, occ_snap}`` is used and three dicts (x, y, z) are returned. With a list of decoding settings,
    one network pass serves all of them and the return value is one list of three dicts per setting.
    """
    if shape_id not in set(shape_ids(split, Path(root))):
        raise ValueError(f"{shape_id} is not in {split}.lst")
    if isinstance(rays, str):
        if rays != "all":
            raise ValueError("rays must be 'all' or (ray_axis, ray_index)")
        prep = prepare_all(shape_id, Path(root))
    else:
        ray_axis, ray_index, uv = _rays_of(rays)
        prep = _prepare(shape_id, ray_axis, ray_index, uv, Path(root))
    if getattr(model, "kind", None) == "points":
        grid_settings = [{"iso": 0.5}] if settings is None else list(settings)
        res = score_points(model, prep, grid_settings, device or next(model.parameters()).device, Path(root), deltas)
        return res[0] if settings is None else res
    outs = run_decoder(model, prep, device, chunk_rays, Path(root))
    if settings is None:
        return score_grid(model.target, outs, prep.ray_axis, prep.gt_sub,
                          [{"thr": thr, "occ_snap": occ_snap}], deltas)[0]
    return score_grid(model.target, outs, prep.ray_axis, prep.gt_sub, list(settings), deltas)


def evaluate_outputs(target: str, outputs: dict, gt: list[RayHits], ray_axis: np.ndarray, ray_index: np.ndarray,
                     thr: float = EVAL_THR, occ_snap: float | None = None, deltas=EVAL_DELTAS) -> list[dict]:
    """Decode the outputs of the given rays and evaluate them against ``gt`` (the three axes, all rays).

    ``ray_axis`` and ``ray_index`` (R,) say which ray each output row is. Returns three ``evaluate_axis``
    dicts (x, y, z). Used by the tests with perfect outputs.
    """
    ray_axis = np.asarray(ray_axis)
    ray_index = np.asarray(ray_index)
    gt_sub = [subset_rays(gt[a], ray_index[np.flatnonzero(ray_axis == a)]) for a in range(3)]
    return score_grid(target, outputs, ray_axis, gt_sub, [{"thr": thr, "occ_snap": occ_snap}], deltas)[0]


# ---------------------------------------------------------------- per-hit diagnostics (H5-6)

# Log-spaced bins of the per-hit distance |t_gt - t_dec| (0.6 % wide). Index DIAG_NB is the overflow: distances
# above 10 and infinite distances. Medians and p90 are read from these histograms (bin centre, error < 0.6 %).
DIAG_EDGES: np.ndarray = np.logspace(-7.0, 1.0, 3201)
DIAG_NB: int = len(DIAG_EDGES) - 1


def nearest_same_type(gt: RayHits, dec: RayHits) -> np.ndarray:
    """Per GT hit: distance to the nearest decoded hit of the same type (enter or exit) on the same ray.

    Infinite if the ray has no decoded hit of that type. The nearest hit is not the greedy match of
    csgbox (``match``): it is the diagnostic distance, independent of the delta.
    """
    out = np.full(gt.t.size, np.inf)
    if gt.t.size == 0 or dec.t.size == 0:
        return out
    ray_g = gt.ray_of()
    ray_d = dec.ray_of()
    lo = min(float(gt.t.min()), float(dec.t.min()))
    span = max(float(gt.t.max()), float(dec.t.max())) - lo + 1.0  # (t - lo) < span: rays do not overlap in the key
    for enter in (True, False):
        mg = gt.enter == enter
        md = dec.enter == enter
        if not mg.any() or not md.any():
            continue
        kd_all = ray_d[md] * span + (dec.t[md] - lo)
        order = np.argsort(kd_all, kind="stable")
        kd = kd_all[order]
        td = dec.t[md][order]
        rd = ray_d[md][order]
        tq, rq = gt.t[mg], ray_g[mg]
        idx = np.searchsorted(kd, rq * span + (tq - lo), side="left")  # first decoded key >= the GT key
        succ = np.minimum(idx, kd.size - 1)
        pred = np.maximum(idx - 1, 0)
        d_succ = np.where((idx < kd.size) & (rd[succ] == rq), np.abs(td[succ] - tq), np.inf)
        d_pred = np.where((idx >= 1) & (rd[pred] == rq), np.abs(td[pred] - tq), np.inf)
        out[mg] = np.minimum(d_succ, d_pred)
    return out


def thin_endpoints(gt: RayHits, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Per GT hit: length of its shorter adjacent GT interval if that interval is thin (< GEN_THIN_CELLS h).

    Returns ``(shorter, thin)``: ``shorter`` is inf for hits that are not an endpoint of a thin interval, and
    ``thin`` is the mask of the others. Thin = material interval or gap, the csgbox definition (decision 48).
    """
    ray = gt.ray_of()
    k = np.flatnonzero(ray[1:] == ray[:-1])  # interval k joins hits k and k + 1 of the same ray
    length = gt.t[k + 1] - gt.t[k]
    sel = length < GEN_THIN_CELLS * h
    k, length = k[sel], length[sel]
    shorter = np.full(gt.t.size, np.inf)
    np.minimum.at(shorter, k, length)
    np.minimum.at(shorter, k + 1, length)
    return shorter, np.isfinite(shorter)


def _hist_index(d: np.ndarray) -> np.ndarray:
    """Histogram index of distances (``DIAG_NB`` for distances >= 10 and for inf)."""
    idx = np.clip(np.searchsorted(DIAG_EDGES, d, side="right") - 1, 0, DIAG_NB - 1)
    return np.where(d >= DIAG_EDGES[-1], DIAG_NB, idx)


def hit_diagnostics(gt: RayHits, dec: RayHits, h: float, deltas=EVAL_DELTAS) -> dict:
    """Per-hit diagnostics of one (shape, axis); every entry is an additive count or a histogram.

    Ordinary GT hits are the hits that are not an endpoint of a thin GT interval; thin GT hits are binned by the
    length of their shorter adjacent thin interval (``thin_bins``, the gate's L/h bins). For both groups:
    ``n``, ``hist`` (distance to the nearest same-type decoded hit, ``hist`` of ``DIAG_NB + 1`` bins; thin: one
    list per bin), ``within`` (count with that distance <= delta, per delta) and ``matched`` (GT hits matched at
    delta by the csgbox greedy matching, per delta: the recall numerator).
    """
    dist = nearest_same_type(gt, dec)
    shorter, thin = thin_endpoints(gt, h)
    n_bins = len(GATE_BIN_EDGES) - 1
    bins = np.zeros(gt.t.size, dtype=np.int64)
    bins[thin] = thin_bins(shorter[thin], h)
    ordn = ~thin
    hidx = _hist_index(dist)
    thin_hist = np.bincount(bins[thin] * (DIAG_NB + 1) + hidx[thin],
                            minlength=n_bins * (DIAG_NB + 1)).reshape(n_bins, DIAG_NB + 1)
    out = {"ord": {"n": int(ordn.sum()), "hist": np.bincount(hidx[ordn], minlength=DIAG_NB + 1).tolist(),
                   "within": {}, "matched": {}},
           "thin": {"n": np.bincount(bins[thin], minlength=n_bins).tolist(), "hist": thin_hist.tolist(),
                    "within": {}, "matched": {}}}
    for d in deltas:
        key = str(d)
        m = match(gt, dec, float(d)).gt_matched
        within = dist <= float(d)
        out["ord"]["within"][key] = int((within & ordn).sum())
        out["ord"]["matched"][key] = int((m & ordn).sum())
        out["thin"]["within"][key] = np.bincount(bins[thin & within], minlength=n_bins).tolist()
        out["thin"]["matched"][key] = np.bincount(bins[thin & m], minlength=n_bins).tolist()
    return out


def evaluate_axis_diag(gt: RayHits, dec: RayHits, axis: int, deltas: tuple[float, ...], h: float) -> dict:
    """``evaluate_axis`` (csgbox, unchanged) plus the per-hit diagnostics under the key ``diag``."""
    out = evaluate_axis(gt, dec, axis, tuple(deltas), h)
    out["diag"] = hit_diagnostics(gt, dec, h, deltas)
    return out


def _hist_quantile(hist: np.ndarray, q: float) -> float:
    """Quantile q of the distances in a histogram of ``DIAG_NB + 1`` bins (bin centre; inf in the overflow)."""
    total = int(hist.sum())
    if total == 0:
        return float("nan")
    idx = int(np.searchsorted(np.cumsum(hist), q * total, side="left"))
    if idx >= DIAG_NB:
        return float("inf")
    return float(np.sqrt(DIAG_EDGES[idx] * DIAG_EDGES[idx + 1]))


def dec_gt_ratio(summary: dict) -> float:
    """Decoded hits / GT hits of a summary (nan without GT hits)."""
    return _ratio(float(summary["dec_hits"]), float(summary["gt_hits"]))


def m_comparable(summary: dict) -> bool:
    """False when the decoded/GT ratio is outside ``RATIO_BAND`` (or undefined): M is then not comparable."""
    r = dec_gt_ratio(summary)
    return bool(RATIO_BAND[0] <= r <= RATIO_BAND[1])


# ---------------------------------------------------------------- aggregation (reimplements gate_ceilings)


def _p95(hist: np.ndarray) -> float:
    total = int(hist.sum())
    if total == 0:
        return float("nan")
    edges = np.logspace(GATE_DT_LOG_LO, GATE_DT_LOG_HI, GATE_DT_BINS + 1)
    idx = int(np.searchsorted(np.cumsum(hist), 0.95 * total))
    return float(edges[min(idx + 1, GATE_DT_BINS)])


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def aggregate(results: list[dict], deltas=EVAL_DELTAS) -> tuple[dict, str]:
    """Sum the per-axis results and derive the metrics (summary dict, markdown).

    ``results`` is a list of ``evaluate_axis`` dicts, one per (shape, axis). Summary keys:
    ``gt_hits``, ``dec_hits``, ``rays``, ``thin_total`` (class x bin counts of the GT population) and,
    per delta key (``str(delta)``): ``precision``, ``recall``, ``f1``, ``dt_mean``, ``dt_p95``,
    ``count_err_mean``, ``thin_recall`` (class x bin, with rows wall, gap, grazing), ``walls_gaps`` (the
    walls and gaps together per bin, and ``pooled_L_lt_1``, the population-weighted value over the
    bins with L/h < 1), ``thin_recalled`` and ``thin_total`` in counts.
    """
    n_cls, n_bins = len(CLASSES), len(GATE_BIN_EDGES) - 1
    total = np.zeros((n_cls, n_bins), dtype=np.int64)
    gt_hits_n = dec_hits_n = rays_n = 0
    count_err = 0.0
    per_delta: dict[str, dict] = {}
    for d in deltas:
        per_delta[str(d)] = {"matched": 0, "dt_sum": 0.0, "dt_n": 0,
                             "dt_hist": np.zeros(GATE_DT_BINS, dtype=np.int64),
                             "recalled": np.zeros((n_cls, n_bins), dtype=np.int64)}
    dg = {"ord_n": 0, "ord_hist": np.zeros(DIAG_NB + 1, dtype=np.int64),
          "ord_within": {str(d): 0 for d in deltas}, "ord_matched": {str(d): 0 for d in deltas},
          "thin_n": np.zeros(n_bins, dtype=np.int64),
          "thin_hist": np.zeros((n_bins, DIAG_NB + 1), dtype=np.int64),
          "thin_within": {str(d): np.zeros(n_bins, dtype=np.int64) for d in deltas},
          "thin_matched": {str(d): np.zeros(n_bins, dtype=np.int64) for d in deltas}}
    for r in results:
        gt_hits_n += int(r["gt_hits"])
        dec_hits_n += int(r["dec_hits"])
        rays_n += int(r["rays"])
        count_err += float(r["count_err_sum"])
        total += np.asarray(r["thin_total"], dtype=np.int64)
        for d in deltas:
            key = str(d)
            x = r[key]
            acc = per_delta[key]
            acc["matched"] += int(x["matched"])
            acc["dt_sum"] += float(x["dt_sum"])
            acc["dt_n"] += int(x["dt_n"])
            acc["dt_hist"] += np.asarray(x["dt_hist"], dtype=np.int64)
            acc["recalled"] += np.asarray(x["thin_recalled"], dtype=np.int64)
        g = r.get("diag")  # results made before H5-6 have no diagnostics: they add nothing to them
        if g is None:
            continue
        dg["ord_n"] += int(g["ord"]["n"])
        dg["ord_hist"] += np.asarray(g["ord"]["hist"], dtype=np.int64)
        dg["thin_n"] += np.asarray(g["thin"]["n"], dtype=np.int64)
        dg["thin_hist"] += np.asarray(g["thin"]["hist"], dtype=np.int64)
        for d in deltas:
            key = str(d)
            dg["ord_within"][key] += int(g["ord"]["within"][key])
            dg["ord_matched"][key] += int(g["ord"]["matched"][key])
            dg["thin_within"][key] += np.asarray(g["thin"]["within"][key], dtype=np.int64)
            dg["thin_matched"][key] += np.asarray(g["thin"]["matched"][key], dtype=np.int64)

    summary: dict = {"gt_hits": gt_hits_n, "dec_hits": dec_hits_n, "rays": rays_n,
                     "thin_total": total.tolist(), "count_err_mean": _ratio(count_err, rays_n), "deltas": {}}
    summary["dec_gt_ratio"] = dec_gt_ratio(summary)
    summary["m_comparable"] = m_comparable(summary)
    thin_n = dg["thin_n"]
    summary["diag"] = {
        "ord_n": dg["ord_n"],
        "ord_median": _hist_quantile(dg["ord_hist"], 0.5), "ord_p90": _hist_quantile(dg["ord_hist"], 0.9),
        "ord_within": {k: _ratio(v, dg["ord_n"]) for k, v in dg["ord_within"].items()},
        "ord_recall": {k: _ratio(v, dg["ord_n"]) for k, v in dg["ord_matched"].items()},
        "thin_n": thin_n.tolist(),
        "thin_median": [_hist_quantile(h, 0.5) for h in dg["thin_hist"]],
        "thin_p90": [_hist_quantile(h, 0.9) for h in dg["thin_hist"]],
        "thin_within": {k: np.where(thin_n > 0, v / np.maximum(thin_n, 1), np.nan).tolist()
                        for k, v in dg["thin_within"].items()},
        "thin_recall": {k: np.where(thin_n > 0, v / np.maximum(thin_n, 1), np.nan).tolist()
                        for k, v in dg["thin_matched"].items()},
    }
    for d in deltas:
        key = str(d)
        acc = per_delta[key]
        prec = _ratio(acc["matched"], dec_hits_n)
        rec = _ratio(acc["matched"], gt_hits_n)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else float("nan")
        rec_cls = np.where(total > 0, acc["recalled"] / np.maximum(total, 1), np.nan)
        wg_tot = total[0] + total[1]
        wg_rec = acc["recalled"][0] + acc["recalled"][1]
        wg = np.where(wg_tot > 0, wg_rec / np.maximum(wg_tot, 1), np.nan)
        small_tot = int(wg_tot[:N_SMALL_BINS].sum())
        small_rec = int(wg_rec[:N_SMALL_BINS].sum())
        summary["deltas"][key] = {
            "precision": prec, "recall": rec, "f1": f1,
            "dt_mean": _ratio(acc["dt_sum"], acc["dt_n"]), "dt_p95": _p95(acc["dt_hist"]),
            "matched": acc["matched"],
            "thin_recall": rec_cls.tolist(),
            "thin_recalled": acc["recalled"].tolist(),
            "walls_gaps": wg.tolist(),
            "walls_gaps_pooled_L_lt_1": _ratio(small_rec, small_tot),
        }
    md = to_markdown(summary, deltas)
    return summary, md


def _f(x: float, digits: int = 4) -> str:
    return "n/a" if x != x else f"{x:.{digits}g}"


def to_markdown(summary: dict, deltas=EVAL_DELTAS, title: str = "Ray-space evaluation") -> str:
    lines = [f"# {title}", "",
             f"- Rays: {summary['rays']}; GT hits: {summary['gt_hits']}; decoded hits: {summary['dec_hits']}.",
             f"- GT thin features (< 4h): {int(np.sum(summary['thin_total']))}.", ""]
    row_names = ["walls (c >= 0.5)", "gaps (c >= 0.5)", "walls + gaps", "tangent chords (c < 0.5)"]
    for d in deltas:
        m = summary["deltas"][str(d)]
        dname = "0.002" if d == GATE_HIT_DELTA else "h/2"
        flag = "" if summary.get("m_comparable", False) else " **(not comparable: decoded/GT outside [0.8, 1.25])**"
        lines += [f"## delta = {dname}", "",
                  f"- Precision {_f(m['precision'])}, recall {_f(m['recall'])}, F1 {_f(m['f1'])}; "
                  f"**M {_f(m['walls_gaps_pooled_L_lt_1'])}**{flag}; decoded/GT {_f(summary.get('dec_gt_ratio'))}; "
                  f"mean |dt| {_f(m['dt_mean'], 3)}, p95 |dt| {_f(m['dt_p95'], 3)}; "
                  f"mean abs count error per ray {_f(summary['count_err_mean'], 3)}.", "",
                  "| thin recall | " + " | ".join(BIN_LABELS) + " | L/h < 1, pooled |",
                  "|---|" + "---|" * (len(BIN_LABELS) + 1)]
        tt = np.asarray(summary["thin_total"])
        rec = np.asarray(m["thin_recall"], dtype=float)
        rec_c = np.asarray(m["thin_recalled"], dtype=float)
        pop = [int(x) for x in tt[0] + tt[1]]
        lines.append("| population (walls + gaps) | " + " | ".join(str(x) for x in pop) + f" | {int(sum(pop[:N_SMALL_BINS]))} |")
        pooled_by_row = {
            0: _ratio(rec_c[0][:N_SMALL_BINS].sum(), tt[0][:N_SMALL_BINS].sum()),
            1: _ratio(rec_c[1][:N_SMALL_BINS].sum(), tt[1][:N_SMALL_BINS].sum()),
            2: m["walls_gaps_pooled_L_lt_1"],
            3: _ratio(rec_c[2][:N_SMALL_BINS].sum(), tt[2][:N_SMALL_BINS].sum()),
        }
        wg = m["walls_gaps"]
        cells_by_row = [rec[0], rec[1], wg, rec[2]]
        for i, name in enumerate(row_names):
                lines.append(f"| {name} | " + " | ".join(_f(x, 3) for x in cells_by_row[i]) + f" | {_f(pooled_by_row[i], 3)} |")
        lines.append("")
    lines += _diag_markdown(summary, deltas)
    return "\n".join(lines) + "\n"


def _dname(d) -> str:
    return "0.002" if d == GATE_HIT_DELTA else "h/2"


def _diag_markdown(summary: dict, deltas) -> list[str]:
    """Markdown of the per-hit diagnostics (section ``Per-hit diagnostics``) of a summary."""
    dg = summary["diag"]
    head = ["| hits | n | median dist | p90 dist | " + " | ".join(f"within {_dname(d)} | recall {_dname(d)}"
                                                                  for d in deltas) + " |",
            "|---|---|---|---|" + "---|---|" * len(deltas)]

    def cells(n, med, p90, within, recall) -> str:
        return (f"| {n} | {_f(med, 3)} | {_f(p90, 3)} | "
                + " | ".join(f"{_f(within[str(d)], 3)} | {_f(recall[str(d)], 3)}" for d in deltas) + " |")

    lines = ["## Per-hit diagnostics", "",
             f"- Decoded / GT hits: {_f(summary.get('dec_gt_ratio'), 4)}; ordinary GT hits: {dg['ord_n']}. "
             "Distance = |t| to the nearest decoded hit of the same type on the same ray (inf if none); "
             "recall = csgbox greedy match. Thin = endpoint of a GT interval shorter than 4h; thin rows binned by "
             "the shorter adjacent thin interval.", ""] + head
    lines.append("| ordinary " + cells(dg["ord_n"], dg["ord_median"], dg["ord_p90"], dg["ord_within"],
                                      dg["ord_recall"]))
    for i, label in enumerate(BIN_LABELS):
        within = {str(d): dg["thin_within"][str(d)][i] for d in deltas}
        recall = {str(d): dg["thin_recall"][str(d)][i] for d in deltas}
        lines.append(f"| thin {label} " + cells(dg["thin_n"][i], dg["thin_median"][i], dg["thin_p90"][i],
                                                within, recall))
    return lines + [""]


# ---------------------------------------------------------------- validation metric


def setting_name(st: dict) -> str:
    if "iso" in st:
        return f"iso={st['iso']:g}"
    if "occ_snap" in st:
        return "literal" if st["occ_snap"] is None else f"occ_snap={st['occ_snap']:g}"
    if "thr" in st:
        return f"thr={st['thr']:g}"
    return "default"


def decoding_grid(target: str) -> list[dict]:
    """The decoding settings evaluated at every validation of ``target`` (one network pass serves all)."""
    if target in TARGET_K or target == "binary_128":
        return [{"thr": t} for t in MASK_THRESHOLDS]
    if target == "occfrac_128":
        return [{"occ_snap": s} for s in OCC_SNAP_GRID]
    if target == "tsdf_128":
        return [{}]
    if target in POINT_TARGET_NAMES:
        return [{"iso": x} for x in POINT_ISO_GRID]
    raise ValueError(f"unknown target {target!r}")


def point_hits(model, prep: Prepared, isos: list[float], device, root: Path = DATA_ROOT) -> list[list[RayHits]]:
    """Ray protocol of a point decoder on the rays of ``prep``: per iso, the three axes' ``RayHits``.

    The encoder runs once; the sampling is shared by the isos (``eval_points.ray_protocol``).
    """
    inputs = torch.as_tensor(shape_points(prep.shape_id, Path(root)), device=device)[None]
    model.eval()
    with deterministic_kernels():
        with torch.no_grad():
            planes = model.encode(inputs)
        origins, dirs = ray_geometry(prep.ray_axis, prep.ray_index)
        decs = ray_protocol(network_prob_fn(model, planes), origins, dirs, list(isos), device=device)
    rows = [np.flatnonzero(prep.ray_axis == a) for a in range(3)]
    return [[subset_rays(dec, rows[a]) for a in range(3)] for dec in decs]


def score_points(model, prep: Prepared, settings: list[dict], device, root: Path = DATA_ROOT,
                 deltas=EVAL_DELTAS) -> list[list[dict]]:
    """``evaluate_axis`` dicts of a point decoder, one list of three per iso setting."""
    per_iso = point_hits(model, prep, [float(s["iso"]) for s in settings], device, root)
    return [[evaluate_axis_diag(prep.gt_sub[a], hits[a], a, tuple(deltas), H) for a in range(3)] for hits in per_iso]


def grid_summaries(model, items: list[Prepared], settings: list[dict], device=None,
                   root: Path = DATA_ROOT, deltas=EVAL_DELTAS) -> list[dict]:
    """One ``aggregate`` summary per decoding setting over the prepared shapes (one network pass per shape)."""
    per: list[list[dict]] = [[] for _ in settings]
    if device is None:
        device = next(model.parameters()).device
    for prep in items:
        if getattr(model, "kind", None) == "points":
            grid = score_points(model, prep, settings, device, Path(root), deltas)
        else:
            outs = run_decoder(model, prep, device, EVAL_CHUNK_RAYS, Path(root))
            grid = score_grid(model.target, outs, prep.ray_axis, prep.gt_sub, settings, deltas)
        for k, res in enumerate(grid):
            per[k].extend(res)
    summaries = []
    for st, res in zip(settings, per):
        summary, _md = aggregate(res, deltas)
        summary["setting"] = dict(st)
        summary["setting_name"] = setting_name(st)
        summary["n_shapes"] = len(items)
        summaries.append(summary)
    return summaries


def val_metric(model, n_shapes: int = 200, n_rays: int = 4096, split: str = "val", thr: float = EVAL_THR,
               device=None, root: Path = DATA_ROOT, occ_snap: float | None = None,
               settings: list[dict] | None = None):
    """F1 at delta 0.002 and the thin-recall summary on the first ``n_shapes`` shapes of ``val.lst``.

    The rays are the fixed validation rays (``n_rays`` per shape), prepared once and cached. With
    ``settings=None`` the single setting ``{thr, occ_snap}`` is used and one summary is returned (metric:
    ``deltas['0.002']['f1']``). With a list of settings, one summary per setting, in order.
    """
    ids = shape_ids(split, Path(root))[:n_shapes]
    items = [prepare_fixed(split, i, n_rays, Path(root)) for i in range(len(ids))]
    if settings is None:
        return grid_summaries(model, items, [{"thr": thr, "occ_snap": occ_snap}], device, Path(root))[0]
    return grid_summaries(model, items, list(settings), device, Path(root))


def median_seconds(fn, repeats: int = 3) -> float:
    """Median wall time of ``repeats`` calls (helper for the timing tests)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)
