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
)
from csgbox.grid import cell_size  # noqa: E402
from csgbox.thin import CLASSES  # noqa: E402
from csgbox.tree import from_json  # noqa: E402

from data import DATA_ROOT, N_AXIS, POINTCLOUD_FILE, fixed_val_rays, ray_fields, shape_ids  # noqa: E402
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
TARGET_K: dict[str, int] = {"hits_8": 2, "hits_16": 4, "hits_8_plane": 2}  # K per hits target
N_ALL: int = 3 * N_AXIS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (float64)."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def _empty(n_rays: int = 0) -> RayHits:
    return RayHits(np.zeros(n_rays + 1, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=bool), None)


def _decode_axis(target: str, out: dict, thr: float, occ_snap: float | None = None) -> RayHits:
    """Decoder rule of one target for the rows of one axis (``out`` values are numpy arrays)."""
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
    ray_axis = np.asarray(ray_axis)
    result = []
    for a in range(3):
        rows = np.flatnonzero(ray_axis == a)
        if rows.size == 0:
            result.append(_empty())
            continue
        sub = {k: np.asarray(v)[rows] for k, v in outputs.items()}
        result.append(_decode_axis(target, sub, thr, occ_snap))
    return result


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


def evaluate_shape(model, shape_id: str, split: str, rays="all", thr: float = EVAL_THR, device=None,
                   chunk_rays: int = EVAL_CHUNK_RAYS, root: Path = DATA_ROOT, deltas=EVAL_DELTAS,
                   occ_snap: float | None = None) -> list[dict]:
    """Per-axis ``evaluate_axis`` results of one shape for the rays requested.

    The encoder runs once; the decoder runs in chunks of ``chunk_rays`` rays. The GT is
    ``gt_hits(tree)`` (float64), restricted to the same rays with ``subset_rays``. Returns three dicts
    (x, y, z), each the output of ``csgbox.ceilings.evaluate_axis``.
    """
    if shape_id not in set(shape_ids(split, Path(root))):
        raise ValueError(f"{shape_id} is not in {split}.lst")
    shape_dir = Path(root) / "csg" / shape_id
    device = torch.device(device) if device is not None else next(model.parameters()).device
    tree = from_json((shape_dir / "tree.json").read_text(encoding="utf-8"))
    gt = _gt_axes(tree)
    ray_axis, ray_index, uv = _rays_of(rays)

    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(shape_points(shape_id, Path(root)), device=device)[None]
        planes = model.encode(inputs)
        chunks: dict[str, list] = {}
        for start in range(0, ray_axis.size, chunk_rays):
            sl = slice(start, start + chunk_rays)
            ax_t = torch.as_tensor(ray_axis[sl], device=device)[None]
            uv_t = torch.as_tensor(uv[sl], device=device)[None]
            out = model.decode(planes, ax_t, uv_t)
            for k, v in out.items():
                chunks.setdefault(k, []).append(v[0].detach().cpu().numpy())
    outputs = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    return evaluate_outputs(model.target, outputs, gt, ray_axis, ray_index, thr, occ_snap, deltas)


def evaluate_outputs(target: str, outputs: dict, gt: list[RayHits], ray_axis: np.ndarray, ray_index: np.ndarray,
                     thr: float = EVAL_THR, occ_snap: float | None = None, deltas=EVAL_DELTAS) -> list[dict]:
    """Decode the outputs of the given rays and evaluate them against ``gt`` (the three axes, all rays).

    ``ray_axis`` and ``ray_index`` (R,) say which ray each output row is. Returns three ``evaluate_axis``
    dicts (x, y, z). Used by ``evaluate_shape`` and, with perfect outputs, by the tests.
    """
    ray_axis = np.asarray(ray_axis)
    ray_index = np.asarray(ray_index)
    dec = outputs_to_rayhits(target, outputs, ray_axis, thr, occ_snap)
    results = []
    for a in range(3):
        rows = np.flatnonzero(ray_axis == a)
        gt_a = subset_rays(gt[a], ray_index[rows])
        results.append(evaluate_axis(gt_a, dec[a], a, tuple(deltas), H))
    return results


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

    summary: dict = {"gt_hits": gt_hits_n, "dec_hits": dec_hits_n, "rays": rays_n,
                     "thin_total": total.tolist(), "count_err_mean": _ratio(count_err, rays_n), "deltas": {}}
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
        lines += [f"## delta = {dname}", "",
                  f"- Precision {_f(m['precision'])}, recall {_f(m['recall'])}, F1 {_f(m['f1'])}; "
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
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- validation metric


def val_metric(model, n_shapes: int = 200, n_rays: int = 4096, split: str = "val", thr: float = EVAL_THR,
               device=None, root: Path = DATA_ROOT, occ_snap: float | None = None) -> dict:
    """F1 at delta 0.002 and the thin-recall summary on the first ``n_shapes`` shapes of ``val.lst``.

    The rays are the fixed validation rays of ``fixed_val_rays`` (``n_rays`` per shape). Returns the
    summary of ``aggregate`` (key ``deltas['0.002']['f1']`` is the metric).
    """
    ids = shape_ids(split, Path(root))[:n_shapes]
    results: list[dict] = []
    for i, sid in enumerate(ids):
        ray_axis, ray_index, _uv = fixed_val_rays(i, n=n_rays, split=split, root=Path(root))
        results.extend(evaluate_shape(model, sid, split, rays=(ray_axis, ray_index), thr=thr, device=device,
                                      root=Path(root), occ_snap=occ_snap))
    summary, _md = aggregate(results)
    summary["n_shapes"] = len(ids)
    return summary


def median_seconds(fn, repeats: int = 3) -> float:
    """Median wall time of ``repeats`` calls (helper for the timing tests)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)
