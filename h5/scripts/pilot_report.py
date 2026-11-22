"""Pilot evaluation and decision report (H5-5b).

Two stages, both run by ``h5/scripts/pilot.sh``:

- ``eval``: evaluates every checkpoint ``model_it{N}.pt`` of the three P2 runs (hits_8, occfrac_128,
  points_strong) on the first ``--n-val-shapes`` shapes of ``val.lst`` with **all rays**, at the decoding
  setting stored next to the checkpoint (``model_it{N}.json``). Writes ``ROOT/eval/{run}_it{N}.json``.
  It also writes ``ROOT/eval/ceilings.json``: the GT ceilings of hits_8 (``decode_hits``, L = 8) and of
  occfrac_128 (``occ_fraction`` with the occ_snap selected for the 20k checkpoint) on the same shapes and rays.
- ``report``: the metrics and the decision. Writes ``OUT/report.md`` and ``OUT/report.json`` and prints the verdict.

Metric M: recall of thin walls and gaps (c >= 0.5) at delta 0.002, pooled over the bins with L/h < 1.
The decision rule, in this order (first match wins):

1. REDESIGN ENCODER if the point check < 0.3 (recall of walls and gaps pooled over L/h in [0.25, 1) of
   points_strong at the last iteration);
2. STOP if the P1 gap <= 0.02 and the P2 gap at the last iteration <= 0.02;
3. CONTINUE if the P2 gap at the last iteration >= 0.05 and >= the P2 gap at the previous iteration;
4. UNCLEAR otherwise (extend hits_8 and occfrac_128 to 60k).

The cost rule is separate: PASS if hits_8 or hits_8_plane is at least 10 times faster than points_strong at
N = 512 (total per shape, from the P0 timing of the pilot), else REPORT ONLY.

The P1 gap is M(hits_8) - M(occfrac_128) on the train subset (final checkpoint, all rays), at the selected
setting of each run. The P2 gap is the same on the validation shapes (all rays).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.ceilings import decode_hits, decode_occfrac, evaluate_axis, occ_fraction  # noqa: E402
from csgbox.constants import GATE_BIN_EDGES, GATE_HIT_DELTA, GEN_CHECK_N  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402

from data import DATA_ROOT, shape_ids  # noqa: E402
from eval_rays import (  # noqa: E402
    EVAL_DELTAS,
    aggregate,
    grid_summaries,
    prepare_all,
)
from models import build_model  # noqa: E402

P2_RUNS: dict[str, str] = {"hits_8": "hits_8", "occfrac_128": "occfrac_128", "points_strong": "points_strong"}
DELTA: str = str(GATE_HIT_DELTA)
H = cell_size(GEN_CHECK_N)
N_SMALL = sum(1 for b in GATE_BIN_EDGES[1:] if b <= 1.0)  # bins with L/h < 1
BIN_LABELS = [f"[{a:g},{b:g})" for a, b in zip(GATE_BIN_EDGES[:-1], GATE_BIN_EDGES[1:])]
THRESHOLDS = {"redesign_encoder": 0.3, "stop_gap": 0.02, "continue_gap": 0.05, "cost_ratio": 10.0}


def _its(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------- stage eval

def run_eval(root: Path, its: list[int], n_val: int, device: torch.device) -> None:
    out = root / "eval"
    out.mkdir(parents=True, exist_ok=True)
    ids = shape_ids("val", DATA_ROOT)[:n_val]
    items = [prepare_all(sid, DATA_ROOT) for sid in ids]  # all rays; GT cached for the ceilings too
    for run, target in P2_RUNS.items():
        for it in its:
            ck = root / f"P2_{run}" / f"model_it{it}.pt"
            side = ck.with_suffix(".json")
            setting = json.loads(side.read_text(encoding="utf-8"))["setting"]
            blob = torch.load(ck, map_location=device, weights_only=False)
            model = build_model(target).to(device)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            summary = grid_summaries(model, items, [setting], device, DATA_ROOT)[0]  # deterministic forward inside
            doc = {"run": run, "iteration": it, "setting": setting, "setting_name": summary["setting_name"],
                   "n_shapes": len(ids), "summary": summary}
            (out / f"{run}_it{it}.json").write_text(json.dumps(doc, default=float), encoding="utf-8")
            print(f"eval {run} it {it} {summary['setting_name']}: M "
                  f"{M(summary):.4f}, F1 {summary['deltas'][DELTA]['f1']:.4f}", flush=True)
    # ceilings of hits_8 and occfrac_128 on the same shapes and rays; occfrac at the snap of the last checkpoint
    last = max(its)
    occ_setting = json.loads((root / "P2_occfrac_128" / f"model_it{last}.json").read_text(encoding="utf-8"))["setting"]
    snap = occ_setting.get("occ_snap")
    ceil = {"hits_8": [], "occfrac_128": []}
    for prep in items:
        for a in range(3):
            gt = prep.gt_sub[a]
            ceil["hits_8"].append(evaluate_axis(gt, decode_hits(gt, 8), a, tuple(EVAL_DELTAS), H))
            f = occ_fraction(gt, 128).astype(np.float32)
            if snap is not None:
                f = np.where(f < snap, 0.0, np.where(f > 1.0 - snap, 1.0, f)).astype(np.float32)
            ceil["occfrac_128"].append(evaluate_axis(gt, decode_occfrac(f), a, tuple(EVAL_DELTAS), H))
    doc = {"occ_snap": snap, "n_shapes": len(ids),
           "hits_8": aggregate(ceil["hits_8"])[0], "occfrac_128": aggregate(ceil["occfrac_128"])[0]}
    (out / "ceilings.json").write_text(json.dumps(doc, default=float), encoding="utf-8")
    print(f"ceilings written (occ_snap {snap})", flush=True)


# ---------------------------------------------------------------- stage report

def M(summary: dict) -> float:
    """Recall of thin walls and gaps (c >= 0.5) at delta 0.002, pooled over L/h < 1."""
    return float(summary["deltas"][DELTA]["walls_gaps_pooled_L_lt_1"])


def M_bins(summary: dict, bins: list[int]) -> float:
    """Recall of walls and gaps pooled over the given bin indices."""
    rec = np.asarray(summary["deltas"][DELTA]["thin_recalled"], dtype=float)
    tot = np.asarray(summary["thin_total"], dtype=float)
    num = rec[0, bins].sum() + rec[1, bins].sum()
    den = tot[0, bins].sum() + tot[1, bins].sum()
    return float(num / den) if den else float("nan")


def selected_setting(run_dir: Path, it: int) -> tuple[dict, str]:
    doc = json.loads((run_dir / f"model_it{it}.json").read_text(encoding="utf-8"))
    return doc["setting"], doc["selected_setting"]


def p1_values(root: Path, p1_it: int) -> dict:
    out = {}
    for t in ("hits_8", "occfrac_128"):
        rec = [json.loads(x) for x in (root / f"P1_{t}" / "val.jsonl").read_text(encoding="utf-8").splitlines() if x]
        r = next(r for r in rec if r["iteration"] == p1_it)
        s = next(s for s in r["settings"] if s["setting_name"] == r["selected"])
        out[t] = {"setting": r["selected"], "M": M(s), "F1": s["deltas"][DELTA]["f1"]}
    out["gap"] = out["hits_8"]["M"] - out["occfrac_128"]["M"]
    return out


def verdict(point_check: float, p1_gap: float, gap_last: float, gap_prev: float) -> tuple[str, str]:
    t = THRESHOLDS
    if point_check < t["redesign_encoder"]:
        return "REDESIGN ENCODER", f"point check {point_check:.3f} < {t['redesign_encoder']}"
    if p1_gap <= t["stop_gap"] and gap_last <= t["stop_gap"]:
        return "STOP", f"P1 gap {p1_gap:.3f} <= {t['stop_gap']} and P2 gap at last iteration {gap_last:.3f} <= {t['stop_gap']}"
    if gap_last >= t["continue_gap"] and gap_last >= gap_prev:
        return "CONTINUE", f"P2 gap at last iteration {gap_last:.3f} >= {t['continue_gap']} and >= previous {gap_prev:.3f}"
    return "UNCLEAR", (f"P2 gap last {gap_last:.3f}, previous {gap_prev:.3f}: not covered by the rules above; "
                       "extend hits_8 and occfrac_128 to 60k")


def run_report(root: Path, its: list[int], p1_it: int, out: Path) -> dict:
    ev = root / "eval"
    rows = {}
    for run in P2_RUNS:
        for it in its:
            d = json.loads((ev / f"{run}_it{it}.json").read_text(encoding="utf-8"))
            rows[(run, it)] = d
    ceil = json.loads((ev / "ceilings.json").read_text(encoding="utf-8"))
    p1 = p1_values(root, p1_it)
    gaps = {it: M(rows[("hits_8", it)]["summary"]) - M(rows[("occfrac_128", it)]["summary"]) for it in its}
    last, prev = its[-1], its[-2]
    point_summary = rows[("points_strong", last)]["summary"]
    point_check = M_bins(point_summary, [2, 3])  # bins [0.25, 0.5) and [0.5, 1)
    code, why = verdict(point_check, p1["gap"], gaps[last], gaps[prev])

    timing = json.loads((root / "timing" / "p2.json").read_text(encoding="utf-8")) if (root / "timing" / "p2.json").exists() else None
    cost = {"available": timing is not None}
    if timing is not None:
        by = {r["label"]: r for r in timing["rows"]}
        n512 = by["points_strong N=512"]["total"]
        ratio_hits = n512 / by["hits_8"]["total"]
        ratio_plane = n512 / by["hits_8_plane"]["total"]
        cost.update({"points_N512_total_s": n512, "hits_8_total_s": by["hits_8"]["total"],
                     "hits_8_plane_total_s": by["hits_8_plane"]["total"], "ratio_hits_8": ratio_hits,
                     "ratio_hits_8_plane": ratio_plane,
                     "hits_8_plane_untrained": "hits_8_plane" in timing.get("untrained_methods", []),
                     "verdict": "PASS" if max(ratio_hits, ratio_plane) >= THRESHOLDS["cost_ratio"] else "REPORT ONLY"})
    doc = {"its": its, "p1_it": p1_it, "P1": p1,
           "P2_gap": {str(it): gaps[it] for it in its},
           "point_check": point_check,
           "M": {f"{run}@{it}": M(rows[(run, it)]["summary"]) for run in P2_RUNS for it in its},
           "F1_all": {f"{run}@{it}": rows[(run, it)]["summary"]["deltas"][DELTA]["f1"] for run in P2_RUNS for it in its},
           "selected": {f"{run}@{it}": rows[(run, it)]["setting_name"] for run in P2_RUNS for it in its},
           "ceiling_hits_8_M": ceil["hits_8"]["deltas"][DELTA]["walls_gaps_pooled_L_lt_1"],
           "ceiling_occfrac_128_M": ceil["occfrac_128"]["deltas"][DELTA]["walls_gaps_pooled_L_lt_1"],
           "ceiling_occ_snap": ceil["occ_snap"],
           "verdict": code, "verdict_reason": why, "cost": cost}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(doc, rows, p1, ceil), encoding="utf-8")
    print(f"VERDICT: {code}  ({why})")
    print(f"point check {point_check:.4f}; P1 gap {p1['gap']:.4f}; P2 gaps " +
          ", ".join(f"{it}: {gaps[it]:.4f}" for it in its))
    if cost["available"]:
        print(f"cost: hits_8 {cost['ratio_hits_8']:.2f}x, hits_8_plane {cost['ratio_hits_8_plane']:.2f}x faster than "
              f"points N=512 -> {cost['verdict']}")
    else:
        print("cost: timing not available")
    return doc


def _f(x, d: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}g}"


def to_markdown(doc: dict, rows: dict, p1: dict, ceil: dict) -> str:
    L = ["# H5 pilot report", "",
         f"**Verdict: {doc['verdict']}.** {doc['verdict_reason']}.", "",
         f"- Point check (points_strong at it {doc['its'][-1]}, recall of walls and gaps, L/h in [0.25, 1)): "
         f"**{doc['point_check']:.3f}** (REDESIGN ENCODER below {THRESHOLDS['redesign_encoder']}).",
         f"- P1 gap (train subset, all rays, final it {doc['p1_it']}): M(hits_8) {p1['hits_8']['M']:.3f} - "
         f"M(occfrac_128) {p1['occfrac_128']['M']:.3f} = **{p1['gap']:.3f}** (STOP needs <= {THRESHOLDS['stop_gap']}).",
         "- P2 gap (validation, all rays): " + ", ".join(f"it {it}: **{doc['P2_gap'][str(it)]:.3f}**" for it in doc["its"])
         + f" (CONTINUE needs >= {THRESHOLDS['continue_gap']} at the last iteration, and >= the previous one).",
         f"- Cost: **{doc['cost'].get('verdict', 'timing not available')}**"
         + (f" (points N=512 / hits_8 = {doc['cost']['ratio_hits_8']:.2f}; points N=512 / hits_8_plane = "
            f"{doc['cost']['ratio_hits_8_plane']:.2f}"
            f"{' (hits_8_plane untrained)' if doc['cost'].get('hits_8_plane_untrained') else ''}; "
            f"PASS needs >= {THRESHOLDS['cost_ratio']:g} for hits_8 or hits_8_plane)"
            if doc["cost"].get("available") else ""), "",
         "## Selected decoding settings", "", "| run | iteration | setting |", "|---|---|---|"]
    for key, name in doc["selected"].items():
        run, it = key.split("@")
        L.append(f"| {run} | {it} | {name} |")
    L += ["", "## Metric M (walls + gaps recall, delta 0.002, L/h < 1) and F1 of all hits", "",
          "| run | iteration | M | F1 |", "|---|---|---|---|"]
    for key in doc["M"]:
        run, it = key.split("@")
        L.append(f"| {run} | {it} | {_f(doc['M'][key])} | {_f(doc['F1_all'][key])} |")
    L += ["", f"Ceilings on the same shapes and rays (occ_snap {doc['ceiling_occ_snap']}): "
          f"hits_8 M {_f(doc['ceiling_hits_8_M'])}, occfrac_128 M {_f(doc['ceiling_occfrac_128_M'])}.", "",
          "## Recall of walls and gaps per L/h bin (delta 0.002, validation, all rays)", ""]
    for run in P2_RUNS:
        it = doc["its"][-1]
        s = rows[(run, it)]["summary"]
        rec = s["deltas"][DELTA]["walls_gaps"]
        L += [f"### {run} at it {it}", "", "| " + " | ".join(BIN_LABELS) + " |", "|" + "---|" * len(BIN_LABELS),
              "| " + " | ".join(_f(x, 3) for x in rec) + " |", ""]
    L += ["## Thin-feature recall per bin (walls, gaps, tangent chords) at the last iteration", ""]
    for run in P2_RUNS:
        it = doc["its"][-1]
        s = rows[(run, it)]["summary"]
        rc = s["deltas"][DELTA]["thin_recall"]
        L += [f"### {run}", "", "| class | " + " | ".join(BIN_LABELS) + " |", "|---|" + "---|" * len(BIN_LABELS),
              "| wall | " + " | ".join(_f(x, 3) for x in rc[0]) + " |",
              "| gap | " + " | ".join(_f(x, 3) for x in rc[1]) + " |",
              "| tangent chord | " + " | ".join(_f(x, 3) for x in rc[2]) + " |", ""]
    L += ["## P2 gap per iteration", "", "| iteration | M(hits_8) | M(occfrac_128) | gap |", "|---|---|---|---|"]
    for it in doc["its"]:
        L.append(f"| {it} | {_f(doc['M'][f'hits_8@{it}'])} | {_f(doc['M'][f'occfrac_128@{it}'])} | "
                 f"{_f(doc['P2_gap'][str(it)])} |")
    L += ["", "## Verdict rule", "", "1. REDESIGN ENCODER if point check < 0.3; 2. STOP if P1 gap <= 0.02 and P2 gap "
          "(last) <= 0.02; 3. CONTINUE if P2 gap (last) >= 0.05 and >= P2 gap (previous); 4. UNCLEAR otherwise."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["eval", "report"])
    p.add_argument("--root", type=Path, required=True, help="pilot run root (contains P1_*, P2_*)")
    p.add_argument("--its", type=_its, default=_its("5000,10000,15000,20000"))
    p.add_argument("--p1-it", type=int, default=10000)
    p.add_argument("--n-val-shapes", type=int, default=200)
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "pilot")
    args = p.parse_args(argv)
    if args.stage == "eval":
        run_eval(args.root, args.its, args.n_val_shapes, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        run_report(args.root, args.its, args.p1_it, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
