"""Memorisation probe Q0 (H5-6): evaluation of the checkpoints and the decision report.

Stages, run by ``h5/scripts/q0.sh``:

- ``eval``: evaluates every ``model_it{N}.pt`` of the Q0 runs (Q0_points_strong, Q0_hits_8, Q0_occfrac_128, the last
  one added by H5-6c) on the 32 training shapes of the run (``train_subset.json``, identical for all runs) with
  **all rays**, at the decoding setting stored next to the checkpoint (``model_it{N}.json``). Writes
  ``ROOT/eval/{run}_it{N}.json``. Checkpoints that do not exist yet are skipped (printed); existing results are kept.
- ``report``: writes ``OUT/report.md`` and ``OUT/report.json`` and prints the verdicts.

Per run and iteration the report holds: ordinary hits (recall at 0.002 and at h/2, median and p90 distance),
the same per thin bin (L/h), F1 of all hits, decoded/GT hits, precision, metric M (with the not-comparable mark) and
the training loss (mean of the logged losses over the validation interval ending at the iteration).

Rule v1 (H5-6, fixed before the first run; verdict at ``final`` = 30000, ``prev`` = 20000; runs Q0_points_strong
and Q0_hits_8 only). First match wins:

- REACHABLE if, for both runs, recall(0.002) of ordinary hits is >= 0.8 at some iteration <= ``final``;
- NOT REACHABLE if, for both runs, recall(0.002) of ordinary hits at ``final`` is < 0.5 and the median distance of
  ordinary hits improves by < 10 % from ``prev`` to ``final`` (improvement = (median_prev - median_final) /
  median_prev; a median that becomes inf counts as no improvement);
- UNCLEAR otherwise.

Rule v2 (H5-6c, fixed before the extension run; at ``v2_final`` = 60000; runs Q0_points_strong and Q0_hits_8 only).
First match wins:

- REACHABLE if, for both runs, recall(0.002) of ordinary hits is >= 0.8 at some iteration <= ``v2_final``;
- NOT REACHABLE IN BUDGET if, for both runs, recall(0.002) of ordinary hits at ``v2_final`` is < 0.6;
- UNCLEAR otherwise.

Descriptive (no rule): ordinary recall and median at the learning-rate step, the midpoint and ``v2_final``, for every
run, and a table at ``v2_final`` of the thin-bin recall(0.002) and median per L/h bin for hits_8, occfrac_128 and
points_strong side by side, with decoded/GT and precision.
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

from data import DATA_ROOT, shape_ids  # noqa: E402
from eval_rays import (  # noqa: E402
    BIN_LABELS,
    EVAL_DELTAS,
    dec_gt_ratio,
    decoding_grid,
    grid_summaries,
    m_comparable,
    prepare_all,
)
from train import _select  # noqa: E402  (the band selection of training, H5-K1-1a)
from models import build_model  # noqa: E402

RUNS: tuple[str, ...] = ("Q0_points_strong", "Q0_hits_8", "Q0_occfrac_128")
V1_RUNS: tuple[str, ...] = ("Q0_points_strong", "Q0_hits_8")  # the rules compare these two runs only
TABLE_ORDER: tuple[tuple[str, str], ...] = (("Q0_hits_8", "hits_8"), ("Q0_occfrac_128", "occfrac_128"),
                                            ("Q0_points_strong", "points_strong"))
DK, DH = str(EVAL_DELTAS[0]), str(EVAL_DELTAS[1])  # the keys of delta 0.002 and delta h/2
RULE = {"reach_recall": 0.8, "low_recall": 0.5, "flat_improvement": 0.10}
RULE_V2 = {"reach_recall": 0.8, "budget_recall": 0.6}


def _its(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def train_loss(run_dir: Path, it: int, val_every: int) -> float:
    """Mean of the logged training losses with ``it - val_every < iteration <= it`` (nan if none)."""
    path = run_dir / "log.jsonl"
    if not path.exists():
        return float("nan")
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    vals = [r["loss"] for r in recs if it - val_every < r["iteration"] <= it]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------- stage eval

def run_eval(root: Path, its: list[int], device: torch.device, runs_wanted: tuple[str, ...] = RUNS,
             split: str | None = None, n_shapes: int | None = None, grid: bool = False) -> None:
    """Evaluates the checkpoints of ``runs_wanted`` with all rays. Shapes: the training subset of the Q0 runs
    (``split`` None), or the first ``n_shapes`` shapes of ``split`` (e.g. ``val`` for H5-K1-1).

    ``grid`` (H5-K1-1b): every setting of the target's decoding grid is evaluated, and the band selection of
    ``train._select`` picks one. Written to ``eval/{run}_it{N}_grid.json``; the per-checkpoint files are not touched."""
    out = root / "eval"
    out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in runs_wanted if (root / r).is_dir()]
    if split is None:
        shapes = {run: json.loads((root / run / "train_subset.json").read_text(encoding="utf-8")) for run in runs}
        if any(shapes[r] != shapes[runs[0]] for r in runs):
            raise ValueError("the Q0 runs do not use the same training shapes")
        shape_list = shapes[runs[0]]
    else:
        shape_list = shape_ids(split, DATA_ROOT)[:n_shapes]
    items = [prepare_all(sid, DATA_ROOT) for sid in shape_list]
    for run in runs:
        run_dir = root / run
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        target = cfg["target"]
        for it in its:
            ck = run_dir / f"model_it{it}.pt"
            if not ck.exists():
                print(f"eval {run} it {it}: no checkpoint yet, skipped", flush=True)
                continue
            name = f"{run}_it{it}_grid.json" if grid else f"{run}_it{it}.json"
            if (out / name).exists():  # resumed: the evaluation of this checkpoint is done
                print(f"eval {run} it {it}: exists, skipped", flush=True)
                continue
            blob = torch.load(ck, map_location=device, weights_only=False)
            model = build_model(target).to(device)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            train_part = {"train_loss": train_loss(run_dir, it, int(cfg["val_every"])),
                          "split": split or "train_subset", "n_shapes": len(items)}
            if grid:
                settings = decoding_grid(target)
                summaries = grid_summaries(model, items, settings, device, DATA_ROOT)
                choice = _select(summaries)  # the band selection of training (H5-K1-1a)
                k = choice["index"]
                doc = {"run": run, "iteration": it, "grid": True, "selected": k,
                       "selected_setting": summaries[k]["setting"], "selected_name": summaries[k]["setting_name"],
                       "ratio": choice["ratio"], "in_band": choice["in_band"],
                       "settings": [s["setting"] for s in summaries], **train_part, "summaries": summaries}
                print(f"eval {run} it {it} grid: selected {summaries[k]['setting_name']} "
                      f"(ratio {choice['ratio']:.3f}, in band {choice['in_band']}), "
                      f"F1 {summaries[k]['deltas'][DK]['f1']:.4f}", flush=True)
            else:
                setting = json.loads(ck.with_suffix(".json").read_text(encoding="utf-8"))["setting"]
                summary = grid_summaries(model, items, [setting], device, DATA_ROOT)[0]  # deterministic forward inside
                doc = {"run": run, "iteration": it, "setting": setting, "setting_name": summary["setting_name"],
                       **train_part, "summary": summary}
                print(f"eval {run} it {it} {summary['setting_name']}: F1 {summary['deltas'][DK]['f1']:.4f}, "
                      f"ordinary recall {summary['diag']['ord_recall'][DK]:.3f}, "
                      f"median {summary['diag']['ord_median']:.4g}", flush=True)
            (out / name).write_text(json.dumps(doc, default=float), encoding="utf-8")


# ---------------------------------------------------------------- stage report

def _improvement(m_prev: float, m_final: float) -> float:
    """Relative improvement of the median distance from prev to final (see the module docstring)."""
    if math.isnan(m_prev):
        return float("nan")
    if math.isinf(m_prev):
        return 0.0 if math.isinf(m_final) else float("inf")
    if math.isinf(m_final):
        return float("-inf")
    return (m_prev - m_final) / m_prev


def run_row(doc: dict, loss: float) -> dict:
    s = doc["summary"]
    d = s["diag"]
    return {
        "iteration": doc["iteration"], "setting": doc["setting_name"], "train_loss": loss,
        "F1": s["deltas"][DK]["f1"], "precision": s["deltas"][DK]["precision"],
        "M": s["deltas"][DK]["walls_gaps_pooled_L_lt_1"], "M_comparable": m_comparable(s),
        "dec_gt_ratio": dec_gt_ratio(s),
        "ord_n": d["ord_n"], "ord_recall": {DK: d["ord_recall"][DK], DH: d["ord_recall"][DH]},
        "ord_median": d["ord_median"], "ord_p90": d["ord_p90"],
        "thin": [{"bin": BIN_LABELS[i], "n": d["thin_n"][i],
                  "recall": {DK: d["thin_recall"][DK][i], DH: d["thin_recall"][DH][i]},
                  "median": d["thin_median"][i], "p90": d["thin_p90"][i]} for i in range(len(BIN_LABELS))],
    }


def q0_rule(series: dict[str, list[dict]], final: int, prev: int) -> tuple[str, str, dict]:
    """Rule v1: the verdict and the per-run values it used."""
    per = {}
    for run, rows in series.items():
        by = {r["iteration"]: r for r in rows}
        reach = [r["iteration"] for r in rows if r["iteration"] <= final
                 and r["ord_recall"][DK] >= RULE["reach_recall"]]
        imp = _improvement(by[prev]["ord_median"], by[final]["ord_median"])
        per[run] = {"first_reach_iteration": min(reach) if reach else None,
                    "recall_final": by[final]["ord_recall"][DK],
                    "median_prev": by[prev]["ord_median"], "median_final": by[final]["ord_median"],
                    "improvement": imp,
                    "reached": bool(reach),
                    "low": by[final]["ord_recall"][DK] < RULE["low_recall"],
                    "flat": bool(imp < RULE["flat_improvement"])}  # nan compares False
    if all(p["reached"] for p in per.values()):
        return "REACHABLE", f"both runs reach ordinary recall(0.002) >= {RULE['reach_recall']} by {final}", per
    if all(p["low"] and p["flat"] for p in per.values()):
        return ("NOT REACHABLE", f"both runs: recall(0.002) < {RULE['low_recall']} at {final} and median improves "
                f"< {RULE['flat_improvement']:.0%} from {prev}", per)
    return "UNCLEAR", "neither rule holds", per


def q0_rule_v2(series: dict[str, list[dict]], final: int) -> tuple[str, str, dict]:
    """Rule v2 (H5-6c): verdict at ``final`` and the per-run values it used."""
    per = {}
    for run, rows in series.items():
        by = {r["iteration"]: r for r in rows}
        if final not in by:
            return "NOT AVAILABLE", f"no evaluation at iteration {final} for {run}", per
        reach = [r["iteration"] for r in rows if r["iteration"] <= final
                 and r["ord_recall"][DK] >= RULE_V2["reach_recall"]]
        rec = by[final]["ord_recall"][DK]
        per[run] = {"first_reach_iteration": min(reach) if reach else None, "recall_final": rec,
                    "reached": bool(reach), "budget_low": rec < RULE_V2["budget_recall"]}
    if all(p["reached"] for p in per.values()):
        return "REACHABLE", f"both runs reach ordinary recall(0.002) >= {RULE_V2['reach_recall']} by {final}", per
    if all(p["budget_low"] for p in per.values()):
        return ("NOT REACHABLE IN BUDGET", f"both runs: recall(0.002) < {RULE_V2['budget_recall']} at {final}", per)
    return "UNCLEAR", "neither rule v2 condition holds", per


def _f(x, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{d}g}"


def _load_series(root: Path, its: list[int]) -> dict[str, list[dict]]:
    """Rows of every run that has evaluations; iterations without an evaluation are left out."""
    ev = root / "eval"
    series: dict[str, list[dict]] = {}
    for run in RUNS:
        rows = []
        for it in its:
            path = ev / f"{run}_it{it}.json"
            if path.exists():
                doc = json.loads(path.read_text(encoding="utf-8"))
                rows.append(run_row(doc, float(doc["train_loss"])))
        if rows:
            series[run] = rows
    return series


def _descriptive(series: dict[str, list[dict]], lr_step: int, final: int) -> dict:
    """Ordinary recall and median at the lr step, the midpoint and the final iteration, per run (no rule)."""
    mid = (lr_step + final) // 2
    out = {}
    for run, rows in series.items():
        by = {r["iteration"]: r for r in rows}
        out[run] = {str(it): {"recall": by[it]["ord_recall"][DK], "median": by[it]["ord_median"]}
                    for it in (lr_step, mid, final) if it in by}
    return {"iterations": [lr_step, mid, final], "runs": out}


def _table_final(series: dict[str, list[dict]], final: int) -> dict | None:
    """Thin-bin recall(0.002) and median per L/h bin at ``final``, plus decoded/GT and precision, per run."""
    cols = {}
    for run, label in TABLE_ORDER:
        rows = series.get(run, [])
        hit = [r for r in rows if r["iteration"] == final]
        if hit:
            cols[label] = hit[0]
    if not cols:
        return None
    bins = [{"bin": b, **{label: {"recall": r["thin"][i]["recall"][DK], "median": r["thin"][i]["median"],
                                  "n": r["thin"][i]["n"]} for label, r in cols.items()}}
            for i, b in enumerate(BIN_LABELS)]
    return {"iteration": final, "bins": bins,
            "dec_gt_ratio": {label: r["dec_gt_ratio"] for label, r in cols.items()},
            "precision": {label: r["precision"] for label, r in cols.items()}}


def run_report(root: Path, its: list[int], final: int, prev: int, out: Path,
               v2_final: int, lr_step: int) -> dict:
    series = _load_series(root, its)
    v1_series = {run: series[run] for run in V1_RUNS if run in series}
    code, why, per = q0_rule(v1_series, final, prev)
    code2, why2, per2 = q0_rule_v2({run: series[run] for run in V1_RUNS if run in series}, v2_final)
    doc = {"its": its, "final_it": final, "prev_it": prev, "rule": RULE, "verdict": code, "verdict_reason": why,
           "per_run_rule": per, "v2": {"final_it": v2_final, "lr_step": lr_step, "rule": RULE_V2,
                                       "verdict": code2, "verdict_reason": why2, "per_run_rule": per2},
           "descriptive_lr": _descriptive(series, lr_step, v2_final),
           "table_final": _table_final(series, v2_final), "runs": series}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(doc), encoding="utf-8")
    print(f"VERDICT v1 at {final}: {code}  ({why})")
    for run, p in per.items():
        print(f"{run}: first recall(0.002) >= {RULE['reach_recall']} at {p['first_reach_iteration']}; "
              f"recall at {final} {_f(p['recall_final'])}; median {prev} {_f(p['median_prev'])} -> {final} "
              f"{_f(p['median_final'])} (improvement {_f(p['improvement'])})")
    print(f"VERDICT v2 at {v2_final}: {code2}  ({why2})")
    for run, p in per2.items():
        print(f"{run}: first recall(0.002) >= {RULE_V2['reach_recall']} at {p['first_reach_iteration']}; "
              f"recall at {v2_final} {_f(p['recall_final'])}")
    for run, v in doc["descriptive_lr"]["runs"].items():
        print(f"descriptive {run}: " + "; ".join(f"it {it}: recall {_f(x['recall'])}, median {_f(x['median'])}"
                                                  for it, x in v.items()))
    return doc


def to_markdown(doc: dict) -> str:
    v2 = doc["v2"]
    L = ["# H5-6 memorisation probe Q0", "",
         f"**Verdict v1 at {doc['final_it']}: {doc['verdict']}.** {doc['verdict_reason']}.", "",
         f"**Verdict v2 at {v2['final_it']}: {v2['verdict']}.** {v2['verdict_reason']}.", "",
         "Rule v1 (fixed before the run): REACHABLE if both runs (points_strong, hits_8) reach ordinary recall(0.002) "
         f">= 0.8 by {doc['final_it']}; NOT REACHABLE if both runs have recall(0.002) < 0.5 at {doc['final_it']} and the "
         f"median distance of ordinary hits improves by < 10 % from {doc['prev_it']} to {doc['final_it']}; UNCLEAR "
         "otherwise.", "",
         f"Rule v2 (fixed before the extension run; points_strong and hits_8): REACHABLE if both reach recall(0.002) "
         f">= 0.8 at any iteration <= {v2['final_it']}; NOT REACHABLE IN BUDGET if both are < 0.6 at "
         f"{v2['final_it']}; UNCLEAR otherwise. Learning-rate step: {v2['lr_step']}.", "",
         "Ordinary hits: not an endpoint of a thin GT interval. Recall = csgbox greedy match (decision 46). "
         "Distance = |t| to the nearest same-type decoded hit on the ray. Training set: 32 shapes, all rays. "
         "M is marked not comparable when decoded/GT hits is outside [0.8, 1.25].", "",
         f"## Rule v1 values (at {doc['final_it']})", "",
         "| run | first iteration with recall(0.002) >= 0.8 | recall(0.002) at final | median prev | median final | "
         "improvement |", "|---|---|---|---|---|---|"]
    for run, p in doc["per_run_rule"].items():
        L.append(f"| {run} | {p['first_reach_iteration'] or 'none'} | {_f(p['recall_final'])} | "
                 f"{_f(p['median_prev'])} | {_f(p['median_final'])} | {_f(p['improvement'])} |")
    L += ["", f"## Rule v2 values (at {v2['final_it']})", "",
          "| run | first iteration with recall(0.002) >= 0.8 | recall(0.002) at final |", "|---|---|---|"]
    for run, p in v2["per_run_rule"].items():
        L.append(f"| {run} | {p['first_reach_iteration'] or 'none'} | {_f(p['recall_final'])} |")
    desc = doc["descriptive_lr"]
    L += ["", f"## Effect of the learning-rate drop (descriptive, no rule; step at {v2['lr_step']})", "",
          "| run | iteration | ordinary recall 0.002 | ordinary median |", "|---|---|---|---|"]
    for run, v in desc["runs"].items():
        for it, x in v.items():
            L.append(f"| {run} | {it} | {_f(x['recall'])} | {_f(x['median'])} |")
    tf = doc["table_final"]
    if tf is not None:
        labels = [label for _, label in TABLE_ORDER if label in tf["dec_gt_ratio"]]
        L += ["", f"## Thin bins at {tf['iteration']} (descriptive): recall(0.002) and median per L/h bin", "",
              "| bin | " + " | ".join(f"{label} recall / median" for label in labels) + " |",
              "|---|" + "---|" * len(labels)]
        for b in tf["bins"]:
            L.append(f"| {b['bin']} | " + " | ".join(f"{_f(b[label]['recall'])} / {_f(b[label]['median'])}"
                                                    for label in labels) + " |")
        L.append("| decoded/GT | " + " | ".join(_f(tf["dec_gt_ratio"][label]) for label in labels) + " |")
        L.append("| precision | " + " | ".join(_f(tf["precision"][label]) for label in labels) + " |")
    for run, rows in doc["runs"].items():
        L += ["", f"## {run}", "",
              "| iteration | setting | train loss | F1 | precision | decoded/GT | M | ord n | ord recall 0.002 | "
              "ord recall h/2 | ord median | ord p90 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            mark = "" if r["M_comparable"] else " (not comparable)"
            L.append(f"| {r['iteration']} | {r['setting']} | {_f(r['train_loss'])} | {_f(r['F1'])} | "
                     f"{_f(r['precision'])} | {_f(r['dec_gt_ratio'])} | {_f(r['M'])}{mark} | {r['ord_n']} | "
                     f"{_f(r['ord_recall'][DK])} | {_f(r['ord_recall'][DH])} | {_f(r['ord_median'])} | "
                     f"{_f(r['ord_p90'])} |")
        L += ["", f"Thin hits per L/h bin ({run}):", "",
              "| iteration | bin | n | recall 0.002 | recall h/2 | median | p90 |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            for t in r["thin"]:
                L.append(f"| {r['iteration']} | {t['bin']} | {t['n']} | {_f(t['recall'][DK])} | "
                         f"{_f(t['recall'][DH])} | {_f(t['median'])} | {_f(t['p90'])} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["eval", "report"])
    p.add_argument("--root", type=Path, required=True, help="Q0 run root (contains the Q0_* run directories)")
    p.add_argument("--its", type=_its, default=_its("5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,"
                                                    "55000,60000"))
    p.add_argument("--final-it", type=int, default=30000, help="rule v1: the verdict iteration")
    p.add_argument("--prev-it", type=int, default=20000, help="rule v1: the iteration the median is compared from")
    p.add_argument("--v2-final", type=int, default=60000, help="rule v2: the verdict iteration")
    p.add_argument("--lr-step", type=int, default=50000, help="the iteration of the learning-rate drop (descriptive)")
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "q0")
    p.add_argument("--runs", type=lambda s: tuple(x for x in s.split(",") if x.strip()), default=RUNS,
                   help="eval: comma-separated run directory names (default: the Q0 runs)")
    p.add_argument("--split", type=str, default=None,
                   help="eval: shape split to evaluate on (default: the training subset of the Q0 runs)")
    p.add_argument("--n-shapes", type=int, default=None, help="eval with --split: the first n shapes of the split")
    p.add_argument("--grid", action="store_true",
                   help="eval: all settings of the decoding grid and the band selection (writes *_grid.json)")
    args = p.parse_args(argv)
    if args.stage == "eval":
        run_eval(args.root, args.its, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 args.runs, args.split, args.n_shapes, args.grid)
    else:
        run_report(args.root, args.its, args.final_it, args.prev_it, args.out, args.v2_final, args.lr_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
