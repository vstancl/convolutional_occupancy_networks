"""Memorisation probe Q0 (H5-6): evaluation of the checkpoints and the decision report.

Two stages, both run by ``h5/scripts/q0.sh``:

- ``eval``: evaluates every ``model_it{N}.pt`` of the two Q0 runs (Q0_points_strong, Q0_hits_8) on the 32 training
  shapes of the run (``train_subset.json``, identical for both runs) with **all rays**, at the decoding setting
  stored next to the checkpoint (``model_it{N}.json``). Writes ``ROOT/eval/{run}_it{N}.json``.
- ``report``: writes ``OUT/report.md`` and ``OUT/report.json`` and prints the verdict of the rule below.

Per run and iteration the report holds: ordinary hits (recall at 0.002 and at h/2, median and p90 distance),
the same per thin bin (L/h), F1 of all hits, decoded/GT hits, metric M (with the not-comparable mark) and the
training loss (mean of the logged losses over the validation interval ending at the iteration).

Rule (fixed before the run; first match wins). Let ``final`` = 30000 and ``prev`` = 20000:

- REACHABLE if, for both runs, recall(0.002) of ordinary hits is >= 0.8 at some iteration <= ``final``;
- NOT REACHABLE if, for both runs, recall(0.002) of ordinary hits at ``final`` is < 0.5 and the median distance of
  ordinary hits improves by < 10 % from ``prev`` to ``final`` (improvement = (median_prev - median_final) /
  median_prev; a median that becomes inf counts as no improvement);
- UNCLEAR otherwise.
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

from csgbox.constants import GATE_HIT_DELTA  # noqa: E402

from data import DATA_ROOT  # noqa: E402
from eval_rays import (  # noqa: E402
    BIN_LABELS,
    EVAL_DELTAS,
    dec_gt_ratio,
    grid_summaries,
    m_comparable,
    prepare_all,
)
from models import build_model  # noqa: E402

RUNS: tuple[str, ...] = ("Q0_points_strong", "Q0_hits_8")
DK, DH = str(EVAL_DELTAS[0]), str(EVAL_DELTAS[1])  # the keys of delta 0.002 and delta h/2
RULE = {"reach_recall": 0.8, "low_recall": 0.5, "flat_improvement": 0.10}


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

def run_eval(root: Path, its: list[int], device: torch.device) -> None:
    out = root / "eval"
    out.mkdir(parents=True, exist_ok=True)
    shapes = {run: json.loads((root / run / "train_subset.json").read_text(encoding="utf-8")) for run in RUNS}
    if any(shapes[r] != shapes[RUNS[0]] for r in RUNS):
        raise ValueError("the Q0 runs do not use the same training shapes")
    items = [prepare_all(sid, DATA_ROOT) for sid in shapes[RUNS[0]]]
    for run in RUNS:
        run_dir = root / run
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        target = cfg["target"]
        for it in its:
            ck = run_dir / f"model_it{it}.pt"
            if not ck.exists():
                raise FileNotFoundError(f"missing checkpoint {ck}")
            if (out / f"{run}_it{it}.json").exists():  # resumed: the evaluation of this checkpoint is done
                print(f"eval {run} it {it}: exists, skipped", flush=True)
                continue
            setting = json.loads(ck.with_suffix(".json").read_text(encoding="utf-8"))["setting"]
            blob = torch.load(ck, map_location=device, weights_only=False)
            model = build_model(target).to(device)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            summary = grid_summaries(model, items, [setting], device, DATA_ROOT)[0]  # deterministic forward inside
            doc = {"run": run, "iteration": it, "setting": setting, "setting_name": summary["setting_name"],
                   "n_shapes": len(items), "train_loss": train_loss(run_dir, it, int(cfg["val_every"])),
                   "summary": summary}
            (out / f"{run}_it{it}.json").write_text(json.dumps(doc, default=float), encoding="utf-8")
            print(f"eval {run} it {it} {summary['setting_name']}: F1 {summary['deltas'][DK]['f1']:.4f}, "
                  f"ordinary recall {summary['diag']['ord_recall'][DK]:.3f}, "
                  f"median {summary['diag']['ord_median']:.4g}", flush=True)


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
    """The verdict of the fixed rule and the per-run values it used."""
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


def _f(x, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{d}g}"


def run_report(root: Path, its: list[int], final: int, prev: int, out: Path) -> dict:
    ev = root / "eval"
    series: dict[str, list[dict]] = {}
    for run in RUNS:
        series[run] = []
        for it in its:
            doc = json.loads((ev / f"{run}_it{it}.json").read_text(encoding="utf-8"))
            series[run].append(run_row(doc, float(doc["train_loss"])))
    code, why, per = q0_rule(series, final, prev)
    doc = {"its": its, "final_it": final, "prev_it": prev, "rule": RULE, "verdict": code, "verdict_reason": why,
           "per_run_rule": per, "runs": series}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(doc), encoding="utf-8")
    print(f"VERDICT: {code}  ({why})")
    for run, p in per.items():
        print(f"{run}: first recall(0.002) >= {RULE['reach_recall']} at {p['first_reach_iteration']}; "
              f"recall at {final} {_f(p['recall_final'])}; median {prev} {_f(p['median_prev'])} -> {final} "
              f"{_f(p['median_final'])} (improvement {_f(p['improvement'])})")
    return doc


def to_markdown(doc: dict) -> str:
    L = ["# H5-6 memorisation probe Q0", "",
         f"**Verdict: {doc['verdict']}.** {doc['verdict_reason']}.", "",
         "Rule (fixed before the run): REACHABLE if both runs reach ordinary recall(0.002) >= 0.8 by "
         f"{doc['final_it']}; NOT REACHABLE if both runs have recall(0.002) < 0.5 at {doc['final_it']} and the median "
         f"distance of ordinary hits improves by < 10 % from {doc['prev_it']} to {doc['final_it']}; UNCLEAR otherwise.",
         "", "Ordinary hits: not an endpoint of a thin GT interval. Recall = csgbox greedy match (decision 46). "
         "Distance = |t| to the nearest same-type decoded hit on the ray. Training set: 32 shapes, all rays. "
         "M is marked not comparable when decoded/GT hits is outside [0.8, 1.25].", "",
         "## Per-run rule values", "",
         "| run | first iteration with recall(0.002) >= 0.8 | recall(0.002) at final | median prev | median final | "
         "improvement |", "|---|---|---|---|---|---|"]
    for run, p in doc["per_run_rule"].items():
        L.append(f"| {run} | {p['first_reach_iteration'] or 'none'} | {_f(p['recall_final'])} | "
                 f"{_f(p['median_prev'])} | {_f(p['median_final'])} | {_f(p['improvement'])} |")
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
    p.add_argument("--root", type=Path, required=True, help="Q0 run root (contains Q0_points_strong, Q0_hits_8)")
    p.add_argument("--its", type=_its, default=_its("5000,10000,15000,20000,25000,30000"))
    p.add_argument("--final-it", type=int, default=30000)
    p.add_argument("--prev-it", type=int, default=20000)
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "q0")
    args = p.parse_args(argv)
    if args.stage == "eval":
        run_eval(args.root, args.its, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        run_report(args.root, args.its, args.final_it, args.prev_it, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
