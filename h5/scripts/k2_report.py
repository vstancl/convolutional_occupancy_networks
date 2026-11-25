"""H5-K2-2 report: point decoders with ray-interval query points against Q0_points_strong (memorisation, 32 shapes).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/k2_report.py --root ~/PhD/runs/k2 \
        --q0-root ~/PhD/runs/h5_q0 [--final-it 60000] [--out h5/results/k2]

Reads ``ROOT/eval/K2_points_rayint_{a,b,c}_it{N}_grid.json`` and ``Q0_ROOT/eval/Q0_points_strong_it{N}_grid.json``
(``q0_report.py eval --grid``: all rays of the 32 training shapes, every iso of the grid, the band selection of
``train._select``). Writes ``OUT/report.json`` and ``OUT/report.md``.

Per run at ``--final-it`` (the selected setting): ordinary recall at 0.002, weighted thin recall L/h < 1 (the measure of
Q0: per-hit thin recall at 0.002, population-weighted over the four L/h < 1 bins), M (walls and gaps, both ends
matched, pooled over L/h < 1), precision, decoded/GT, F1 of all hits at 0.002, and the per-bin thin tables (0.002 and
h/2). The K2 runs are also listed per checkpoint.

Rule (fixed on 24. 11. 2026, at ``--final-it``):

- a candidate meets the thresholds if its weighted thin recall >= 0.297 AND ordinary recall >= 0.702 AND decoded/GT
  in [0.8, 1.25]. The candidates are points_rayint_a and (H5-K2-1c) the thin-focused points_rayint_c, under the same
  rule; CONTINUE if at least one of them meets it (the summary names which), otherwise STOP;
- points_rayint_b stays the control: if it meets the same thresholds, "lines, not the weight" (continue with b).

The thresholds are the Q0_points_strong values at 60000 (weighted thin recall 0.247, ordinary recall 0.722) + 0.05 and
- 0.02. The same rule against the re-evaluated Q0_points_strong values is printed as well; it is descriptive only, the
fixed numbers decide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from eval_rays import BIN_LABELS, RATIO_BAND  # noqa: E402
from q0_report import DH, DK, _f, run_row  # noqa: E402

K2_RUNS: tuple[str, ...] = ("K2_points_rayint_a", "K2_points_rayint_b", "K2_points_rayint_c")
CANDIDATES: tuple[str, ...] = ("a", "c")  # the variants the rule decides on; b is the control
Q0_RUN = "Q0_points_strong"
SMALL_BINS = 4  # the L/h < 1 bins
Q0_FIXED = {"thin": 0.247, "ord": 0.722}  # Q0_points_strong at 60000 (24. 11. 2026)
GAIN_THIN = 0.05
LOSS_ORD = 0.02
THRESHOLDS = {"thin": round(Q0_FIXED["thin"] + GAIN_THIN, 6), "ord": round(Q0_FIXED["ord"] - LOSS_ORD, 6)}


def weighted_thin(row: dict) -> float:
    """Per-hit thin recall at 0.002, population-weighted over the L/h < 1 bins (the Q0 measure)."""
    bins = row["thin"][:SMALL_BINS]
    n = sum(b["n"] for b in bins)
    return sum(b["recall"][DK] * b["n"] for b in bins if b["n"]) / n if n else float("nan")


def in_band(ratio: float) -> bool:
    return bool(RATIO_BAND[0] <= ratio <= RATIO_BAND[1])


def _row(path: Path) -> dict | None:
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    sel = doc["summaries"][doc["selected"]]
    row = run_row({"iteration": doc["iteration"], "setting_name": doc["selected_name"], "summary": sel},
                  float(doc["train_loss"]))
    row["weighted_thin"] = weighted_thin(row)
    row["in_band"] = in_band(row["dec_gt_ratio"])
    row["split"], row["n_shapes"] = doc.get("split"), doc.get("n_shapes")
    return row


def meets(row: dict | None, thr: dict) -> bool:
    return (row is not None and row["weighted_thin"] >= thr["thin"] and row["ord_recall"][DK] >= thr["ord"]
            and row["in_band"])


def rule(rows: dict[str, dict | None], thr: dict) -> dict:
    """The fixed rule on the variants ``rows`` (keys a, b, c): candidates a and c, control b."""
    evaluated = [v for v in CANDIDATES if rows.get(v) is not None]
    if not evaluated:
        return {"decision": "not evaluable", "reason": "neither K2_points_rayint_a nor K2_points_rayint_c evaluated"}
    met = [v for v in CANDIDATES if meets(rows.get(v), thr)]
    return {"decision": "CONTINUE" if met else "STOP", "meets": met, "evaluated": evaluated,
            "a_meets": "a" in met, "c_meets": "c" in met, "b_meets": meets(rows.get("b"), thr),
            "lines_not_weight": meets(rows.get("b"), thr), "thresholds": thr}


def _subset(root: Path, run: str) -> list[str] | None:
    p = root / run / "train_subset.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build(root: Path, q0_root: Path, final: int) -> dict:
    per_it: dict[str, list[dict]] = {}
    for run in K2_RUNS:
        rows = []
        for path in sorted((root / "eval").glob(f"{run}_it*_grid.json"),
                           key=lambda p: int(p.stem.split("_it")[1].split("_")[0])):
            rows.append(_row(path))
        per_it[run] = [r for r in rows if r is not None]
    final_rows = {run: next((r for r in per_it[run] if r["iteration"] == final), None) for run in K2_RUNS}
    final_rows[Q0_RUN] = _row(q0_root / "eval" / f"{Q0_RUN}_it{final}_grid.json")
    subsets = {run: _subset(root, run) for run in K2_RUNS}
    subsets[Q0_RUN] = _subset(q0_root, Q0_RUN)
    known = [s for s in subsets.values() if s is not None]
    same_shapes = bool(known) and all(s == known[0] for s in known)
    variants = {v: final_rows[f"K2_points_rayint_{v}"] for v in ("a", "b", "c")}
    q = final_rows[Q0_RUN]
    fixed = rule(variants, THRESHOLDS)
    if q is not None:
        thr_q = {"thin": q["weighted_thin"] + GAIN_THIN, "ord": q["ord_recall"][DK] - LOSS_ORD}
        descriptive = rule(variants, thr_q)
    else:
        descriptive = {"decision": "not evaluable", "reason": f"{Q0_RUN} it {final} (grid) not evaluated"}
    missing = [run for run, r in final_rows.items() if r is None]
    return {"final_it": final, "ratio_band": list(RATIO_BAND), "q0_fixed": Q0_FIXED, "thresholds": THRESHOLDS,
            "same_training_shapes": same_shapes, "missing": missing, "final": final_rows, "per_it": per_it,
            "rule_fixed": fixed, "rule_descriptive": descriptive}


def _rule_lines(r: dict) -> list[str]:
    if r["decision"] == "not evaluable":
        return [f"**not evaluable**: {r['reason']}."]
    thr = r["thresholds"]
    who = ", ".join(f"points_rayint_{v}" for v in r["meets"]) or "none of points_rayint_a / points_rayint_c"
    L = [f"**{r['decision']}**: meets the thresholds: {who} (evaluated: "
         f"{', '.join('points_rayint_' + v for v in r['evaluated'])}; weighted thin recall >= {_f(thr['thin'], 4)}, "
         f"ordinary recall >= {_f(thr['ord'], 4)}, decoded/GT in band)."]
    if r["lines_not_weight"]:
        L.append("points_rayint_b meets the same thresholds: **lines, not the weight** (continue with b).")
    else:
        L.append("points_rayint_b does not meet the same thresholds.")
    return L


def to_markdown(doc: dict) -> str:
    f = doc["final_it"]
    n = sorted({r["n_shapes"] for r in doc["final"].values() if r is not None})
    L = [f"# H5-K2-2 report: ray-interval query points (memorisation, {', '.join(map(str, n)) or '?'} training shapes)", "",
         f"Evaluation: all rays of the training shapes, the band selection over the iso grid (decoded/GT in "
         f"[{doc['ratio_band'][0]:g}, {doc['ratio_band'][1]:g}], best F1 at 0.002; otherwise the closest ratio). "
         "Weighted thin recall L/h < 1: per-hit thin recall at 0.002, population-weighted over the four L/h < 1 bins "
         "(the Q0 measure). M: walls and gaps with both ends matched, pooled over L/h < 1.", "",
         f"Same training shapes in all three runs: {'yes' if doc['same_training_shapes'] else '**no**'}.", ""]
    if doc["missing"]:
        L += [f"Not evaluated at {f}: {', '.join(doc['missing'])}.", ""]
    L += [f"## Rule (fixed on 24. 11., at iteration {f})", ""] + _rule_lines(doc["rule_fixed"]) + [""]
    L += ["## The same rule against the re-evaluated Q0_points_strong (descriptive only; the fixed numbers decide)", ""]
    L += _rule_lines(doc["rule_descriptive"]) + [""]
    L += [f"## Iteration {f}", "",
          "| run | setting | ordinary recall 0.002 | weighted thin recall L/h < 1 | M | precision | decoded/GT | in band | F1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for run, r in doc["final"].items():
        if r is None:
            L.append(f"| {run} | not evaluated | | | | | | | |")
            continue
        L.append(f"| {run} | {r['setting']} | {_f(r['ord_recall'][DK])} | {_f(r['weighted_thin'])} | {_f(r['M'])} | "
                 f"{_f(r['precision'])} | {_f(r['dec_gt_ratio'])} | {'yes' if r['in_band'] else 'no'} | {_f(r['F1'])} |")
    for key, title in ((DK, "0.002"), (DH, "h/2 (descriptive)")):
        L += ["", f"## Thin recall per L/h bin at {title}, iteration {f} (n)", "",
              "| L/h bin | " + " | ".join(doc["final"]) + " |", "|---|" + "---|" * len(doc["final"])]
        for i, lab in enumerate(BIN_LABELS):
            cells = ["" if r is None else f"{_f(r['thin'][i]['recall'][key])} ({r['thin'][i]['n']})"
                     for r in doc["final"].values()]
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "## K2 runs per checkpoint", "",
          "| run | iteration | setting | ordinary recall 0.002 | weighted thin recall L/h < 1 | M | precision | "
          "decoded/GT | in band | F1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run, rows in doc["per_it"].items():
        for r in rows:
            L.append(f"| {run} | {r['iteration']} | {r['setting']} | {_f(r['ord_recall'][DK])} | "
                     f"{_f(r['weighted_thin'])} | {_f(r['M'])} | {_f(r['precision'])} | {_f(r['dec_gt_ratio'])} | "
                     f"{'yes' if r['in_band'] else 'no'} | {_f(r['F1'])} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="K2 run root (contains eval/ and the K2 run directories)")
    p.add_argument("--q0-root", type=Path, default=Path.home() / "PhD" / "runs" / "h5_q0")
    p.add_argument("--final-it", type=int, default=60000)
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "k2")
    args = p.parse_args(argv)
    doc = build(args.root, args.q0_root, args.final_it)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (args.out / "report.md").write_text(to_markdown(doc), encoding="utf-8")
    d = doc["rule_descriptive"]["decision"]
    r = doc["rule_fixed"]
    met = ", ".join(r.get("meets", [])) or "none"
    lines = "; control b meets it: lines, not the weight" if r.get("lines_not_weight") else ""
    print(f"report: rule {r['decision']} (meets: {met}){lines}; against re-evaluated Q0 (descriptive): {d}; "
          f"same shapes {doc['same_training_shapes']}; missing {doc['missing']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
