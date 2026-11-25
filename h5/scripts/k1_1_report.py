"""H5-K1-1 report: the checkpoints of the K1-1 runs, evaluated by ``q0_report.py eval`` (all rays, first 200 val shapes).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/k1_1_report.py --root ~/PhD/runs/k1_1 \
        --its 40000,80000,120000 --out h5/results/k1_1

Reads ``ROOT/eval/{run}_it{N}.json`` (the evaluation with the setting of training; missing checkpoints are listed, not
an error) and, for occfrac_128, ``ROOT/eval/occfrac_128_it{N}_grid.json`` (H5-K1-1b: every setting of the extended
grid, with the band selection). Writes ``OUT/report.json`` and ``OUT/report.md``.

Per row (run and checkpoint): the values of ``q0_report.run_row``: ordinary recall at 0.002 and h/2, ordinary median and
p90 distance, thin-bin recall per L/h bin, the weighted recall of the L/h < 1 bins, F1 of all hits at 0.002, precision,
decoded/GT, whether that ratio is in ``RATIO_BAND``, and the setting.

Verdict (H5-K1-1b, rule of SPEC section 5, layer 1, delta 0.002, at the last checkpoint): hits_8 against the best dense
run that is in band (dense = occfrac_128, with the grid selection, and points_strong). Rule, fixed for this report:

- H5 supported if the recall of the thin features with L/h < 1 (population-weighted over the four bins) is at least
  0.05 higher for hits_8, and the F1 of all hits is not lower by more than 0.01 (the t/h > 2 condition of the SPEC,
  checked with F1 because the rule does not give a separate t/h > 2 quantity; IoU is not evaluated here);
- H5 refuted if, in every L/h bin, the recall of hits_8 is at most 0.02 higher than that of the dense run;
- otherwise undecided (the SPEC calls this "compactness only"). If no dense run is in band: "not evaluable".
"""

from __future__ import annotations

import argparse
import json
import math
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

from eval_rays import BIN_LABELS, RATIO_BAND, dec_gt_ratio  # noqa: E402
from q0_report import DH, DK, _f, run_row  # noqa: E402

K1_RUNS: tuple[str, ...] = ("hits_8", "occfrac_128", "points_strong")
GRID_LABEL = "occfrac_128 (grid)"
RULE = {"small_bins": 4, "small_gain": 0.05, "f1_tolerance": 0.01, "refute_gain": 0.02}


def _its(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def in_band(ratio: float) -> bool:
    return bool(RATIO_BAND[0] <= ratio <= RATIO_BAND[1])


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build(root: Path, its: list[int]) -> dict:
    """Rows per run label (checkpoints in ``its`` order), the evaluation set, and the verdict."""
    ev = root / "eval"
    runs: dict[str, list[dict]] = {}
    missing: list[str] = []
    split_info: set[tuple[str, int]] = set()
    for run in K1_RUNS:
        rows = []
        for it in its:
            doc = _load(ev / f"{run}_it{it}.json")
            if doc is None:
                missing.append(f"{run}_it{it}")
                continue
            split_info.add((doc.get("split", "?"), int(doc.get("n_shapes", -1))))
            row = run_row(doc, float(doc["train_loss"]))
            row["in_band"] = in_band(row["dec_gt_ratio"])
            rows.append(row)
        if rows:
            runs[run] = rows
    if "occfrac_128" in K1_RUNS:
        rows = []
        for it in its:
            doc = _load(ev / f"occfrac_128_it{it}_grid.json")
            if doc is None:
                missing.append(f"occfrac_128_it{it} (grid)")
                continue
            split_info.add((doc.get("split", "?"), int(doc.get("n_shapes", -1))))
            sel = doc["summaries"][doc["selected"]]
            row_doc = {"iteration": it, "setting_name": doc["selected_name"], "summary": sel,
                       "train_loss": doc["train_loss"]}
            row = run_row(row_doc, float(doc["train_loss"]))
            row["in_band"] = in_band(row["dec_gt_ratio"])
            row["grid_all"] = [{"setting": s["setting_name"], "F1": s["deltas"][DK]["f1"],
                                "ratio": dec_gt_ratio(s), "in_band": in_band(dec_gt_ratio(s))}
                               for s in doc["summaries"]]
            rows.append(row)
        if rows:
            runs[GRID_LABEL] = rows
    return {"its": its, "ratio_band": list(RATIO_BAND),
            "evaluation": [{"split": s, "n_shapes": n} for s, n in sorted(split_info)],
            "missing": missing, "runs": runs, "verdict": verdict(runs, its)}


def _row(rows: list[dict], it: int) -> dict | None:
    for r in rows:
        if r["iteration"] == it:
            return r
    return None


def small_recall(row: dict) -> float:
    """Population-weighted recall at 0.002 of the thin features in the L/h < 1 bins (as the M of H5-6, all classes)."""
    bins = row["thin"][:RULE["small_bins"]]
    n = sum(b["n"] for b in bins)
    return sum(b["recall"][DK] * b["n"] for b in bins if b["n"]) / n if n else float("nan")


def verdict(runs: dict[str, list[dict]], its: list[int]) -> dict:
    final = its[-1]
    hits = _row(runs.get("hits_8", []), final)
    dense = []
    for label in (GRID_LABEL, "occfrac_128", "points_strong"):
        row = _row(runs.get(label, []), final)
        if row is not None and row["in_band"]:
            dense.append((label, row))
    if hits is None or not dense:
        return {"iteration": final, "status": "not evaluable",
                "reason": "no dense run is in band" if hits is not None else "hits_8 not evaluated"}
    label, drow = max(dense, key=lambda x: x[1]["F1"] if x[1]["F1"] == x[1]["F1"] else -math.inf)
    d_small = small_recall(hits) - small_recall(drow)
    d_f1 = hits["F1"] - drow["F1"]
    gains = [h["recall"][DK] - d["recall"][DK] for h, d in zip(hits["thin"], drow["thin"])
             if h["n"] and d["n"]]
    max_gain = max(gains) if gains else float("nan")
    if d_small >= RULE["small_gain"] and d_f1 >= -RULE["f1_tolerance"]:
        status = "H5 supported"
    elif max_gain <= RULE["refute_gain"]:
        status = "H5 refuted"
    else:
        status = "undecided (SPEC: compactness only)"
    return {"iteration": final, "status": status, "dense_run": label, "in_band_dense": [x[0] for x in dense],
            "small_recall_hits": small_recall(hits), "small_recall_dense": small_recall(drow),
            "gain_small": d_small, "F1_hits": hits["F1"], "F1_dense": drow["F1"], "gain_F1": d_f1,
            "max_bin_gain": max_gain, "rule": RULE}


def to_markdown(doc: dict) -> str:
    ev = ", ".join(f"{e['split']} (first {e['n_shapes']} shapes, all rays)" for e in doc["evaluation"]) or "none"
    v = doc["verdict"]
    L = ["# H5-K1-1 report", "",
         f"Evaluation: {ev}. Setting of each row: the H5-K1-1a band selection of training (rows without a label), and for "
         "occfrac_128 (grid) the band selection over the extended grid, which includes `binarize` (H5-K1-1b). Best F1 "
         f"at 0.002 among the settings with decoded/GT in [{doc['ratio_band'][0]:g}, {doc['ratio_band'][1]:g}]; otherwise "
         "the closest ratio. Recall and distances at delta 0.002 unless labelled h/2.", "",
         "M (H5-6) in the per-checkpoint table: pooled recall at 0.002 of the walls and gaps with L/h < 1 only. The "
         "verdict instead uses the population-weighted recall at 0.002 of all thin features (material and gap) in the "
         "four L/h < 1 bins of the thin-bin tables, so the two numbers differ.", ""]
    if doc["missing"]:
        L += [f"Not evaluated yet: {', '.join(doc['missing'])}.", ""]
    L += ["## Verdict (H5, SPEC section 5, layer 1, delta 0.002)", ""]
    if v["status"] == "not evaluable":
        L += [f"**not evaluable** at iteration {v['iteration']}: {v['reason']}.", ""]
    else:
        L += [f"**{v['status']}** at iteration {v['iteration']}: hits_8 against {v['dense_run']} (the best F1 of the "
              f"dense runs in band: {', '.join(v['in_band_dense'])}).", "",
              "| quantity | hits_8 | dense | difference | rule |", "|---|---|---|---|---|",
              f"| weighted recall of all thin features, L/h < 1 | {_f(v['small_recall_hits'])} | {_f(v['small_recall_dense'])} | "
              f"{_f(v['gain_small'])} | at least +{v['rule']['small_gain']} |",
              f"| F1 of all hits, 0.002 | {_f(v['F1_hits'])} | {_f(v['F1_dense'])} | {_f(v['gain_F1'])} | "
              f"not below -{v['rule']['f1_tolerance']} (t/h > 2 condition) |",
              f"| largest recall gain of hits_8 over the dense run, any L/h bin | | | {_f(v['max_bin_gain'])} | "
              f"at most +{v['rule']['refute_gain']} in every bin for refuted |", "",
              "The rule is applied to the ray layer only; IoU and the mesh layer are not evaluated in this report. "
              "The SPEC condition \"with a smaller output\" is not checked here, and the SPEC compares against TSDF "
              "M = 128, which K1-1 did not train: the best dense run in band stands in for it.", ""]
    L += ["## Per checkpoint", "",
          "| run | iteration | selected setting | F1 (all hits, 0.002) | precision | decoded/GT | in band | "
          "ord n | ord recall 0.002 | ord recall h/2 | ord median | ord p90 | M (H5-6, walls+gaps L/h < 1) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, rows in doc["runs"].items():
        for r in rows:
            L.append(f"| {run} | {r['iteration']} | {r['setting']} | {_f(r['F1'])} | {_f(r['precision'])} | "
                     f"{_f(r['dec_gt_ratio'])} | {'yes' if r['in_band'] else 'no'} | {r['ord_n']} | "
                     f"{_f(r['ord_recall'][DK])} | {_f(r['ord_recall'][DH])} | {_f(r['ord_median'])} | "
                     f"{_f(r['ord_p90'])} | {_f(r['M'])} |")
    for key, title in ((DK, "thin recall at 0.002"), (DH, "thin recall at h/2 (descriptive, not the criterion)")):
        for run, rows in doc["runs"].items():
            L += ["", f"## Thin bins, {run}: {title} (n) per L/h bin", "",
                  "| L/h bin | " + " | ".join(f"it {r['iteration']}" for r in rows) + " |",
                  "|---|" + "---|" * len(rows)]
            for i, lab in enumerate(BIN_LABELS):
                cells = [f"{_f(r['thin'][i]['recall'][key])} ({r['thin'][i]['n']})" for r in rows]
                L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="K1-1 run root (contains eval/ and the run directories)")
    p.add_argument("--its", type=_its, default=_its("40000,80000,120000"))
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "k1_1")
    args = p.parse_args(argv)
    doc = build(args.root, args.its)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (args.out / "report.md").write_text(to_markdown(doc), encoding="utf-8")
    n = sum(len(v) for v in doc["runs"].values())
    print(f"report: {n} rows, {len(doc['missing'])} missing, verdict: {doc['verdict']['status']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
