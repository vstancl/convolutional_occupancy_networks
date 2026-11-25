"""H5-K1-1 report: the checkpoints of the K1-1 runs, evaluated by ``q0_report.py eval`` (all rays, first 200 val shapes).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/k1_1_report.py --root ~/PhD/runs/k1_1 \
        --its 40000,80000,120000 --out h5/results/k1_1

Reads ``ROOT/eval/{run}_it{N}.json`` (the evaluation with the setting of training; missing checkpoints are listed, not
an error) and, for occfrac_128, ``ROOT/eval/occfrac_128_it{N}_grid.json`` (H5-K1-1b: every setting of the extended
grid, with the band selection). Writes ``OUT/report.json`` and ``OUT/report.md``.

Per row (run and checkpoint): the values of ``q0_report.run_row``: ordinary recall at 0.002 and h/2, ordinary median and
p90 distance, thin-bin recall per L/h bin, the weighted recall of the L/h < 1 bins, F1 of all hits at 0.002, precision,
decoded/GT, whether that ratio is in ``RATIO_BAND``, and the setting.

Verdict (H5-K1-1c, rule of SPEC section 5, layer 1, delta 0.002, at the last checkpoint): hits_8 against the dense ray
representations that are in band: occfrac_128 with the grid selection, and tsdf_128 and binary_128 once their eval
files exist. points_strong is never a dense competitor; it enters the H5 rule only through the cost criterion of the
SPEC, which this report does not evaluate. The dense value of a quantity is the better of the dense runs in band, per L/h
bin for the per-bin values. Thin-feature quantity: the walls-and-gaps recall with both ends matched (``walls_gaps`` per
L/h bin and ``walls_gaps_pooled_L_lt_1`` = the M of H5-6, from the evaluation summary), not the per-hit thin recall.
Rule, fixed for this report:

- H5 supported if M of hits_8 is at least 0.05 higher than the dense M, and the F1 of all hits is not lower by more
  than 0.01 (the t/h > 2 condition of the SPEC, checked with F1; IoU is not evaluated here);
- H5 refuted if, in every L/h bin, the walls-and-gaps recall of hits_8 is at most 0.02 higher than the dense value;
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
OPTIONAL_DENSE: tuple[str, ...] = ("tsdf_128", "binary_128")  # loaded when their eval files exist; not listed missing
GRID_LABEL = "occfrac_128 (grid)"
DENSE: tuple[str, ...] = (GRID_LABEL,) + OPTIONAL_DENSE  # the dense ray competitors of the H5 rule (H5-K1-1c)
RULE = {"small_bins": 4, "small_gain": 0.05, "f1_tolerance": 0.01, "refute_gain": 0.02}


def _its(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def in_band(ratio: float) -> bool:
    return bool(RATIO_BAND[0] <= ratio <= RATIO_BAND[1])


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _walls_gaps(row: dict, summary: dict) -> dict:
    """Adds the walls-and-gaps recall (both ends matched) per L/h bin at 0.002 and h/2 to ``row``."""
    row["walls_gaps"] = {DK: list(summary["deltas"][DK]["walls_gaps"]),
                         DH: list(summary["deltas"][DH]["walls_gaps"])}
    return row


def build(root: Path, its: list[int]) -> dict:
    """Rows per run label (checkpoints in ``its`` order), the evaluation set, and the verdict."""
    ev = root / "eval"
    runs: dict[str, list[dict]] = {}
    missing: list[str] = []
    split_info: set[tuple[str, int]] = set()
    for run in K1_RUNS + OPTIONAL_DENSE:
        rows = []
        for it in its:
            doc = _load(ev / f"{run}_it{it}.json")
            if doc is None:
                if run in K1_RUNS:
                    missing.append(f"{run}_it{it}")
                continue
            split_info.add((doc.get("split", "?"), int(doc.get("n_shapes", -1))))
            row = _walls_gaps(run_row(doc, float(doc["train_loss"])), doc["summary"])
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
            row = _walls_gaps(run_row(row_doc, float(doc["train_loss"])), sel)
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


def _nanmax(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return max(xs) if xs else float("nan")


def verdict(runs: dict[str, list[dict]], its: list[int]) -> dict:
    final = its[-1]
    hits = _row(runs.get("hits_8", []), final)
    dense = []
    for label in DENSE:
        row = _row(runs.get(label, []), final)
        if row is not None and row["in_band"]:
            dense.append((label, row))
    absolute = {label: {"setting": r["setting"], "in_band": r["in_band"],
                        "walls_gaps_L_lt_1": r["walls_gaps"][DK][:RULE["small_bins"]], "M": r["M"]}
                for label, rows in runs.items() if (r := _row(rows, final)) is not None}
    if hits is None or not dense:
        return {"iteration": final, "status": "not evaluable", "absolute": absolute,
                "reason": "no dense run is in band" if hits is not None else "hits_8 not evaluated"}
    m_dense = _nanmax([r["M"] for _, r in dense])
    f1_dense = _nanmax([r["F1"] for _, r in dense])
    wg_dense = [_nanmax([r["walls_gaps"][DK][i] for _, r in dense]) for i in range(len(BIN_LABELS))]
    d_m = hits["M"] - m_dense
    d_f1 = hits["F1"] - f1_dense
    gains = [h - d for h, d in zip(hits["walls_gaps"][DK], wg_dense) if h == h and d == d]
    max_gain = max(gains) if gains else float("nan")
    if d_m >= RULE["small_gain"] and d_f1 >= -RULE["f1_tolerance"]:
        status = "H5 supported"
    elif max_gain <= RULE["refute_gain"]:
        status = "H5 refuted"
    else:
        status = "undecided (SPEC: compactness only)"
    return {"iteration": final, "status": status, "dense_runs": [x[0] for x in dense],
            "M_hits": hits["M"], "M_dense": m_dense, "gain_M": d_m,
            "F1_hits": hits["F1"], "F1_dense": f1_dense, "gain_F1": d_f1,
            "walls_gaps_hits": hits["walls_gaps"][DK], "walls_gaps_dense": wg_dense,
            "bin_gains": [h - d for h, d in zip(hits["walls_gaps"][DK], wg_dense)],
            "max_bin_gain": max_gain, "rule": RULE, "absolute": absolute}


def to_markdown(doc: dict) -> str:
    ev = ", ".join(f"{e['split']} (first {e['n_shapes']} shapes, all rays)" for e in doc["evaluation"]) or "none"
    v = doc["verdict"]
    L = ["# H5-K1-1 report", "",
         f"Evaluation: {ev}. Setting of each row: the H5-K1-1a band selection of training (rows without a label), and for "
         "occfrac_128 (grid) the band selection over the extended grid, which includes `binarize` (H5-K1-1b). Best F1 "
         f"at 0.002 among the settings with decoded/GT in [{doc['ratio_band'][0]:g}, {doc['ratio_band'][1]:g}]; otherwise "
         "the closest ratio. Recall and distances at delta 0.002 unless labelled h/2.", "",
         "M (H5-6): recall at 0.002 of the walls and gaps with L/h < 1, both ends matched, pooled over the four L/h < 1 "
         "bins; walls+gaps per bin: the same per L/h bin. These are the thin-feature quantities of the verdict. The "
         "thin-bin tables below show the per-hit thin recall, which is descriptive.", ""]
    if doc["missing"]:
        L += [f"Not evaluated yet: {', '.join(doc['missing'])}.", ""]
    L += ["## Verdict (H5, SPEC section 5, layer 1, delta 0.002)", ""]
    if v["status"] == "not evaluable":
        L += [f"**not evaluable** at iteration {v['iteration']}: {v['reason']}.", ""]
    else:
        L += [f"**{v['status']}** at iteration {v['iteration']}: hits_8 against the dense ray runs in band "
              f"({', '.join(v['dense_runs'])}); the dense value is the better of them, per L/h bin for the bins. "
              "points_strong is not a dense competitor (it enters H5 only through the cost criterion, not evaluated "
              "here).", "",
              "| quantity | hits_8 | dense | difference | rule |", "|---|---|---|---|---|",
              f"| M (walls+gaps, L/h < 1) | {_f(v['M_hits'])} | {_f(v['M_dense'])} | {_f(v['gain_M'])} | "
              f"at least +{v['rule']['small_gain']} |",
              f"| F1 of all hits, 0.002 | {_f(v['F1_hits'])} | {_f(v['F1_dense'])} | {_f(v['gain_F1'])} | "
              f"not below -{v['rule']['f1_tolerance']} (t/h > 2 condition) |"]
        for i, lab in enumerate(BIN_LABELS):
            L.append(f"| walls+gaps {lab} | {_f(v['walls_gaps_hits'][i])} | {_f(v['walls_gaps_dense'][i])} | "
                     f"{_f(v['bin_gains'][i])} | refuted if at most +{v['rule']['refute_gain']} in every bin |")
        L += ["", f"Largest per-bin walls+gaps gain of hits_8: {_f(v['max_bin_gain'])}.", ""]
    abs_txt = "; ".join(f"{run} ({a['setting']}{'' if a['in_band'] else ', out of band'}): "
                        + ", ".join(_f(x) for x in a["walls_gaps_L_lt_1"])
                        for run, a in v["absolute"].items())
    L += [f"Absolute walls+gaps recall at 0.002 in the L/h < 1 bins ({', '.join(BIN_LABELS[:RULE['small_bins']])}), "
          f"iteration {v['iteration']}: {abs_txt}.", ""]
    if v["status"] != "not evaluable":
        L += ["The rule is applied to the ray layer only; IoU and the mesh layer are not evaluated in this report. "
              "The SPEC condition \"with a smaller output\" and the cost criterion are not checked here, and the SPEC "
              "compares against TSDF M = 128, which K1-1 did not train: the dense ray runs in band stand in for it.", ""]
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
    order = [r for r in doc["runs"] if r != "occfrac_128"] + (["occfrac_128"] if "occfrac_128" in doc["runs"] else [])
    for key, title in ((DK, "thin recall at 0.002"), (DH, "thin recall at h/2 (descriptive, not the criterion)")):
        for run in order:
            rows = doc["runs"][run]
            names = sorted({r["setting"] for r in rows})
            if run == GRID_LABEL:
                head = f"occfrac_128, grid-selected setting ({', '.join(names)})"
            elif run == "occfrac_128":
                head = f"occfrac_128, {', '.join(names)}: out of band, not comparable"
            else:
                head = run
            L += ["", f"## Thin bins, {head}: {title} (n) per L/h bin", "",
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
