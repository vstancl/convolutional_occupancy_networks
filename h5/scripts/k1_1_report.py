"""H5-K1-1 report: the checkpoints of the K1-1 runs, evaluated by ``q0_report.py eval`` (all rays, first 200 val shapes).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/k1_1_report.py --root ~/PhD/runs/k1_1 \
        --its 40000,80000,120000 --out h5/results/k1_1

Reads ``ROOT/eval/{run}_it{N}.json`` (written once by the eval stage; missing checkpoints are listed, not an error)
and writes ``OUT/report.json`` and ``OUT/report.md``. Per run and checkpoint the rows hold the values of
``q0_report.run_row``: ordinary recall at 0.002 and h/2, ordinary median and p90 distance, the thin-bin recall per
L/h bin, the weighted recall of the L/h < 1 bins (``M``), F1 of all hits at 0.002, precision, decoded/GT ratio,
whether that ratio is in ``RATIO_BAND``, and the selected decoding setting.
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

K1_RUNS: tuple[str, ...] = ("hits_8", "occfrac_128", "points_strong")


def _its(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def in_band(ratio: float) -> bool:
    return bool(RATIO_BAND[0] <= ratio <= RATIO_BAND[1])


def build(root: Path, its: list[int]) -> dict:
    """Rows per run (checkpoints in ``its`` order, missing ones listed) and the evaluation set of the rows."""
    ev = root / "eval"
    runs: dict[str, list[dict]] = {}
    missing: list[str] = []
    split_info: set[tuple[str, int]] = set()
    for run in K1_RUNS:
        rows = []
        for it in its:
            path = ev / f"{run}_it{it}.json"
            if not path.exists():
                missing.append(f"{run}_it{it}")
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            split_info.add((doc.get("split", "?"), int(doc.get("n_shapes", -1))))
            row = run_row(doc, float(doc["train_loss"]))
            row["in_band"] = in_band(row["dec_gt_ratio"])
            rows.append(row)
        if rows:
            runs[run] = rows
    return {"its": its, "ratio_band": list(RATIO_BAND), "evaluation": [
                {"split": s, "n_shapes": n} for s, n in sorted(split_info)],
            "missing": missing, "runs": runs}


def to_markdown(doc: dict) -> str:
    ev = ", ".join(f"{e['split']} (first {e['n_shapes']} shapes, all rays)" for e in doc["evaluation"]) or "none"
    L = ["# H5-K1-1 report", "",
         f"Evaluation: {ev}. Decoding setting of every checkpoint: the H5-K1-1a band selection (best F1 at 0.002 "
         f"among the settings with decoded/GT in [{doc['ratio_band'][0]:g}, {doc['ratio_band'][1]:g}]; otherwise the "
         "closest ratio). Recall and distances at delta 0.002 unless labelled h/2.", "",
         "Weighted recall L/h < 1: population-weighted recall of the walls and gaps with L/h < 1 at 0.002 (the M of "
         "H5-6); it is not comparable when decoded/GT is outside the band.", ""]
    if doc["missing"]:
        L += [f"Not evaluated yet: {', '.join(doc['missing'])}.", ""]
    L += ["## Per checkpoint", "",
          "| run | iteration | selected setting | F1 (all hits, 0.002) | precision | decoded/GT | in band | "
          "ord n | ord recall 0.002 | ord recall h/2 | ord median | ord p90 | weighted recall L/h < 1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, rows in doc["runs"].items():
        for r in rows:
            L.append(f"| {run} | {r['iteration']} | {r['setting']} | {_f(r['F1'])} | {_f(r['precision'])} | "
                     f"{_f(r['dec_gt_ratio'])} | {'yes' if r['in_band'] else 'no'} | {r['ord_n']} | "
                     f"{_f(r['ord_recall'][DK])} | {_f(r['ord_recall'][DH])} | {_f(r['ord_median'])} | "
                     f"{_f(r['ord_p90'])} | {_f(r['M'])} |")
    for run, rows in doc["runs"].items():
        L += ["", f"## Thin bins, {run}: ordinary-thin recall at 0.002 (n) per L/h bin", "",
              "| L/h bin | " + " | ".join(f"it {r['iteration']}" for r in rows) + " |",
              "|---|" + "---|" * len(rows)]
        for i, label in enumerate(BIN_LABELS):
            cells = [f"{_f(r['thin'][i]['recall'][DK])} ({r['thin'][i]['n']})" for r in rows]
            L.append(f"| {label} | " + " | ".join(cells) + " |")
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
    print(f"report: {n} checkpoints, {len(doc['missing'])} missing -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
