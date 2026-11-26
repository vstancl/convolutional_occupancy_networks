"""H5-G1b: re-evaluate the last checkpoint of every K1 run at the setting the combined report selected, with the
distance histograms of ``eval_rays`` (GPU; no grid search, no training).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/eval_hist.py [--report h5/results/k1/report.json] [--it 120000]

Runs: every run of the report's roots (directories with ``config.json``). The setting of a run is the one of its row
in the combined report: the ``(grid)`` row if it has one, else the plain row; the exact setting dict is read from that
row's evaluation file (``selected_setting`` of ``{run}_it{N}_grid.json``, or ``setting`` of ``{run}_it{N}.json``).
Shapes: the report's evaluation (the first ``n_shapes`` shapes of ``split``), all rays, as ``q0_report.py eval``.

Output: ``<root>/eval/{run}_it{N}_hist.json`` (same layout as the plain evaluation, ``summary`` with
``diag.dist_hist``). An existing file is kept. Check: F1 at 0.002, precision, decoded/GT and M
(``walls_gaps_pooled_L_lt_1``) must equal the report's row exactly; otherwise the script stops with the difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from data import DATA_ROOT, shape_ids  # noqa: E402
from eval_rays import grid_summaries, prepare_all  # noqa: E402
from models import build_model  # noqa: E402

DK = "0.002"


def row_of(report: dict, run: str, it: int) -> tuple[str, dict]:
    for label in (f"{run} (grid)", run):
        if label in report["runs"]:
            rows = [r for r in report["runs"][label] if r["iteration"] == it]
            if rows:
                return label, rows[0]
    raise SystemExit(f"{run}: no row at iteration {it} in the report")


def setting_of(root: Path, label: str, run: str, it: int) -> dict:
    if label.endswith(" (grid)"):
        return json.loads((root / "eval" / f"{run}_it{it}_grid.json").read_text(encoding="utf-8"))["selected_setting"]
    return json.loads((root / "eval" / f"{run}_it{it}.json").read_text(encoding="utf-8"))["setting"]


def check(run: str, summary: dict, row: dict) -> list[str]:
    got = {"F1": summary["deltas"][DK]["f1"], "precision": summary["deltas"][DK]["precision"],
           "dec_gt_ratio": summary["dec_gt_ratio"], "M": summary["deltas"][DK]["walls_gaps_pooled_L_lt_1"]}
    return [f"{run}: {k} {got[k]!r} != report {row[k]!r}" for k in got if got[k] != row[k]]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", type=Path, default=H5_DIR / "results" / "k1" / "report.json")
    p.add_argument("--it", type=int, default=120000)
    args = p.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    (ev,) = report["evaluation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = None
    failures: list[str] = []
    for root in map(Path, report["roots"]):
        for run_dir in sorted(d for d in root.iterdir() if (d / "config.json").is_file()):
            run = run_dir.name
            out = root / "eval" / f"{run}_it{args.it}_hist.json"
            label, row = row_of(report, run, args.it)
            if out.exists():
                print(f"{run}: {out.name} exists, kept", flush=True)
                summary = json.loads(out.read_text(encoding="utf-8"))["summary"]
            else:
                if items is None:
                    items = [prepare_all(sid, DATA_ROOT) for sid in shape_ids(ev["split"], DATA_ROOT)[: ev["n_shapes"]]]
                setting = setting_of(root, label, run, args.it)
                target = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["target"]
                blob = torch.load(run_dir / f"model_it{args.it}.pt", map_location=device, weights_only=False)
                model = build_model(target).to(device)
                model.load_state_dict(blob["state_dict"])
                model.eval()
                summary = grid_summaries(model, items, [setting], device, DATA_ROOT)[0]
                doc = {"run": run, "iteration": args.it, "row": label, "setting": setting,
                       "setting_name": summary["setting_name"], "split": ev["split"], "n_shapes": len(items),
                       "summary": summary}
                out.write_text(json.dumps(doc, default=float), encoding="utf-8")
            bad = check(run, summary, row)
            failures += bad
            print(f"{run} ({label}, {summary['setting_name']}): F1 {summary['deltas'][DK]['f1']:.6g}, "
                  f"M {summary['deltas'][DK]['walls_gaps_pooled_L_lt_1']:.6g}, "
                  f"{'reproduces the report' if not bad else 'DIFFERS: ' + '; '.join(bad)}", flush=True)
    if failures:
        print("STOP: the re-evaluation does not reproduce the report", flush=True)
        return 1
    print("all runs reproduce the report", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
