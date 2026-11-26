"""H5-G1 learning curves: training loss and validation F1 per run (CPU, reads only what training wrote).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/plot_runs.py --root ~/PhD/runs/k1_1 --root ~/PhD/runs/k1_4 \
        --group k1 [--out h5/results/figures]

A run is any directory of a root with a ``config.json``. Per run:

- training loss (``log.jsonl``, field ``loss``): the raw values as a thin faint line and an EMA (alpha 0.01, started at
  the first value) as a solid line, log y;
- validation F1 of all hits at delta 0.002 (``val.jsonl``, field ``selected_f1``: the setting chosen by the band
  selection of training).

A dashed vertical line marks every learning-rate drop of the run's config (``lr_steps``). A run resumed with a new
schedule keeps the old config as ``config_resumed_at_<it>.json`` next to the new one; the drops of all of them are
taken. A run whose log lacks a field is skipped for that quantity, with a message.

Two figures per group, written to ``<out>/<group>/`` as PNG (dpi 200) and SVG:
``curves_all`` (validation F1 of all runs in one panel; the training losses of different targets are different
quantities and are not compared, so they are only in the other figure) and ``curves_per_run`` (one panel per run,
loss on the left axis and F1 on the right axis).

With ``--report`` (a report JSON of k1_1_report.py or q0_report.py; repeatable), a run whose final evaluation used a
different decoding than its training-time validation at that iteration (the last ``selected`` of ``val.jsonl``;
for K1 the band selection over the whole decoding grid, the ``(grid)`` rows) gets a hollow marker at its last
report iteration with the report's F1 and the label "final decoding: <setting>". The report's F1 is measured on all
rays, the validation F1 on the fixed validation rays.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fig_style import colour, plt, save  # noqa: E402

EMA_ALPHA = 0.01
H5_DIR = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ema(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def lr_drops(run_dir: Path) -> list[tuple[int, float]]:
    drops: set[tuple[int, float]] = set()
    for cfg_path in sorted(run_dir.glob("config*.json")):
        steps = json.loads(cfg_path.read_text(encoding="utf-8")).get("lr_steps") or []
        drops.update((int(it), float(lr)) for it, lr in steps)
    return sorted(drops)


def load_run(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    run = {"name": run_dir.name, "dir": str(run_dir), "target": cfg.get("target"), "drops": lr_drops(run_dir), "loss": None, "f1": None}
    log = [r for r in read_jsonl(run_dir / "log.jsonl") if "iteration" in r]
    loss = [(r["iteration"], r["loss"]) for r in log if isinstance(r.get("loss"), (int, float))]
    if loss:
        it, v = np.array(loss, dtype=np.float64).T
        run["loss"] = (it, v)
    else:
        print(f"{run['name']}: log.jsonl has no field 'loss'; training loss skipped")
    val = [(r["iteration"], r["selected_f1"]) for r in read_jsonl(run_dir / "val.jsonl")
           if isinstance(r.get("selected_f1"), (int, float)) and math.isfinite(r["selected_f1"])]
    if val:
        it, v = np.array(val, dtype=np.float64).T
        run["f1"] = (it, v)
    else:
        print(f"{run['name']}: val.jsonl has no field 'selected_f1'; validation F1 skipped")
    return run


def discover(roots: list[Path]) -> list[dict]:
    runs, seen = [], set()
    for root in roots:
        for d in sorted(p for p in root.iterdir() if (p / "config.json").is_file()):
            if d.name in seen:
                raise SystemExit(f"run name {d.name} occurs in more than one root")
            seen.add(d.name)
            runs.append(load_run(d))
    return runs


def final_decodings(report_paths: list[Path], runs: list[dict]) -> dict[str, tuple[int, float, str]]:
    """Run name -> (iteration, F1, setting) of its final evaluation when the decoding differs from validation."""
    out: dict[str, tuple[int, float, str]] = {}
    val_setting = {}
    for r in runs:
        rows = read_jsonl(Path(r["dir"]) / "val.jsonl")
        val_setting[r["name"]] = {row["iteration"]: row.get("selected") for row in rows}
    for rp in report_paths:
        rep = json.loads(rp.expanduser().read_text(encoding="utf-8"))
        labels = list(rep["runs"])
        for label in labels:
            base = label.removesuffix(" (grid)")
            if base not in val_setting or (label == base and f"{base} (grid)" in labels):
                continue  # not a run of this group, or superseded by its grid row
            last = max(rep["runs"][label], key=lambda e: e["iteration"])
            if val_setting[base].get(last["iteration"]) != last["setting"]:
                out[base] = (last["iteration"], last["F1"], last["setting"])
    return out


def mark_final(ax, run: str, final: dict[str, tuple[int, float, str]], inside: bool = False) -> None:
    """Hollow marker and label; ``inside``: the label below-left of the marker (small panels), else to its right."""
    if run not in final:
        return
    it, f1, setting = final[run]
    ax.plot([it], [f1], ls="none", marker="o", ms=9, mfc="none", mec=colour(run), mew=1.6, zorder=4)
    if inside:
        ax.annotate(f"final decoding: {setting}", (it, f1), xytext=(-4, -12), textcoords="offset points",
                    ha="right", va="top", fontsize=7, color=colour(run))
        return
    ax.annotate(f"final decoding:\n{setting}", (it, f1), xytext=(9, 0), textcoords="offset points", ha="left",
                va="center", fontsize=7, color=colour(run))
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, max(hi, it * 1.22))  # room for the label right of the marker


def mark_drops(ax, drops: list[tuple[int, float]], label: bool) -> None:
    for it, lr in drops:
        ax.axvline(it, color="0.3", ls="--", lw=0.8)
        if label:
            ax.text(it, 0.98, f" lr -> {lr:g}", transform=ax.get_xaxis_transform(), rotation=90, va="top",
                    ha="right", fontsize=7, color="0.3")


def style(name: str) -> str:
    """Solid, except P1 pilot runs (dashed), which share their decoder's colour with the P2 runs."""
    return "--" if name.startswith("P1_") else "-"


def plot_loss(ax, run: dict) -> None:
    it, v = run["loss"]
    c = colour(run["name"])
    ax.plot(it, v, color=c, lw=0.4, alpha=0.25, ls=style(run["name"]))
    ax.plot(it, ema(v), color=c, lw=1.4, ls=style(run["name"]), label=run["name"])


def figure_all(runs: list[dict], title: str, final: dict[str, tuple[int, float, str]]) -> plt.Figure:
    fig, a_f1 = plt.subplots(1, 1, figsize=(8, 5))
    drops = sorted({d for r in runs for d in r["drops"]})
    for r in runs:
        if r["f1"] is not None:
            it, v = r["f1"]
            a_f1.plot(it, v, color=colour(r["name"]), ls=style(r["name"]), marker="o", ms=3, lw=1.4, label=r["name"])
        mark_final(a_f1, r["name"], final)
    a_f1.set(xlabel="iteration", ylabel="F1 of all hits, delta 0.002",
             title="validation F1 (band-selected setting of training)")
    mark_drops(a_f1, drops, label=True)
    a_f1.grid(alpha=0.3)
    a_f1.legend(fontsize=7)
    if final:
        fig.text(0.01, -0.02, "Hollow markers: the final evaluation (all rays) at the decoding chosen over the whole "
                 "decoding grid, where it differs from the training-time validation.", fontsize=7, ha="left")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def figure_per_run(runs: list[dict], title: str, final: dict[str, tuple[int, float, str]]) -> plt.Figure:
    n = len(runs)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False)
    for ax, r in zip(axes.ravel(), runs):
        if r["loss"] is not None:
            plot_loss(ax, r)
            ax.set_yscale("log")
        ax.set_ylabel("loss", fontsize=8)
        ax.set_title(f"{r['name']}", fontsize=9, color=colour(r["name"]))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        mark_drops(ax, r["drops"], label=True)
        if r["f1"] is not None:
            a2 = ax.twinx()
            it, v = r["f1"]
            a2.plot(it, v, color="k", marker="o", ms=2.5, lw=1.0)
            mark_final(a2, r["name"], final, inside=True)
            a2.set_ylabel("val F1 (black)", fontsize=8)
            a2.tick_params(labelsize=7)
            a2.set_ylim(bottom=0)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(title + "\nper run: loss (colour, log, left) and validation F1 at delta 0.002 (black, right)")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--title", default=None, help="figure title (default: the group and its roots)")
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "figures")
    p.add_argument("--report", type=Path, action="append", default=[],
                   help="report JSON: marks the final evaluation where its decoding differs from validation")
    args = p.parse_args(argv)
    runs = discover([r.expanduser() for r in args.root])
    if not runs:
        raise SystemExit("no runs found")
    title = args.title or f"{args.group}: learning curves ({', '.join(r.name for r in args.root)})"
    out = args.out / args.group
    final = final_decodings(args.report, runs)
    for run, (it, f1, setting) in sorted(final.items()):
        print(f"{run}: final decoding {setting} at {it} (F1 {f1:.4g}) differs from validation; marked")
    written = save(figure_all(runs, title, final), out, "curves_all")
    written += save(figure_per_run(runs, title, final), out, "curves_per_run")
    for w in written:
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
