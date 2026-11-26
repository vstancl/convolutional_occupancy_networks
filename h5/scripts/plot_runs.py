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
``curves_all`` (one panel per quantity, all runs) and ``curves_per_run`` (one panel per run, loss on the left axis
and F1 on the right axis).
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
    run = {"name": run_dir.name, "target": cfg.get("target"), "drops": lr_drops(run_dir), "loss": None, "f1": None}
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


def figure_all(runs: list[dict], group: str, title: str) -> plt.Figure:
    fig, (a_loss, a_f1) = plt.subplots(1, 2, figsize=(12, 4.6))
    drops = sorted({d for r in runs for d in r["drops"]})
    for r in runs:
        if r["loss"] is not None:
            plot_loss(a_loss, r)
        if r["f1"] is not None:
            it, v = r["f1"]
            a_f1.plot(it, v, color=colour(r["name"]), ls=style(r["name"]), marker="o", ms=3, lw=1.4, label=r["name"])
    a_loss.set_yscale("log")
    a_loss.set(xlabel="iteration", ylabel="training loss (EMA solid, raw faint)", title="(a) training loss")
    a_f1.set(xlabel="iteration", ylabel="F1 of all hits, delta 0.002", title="(b) validation F1 (band-selected setting)")
    for ax in (a_loss, a_f1):
        mark_drops(ax, drops, label=True)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def figure_per_run(runs: list[dict], title: str) -> plt.Figure:
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
    args = p.parse_args(argv)
    runs = discover([r.expanduser() for r in args.root])
    if not runs:
        raise SystemExit("no runs found")
    title = args.title or f"{args.group}: learning curves ({', '.join(r.name for r in args.root)})"
    out = args.out / args.group
    written = save(figure_all(runs, args.group, title), out, "curves_all")
    written += save(figure_per_run(runs, title), out, "curves_per_run")
    for w in written:
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
