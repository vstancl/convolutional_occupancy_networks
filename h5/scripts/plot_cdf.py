"""H5-G1b G3 "error CDF" from the stored distance histograms (CPU; reads ``*_it{N}_hist.json`` of eval_hist.py).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/plot_cdf.py --report h5/results/k1/report.json --group k1

Per run (every run of the report's roots with a ``{run}_it{N}_hist.json``): the cumulative share of GT hits whose
distance to the nearest decoded hit of the same type on the same ray is <= x, for (a) ordinary hits and (b) thin hits
(endpoints of a wall or gap, grazing excluded, with L/h < 1). The histogram layout is [underflow (< 1e-5), 200
log-spaced bins 1e-5..1, overflow (>= 1, finite), inf]: the curve is evaluated at the bin edges, the denominator is
every GT hit of the group (an overflow or inf hit never enters the curve, so the curve ends below 1 by that share).
Log x, vertical lines at 0.002 and h/2. Output ``<out>/<group>/g3_error_cdf`` (PNG dpi 200 + SVG).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fig_style import colour, plt, save  # noqa: E402

H5_DIR = Path(__file__).resolve().parents[1]
H_HALF = 0.0042969140625  # h(128) / 2
GROUPS = (("ord", "(a) ordinary GT hits"), ("thin_wg_lt1", "(b) thin GT hits: walls and gaps with L/h < 1"))


def cdf(counts: list[int], edges: list[float]) -> tuple[np.ndarray, np.ndarray, float]:
    """x = the bin edges, y = share of hits <= x; also the share that is overflow or inf."""
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    nb = len(edges) - 1
    cum = np.cumsum(c[: nb + 1])  # [underflow, bin 0, ..., bin nb-1]: cum[i] = count of distances < edges[i]
    y = cum / total
    return np.asarray(edges), y, float((c[nb + 1] + c[nb + 2]) / total)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--it", type=int, default=120000)
    p.add_argument("--title", default="K1: error CDF at the last checkpoint (200 val shapes, all rays)")
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "figures")
    args = p.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    docs = []
    for root in map(Path, report["roots"]):
        docs += [json.loads(f.read_text(encoding="utf-8")) for f in sorted((root / "eval").glob(f"*_it{args.it}_hist.json"))]
    if not docs:
        raise SystemExit("no *_hist.json evaluations found")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, (grp, name) in zip(axes, GROUPS):
        for doc in docs:
            dh = doc["summary"]["diag"]["dist_hist"]
            x, y, tail = cdf(dh[grp], dh["edges"])
            n = int(sum(dh[grp]))
            ax.plot(x, y, color=colour(doc["run"]), lw=1.6,
                    label=f"{doc['row']} ({doc['setting_name']}), n={n:,}, no hit within 1: {tail:.3f}")
        for v, lab in ((0.002, "0.002"), (H_HALF, "h/2")):
            ax.axvline(v, color="0.3", ls="--", lw=0.9)
            ax.text(v, 0.01, f" {lab}", transform=ax.get_xaxis_transform(), va="bottom", fontsize=8, color="0.3")
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("distance to the nearest decoded hit of the same type on the ray (log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("cumulative share of GT hits with distance <= x")
    fig.suptitle(f"{args.title}\nsetting of each run: the one the combined report selected")
    fig.tight_layout()
    for w in save(fig, args.out / args.group, "g3_error_cdf"):
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
