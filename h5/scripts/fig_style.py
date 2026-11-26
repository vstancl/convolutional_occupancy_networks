"""H5-G1 figure style: one fixed colour per decoder, shared by plot_runs.py and plot_results.py.

Run names carry prefixes (P1_, P2_, Q0_, K2_) and suffixes (" (grid)"); ``decoder_of`` strips them, so a decoder has
the same colour in every figure. Out-of-band rows are drawn in ``OUT_OF_BAND``.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS: dict[str, str] = {
    "hits_8": "#d62728",
    "hits_8_plane": "#ff7f0e",
    "hits_16": "#e7969c",
    "occfrac_128": "#1f77b4",
    "tsdf_128": "#2ca02c",
    "binary_128": "#9467bd",
    "points_strong": "#8c564b",
    "points_std": "#e377c2",
    "points_rayint_a": "#bcbd22",
    "points_rayint_b": "#7b4173",
    "points_rayint_c": "#17becf",
    "points_512": "#393b79",  # gate-B point-sample ceiling (no network)
}
OTHER = "#000000"
OUT_OF_BAND = "#a0a0a0"
DPI = 200

# The six L/h bins of gate B (csgbox.constants.GATE_BIN_EDGES = 0, 1/8, 1/4, 1/2, 1, 2, 4).
BIN_LABELS: tuple[str, ...] = ("<1/8", "1/8-1/4", "1/4-1/2", "1/2-1", "1-2", "2-4")
DELTA_002 = "0.002"
DELTA_HALF_H = "0.0042969140625"  # h(128) / 2, the key the evaluations use

_PREFIX = re.compile(r"^(P\d+|Q\d+|K\d+)_")


def decoder_of(name: str) -> str:
    """``Q0_hits_8`` -> ``hits_8``; ``tsdf_128 (grid)`` -> ``tsdf_128``."""
    return _PREFIX.sub("", name.split(" (")[0])


def colour(name: str) -> str:
    return COLOURS.get(decoder_of(name), OTHER)


def save(fig, out_dir: Path, stem: str) -> list[Path]:
    """PNG (dpi 200) and SVG, without timestamps in the metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"]
    plt.rcParams["svg.hashsalt"] = "h5-g1"
    fig.savefig(paths[0], dpi=DPI, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(paths[1], bbox_inches="tight", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    return paths
