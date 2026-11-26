"""H5-G1 result figures and table from the written reports and evaluations (CPU; nothing is re-evaluated).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/plot_results.py --report h5/results/k1/report.json --group k1
    cd ~/PhD/convonet && .venv/bin/python h5/scripts/plot_results.py --report h5/results/q0/report.json --group q0 \
        --runs-root ~/PhD/runs/h5_q0 --title "Q0: memorisation of 32 training shapes" \
        --header "memorisation of 32 training shapes: 1 run per decoder, the 32 training shapes, all rays"

Rows (networks): every run of the report(s) at its last checkpoint, with the report's setting, F1, precision, ratio
and ordinary-hit diagnostics. If a run has a ``(grid)`` row (band selection over the whole decoding grid), the grid
row is used; its plain row is kept only if it is out of band (the old occfrac_128 ``occ_snap=0.45`` row of K1-1).
Out-of-band rows are listed in the G4 table and named in a caption line under the G2 figures, but not drawn. In band: the report's ``in_band``, else decoded/GT in the report's
``ratio_band`` (default [0.8, 1.25]). The per-bin walls+gaps recall and the GT feature counts come from the
evaluation file of the row (``<root>/eval/{run}_it{N}.json``, or ``..._grid.json`` with the selected summary);
roots: ``--runs-root``, else the report's ``roots``. Training s/it: median of ``s_per_it`` in the run's ``log.jsonl``.

Ceilings (representations without a network), gate B of csgbox (``--gate``, the JSON that
``csgbox/docs/results/gate_v2.md`` was made from: 200 shapes of the gate_v1 dataset, not the val shapes):
walls+gaps per L/h bin = (recalled walls + recalled gaps) / (walls + gaps), from ``metrics[rep].deltas[d]
.thin_recalled`` and ``agg.thin_total``, the same definition as ``eval_rays.summarize`` (``walls_gaps``); M pools the
four L/h < 1 bins. Decoded/GT of a ceiling = recall / precision. occfrac_128 binarize (``--binarize``,
``h5/results/k1_1/occfrac_ceiling.json``, same 200 gate shapes): the walls+gaps fields added in H5-G1b
(``walls_gaps``, ``walls_gaps_pooled_L_lt_1``, from the per-class counts); its population must equal gate B's, and
its occfrac_128 row must reproduce gate B's walls+gaps.

Outputs in ``<out>/<group>/`` (PNG dpi 200 + SVG):

- ``g2_recall_by_thickness_{d0002,half_h}``: (a) the ceilings, (b) the in-band networks, each panel its own y-scale;
- ``g2c_ceiling_vs_network_{d0002,half_h}``: one panel, log y from 1e-3 to 1, each in-band network solid and its
  representation's ceiling dashed in the same colour (hits_8 and hits_8_plane share the hits_8 ceiling, points_strong
  and points_std share points_512, occfrac_128 uses the ceiling of its decoding: binarize or the gate decoding);
  values below 1e-3 are drawn at 1e-3 with an open marker;
- ``g4_table.md``, ``g4_table.csv``.

The error CDF (G3) is drawn by ``plot_cdf.py`` from the ``*_hist.json`` evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fig_style import BIN_LABELS, DELTA_002, DELTA_HALF_H, colour, decoder_of, plt, save  # noqa: E402

H5_DIR = Path(__file__).resolve().parents[1]
GATE_DEFAULT = Path.home() / "PhD" / "data" / "csgbox" / "gate_v1" / "gate_v2.json"
BINARIZE_DEFAULT = H5_DIR / "results" / "k1_1" / "occfrac_ceiling.json"
BAND_DEFAULT = (0.8, 1.25)
N_SMALL = 4  # bins with L/h < 1
CEILING_REPS: tuple[str, ...] = ("hits_8", "occfrac_128", "tsdf_128", "binary_128", "points_512")
CEILING_LABEL = {"hits_8": "hits_8 (= hits_8_plane)"}
N_RES = 128
VALUES_PER_SHAPE = {"hits": 3 * N_RES**2 * 12, "dense": 3 * N_RES**2 * N_RES}
DENSE = ("occfrac_128", "tsdf_128", "binary_128")
HITS = ("hits_8", "hits_8_plane")


def values_per_shape(decoder: str) -> str:
    if decoder in HITS:
        return str(VALUES_PER_SHAPE["hits"])
    if decoder in DENSE:
        return str(VALUES_PER_SHAPE["dense"])
    return "n/a"


def load(path: Path) -> dict:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


# ---------------------------------------------------------------- network rows

def eval_summary(roots: list[Path], label: str, it: int) -> dict:
    base, grid = label.removesuffix(" (grid)"), label.endswith(" (grid)")
    name = f"{base}_it{it}_grid.json" if grid else f"{base}_it{it}.json"
    for root in roots:
        f = root / "eval" / name
        if f.is_file():
            doc = load(f)
            return doc["summaries"][doc["selected"]] if grid else doc["summary"]
    raise SystemExit(f"evaluation {name} not found in {[str(r) for r in roots]}")


def s_per_it(roots: list[Path], label: str) -> float | None:
    base = label.removesuffix(" (grid)")
    for root in roots:
        log = root / base / "log.jsonl"
        if log.is_file():
            vals = [json.loads(l).get("s_per_it") for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
            vals = [v for v in vals if isinstance(v, (int, float))]
            return statistics.median(vals) if vals else None
    return None


def network_rows(report_paths: list[Path], runs_roots: list[Path]) -> tuple[list[dict], dict]:
    rows, meta = [], {"evaluation": None}
    for rp in report_paths:
        rep = load(rp)
        band = tuple(rep.get("ratio_band") or BAND_DEFAULT)
        roots = runs_roots or [Path(r) for r in rep.get("roots", [])]
        meta["evaluation"] = meta["evaluation"] or rep.get("evaluation")
        labels = list(rep["runs"])
        for label in labels:
            entry = max(rep["runs"][label], key=lambda e: e["iteration"])
            ratio = entry["dec_gt_ratio"]
            in_band = entry.get("in_band")
            if in_band is None:
                in_band = band[0] <= ratio <= band[1]
            has_grid = not label.endswith(" (grid)") and f"{label} (grid)" in labels
            if has_grid and in_band:
                continue  # superseded by its grid row
            summ = eval_summary(roots, label, entry["iteration"])
            d002 = summ["deltas"][DELTA_002]
            m_eval = d002["walls_gaps_pooled_L_lt_1"]
            if entry.get("M") is not None and not math.isclose(entry["M"], m_eval, rel_tol=1e-9, abs_tol=1e-12):
                raise SystemExit(f"{label}: M of the report {entry['M']} != evaluation {m_eval}")
            tt = summ["thin_total"]
            rows.append({
                "label": label, "decoder": decoder_of(label), "iteration": entry["iteration"],
                "setting": entry["setting"], "F1": entry["F1"], "precision": entry["precision"], "ratio": ratio,
                "in_band": bool(in_band), "ord_recall_002": entry["ord_recall"][DELTA_002],
                "ord_recall_half_h": entry["ord_recall"][DELTA_HALF_H], "ord_median": entry["ord_median"],
                "M": m_eval, "values": values_per_shape(decoder_of(label)), "s_per_it": s_per_it(roots, label),
                "wg": {d: summ["deltas"][d]["walls_gaps"] for d in (DELTA_002, DELTA_HALF_H)},
                "n_wg": [tt[0][b] + tt[1][b] for b in range(len(BIN_LABELS))],
            })
    return rows, meta


# ---------------------------------------------------------------- ceiling rows

def ceiling_rows(gate_path: Path, binarize_path: Path) -> tuple[list[dict], list[int]]:
    gate = load(gate_path)
    tt = gate["agg"]["thin_total"]
    n_wg = [tt[0][b] + tt[1][b] for b in range(len(BIN_LABELS))]
    rows = []
    for rep in CEILING_REPS:
        m = gate["metrics"][rep]["deltas"]
        wg, M = {}, None
        for d in (DELTA_002, DELTA_HALF_H):
            rec = m[d]["thin_recalled"]
            r_wg = [rec[0][b] + rec[1][b] for b in range(len(BIN_LABELS))]
            wg[d] = [r / n if n else float("nan") for r, n in zip(r_wg, n_wg)]
            if d == DELTA_002:
                M = sum(r_wg[:N_SMALL]) / sum(n_wg[:N_SMALL])
        d0 = m[DELTA_002]
        rows.append({"label": f"ceiling {CEILING_LABEL.get(rep, rep)}", "decoder": rep, "setting": "gate B",
                     "F1": d0["f1"], "precision": d0["precision"], "ratio": d0["recall"] / d0["precision"],
                     "M": M, "values": values_per_shape(rep), "wg": wg})
    # occfrac binarize: walls+gaps fields of occfrac_ceiling.json (H5-G1b), checked against gate B
    binz = load(binarize_path)["reps"]
    keys = {DELTA_002: "0.002", DELTA_HALF_H: "h/2"}
    for d, k in keys.items():
        if binz["occfrac_128_bin"][k]["thin_total"] != tt:
            raise SystemExit("occfrac_ceiling.json: the thin population differs from gate B")
        occ = next(r for r in rows if r["decoder"] == "occfrac_128")
        if binz["occfrac_128"][k]["walls_gaps"] != occ["wg"][d]:
            raise SystemExit("occfrac_ceiling.json: occfrac_128 walls+gaps differ from gate B")
    b = binz["occfrac_128_bin"]
    rows.append({"label": "ceiling occfrac_128 binarize", "decoder": "occfrac_128", "setting": "binarize",
                 "F1": b["0.002"]["f1"], "precision": b["0.002"]["precision"],
                 "ratio": b["0.002"]["dec_hits"] / b["0.002"]["gt_hits"], "M": b["0.002"]["walls_gaps_pooled_L_lt_1"],
                 "values": values_per_shape("occfrac_128"),
                 "wg": {d: b[k]["walls_gaps"] for d, k in keys.items()}})
    return rows, n_wg


# ---------------------------------------------------------------- G2

def tick_labels(n: list[int]) -> list[str]:
    return [f"{lab}\nn={k:,}" for lab, k in zip(BIN_LABELS, n)]


def figure_g2(net: list[dict], ceil: list[dict], n_gate: list[int], delta: str, title: str) -> plt.Figure:
    x = list(range(len(BIN_LABELS)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.6))
    for r in ceil:
        lab = r["label"].removeprefix("ceiling ")
        binarize = r["setting"] == "binarize"  # same colour as the gate occfrac ceiling: told apart by the marker
        a.plot(x, r["wg"][delta], color=colour(r["decoder"]), ls="--", marker="D" if binarize else "s",
               mfc="none" if binarize else None, ms=4, lw=1.4, label=lab)
    a.set_title("(a) representation ceilings, no network (gate B, 200 gate shapes)", fontsize=10)
    a.set_xticks(x, tick_labels(n_gate), fontsize=8)
    a.set_ylim(-0.02, 1.02)
    n_net = net[0]["n_wg"]
    for r in net:
        if r["n_wg"] != n_net:
            raise SystemExit(f"{r['label']}: GT feature counts differ from {net[0]['label']}")
        if not r["in_band"]:
            continue
        b.plot(x, r["wg"][delta], color=colour(r["label"]), ls="-", marker="o", ms=4, lw=1.6,
               label=f"{r['label']} ({r['setting']})")
    b.set_title(f"(b) networks at the last checkpoint (it {net[0]['iteration']}), own y-scale", fontsize=10)
    b.set_xticks(x, tick_labels(n_net), fontsize=8)
    b.set_ylim(bottom=0)
    dname = "0.002" if delta == DELTA_002 else "h/2 (descriptive)"
    for ax in (a, b):
        ax.set_xlabel("L/h bin (thickness of the wall or gap / cell size h); n = GT walls+gaps", fontsize=9)
        ax.set_ylabel(f"walls+gaps recall, both ends matched, delta {dname}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    caption(fig, net)
    return fig


def caption(fig, net: list[dict]) -> None:
    out = [f"{r['label']} ({r['setting']}, decoded/GT {r['ratio']:.2f})" for r in net if not r["in_band"]]
    if out:
        fig.text(0.01, -0.01, "Not drawn, out of band (decoded/GT outside [0.8, 1.25], not comparable): "
                 + "; ".join(out) + ".", fontsize=8, ha="left", va="top")


# ---------------------------------------------------------------- G2c

FLOOR = 1e-3
SHARED_CEILING = {"hits_8": "hits_8", "hits_8_plane": "hits_8", "tsdf_128": "tsdf_128",
                  "binary_128": "binary_128", "points_strong": "points_512", "points_std": "points_512"}


def ceiling_key(r: dict) -> str:
    """Label of the ceiling row of a network row (occfrac_128: the ceiling of its decoding)."""
    dec = r["decoder"]
    if dec == "occfrac_128":
        return "ceiling occfrac_128 binarize" if r["setting"] == "binarize" else "ceiling occfrac_128"
    rep = SHARED_CEILING[dec]
    return f"ceiling {CEILING_LABEL.get(rep, rep)}"


def floored(ax, x, y, **kw) -> None:
    """Line through max(y, FLOOR); markers filled at real values, open where the value is below FLOOR."""
    y = list(y)
    yy = [max(v, FLOOR) for v in y]
    c = kw.pop("color")
    ax.plot(x, yy, color=c, marker=None, **kw)
    low = [v < FLOOR for v in y]
    ax.plot([xi for xi, lo in zip(x, low) if not lo], [v for v, lo in zip(yy, low) if not lo], ls="none",
            marker="o", ms=4, color=c)
    ax.plot([xi for xi, lo in zip(x, low) if lo], [FLOOR] * sum(low), ls="none", marker="o", ms=6, mfc="none",
            mec=c, mew=1.3)


def figure_g2c(net: list[dict], ceil: list[dict], delta: str, title: str) -> plt.Figure:
    x = list(range(len(BIN_LABELS)))
    fig, ax = plt.subplots(figsize=(10, 6.4))
    by_label = {r["label"]: r for r in ceil}
    drawn: dict[str, str] = {}
    for r in net:
        if not r["in_band"]:
            continue
        key = ceiling_key(r)
        c = colour(r["label"])
        if key not in drawn:  # a shared ceiling is drawn once, in the colour of its first network
            users = [n["label"] for n in net if n["in_band"] and ceiling_key(n) == key]
            floored(ax, x, by_label[key]["wg"][delta], color=c, ls="--", lw=1.3,
                    label=f"{key} (for {', '.join(users)})")
            drawn[key] = c
        floored(ax, x, r["wg"][delta], color=c, ls="-", lw=1.8, label=f"{r['label']} ({r['setting']})")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 1.05)
    ax.set_xticks(x, BIN_LABELS)
    dname = "0.002" if delta == DELTA_002 else "h/2 (descriptive)"
    ax.set_xlabel("L/h bin (thickness of the wall or gap / cell size h)")
    ax.set_ylabel(f"walls+gaps recall, both ends matched, delta {dname} (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.set_title("ceiling (dashed, no network, gate B shapes) and network (solid, evaluation shapes); "
                 f"open markers: below {FLOOR:g}, drawn at {FLOOR:g}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    caption(fig, net)
    return fig


# ---------------------------------------------------------------- G4

COLUMNS = ("row", "decoding setting", "F1 0.002 (up)", "precision (up)", "decoded/GT (target 1)", "in band",
           "ord recall 0.002 (up)", "ord recall h/2 (up)", "ord median (down)", "M walls+gaps L/h<1 (up)",
           "output values per shape (down)", "training s/it")


def fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:.{nd}g}"


def table_rows(net: list[dict], ceil: list[dict]) -> list[list[str]]:
    out = []
    for r in net:
        out.append([r["label"], r["setting"], fmt(r["F1"]), fmt(r["precision"]), fmt(r["ratio"]), fmt(r["in_band"]),
                    fmt(r["ord_recall_002"]), fmt(r["ord_recall_half_h"]), fmt(r["ord_median"]), fmt(r["M"]),
                    r["values"], fmt(r["s_per_it"])])
    for r in ceil:
        out.append([r["label"], r["setting"], fmt(r["F1"]), fmt(r["precision"]), fmt(r["ratio"]), "n/a",
                    "n/a", "n/a", "n/a", fmt(r["M"]), r["values"], "n/a"])
    return out


def write_table(rows: list[list[str]], header: str, out_dir: Path) -> list[Path]:
    md = [header, "", "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", ("Network rows: last checkpoint, band-selected setting (the (grid) rows select over the whole decoding "
           "grid); an out-of-band row is not comparable. Ceiling rows: gate B (csgbox gate_v2, 200 gate shapes, not "
           "the evaluation shapes), no network; the binarize ceiling is computed on the same shapes "
           "(h5/results/k1_1/occfrac_ceiling.json). "
           "Output values per shape: hits = 3 x 128^2 x 12 (4K hit slots + 2K mask logits per ray, K = 2), dense = "
           "3 x 128^2 x 128, point decoders n/a. Training s/it: median of the training log.")]
    paths = [out_dir / "g4_table.md", out_dir / "g4_table.csv"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text("\n".join(md) + "\n", encoding="utf-8")
    with paths[1].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", type=Path, action="append", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--runs-root", type=Path, action="append", default=[],
                   help="roots with eval/ and the run directories (default: the report's roots)")
    p.add_argument("--gate", type=Path, default=GATE_DEFAULT)
    p.add_argument("--binarize", type=Path, default=BINARIZE_DEFAULT)
    p.add_argument("--title", default=None)
    p.add_argument("--header", default=None, help="first line of the table (default from the report's evaluation)")
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "figures")
    args = p.parse_args(argv)
    net, meta = network_rows(args.report, [r.expanduser() for r in args.runs_root])
    ceil, n_gate = ceiling_rows(args.gate.expanduser(), args.binarize)
    ev = meta["evaluation"] or {}
    if isinstance(ev, list):  # k1_1_report writes the list of distinct (split, n_shapes); one is expected
        if len(ev) != 1:
            raise SystemExit(f"more than one evaluation setting in the reports: {ev}; pass --header")
        ev = ev[0]
    header = args.header or (f"1 run per decoder, {ev.get('n_shapes', '?')} {ev.get('split', '?')} shapes, all rays")
    title = args.title or f"{args.group}: recall by thickness"
    out = args.out / args.group
    written = save(figure_g2(net, ceil, n_gate, DELTA_002, f"{title}\nwalls+gaps recall by thickness, delta 0.002"),
                   out, "g2_recall_by_thickness_d0002")
    written += save(figure_g2(net, ceil, n_gate, DELTA_HALF_H,
                              f"{title}\nwalls+gaps recall by thickness, delta h/2 (descriptive)"),
                    out, "g2_recall_by_thickness_half_h")
    for delta, stem, dname in ((DELTA_002, "d0002", "0.002"), (DELTA_HALF_H, "half_h", "h/2 (descriptive)")):
        written += save(figure_g2c(net, ceil, delta, f"{title}\nceiling against network per representation, "
                                   f"delta {dname}"), out, f"g2c_ceiling_vs_network_{stem}")
    written += write_table(table_rows(net, ceil), header, out)
    for w in written:
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
