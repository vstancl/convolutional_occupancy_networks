"""H5-G6/G7: K2 figures for the meeting, from the written reports only (CPU; nothing is evaluated).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/plot_k2.py

Sources (numbers are read from these files and nothing else):

- ``h5/results/k2/report.json``, ``final`` (iteration 60000, the 32 training shapes): K2_points_rayint_a,
  K2_points_rayint_b, Q0_points_strong;
- ``h5/results/k2p/report.json``, ``final`` (iteration 60000, the same 32 shapes): K2p_points_mix_a, K2p_points_mix_b.
  The K2' report repeats the K2 rows; they must be equal to the K2 report (checked);
- ``h5/results/k2_3/report.json``, ``rows`` at iteration 120000 (first 200 val shapes): K2_3_points_mix_a, and the K1
  references points_strong and hits_8.

Quantities (as the reports define them, delta 0.002): ordinary recall ``ord_recall["0.002"]``; weighted thin recall
L/h < 1 ``weighted_thin`` (per-hit thin recall, population-weighted over the four L/h < 1 bins); M (walls and gaps with
L/h < 1, both ends matched, pooled); per-bin thin recall ``thin[i].recall["0.002"]``. Every row used here is in band
(checked).

Outputs in ``h5/results/figures/k2/`` (PNG dpi 200 + PDF) and ``README.md`` there with the numbers of each figure:

- ``g6_2x2``: the four memorisation runs as a 2x2 design (rows: without / with the near-surface half; columns:
  length-proportional (variant b) / equal weight per segment (variant a)); panels ordinary recall and weighted thin
  recall, grouped bars, Q0_points_strong dashed;
- ``g6b_thin_bins``: per-bin thin recall at 0.002, six L/h bins (categorical), the four runs and Q0_points_strong;
- ``g7_memorisation_vs_new``: points_strong against points_mix_a, memorisation (60k) and new shapes (120k); panels
  ordinary recall, weighted thin recall, M (log y: the new-shape M is two orders of magnitude below the memorisation
  M); hits_8 at 120k dashed in the new-shapes group of the thin recall and M panels; ratio mix_a / points_strong above
  each pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fig_style import BIN_LABELS, DELTA_002, colour, plt, save  # noqa: E402

H5_DIR = Path(__file__).resolve().parents[1]
RESULTS = H5_DIR / "results"
K2, K2P, K2_3 = RESULTS / "k2" / "report.json", RESULTS / "k2p" / "report.json", RESULTS / "k2_3" / "report.json"
OUT_DEFAULT = RESULTS / "figures" / "k2"
IT_MEM, IT_NEW = 60000, 120000
FORMATS = ("png", "pdf")

# The 2x2 design of G6: (row, column) -> run. Rows: near-surface half; columns: weighting of the ray segments.
ROWS = ("without near-surface half", "with near-surface half")
COLS = ("length-proportional (b)", "equal weight per segment (a)")
DESIGN = {(0, 0): "K2_points_rayint_b", (0, 1): "K2_points_rayint_a",
          (1, 0): "K2p_points_mix_b", (1, 1): "K2p_points_mix_a"}
COL_SHORT = ("length-prop.", "equal weight")
REF_MEM = "Q0_points_strong"
QUANT = {"ord": "ordinary recall, 0.002", "thin": "weighted thin recall L/h < 1, 0.002", "M": "M (walls+gaps L/h < 1)"}


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def value(row: dict, q: str) -> float:
    return {"ord": lambda r: r["ord_recall"][DELTA_002], "thin": lambda r: r["weighted_thin"],
            "M": lambda r: r["M"]}[q](row)


def gather() -> dict:
    """The rows of every run in the figures, with consistency checks."""
    k2, k2p, k23 = _load(K2), _load(K2P), _load(K2_3)
    if k2["final_it"] != IT_MEM or k2p["final_it"] != IT_MEM:
        raise ValueError(f"memorisation reports are not at {IT_MEM}")
    mem = {r: k2["final"][r] for r in ("K2_points_rayint_a", "K2_points_rayint_b", REF_MEM)}
    mem.update({r: k2p["final"][r] for r in ("K2p_points_mix_a", "K2p_points_mix_b")})
    for r in ("K2_points_rayint_a", "K2_points_rayint_b", REF_MEM):  # the K2' report repeats the K2 rows
        if k2p["final"][r] != k2["final"][r]:
            raise ValueError(f"{r} differs between {K2} and {K2P}")
    new = {}
    for lab, key in (("K2_3_points_mix_a", "K2_3_points_mix_a"), ("K1 points_strong", "points_strong"),
                     ("K1 hits_8", "hits_8")):
        rows = [x for x in k23["rows"][key] if x["iteration"] == IT_NEW]
        if len(rows) != 1:
            raise ValueError(f"{key} at {IT_NEW} not in {K2_3}")
        new[lab] = rows[0]
    for name, row in {**mem, **new}.items():
        if not row["in_band"] or row["iteration"] != (IT_NEW if name in new else IT_MEM):
            raise ValueError(f"{name}: out of band or wrong iteration")
        if [b["bin"] for b in row["thin"]] != ["[0,0.125)", "[0.125,0.25)", "[0.25,0.5)", "[0.5,1)", "[1,2)", "[2,4)"]:
            raise ValueError(f"{name}: unexpected L/h bins")
    return {"mem": mem, "new": new, "eval_mem": f"{k2['final'][REF_MEM]['split']}, {k2['final'][REF_MEM]['n_shapes']}",
            "eval_new": k23["evaluation"]}


def _short(run: str) -> str:
    return run.split("_", 1)[1] if run.startswith(("K2_", "K2p_", "Q0_")) else run


def _bar_labels(ax, bars, vals, fmt="{:.3f}", fontsize=7):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fontsize,
                    bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8})


def fig_g6(d: dict, out: Path) -> list[Path]:
    mem = d["mem"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    w = 0.36
    for ax, q in zip(axes, ("ord", "thin")):
        for c in range(2):
            xs = [r + (c - 0.5) * w for r in range(2)]
            runs = [DESIGN[(r, c)] for r in range(2)]
            vals = [value(mem[run], q) for run in runs]
            bars = ax.bar(xs, vals, w * 0.95, color=[colour(run) for run in runs], edgecolor="black", linewidth=0.5)
            _bar_labels(ax, bars, vals)
            for x, run in zip(xs, runs):
                ax.text(x, -0.02, f"{COL_SHORT[c]}\n{_short(run).removeprefix('points_')}", ha="center", va="top",
                        fontsize=7, transform=ax.get_xaxis_transform())
        ref = value(mem[REF_MEM], q)
        ax.axhline(ref, color=colour(REF_MEM), ls="--", lw=1.2, label=f"{REF_MEM} {ref:.3f}")
        ax.set_xticks([0, 1], [f"\n\n\n{r}" for r in ROWS], fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(QUANT[q], fontsize=9)
        ax.set_ylim(0, max(1.0 if q == "ord" else 0.0, 1.15 * max(value(mem[r], q) for r in [*DESIGN.values(), REF_MEM])))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax.set_title(QUANT[q].split(",")[0], fontsize=10)
    fig.suptitle("Memorisation, 32 shapes, 60k")
    fig.text(0.5, -0.06, "Columns: ray segments drawn length-proportional (variant b) or with equal weight per segment "
             "(variant a). Rows: without (rayint) or with (mix) the near-surface half of points_strong. "
             "Recall at delta 0.002 on the 32 training shapes, all rays; every run in band.",
             ha="center", va="top", fontsize=8, wrap=True)
    return save(fig, out, "g6_2x2", FORMATS)


def fig_g6b(d: dict, out: Path) -> list[Path]:
    mem = d["mem"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = list(range(len(BIN_LABELS)))
    styles = {"K2_points_rayint_b": ":", "K2_points_rayint_a": ":", "K2p_points_mix_b": "-", "K2p_points_mix_a": "-"}
    for run in (*DESIGN.values(), REF_MEM):
        ys = [b["recall"][DELTA_002] for b in mem[run]["thin"]]
        ax.plot(x, ys, styles.get(run, "--"), marker="o", ms=4, lw=1.6, color=colour(run), label=_short(run))
    n = [b["n"] for b in mem[REF_MEM]["thin"]]
    ax.set_xticks(x, [f"{lab}\nn={k}" for lab, k in zip(BIN_LABELS, n)], fontsize=8)
    ax.axvline(3.5, color="grey", lw=0.8, alpha=0.6)
    ax.text(3.45, 0.98, "L/h < 1 | L/h >= 1", ha="center", va="top", fontsize=7, color="grey",
            transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1)
    ax.set_xlabel("L/h bin (thickness of the wall or gap / cell size h); n = GT thin hits (same for every run)",
                  fontsize=9)
    ax.set_ylabel("thin recall at delta 0.002 (per hit)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", frameon=False,
              title="dotted: without near-surface half, solid: with, dashed: reference", title_fontsize=7)
    ax.set_title("Memorisation, 32 shapes, 60k: thin recall per L/h bin", fontsize=11)
    return save(fig, out, "g6b_thin_bins", FORMATS)


def fig_g7(d: dict, out: Path) -> list[Path]:
    mem, new = d["mem"], d["new"]
    groups = (("memorisation\n32 train shapes, 60k", mem[REF_MEM], mem["K2p_points_mix_a"],
               (REF_MEM, "K2p_points_mix_a")),
              ("new shapes\n200 val shapes, 120k", new["K1 points_strong"], new["K2_3_points_mix_a"],
               ("K1 points_strong", "K2_3_points_mix_a")))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    w = 0.36
    for ax, q in zip(axes, ("ord", "thin", "M")):
        top = 0.0
        for g, (gname, ps, mix, names) in enumerate(groups):
            vals = [value(ps, q), value(mix, q)]
            xs = [g - w / 2, g + w / 2]
            bars = ax.bar(xs, vals, w * 0.95, color=[colour("points_strong"), colour("points_mix_a")],
                          edgecolor="black", linewidth=0.5)
            _bar_labels(ax, bars, vals)
            top = max(top, *vals)
            ratio = vals[1] / vals[0]
            anchor = max(vals + ([value(new["K1 hits_8"], q)] if g == 1 and q in ("thin", "M") else []))
            ax.annotate(f"mix_a / points_strong\n= {ratio:.2f}x", (g, anchor), xytext=(0, 14),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
            if g == 1 and q in ("thin", "M"):
                h8 = value(new["K1 hits_8"], q)
                ax.hlines(h8, g - w - 0.08, g + w + 0.08, colors=colour("hits_8"), linestyles="--", lw=1.4)
                top = max(top, h8)
        ax.set_xticks([0, 1], [grp[0] for grp in groups], fontsize=9)
        ax.set_ylabel(QUANT[q] + (" (log)" if q == "M" else ""), fontsize=9)
        if q == "M":
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 5.0)
        else:
            ax.set_ylim(0, top * 1.35)
        ax.grid(axis="y", alpha=0.3, which="both" if q == "M" else "major")
        ax.set_title(QUANT[q].split(",")[0], fontsize=10)
    h8 = new["K1 hits_8"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour("points_strong")),
               plt.Rectangle((0, 0), 1, 1, color=colour("points_mix_a")),
               plt.Line2D([], [], color=colour("hits_8"), ls="--", lw=1.4)]
    labels = ["points_strong (Q0 / K1)", "points_mix_a (K2' / K2-3)",
              f"hits_8, new shapes 120k (thin {value(h8, 'thin'):.3f}, M {value(h8, 'M'):.3f})"]
    fig.legend(handles, labels, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.suptitle("points_strong vs points_mix_a: memorisation vs new shapes")
    fig.text(0.5, -0.07, "Memorisation: Q0_points_strong and K2p_points_mix_a on their 32 training shapes at 60000. "
             "New shapes: K1 points_strong and K2_3_points_mix_a on the first 200 val shapes at 120000. All rays, "
             "delta 0.002, every run in band. M on a log axis.", ha="center", va="top", fontsize=8)
    return save(fig, out, "g7_memorisation_vs_new", FORMATS)


def _num(row: dict) -> str:
    return (f"ordinary recall {value(row, 'ord'):.4f}, weighted thin {value(row, 'thin'):.4f}, M {value(row, 'M'):.4f}, "
            f"decoded/GT {row['dec_gt_ratio']:.3f}, setting {row['setting']}")


def readme(d: dict, files: dict[str, list[Path]], out: Path) -> Path:
    mem, new = d["mem"], d["new"]
    rel = lambda p: p.relative_to(H5_DIR.parent).as_posix()  # noqa: E731
    L = ["# H5-G6/G7: K2 figures for the meeting", "",
         "Made on the CPU from the written reports; nothing was trained or evaluated. Each figure is a PNG (dpi 200) "
         "plus a PDF of the same name. Colours from `h5/scripts/fig_style.py` (points_mix_a / points_mix_b in the "
         "hue family of their ray-interval variant). Regenerate:", "",
         "```", "cd ~/PhD/convonet && .venv/bin/python h5/scripts/plot_k2.py", "```", "",
         "Sources: `h5/results/k2/report.json` and `h5/results/k2p/report.json` (`final`, iteration 60000, the 32 "
         f"training shapes of Q0, all rays; split {d['eval_mem']}), `h5/results/k2_3/report.json` (`rows` at "
         f"iteration 120000, {', '.join(f'{s} first {n} shapes' for s, n in d['eval_new'])}, all rays). The rows of "
         "K2_points_rayint_a/b and Q0_points_strong are equal in the K2 and K2' reports (checked). Delta 0.002; "
         "weighted thin recall = per-hit thin recall population-weighted over the four L/h < 1 bins; M = walls+gaps "
         "with L/h < 1, both ends matched, pooled. Every row is in band.", ""]
    L += ["## g6_2x2", "", f"Files: {', '.join(f'`{rel(p)}`' for p in files['g6'])}.", "",
          "Memorisation, 32 shapes, 60k. 2x2 design, rows without / with the near-surface half, columns "
          "length-proportional (variant b) / equal weight per segment (variant a). Panels: ordinary recall and "
          "weighted thin recall; Q0_points_strong dashed.", "",
          "| run | near-surface half | segment weighting | ordinary recall 0.002 | weighted thin L/h < 1 |",
          "|---|---|---|---|---|"]
    for (r, c), run in DESIGN.items():
        L.append(f"| {run} | {'with' if r else 'without'} | {COLS[c]} | {value(mem[run], 'ord'):.4f} | "
                 f"{value(mem[run], 'thin'):.4f} |")
    L.append(f"| {REF_MEM} (reference) | | | {value(mem[REF_MEM], 'ord'):.4f} | {value(mem[REF_MEM], 'thin'):.4f} |")
    L += ["", "## g6b_thin_bins", "", f"Files: {', '.join(f'`{rel(p)}`' for p in files['g6b'])}.", "",
          "Per-hit thin recall at 0.002 per L/h bin (categorical x), the four memorisation runs and Q0_points_strong "
          "(dotted: without near-surface half, solid: with, dashed: reference).", "",
          "| L/h bin | n | " + " | ".join(_short(r) for r in (*DESIGN.values(), REF_MEM)) + " |",
          "|---|---|" + "---|" * 5]
    for i, lab in enumerate(BIN_LABELS):
        L.append(f"| {lab} | {mem[REF_MEM]['thin'][i]['n']} | "
                 + " | ".join(f"{mem[r]['thin'][i]['recall'][DELTA_002]:.4f}" for r in (*DESIGN.values(), REF_MEM))
                 + " |")
    L += ["", "## g7_memorisation_vs_new", "", f"Files: {', '.join(f'`{rel(p)}`' for p in files['g7'])}.", "",
          "points_strong vs points_mix_a. Panels ordinary recall, weighted thin recall, M (log y). Groups: "
          "memorisation (Q0_points_strong vs K2p_points_mix_a, 60k) and new shapes (K1 points_strong vs "
          "K2_3_points_mix_a, 120k); hits_8 at 120k dashed in the new-shapes group of the thin and M panels; the "
          "ratio mix_a / points_strong above each pair.", "",
          "| group | quantity | points_strong | points_mix_a | ratio mix_a / points_strong | hits_8 (120k) |",
          "|---|---|---|---|---|---|"]
    for gname, ps, mix in (("memorisation 60k", mem[REF_MEM], mem["K2p_points_mix_a"]),
                           ("new shapes 120k", new["K1 points_strong"], new["K2_3_points_mix_a"])):
        for q in ("ord", "thin", "M"):
            h8 = f"{value(new['K1 hits_8'], q):.4f}" if gname.startswith("new") and q != "ord" else ""
            L.append(f"| {gname} | {QUANT[q]} | {value(ps, q):.4f} | {value(mix, q):.4f} | "
                     f"{value(mix, q) / value(ps, q):.2f} | {h8} |")
    L += ["", "## All rows used", ""]
    for name, row in {**mem, **new}.items():
        L.append(f"- {name} (it {row['iteration']}): {_num(row)}.")
    path = out / "README.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args(argv)
    d = gather()
    files = {"g6": fig_g6(d, args.out), "g6b": fig_g6b(d, args.out), "g7": fig_g7(d, args.out)}
    files["readme"] = [readme(d, files, args.out)]
    for k, ps in files.items():
        for path in ps:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
