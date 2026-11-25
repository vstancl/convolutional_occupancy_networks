"""H5-K1 report: the checkpoints of the K1 runs from one or more run roots, evaluated by ``q0_report.py eval`` (all rays,
first 200 val shapes).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/k1_1_report.py --root ~/PhD/runs/k1_1 --root ~/PhD/runs/k1_4 \
        --its 40000,80000,120000 --out h5/results/k1

``--root`` may be repeated (H5-K1-4); with the K1-1 root alone it gives the K1-1c report. A run is a directory of a root
with a ``config.json`` (its target comes from there); run names must be unique over the roots. Per root it reads
``ROOT/eval/{run}_it{N}.json`` (the evaluation with the setting of training; row label ``{run}``) and
``ROOT/eval/{run}_it{N}_grid.json`` (every setting of the decoding grid with the band selection, H5-K1-1b; row label
``{run} (grid)``). A run with neither file at a checkpoint is listed as missing. Writes ``OUT/report.json`` and
``OUT/report.md``.

Per row (run and checkpoint): the values of ``q0_report.run_row``: ordinary recall at 0.002 and h/2, ordinary median and
p90 distance, thin-bin recall per L/h bin, the weighted recall of the L/h < 1 bins, F1 of all hits at 0.002, precision,
decoded/GT, whether that ratio is in ``RATIO_BAND``, and the setting.

Verdict (H5-K1-1c, rule of SPEC section 5, layer 1, delta 0.002, at the last checkpoint): hits_8 against the dense ray
representations that are in band: occfrac_128 (grid selection only), tsdf_128 and binary_128 (the grid row if it
exists, else the row of the training setting). points_std and points_strong are never dense competitors; they enter the
H5 rule only through the cost criterion of the SPEC, which this report does not evaluate. The dense value of a quantity
is the better of the dense runs in band, per L/h bin for the per-bin values. Thin-feature quantity: the walls-and-gaps
recall with both ends matched (``walls_gaps`` per L/h bin and ``walls_gaps_pooled_L_lt_1`` = the M of H5-6, from the
evaluation summary), not the per-hit thin recall. Rule, fixed for this report:

- H5 supported if M of hits_8 is at least 0.05 higher than the dense M, and the F1 of all hits is not lower by more
  than 0.01 (the t/h > 2 condition of the SPEC, checked with F1; IoU is not evaluated here);
- H5 refuted if, in every L/h bin, the walls-and-gaps recall of hits_8 is at most 0.02 higher than the dense value;
- otherwise undecided (the SPEC calls this "compactness only"). If no dense run is in band: "not evaluable".

A separate section "Cost (not evaluated yet)" names the hits variant (hits_8 or hits_8_plane) that would carry the cost
criterion of the SPEC (the same accuracy as the point decoder, at least ``COST_SPEED_NEED`` times faster), from the
speed ratios measured in the pilot (``PILOT_SPEED``). The cost criterion itself is not computed.
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

DENSE_TARGETS: tuple[str, ...] = ("occfrac_128", "tsdf_128", "binary_128")  # dense ray competitors (H5-K1-1c)
GRID_ONLY: tuple[str, ...] = ("occfrac_128",)  # dense targets that compete only with their grid selection
HITS_RUN = "hits_8"  # the hits run of the verdict
GRID_SUFFIX = " (grid)"
GRID_LABEL = "occfrac_128" + GRID_SUFFIX
RULE = {"small_bins": 4, "small_gain": 0.05, "f1_tolerance": 0.01, "refute_gain": 0.02}
# Cost criterion (SPEC section 5): the same accuracy as the point decoder, at least this many times faster.
COST_SPEED_NEED: float = 10.0
# Speed ratio points_strong N = 512 / method, trained timings of the pilot (h5/results/pilot/report.md, P2 at 20000).
PILOT_SPEED: dict[str, float] = {"hits_8": 5.38, "hits_8_plane": 25.53}


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


def _target(run_dir: Path) -> str | None:
    cfg = run_dir / "config.json"
    return json.loads(cfg.read_text(encoding="utf-8")).get("target") if cfg.exists() else None


def _plain_row(doc: dict) -> dict:
    row = _walls_gaps(run_row(doc, float(doc["train_loss"])), doc["summary"])
    row["in_band"] = in_band(row["dec_gt_ratio"])
    return row


def _grid_row(doc: dict, it: int) -> dict:
    sel = doc["summaries"][doc["selected"]]
    row_doc = {"iteration": it, "setting_name": doc["selected_name"], "summary": sel, "train_loss": doc["train_loss"]}
    row = _walls_gaps(run_row(row_doc, float(doc["train_loss"])), sel)
    row["in_band"] = in_band(row["dec_gt_ratio"])
    row["grid_all"] = [{"setting": s["setting_name"], "F1": s["deltas"][DK]["f1"],
                        "ratio": dec_gt_ratio(s), "in_band": in_band(dec_gt_ratio(s))} for s in doc["summaries"]]
    return row


def build(roots: list[Path], its: list[int]) -> dict:
    """Rows per run label (checkpoints in ``its`` order) over all roots, the evaluation set, and the verdict."""
    plain: dict[str, list[dict]] = {}
    grid: dict[str, list[dict]] = {}
    targets: dict[str, str] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    split_info: set[tuple[str, int]] = set()
    for root in roots:
        ev = root / "eval"
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.json").exists()):
            run = run_dir.name
            if run in targets:
                raise ValueError(f"run {run!r} is in both {sources[run]} and {root}")
            targets[run], sources[run] = _target(run_dir), str(root)
            for it in its:
                doc_p, doc_g = _load(ev / f"{run}_it{it}.json"), _load(ev / f"{run}_it{it}_grid.json")
                if doc_p is None and doc_g is None:
                    missing.append(f"{run}_it{it}")
                for doc in (doc_p, doc_g):
                    if doc is not None:
                        split_info.add((doc.get("split", "?"), int(doc.get("n_shapes", -1))))
                if doc_p is not None:
                    plain.setdefault(run, []).append(_plain_row(doc_p))
                if doc_g is not None:
                    grid.setdefault(run, []).append(_grid_row(doc_g, it))
    runs = dict(plain)
    runs.update({run + GRID_SUFFIX: rows for run, rows in grid.items()})
    label_target = {lab: targets[lab.removesuffix(GRID_SUFFIX)] for lab in runs}
    return {"its": its, "roots": [str(r) for r in roots], "ratio_band": list(RATIO_BAND),
            "evaluation": [{"split": s, "n_shapes": n} for s, n in sorted(split_info)],
            "missing": missing, "targets": label_target, "runs": runs,
            "verdict": verdict(runs, label_target, its), "cost": cost(runs, its)}


def dense_labels(runs: dict[str, list[dict]], label_target: dict[str, str]) -> list[str]:
    """The row label of each dense run: its grid row if it exists (required for ``GRID_ONLY``), else its plain row."""
    out = []
    for target in DENSE_TARGETS:
        for lab in runs:
            if label_target[lab] != target or lab.endswith(GRID_SUFFIX):
                continue
            if lab + GRID_SUFFIX in runs:
                out.append(lab + GRID_SUFFIX)
            elif target not in GRID_ONLY:
                out.append(lab)
        for lab in runs:  # a dense run evaluated with the grid only
            base = lab.removesuffix(GRID_SUFFIX)
            if lab.endswith(GRID_SUFFIX) and label_target[lab] == target and base not in runs and lab not in out:
                out.append(lab)
    return out


def _row(rows: list[dict], it: int) -> dict | None:
    for r in rows:
        if r["iteration"] == it:
            return r
    return None


def _nanmax(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return max(xs) if xs else float("nan")


def verdict(runs: dict[str, list[dict]], label_target: dict[str, str], its: list[int]) -> dict:
    final = its[-1]
    hits = _row(runs.get(HITS_RUN, runs.get(HITS_RUN + GRID_SUFFIX, [])), final)
    dense = []
    for label in dense_labels(runs, label_target):
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


def cost(runs: dict[str, list[dict]], its: list[int]) -> dict:
    """Which hits variant would carry the cost criterion (from the pilot speed ratios); the criterion is not computed."""
    final = its[-1]
    fast = [v for v, ratio in PILOT_SPEED.items() if ratio >= COST_SPEED_NEED]
    rows = {}
    for v in PILOT_SPEED:
        r = _row(runs.get(v, runs.get(v + GRID_SUFFIX, [])), final)
        rows[v] = None if r is None else {"setting": r["setting"], "M": r["M"], "F1": r["F1"],
                                          "ord_recall": r["ord_recall"][DK], "in_band": r["in_band"]}
    return {"iteration": final, "speed_need": COST_SPEED_NEED, "pilot_speed": PILOT_SPEED,
            "carrier": fast[0] if len(fast) == 1 else (fast or None), "rows": rows}


def to_markdown(doc: dict) -> str:
    ev = ", ".join(f"{e['split']} (first {e['n_shapes']} shapes, all rays)" for e in doc["evaluation"]) or "none"
    v = doc["verdict"]
    L = ["# H5-K1 report", "",
         f"Run roots: {', '.join(doc['roots'])}.", "",
         f"Evaluation: {ev}. Setting of each row: the H5-K1-1a band selection of training (rows without a label), and for "
         "the rows labelled (grid) the band selection over the whole decoding grid of the target, which for occfrac_128 "
         "includes `binarize` (H5-K1-1b). Best F1 "
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
              "points_std and points_strong are not dense competitors (they enter H5 only through the cost criterion, "
              "not evaluated here).", "",
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
              "The SPEC condition \"with a smaller output\" and the cost criterion are not checked here. The SPEC "
              "compares against TSDF M = 128; here the dense value is the better of the dense ray runs in band "
              "(tsdf_128 counts among them only when it is evaluated and in band).", ""]
    c = doc["cost"]
    carrier = c["carrier"] if isinstance(c["carrier"], str) else (", ".join(c["carrier"]) if c["carrier"] else "none")
    L += ["## Cost (not evaluated yet)", "",
          f"SPEC cost criterion: the same accuracy as the point decoder, at least {c['speed_need']:g}x faster. Speed "
          "ratios from the pilot (points_strong N = 512 / method, trained timings, h5/results/pilot/report.md): "
          + ", ".join(f"{v} {r:g}x" for v, r in c["pilot_speed"].items())
          + f". The variant that would carry the cost criterion: **{carrier}** (at or above {c['speed_need']:g}x; "
          + (", ".join(v for v, r in c["pilot_speed"].items() if r < c["speed_need"]) or "none")
          + " below it, so it cannot meet the criterion whatever its accuracy). The criterion itself (accuracy of the "
          "carrier against the point decoder at equal settings, and a timing of the K1 checkpoints) is not computed "
          "here. For orientation only, the ray-layer values at iteration "
          f"{c['iteration']}:", "",
          "| variant | setting | M | F1 | ord recall 0.002 | in band |", "|---|---|---|---|---|---|"]
    for v, r in c["rows"].items():
        L.append(f"| {v} | not evaluated | | | | |" if r is None else
                 f"| {v} | {r['setting']} | {_f(r['M'])} | {_f(r['F1'])} | {_f(r['ord_recall'])} | "
                 f"{'yes' if r['in_band'] else 'no'} |")
    L.append("")
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
    superseded = [r for r in doc["runs"] if not r.endswith(GRID_SUFFIX) and r + GRID_SUFFIX in doc["runs"]]
    order = [r for r in doc["runs"] if r not in superseded] + superseded
    for key, title in ((DK, "thin recall at 0.002"), (DH, "thin recall at h/2 (descriptive, not the criterion)")):
        for run in order:
            rows = doc["runs"][run]
            names = sorted({r["setting"] for r in rows})
            if run.endswith(GRID_SUFFIX):
                head = f"{run.removesuffix(GRID_SUFFIX)}, grid-selected setting ({', '.join(names)})"
            elif run in superseded:
                band = "out of band, not comparable" if not all(r["in_band"] for r in rows) else "see the grid row"
                head = f"{run}, {', '.join(names)}: {band}"
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
    p.add_argument("--root", type=Path, action="append", required=True,
                   help="run root (contains eval/ and the run directories); may be repeated")
    p.add_argument("--its", type=_its, default=_its("40000,80000,120000"))
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "k1")
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
