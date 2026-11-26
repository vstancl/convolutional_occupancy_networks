"""Ceiling of occfrac_128 on the gate shapes with the existing decoding and with the binarize decoding (H5-K1-1b).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/occ_ceiling_binarize.py [--data ~/PhD/data/csgbox/gate_v1]

The gate pipeline of the csgbox repository is used unchanged: ``csgbox.gate.process_shape`` and the ``merge``,
``metrics`` and ``diagnostics`` of ``csgbox/scripts/gate_ceilings.py``. Only two names are replaced before those
modules are imported: the representation list (two occfrac representations) and the decoder of the new one, which is
``h5/occ_binarize.decode_occfrac_binarize`` on the exact GT fractions (``occ_fraction``, M = 128). The existing one,
``occfrac_128``, is the csgbox ``decode_occfrac``, so its row must reproduce ``gate_v2.md``. That is the check of
this run.

Output: ``h5/results/k1_1/occfrac_ceiling.json`` and ``.md`` (delta 0.002 and h/2, per L/h bin pooled over the
classes, precision, recall, F1, weighted recall of the L/h < 1 bins, |dt|, count error).

H5-G1b added, per representation and delta, the walls+gaps recall with the definition of ``h5/eval_rays.py``
(walls and gaps together, both ends matched; grazing excluded): ``thin_total`` and ``thin_recalled`` (class x bin,
rows wall, gap, grazing), ``walls_gaps`` (per L/h bin) and ``walls_gaps_pooled_L_lt_1``. With ``--add-walls-gaps``
the existing JSON is kept: the run is repeated, every stored field must come out identical (``runtime_s`` aside,
which keeps its old value), and only the new fields are added.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

H5_DIR = Path(__file__).resolve().parents[1]
CSGBOX = Path.home() / "PhD" / "csgbox"
sys.path.insert(0, str(H5_DIR))
sys.path.insert(0, str(CSGBOX))
sys.path.insert(0, str(CSGBOX / "scripts"))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

import csgbox.gate as gate  # noqa: E402
from csgbox.ceilings import occ_fraction  # noqa: E402
from csgbox.constants import GEN_CHECK_N  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402
from occ_binarize import decode_occfrac_binarize  # noqa: E402

REP_OLD = "occfrac_128"
REP_NEW = "occfrac_128_bin"
_orig_decode = gate._decode


def _decode(rep, tree, gt, axis):
    if rep == REP_NEW:
        return decode_occfrac_binarize(occ_fraction(gt, GEN_CHECK_N).astype(np.float32))
    return _orig_decode(rep, tree, gt, axis)


# patched before the gate modules are imported, so that process_shape and merge see the two representations
gate._decode = _decode
gate.REPRESENTATIONS = (REP_OLD, REP_NEW)

import gate_ceilings as gc  # noqa: E402

N_SMALL_BINS = 4  # L/h < 1: bins [0, 0.125) .. [0.5, 1)
BIN_LABELS = ("[0,0.125)", "[0.125,0.25)", "[0.25,0.5)", "[0.5,1)", "[1,2)", "[2,4)")


def pooled_bins(agg: dict, rep: str, key: str) -> dict:
    """Recall per L/h bin pooled over the classes, and the population-weighted recall of the L/h < 1 bins."""
    d = agg["reps"][rep]["deltas"][key]
    total = np.asarray(agg["thin_total"], dtype=np.int64).sum(axis=0)
    recalled = np.asarray(d["thin_recalled"], dtype=np.int64).sum(axis=0)
    recall = np.where(total > 0, recalled / np.maximum(total, 1), np.nan)
    small = recalled[:N_SMALL_BINS].sum() / total[:N_SMALL_BINS].sum()
    return {"n": total.tolist(), "bin_recall": recall.tolist(), "weighted_recall_lt1": float(small)}


WG_FIELDS = ("thin_total", "thin_recalled", "walls_gaps", "walls_gaps_pooled_L_lt_1")


def walls_gaps(agg: dict, rep: str, key: str) -> dict:
    """Walls+gaps recall per L/h bin and pooled over L/h < 1 (rows 0 and 1 of the class x bin counts)."""
    total = np.asarray(agg["thin_total"], dtype=np.int64)
    recalled = np.asarray(agg["reps"][rep]["deltas"][key]["thin_recalled"], dtype=np.int64)
    wg_tot, wg_rec = total[0] + total[1], recalled[0] + recalled[1]
    wg = np.where(wg_tot > 0, wg_rec / np.maximum(wg_tot, 1), np.nan)
    return {"thin_total": total.tolist(), "thin_recalled": recalled.tolist(), "walls_gaps": wg.tolist(),
            "walls_gaps_pooled_L_lt_1": float(wg_rec[:N_SMALL_BINS].sum() / wg_tot[:N_SMALL_BINS].sum())}


def add_to_existing(path: Path, doc: dict) -> dict:
    """The stored document with the walls+gaps fields added; every stored field must be reproduced exactly."""
    old = json.loads(path.read_text(encoding="utf-8"))
    for k in old:
        if k in ("runtime_s", "reps"):
            continue
        if old[k] != doc[k]:
            raise SystemExit(f"stored field {k} differs: {old[k]} != {doc[k]}")
    for rep, per_delta in old["reps"].items():
        for label, stored in per_delta.items():
            new = doc["reps"][rep][label]
            for k, v in stored.items():
                if k in WG_FIELDS:
                    continue
                if json.dumps(v) != json.dumps(new[k]):
                    raise SystemExit(f"stored field reps.{rep}.{label}.{k} differs: {v} != {new[k]}")
            for k in WG_FIELDS:
                stored[k] = new[k]
    return old


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path.home() / "PhD" / "data" / "csgbox" / "gate_v1")
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "k1_1")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--add-walls-gaps", action="store_true",
                   help="keep the stored JSON, check it is reproduced, add only the walls+gaps fields")
    args = p.parse_args(argv)
    start = time.perf_counter()
    dirs = sorted((d for d in args.data.glob("seed_*") if d.is_dir()), key=lambda d: int(d.name.split("_")[1]))
    with mp.Pool(args.workers) as pool:
        results = list(pool.imap(gate.process_shape, [str(d) for d in dirs]))
    agg = gc.merge(results)
    met = gc.metrics(agg)
    diag = gc.diagnostics(agg)
    h = cell_size(GEN_CHECK_N)
    keys = {"0.002": str(0.002), "h/2": str(h / 2.0)}
    doc = {"shapes": len(results), "data": str(args.data), "runtime_s": time.perf_counter() - start,
           "reps": {}, "bins": list(BIN_LABELS), "diagnostics": {"area_median": diag["area_median"]}}
    for rep in (REP_OLD, REP_NEW):
        doc["reps"][rep] = {}
        for label, key in keys.items():
            m = met[rep]["deltas"][key]
            doc["reps"][rep][label] = {
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "dt_mean": m["dt_mean"], "dt_p95": m["dt_p95"], "count_err_mean": m["count_err_mean"],
                "dec_hits": agg["reps"][rep]["deltas"][key]["dec_hits"],
                "gt_hits": agg["reps"][rep]["deltas"][key]["gt_hits"],
                **pooled_bins(agg, rep, key),
                **walls_gaps(agg, rep, key),
            }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.add_walls_gaps:
        doc = add_to_existing(args.out / "occfrac_ceiling.json", doc)
    (args.out / "occfrac_ceiling.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (args.out / "occfrac_ceiling.md").write_text(to_markdown(doc), encoding="utf-8")
    print(f"shapes {len(results)}, runtime {doc['runtime_s']:.1f} s -> {args.out / 'occfrac_ceiling.md'}")
    return 0


def _f(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.4g}"


def to_markdown(doc: dict) -> str:
    L = [f"# occfrac_128 ceiling on the gate shapes ({doc['shapes']} shapes)", "",
         f"Data: `{doc['data']}`, N = {GEN_CHECK_N}. `occfrac_128` is the csgbox `decode_occfrac` (the row must match "
         "gate_v2.md); `occfrac_128 binarize` is the K1-1b rule (h5/occ_binarize.py) on the same GT fractions. Thin "
         "features: material and gap intervals shorter than 4 h, pooled over the classes per L/h bin.", ""]
    for label in ("0.002", "h/2"):
        L += [f"## delta {label}", "",
              "| representation | precision | recall | F1 | |dt| mean | |dt| p95 | count err | recall of L/h < 1 (weighted) | "
              + " | ".join(f"recall {b}" for b in doc["bins"]) + " |",
              "|---|---|---|---|---|---|---|---|" + "---|" * len(doc["bins"])]
        for rep, name in ((REP_OLD, "occfrac_128 (decode_occfrac)"), (REP_NEW, "occfrac_128 binarize")):
            r = doc["reps"][rep][label]
            L.append(f"| {name} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} | {_f(r['dt_mean'])} | "
                     f"{_f(r['dt_p95'])} | {_f(r['count_err_mean'])} | {_f(r['weighted_recall_lt1'])} | "
                     + " | ".join(_f(x) for x in r["bin_recall"]) + " |")
        n = doc["reps"][REP_OLD][label]["n"]
        L += ["", "Thin population per bin (pooled over the classes): " + ", ".join(
            f"{b} {c}" for b, c in zip(doc["bins"], n)), ""]
        if "walls_gaps" in doc["reps"][REP_OLD][label]:
            L += ["Walls+gaps recall (both ends matched, grazing excluded; the definition of h5/eval_rays.py):", "",
                  "| representation | pooled L/h < 1 (M) | " + " | ".join(doc["bins"]) + " |",
                  "|---|---|" + "---|" * len(doc["bins"])]
            for rep, name in ((REP_OLD, "occfrac_128 (decode_occfrac)"), (REP_NEW, "occfrac_128 binarize")):
                r = doc["reps"][rep][label]
                L.append(f"| {name} | {_f(r['walls_gaps_pooled_L_lt_1'])} | "
                         + " | ".join(_f(x) for x in r["walls_gaps"]) + " |")
            tt = doc["reps"][REP_OLD][label]["thin_total"]
            L += ["", "Walls+gaps population per bin: " + ", ".join(
                f"{b} {tt[0][i] + tt[1][i]}" for i, b in enumerate(doc["bins"])), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
