#!/usr/bin/env python3
"""Ray statistics for the h5 survey of ray decoders (h5/docs/survey.md, question Q8).

Selects ``SAMPLE_SHAPES`` training shapes with ``numpy.random.default_rng(SELECTION_SEED)``
and, for every ray axis of their ``dexels_128.npz`` files (read with ``dexio``):

1. the fraction of rays with 0, 2, 4, 6, 8 hits, more than 8 hits, or an odd count;
2. the positive rate of every slot under the CURRENT ``hits_8`` layout (see ``current_layout``);
3. the positive rate of every slot under the TWO-SIDED layout (see ``two_sided_layout``);
4. per ray, whether both layouts decode the same set of hits, and so the same intervals.

Dependencies: numpy and dexio only. Nothing is imported from csgbox, ConvONet or phd_work.

Usage (from the ConvONet venv, numpy 1.26; dexio is found through h5/_paths.py):

    python h5/scripts/ray_stats.py --data-root ~/PhD/data/csgbox/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # h5/, for _paths
import _paths  # noqa: E402,F401  (puts csgbox and dexio on sys.path)

SAMPLE_SHAPES: int = 200  # training shapes examined
SELECTION_SEED: int = 0  # seed of the shape selection (numpy default_rng)
L_HITS: int = 8  # slots per ray of the hits_8 representation
HALF_SLOTS: int = L_HITS // 2  # slots per half (front and back)
TWO_SIDED_SLOTS: int = 2  # intervals kept from each end in the two-sided layout
AXES: tuple[str, ...] = ("x", "y", "z")
BUCKETS: tuple[str, ...] = ("0", "2", "4", "6", "8", ">8", "odd")


def training_ids(data_root: Path) -> list[str]:
    """Model ids of the training shapes with status ok, from generation.jsonl (last record wins)."""
    records: dict[str, dict] = {}
    with open(data_root / "generation.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "model_id" in rec:
                records[rec["model_id"]] = rec
    ids = sorted(k for k, r in records.items() if r.get("split") == "train" and r.get("status") == "ok")
    return ids


def select_shapes(ids: list[str]) -> list[str]:
    """SAMPLE_SHAPES ids drawn without replacement with the seeded generator; returned in sorted order."""
    rng = np.random.default_rng(SELECTION_SEED)
    picked = rng.choice(len(ids), size=SAMPLE_SHAPES, replace=False)
    return sorted(ids[i] for i in picked)


def current_layout(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank and validity of the 8 slots of hits_8 as fixed slots (an inferred reading, see survey.md Q1).

    Slots 0..3 hold the hits of rank 0..3 (from the front), valid if rank < n.
    Slots 4..7 hold the hits of rank n-4..n-1 (right-aligned, so slot 7 is the last hit),
    valid if rank >= 0. For n < 8 the two halves overlap, so the same hit can fill two slots.
    """
    j = np.arange(HALF_SLOTS)
    front_rank = np.broadcast_to(j, (n.size, HALF_SLOTS))
    front_valid = front_rank < n[:, None]
    back_rank = n[:, None] - HALF_SLOTS + j
    back_valid = back_rank >= 0
    return np.concatenate([front_rank, back_rank], axis=1), np.concatenate([front_valid, back_valid], axis=1)


def two_sided_layout(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interval index and validity of the 4 slots of the two-sided layout.

    Slots 0..1 hold intervals 0 and 1 from the front; slots 2..3 hold intervals K-1 and K-2 from the
    back, where K = n // 2 is the number of intervals. Both halves are filled independently, so with
    K < 4 an interval can fill one slot from each side.
    """
    K = n // 2
    j = np.arange(TWO_SIDED_SLOTS)
    front_i = np.broadcast_to(j, (n.size, TWO_SIDED_SLOTS))
    front_valid = front_i < K[:, None]
    back_i = K[:, None] - 1 - j
    back_valid = back_i >= 0
    return np.concatenate([front_i, back_i], axis=1), np.concatenate([front_valid, back_valid], axis=1)


def select_ranks(width: int, rank_lo: np.ndarray, valid: np.ndarray, both: bool) -> np.ndarray:
    """(R, width) bool matrix of the hit ranks a layout selects; ``both`` adds rank + 1 (an interval)."""
    sel = np.zeros((valid.shape[0], width), dtype=bool)
    rows, cols = np.nonzero(valid)
    lo = rank_lo[rows, cols]
    sel[rows, lo] = True
    if both:
        sel[rows, lo + 1] = True
    return sel


def bucket_of(n: np.ndarray) -> np.ndarray:
    """Index into BUCKETS for every ray."""
    out = np.full(n.shape, BUCKETS.index(">8"), dtype=np.int64)
    for i, label in enumerate(("0", "2", "4", "6", "8")):
        out[n == int(label)] = i
    out[n % 2 == 1] = BUCKETS.index("odd")
    return out


def axis_stats(offsets: np.ndarray) -> dict:
    """Counts of one axis of one shape: bucket counts, slot valid counts, and the layout comparison."""
    n = np.diff(offsets).astype(np.int64)
    R = n.size
    width = int(n.max()) if R else 0
    width += width % 2  # even width, so whole intervals can be compared pairwise
    width = max(width, 2)

    cur_rank, cur_valid = current_layout(n)
    cur_sel = select_ranks(width, cur_rank, cur_valid, both=False)
    two_idx, two_valid = two_sided_layout(n)
    two_sel = select_ranks(width, 2 * two_idx, two_valid, both=True)

    cur_slots = cur_valid.sum(axis=1)
    cur_unique = cur_sel.sum(axis=1)
    two_slots = two_valid.sum(axis=1)
    two_unique = two_sel.sum(axis=1) // 2  # two ranks per interval: count intervals, not ranks

    even = (n % 2 == 0)
    # intervals of the current layout: ranks (2k, 2k+1) must both be selected or both be absent
    pairs_cur = cur_sel.reshape(R, -1, 2)
    split_rays = np.any(pairs_cur[:, :, 0] != pairs_cur[:, :, 1], axis=1) & even

    return {
        "rays": R,
        "bucket_counts": np.bincount(bucket_of(n), minlength=len(BUCKETS)).astype(np.int64),
        "current_slot_valid": cur_valid.sum(axis=0).astype(np.int64),
        "two_sided_slot_valid": two_valid.sum(axis=0).astype(np.int64),
        "current_dup_rays": int(np.count_nonzero(cur_slots > cur_unique)),
        "current_dup_slots": int(np.sum(cur_slots - cur_unique)),
        "two_sided_dup_rays": int(np.count_nonzero(two_slots > two_unique)),
        "two_sided_dup_slots": int(np.sum(two_slots - two_unique)),
        "layout_mismatch_rays": int(np.count_nonzero(np.any(cur_sel != two_sel, axis=1))),
        "current_split_interval_rays": int(np.count_nonzero(split_rays)),
        "rays_with_hits_truncated": int(np.count_nonzero(n > L_HITS)),
    }


def add_stats(total: dict | None, part: dict) -> dict:
    if total is None:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in part.items()}
    for k, v in part.items():
        total[k] = total[k] + v
    return total


def report(per_axis: dict[str, dict], shapes: int) -> str:
    lines = [f"shapes: {shapes}"]
    for axis in AXES:
        s = per_axis[axis]
        R = s["rays"]
        lines.append(f"\n## axis {axis}  (rays {R})")
        lines.append("hit count: " + "  ".join(f"{b}={s['bucket_counts'][i] / R:.6f}" for i, b in enumerate(BUCKETS)))
        lines.append("current hits_8 slot positive rate (slots 0..3 front, 4..7 back):")
        lines.append("  " + " ".join(f"{v / R:.6f}" for v in s["current_slot_valid"]))
        lines.append("two-sided slot positive rate (slots 0..1 front, 2..3 back):")
        lines.append("  " + " ".join(f"{v / R:.6f}" for v in s["two_sided_slot_valid"]))
        lines.append(f"rays with duplicate slots: current {s['current_dup_rays']} ({s['current_dup_slots']} slots), "
                     f"two-sided {s['two_sided_dup_rays']} ({s['two_sided_dup_slots']} slots)")
        lines.append(f"rays with more than {L_HITS} hits (middle hits dropped): {s['rays_with_hits_truncated']}")
        lines.append(f"rays where current decodes a split interval (even n): {s['current_split_interval_rays']}")
        lines.append(f"rays where the two layouts decode different hit sets: {s['layout_mismatch_rays']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="csgbox v1 root (contains generation.jsonl and csg/)")
    args = ap.parse_args()
    import dexio  # noqa: E402  (importable through h5/_paths.py)

    ids = select_shapes(training_ids(args.data_root))
    per_axis: dict[str, dict | None] = {a: None for a in AXES}
    for model_id in ids:
        dex = dexio.read(args.data_root / "csg" / model_id / "dexels_128.npz")
        for axis in AXES:
            per_axis[axis] = add_stats(per_axis[axis], axis_stats(dex.axes[axis].offsets))
    print(report({a: per_axis[a] for a in AXES}, len(ids)))


if __name__ == "__main__":
    main()
