#!/usr/bin/env python3
"""Per-slot positive rate under the training ray sampler, and the pos_weight it implies (h5 step H5-1a, f).

Sampler (the mixture the training loop draws rays from):

- with probability 1/2, a ray uniform over all 3 * 128^2 rays of the shape (all axes pooled);
- with probability 1/2, a ray uniform over the rays with at least 2 hits (all axes pooled).

Per slot s the mixture positive rate is ``p_s = 0.5 * P_all(mask_s) + 0.5 * P_ge2(mask_s)``, where
the probabilities are pooled over the selected shapes. ``pos_weight_s = (1 - p_s) / p_s`` is the
weight of the positive class in a BCE loss that balances the slot against its negatives.

Shapes: SAMPLE_SHAPES training shapes from csg/train.lst, chosen with numpy Generator(PCG64(0)).

    ~/PhD/convonet/.venv/bin/python h5/scripts/pos_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from rays import load_axis_rayhits  # noqa: E402
from slots import encode_slots  # noqa: E402

DATA_ROOT = Path.home() / "PhD" / "data" / "csgbox" / "v1"
SAMPLE_SHAPES = 200
SELECTION_SEED = 0
N = 128
K_VALUES = (2, 4)  # hits_8 (K = 2) and hits_16 (K = 4)


def train_shape_ids(data_root: Path = DATA_ROOT) -> list[str]:
    ids = (data_root / "csg" / "train.lst").read_text(encoding="utf-8").split()
    rng = np.random.Generator(np.random.PCG64(SELECTION_SEED))
    picked = rng.choice(len(ids), size=SAMPLE_SHAPES, replace=False)
    return sorted(ids[i] for i in picked)


def per_slot_counts(shapes: list[str], K: int, data_root: Path = DATA_ROOT) -> dict:
    """Pooled counts over the shapes and the three axes: rays, positives per slot, for all rays and for rays with >= 2 hits."""
    rays_all = 0
    rays_ge2 = 0
    pos_all = np.zeros(2 * K, dtype=np.int64)
    pos_ge2 = np.zeros(2 * K, dtype=np.int64)
    for shape in shapes:
        for axis in range(3):
            gt = load_axis_rayhits(data_root / "csg" / shape / "dexels_128.npz", axis)
            _t, mask = encode_slots(gt, K)
            ge2 = np.diff(gt.offsets) >= 2
            rays_all += gt.n_rays
            rays_ge2 += int(ge2.sum())
            pos_all += mask.sum(axis=0)
            pos_ge2 += mask[ge2].sum(axis=0)
    return {"K": K, "rays_all": rays_all, "rays_ge2": rays_ge2, "pos_all": pos_all, "pos_ge2": pos_ge2}


def slot_names(K: int) -> list[str]:
    return [f"F{j}" for j in range(K)] + [f"B{j}" for j in reversed(range(K))]


def main() -> None:
    shapes = train_shape_ids()
    print(f"{len(shapes)} train shapes, 3 axes, N = {N}")
    for K in K_VALUES:
        c = per_slot_counts(shapes, K)
        p_all = c["pos_all"] / c["rays_all"]
        p_ge2 = c["pos_ge2"] / c["rays_ge2"]
        p_mix = 0.5 * p_all + 0.5 * p_ge2
        pos_weight = (1.0 - p_mix) / p_mix
        print(f"\n## K = {K} (hits_{4 * K}), rays_all = {c['rays_all']}, rays_ge2 = {c['rays_ge2']}")
        print("| slot | P_all | P_ge2 | p_mix | pos_weight |")
        print("|---|---|---|---|---|")
        for i, name in enumerate(slot_names(K)):
            print(f"| {name} | {p_all[i]:.6f} | {p_ge2[i]:.6f} | {p_mix[i]:.6f} | {pos_weight[i]:.4f} |")


if __name__ == "__main__":
    main()
