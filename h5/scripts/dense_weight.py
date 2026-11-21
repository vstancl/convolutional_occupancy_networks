"""Loss weight of the informative cells of the dense targets (h5 step H5-1b).

On 200 train shapes (``Generator(PCG64(0))`` chooses them from ``csg/train.lst``), under the training
sampler (``RayShapeDataset`` with ``seed=0``), the fraction ``f`` of informative cells is measured per
dense target. The weight ``w = (1 - f) / f`` makes the informative cells 50 % of the loss when every
cell has the same per-cell loss: ``w f = (1 - f)``.

Run from the ConvONet venv:

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python h5/scripts/dense_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from data import DENSE_TARGETS, RayShapeDataset, DATA_ROOT  # noqa: E402

N_SHAPES = 200
SELECTION_SEED = 0
TRAIN_SEED = 0


def main() -> None:
    ids_len = len((DATA_ROOT / "csg" / "train.lst").read_text(encoding="utf-8").split())
    rng = np.random.Generator(np.random.PCG64(SELECTION_SEED))
    picked = rng.choice(ids_len, size=N_SHAPES, replace=False)

    print(f"{N_SHAPES} train shapes, Generator(PCG64({SELECTION_SEED})), training sampler seed={TRAIN_SEED}")
    print()
    print("| target | informative cells | total cells | f (pooled) | mean of per-shape f | w = (1-f)/f |")
    print("|---|---|---|---|---|---|")
    for target in DENSE_TARGETS:
        ds = RayShapeDataset("train", target, seed=TRAIN_SEED)
        inf_sum = 0
        cells = 0
        per_shape = []
        for index in picked:
            item = ds[int(index)]
            inf = item["informative"]
            inf_sum += int(inf.sum())
            cells += inf.size
            per_shape.append(inf.mean())
        f = inf_sum / cells
        w = (1.0 - f) / f
        print(f"| {target} | {inf_sum} | {cells} | {f:.4f} | {np.mean(per_shape):.4f} | {w:.3f} |")


if __name__ == "__main__":
    main()
