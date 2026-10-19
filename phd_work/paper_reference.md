# Paper reference numbers

Source: Peng et al., *Convolutional Occupancy Networks*, ECCV 2020, arXiv:2003.04618, Table 2
(voxel super-resolution, 32³ input; Choy et al. split, same preprocessed data).

| Method                 | IoU   | Chamfer-L1 (paper, ×10) | Normal C. | F-Score |
| ---------------------- | ----- | ----------------------- | --------- | ------- |
| Ours-2D (3×64²)        | 0.752 | 0.092                   | 0.905     | 0.735   |

Notes:
- `eval_meshes.py` reports **raw** Chamfer-L1; the paper multiplies by 10. Paper 0.092 ⇒ 0.0092 raw.
- The `mean` row printed by `eval_meshes.py` is the mean over classes (not over objects), as in the paper.
