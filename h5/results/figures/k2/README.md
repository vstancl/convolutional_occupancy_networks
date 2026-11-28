# H5-G6/G7: K2 figures for the meeting

Made on the CPU from the written reports; nothing was trained or evaluated. Each figure is a PNG (dpi 200) plus a PDF of the same name. Colours from `h5/scripts/fig_style.py` (points_mix_a / points_mix_b in the hue family of their ray-interval variant). Regenerate:

```
cd ~/PhD/convonet && .venv/bin/python h5/scripts/plot_k2.py
```

Sources: `h5/results/k2/report.json` and `h5/results/k2p/report.json` (`final`, iteration 60000, the 32 training shapes of Q0, all rays; split train_subset, 32), `h5/results/k2_3/report.json` (`rows` at iteration 120000, val first 200 shapes, all rays). The rows of K2_points_rayint_a/b and Q0_points_strong are equal in the K2 and K2' reports (checked). Delta 0.002; weighted thin recall = per-hit thin recall population-weighted over the four L/h < 1 bins; M = walls+gaps with L/h < 1, both ends matched, pooled. Every row is in band.

## g6_2x2

Files: `h5/results/figures/k2/g6_2x2.png`, `h5/results/figures/k2/g6_2x2.pdf`.

Memorisation, 32 shapes, 60k. 2x2 design, rows without / with the near-surface half, columns length-proportional (variant b) / equal weight per segment (variant a). Panels: ordinary recall and weighted thin recall; Q0_points_strong dashed.

| run | near-surface half | segment weighting | ordinary recall 0.002 | weighted thin L/h < 1 |
|---|---|---|---|---|
| K2_points_rayint_b | without | length-proportional (b) | 0.6548 | 0.0688 |
| K2_points_rayint_a | without | equal weight per segment (a) | 0.5935 | 0.4287 |
| K2p_points_mix_b | with | length-proportional (b) | 0.7169 | 0.1949 |
| K2p_points_mix_a | with | equal weight per segment (a) | 0.7071 | 0.4638 |
| Q0_points_strong (reference) | | | 0.7224 | 0.2473 |

## g6b_thin_bins

Files: `h5/results/figures/k2/g6b_thin_bins.png`, `h5/results/figures/k2/g6b_thin_bins.pdf`.

Per-hit thin recall at 0.002 per L/h bin (categorical x), the four memorisation runs and Q0_points_strong (dotted: without near-surface half, solid: with, dashed: reference).

| L/h bin | n | points_rayint_b | points_rayint_a | points_mix_b | points_mix_a | points_strong |
|---|---|---|---|---|---|---|
| <1/8 | 3928 | 0.0262 | 0.1265 | 0.0509 | 0.0860 | 0.0591 |
| 1/8-1/4 | 4048 | 0.0371 | 0.1638 | 0.0699 | 0.1057 | 0.0840 |
| 1/4-1/2 | 100649 | 0.0799 | 0.5242 | 0.2206 | 0.5331 | 0.2792 |
| 1/2-1 | 81472 | 0.0588 | 0.3385 | 0.1763 | 0.4141 | 0.2251 |
| 1-2 | 126373 | 0.2183 | 0.4636 | 0.3914 | 0.4962 | 0.4306 |
| 2-4 | 145027 | 0.2526 | 0.3897 | 0.3338 | 0.3877 | 0.3519 |

## g7_memorisation_vs_new

Files: `h5/results/figures/k2/g7_memorisation_vs_new.png`, `h5/results/figures/k2/g7_memorisation_vs_new.pdf`.

points_strong vs points_mix_a. Panels ordinary recall, weighted thin recall, M (log y). Groups: memorisation (Q0_points_strong vs K2p_points_mix_a, 60k) and new shapes (K1 points_strong vs K2_3_points_mix_a, 120k); hits_8 at 120k dashed in the new-shapes group of the thin and M panels; the ratio mix_a / points_strong above each pair.

| group | quantity | points_strong | points_mix_a | ratio mix_a / points_strong | hits_8 (120k) |
|---|---|---|---|---|---|
| memorisation 60k | ordinary recall, 0.002 | 0.7224 | 0.7071 | 0.98 |  |
| memorisation 60k | weighted thin recall L/h < 1, 0.002 | 0.2473 | 0.4638 | 1.88 |  |
| memorisation 60k | M (walls+gaps L/h < 1) | 0.1840 | 0.4227 | 2.30 |  |
| new shapes 120k | ordinary recall, 0.002 | 0.2911 | 0.2825 | 0.97 |  |
| new shapes 120k | weighted thin recall L/h < 1, 0.002 | 0.0606 | 0.0759 | 1.25 | 0.1294 |
| new shapes 120k | M (walls+gaps L/h < 1) | 0.0029 | 0.0047 | 1.61 | 0.0306 |

## All rows used

- K2_points_rayint_a (it 60000): ordinary recall 0.5935, weighted thin 0.4287, M 0.3530, decoded/GT 1.035, setting iso=0.5.
- K2_points_rayint_b (it 60000): ordinary recall 0.6548, weighted thin 0.0688, M 0.0108, decoded/GT 0.867, setting iso=0.5.
- Q0_points_strong (it 60000): ordinary recall 0.7224, weighted thin 0.2473, M 0.1840, decoded/GT 1.020, setting iso=0.5.
- K2p_points_mix_a (it 60000): ordinary recall 0.7071, weighted thin 0.4638, M 0.4227, decoded/GT 1.061, setting iso=0.5.
- K2p_points_mix_b (it 60000): ordinary recall 0.7169, weighted thin 0.1949, M 0.1279, decoded/GT 0.987, setting iso=0.5.
- K2_3_points_mix_a (it 120000): ordinary recall 0.2825, weighted thin 0.0759, M 0.0047, decoded/GT 0.964, setting iso=0.5.
- K1 points_strong (it 120000): ordinary recall 0.2911, weighted thin 0.0606, M 0.0029, decoded/GT 0.877, setting iso=0.5.
- K1 hits_8 (it 120000): ordinary recall 0.1976, weighted thin 0.1294, M 0.0306, decoded/GT 0.942, setting thr=0.6.
