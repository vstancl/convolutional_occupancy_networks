# H5-K2-3 report: points_mix_a on the full train split against the K1 references

New run: K2_3_points_mix_a (/home/wish/PhD/runs/k2_3, checkpoints 40000, 80000, 120000), the settings of the K1-1 points_strong run, only the target differs. References (read from their evaluations, not re-evaluated): points_strong, hits_8, occfrac_128 (grid), tsdf_128 (grid) (/home/wish/PhD/runs/k1_1, /home/wish/PhD/runs/k1_4, checkpoints 40000, 80000, 120000).

Evaluation: val (first 200 shapes, all rays). Setting: the band selection of training (decoded/GT in [0.8, 1.25], best F1 at 0.002, otherwise the closest ratio); for the rows labelled (grid) the same selection over the whole decoding grid. Weighted thin recall L/h < 1: per-hit thin recall at 0.002, population-weighted over the four L/h < 1 bins. M: walls and gaps with L/h < 1, both ends matched, pooled.

## Rule (fixed on 27. 11. 2026, at iteration 120000)

**STOP**: K2_3_points_mix_a ordinary recall 0.2825 (threshold 0.271, margin +0.0115), M 0.004691 (threshold 0.031, margin -0.0263), decoded/GT 0.9638 (in band, margin to the nearer band edge +0.1638).

Check: points_strong at 120000 read here: ordinary recall 0.2911, M 0.002911; equal to /home/wish/PhD/convonet/h5/results/k1/report.json: yes; rounded 0.291 / 0.003 as stated (0.291 / 0.003): yes.

## Per checkpoint

| run | iteration | setting | ordinary recall 0.002 | weighted thin recall L/h < 1 | M | precision | decoded/GT | in band | F1 | ordinary median |
|---|---|---|---|---|---|---|---|---|---|---|
| K2_3_points_mix_a | 40000 | iso=0.5 | 0.22 | 0.0607 | 0.00273 | 0.177 | 0.957 | yes | 0.173 | 0.00615 |
| K2_3_points_mix_a | 80000 | iso=0.5 | 0.265 | 0.0683 | 0.00386 | 0.216 | 0.938 | yes | 0.209 | 0.00488 |
| K2_3_points_mix_a | 120000 | iso=0.5 | 0.282 | 0.0759 | 0.00469 | 0.225 | 0.964 | yes | 0.221 | 0.00453 |
| points_strong | 40000 | iso=0.5 | 0.223 | 0.0476 | 0.0019 | 0.191 | 0.873 | yes | 0.178 | 0.00615 |
| points_strong | 80000 | iso=0.5 | 0.272 | 0.053 | 0.00208 | 0.24 | 0.846 | yes | 0.219 | 0.0048 |
| points_strong | 120000 | iso=0.5 | 0.291 | 0.0606 | 0.00291 | 0.248 | 0.877 | yes | 0.232 | 0.00443 |
| hits_8 | 40000 | thr=0.6 | 0.134 | 0.102 | 0.0192 | 0.129 | 0.925 | yes | 0.124 | 0.00931 |
| hits_8 | 80000 | thr=0.6 | 0.172 | 0.13 | 0.0313 | 0.159 | 0.961 | yes | 0.156 | 0.00718 |
| hits_8 | 120000 | thr=0.6 | 0.198 | 0.129 | 0.0306 | 0.182 | 0.942 | yes | 0.176 | 0.00622 |
| occfrac_128 (grid) | 40000 | binarize | 0.192 | 0.0464 | 0.00206 | 0.164 | 0.884 | yes | 0.154 | 0.00702 |
| occfrac_128 (grid) | 80000 | binarize | 0.244 | 0.0567 | 0.00284 | 0.206 | 0.9 | yes | 0.195 | 0.00535 |
| occfrac_128 (grid) | 120000 | binarize | 0.264 | 0.0605 | 0.00338 | 0.22 | 0.908 | yes | 0.21 | 0.00494 |
| tsdf_128 (grid) | 40000 | default | 0.201 | 0.0476 | 0.00115 | 0.183 | 0.825 | yes | 0.166 | 0.00651 |
| tsdf_128 (grid) | 80000 | default | 0.243 | 0.0575 | 0.00192 | 0.218 | 0.843 | yes | 0.2 | 0.0052 |
| tsdf_128 (grid) | 120000 | default | 0.257 | 0.0605 | 0.00226 | 0.23 | 0.849 | yes | 0.211 | 0.00491 |

## Thin recall per L/h bin at 0.002, last checkpoint (n)

| L/h bin | K2_3_points_mix_a (it 120000) | points_strong (it 120000) | hits_8 (it 120000) | occfrac_128 (grid) (it 120000) | tsdf_128 (grid) (it 120000) |
|---|---|---|---|---|---|
| [0,0.125) | 0.0363 (26933) | 0.0288 (26933) | 0.0362 (26933) | 0.025 (26933) | 0.0252 (26933) |
| [0.125,0.25) | 0.0737 (33718) | 0.0606 (33718) | 0.0854 (33718) | 0.0569 (33718) | 0.0568 (33718) |
| [0.25,0.5) | 0.0753 (495361) | 0.0664 (495361) | 0.151 (495361) | 0.0695 (495361) | 0.07 (495361) |
| [0.5,1) | 0.0776 (861490) | 0.0582 (861490) | 0.122 (861490) | 0.0566 (861490) | 0.0562 (861490) |
| [1,2) | 0.0712 (1058120) | 0.0514 (1058120) | 0.11 (1058120) | 0.056 (1058120) | 0.0541 (1058120) |
| [2,4) | 0.0864 (1034922) | 0.0738 (1034922) | 0.0978 (1034922) | 0.0732 (1034922) | 0.0713 (1034922) |

## Thin recall per L/h bin at h/2 (descriptive), last checkpoint (n)

| L/h bin | K2_3_points_mix_a (it 120000) | points_strong (it 120000) | hits_8 (it 120000) | occfrac_128 (grid) (it 120000) | tsdf_128 (grid) (it 120000) |
|---|---|---|---|---|---|
| [0,0.125) | 0.0731 (26933) | 0.0546 (26933) | 0.0708 (26933) | 0.0476 (26933) | 0.0489 (26933) |
| [0.125,0.25) | 0.111 (33718) | 0.0904 (33718) | 0.136 (33718) | 0.0816 (33718) | 0.0841 (33718) |
| [0.25,0.5) | 0.139 (495361) | 0.114 (495361) | 0.269 (495361) | 0.119 (495361) | 0.119 (495361) |
| [0.5,1) | 0.154 (861490) | 0.116 (861490) | 0.251 (861490) | 0.12 (861490) | 0.117 (861490) |
| [1,2) | 0.145 (1058120) | 0.106 (1058120) | 0.227 (1058120) | 0.119 (1058120) | 0.113 (1058120) |
| [2,4) | 0.175 (1034922) | 0.147 (1034922) | 0.203 (1034922) | 0.151 (1034922) | 0.146 (1034922) |
