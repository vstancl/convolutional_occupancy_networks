# H5-K1-1 report

Evaluation: val (first 200 shapes, all rays). Setting of each row: the H5-K1-1a band selection of training (rows without a label), and for occfrac_128 (grid) the band selection over the extended grid, which includes `binarize` (H5-K1-1b). Best F1 at 0.002 among the settings with decoded/GT in [0.8, 1.25]; otherwise the closest ratio. Recall and distances at delta 0.002 unless labelled h/2.

M (H5-6): recall at 0.002 of the walls and gaps with L/h < 1, both ends matched, pooled over the four L/h < 1 bins; walls+gaps per bin: the same per L/h bin. These are the thin-feature quantities of the verdict. The thin-bin tables below show the per-hit thin recall, which is descriptive.

## Verdict (H5, SPEC section 5, layer 1, delta 0.002)

**undecided (SPEC: compactness only)** at iteration 120000: hits_8 against the dense ray runs in band (occfrac_128 (grid)); the dense value is the better of them, per L/h bin for the bins. points_strong is not a dense competitor (it enters H5 only through the cost criterion, not evaluated here).

| quantity | hits_8 | dense | difference | rule |
|---|---|---|---|---|
| M (walls+gaps, L/h < 1) | 0.0306 | 0.00338 | 0.0272 | at least +0.05 |
| F1 of all hits, 0.002 | 0.176 | 0.21 | -0.0335 | not below -0.01 (t/h > 2 condition) |
| walls+gaps [0,0.125) | 0.000348 | 0 | 0.000348 | refuted if at most +0.02 in every bin |
| walls+gaps [0.125,0.25) | 0.0376 | 0 | 0.0376 | refuted if at most +0.02 in every bin |
| walls+gaps [0.25,0.5) | 0.0342 | 0.000228 | 0.034 | refuted if at most +0.02 in every bin |
| walls+gaps [0.5,1) | 0.0282 | 0.00579 | 0.0224 | refuted if at most +0.02 in every bin |
| walls+gaps [1,2) | 0.0291 | 0.00447 | 0.0246 | refuted if at most +0.02 in every bin |
| walls+gaps [2,4) | 0.0189 | 0.00861 | 0.0103 | refuted if at most +0.02 in every bin |

Largest per-bin walls+gaps gain of hits_8: 0.0376.

Absolute walls+gaps recall at 0.002 in the L/h < 1 bins ([0,0.125), [0.125,0.25), [0.25,0.5), [0.5,1)), iteration 120000: hits_8 (thr=0.6): 0.000348, 0.0376, 0.0342, 0.0282; occfrac_128 (occ_snap=0.45, out of band): 0.00854, 0.0194, 0.081, 0.0763; points_strong (iso=0.5): 0.00174, 0.00141, 0.00274, 0.00309; occfrac_128 (grid) (binarize): 0, 0, 0.000228, 0.00579.

The rule is applied to the ray layer only; IoU and the mesh layer are not evaluated in this report. The SPEC condition "with a smaller output" and the cost criterion are not checked here, and the SPEC compares against TSDF M = 128, which K1-1 did not train: the dense ray runs in band stand in for it.

## Per checkpoint

| run | iteration | selected setting | F1 (all hits, 0.002) | precision | decoded/GT | in band | ord n | ord recall 0.002 | ord recall h/2 | ord median | ord p90 | M (H5-6, walls+gaps L/h < 1) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hits_8 | 40000 | thr=0.6 | 0.124 | 0.129 | 0.925 | yes | 7485602 | 0.134 | 0.275 | 0.00931 | 0.0445 | 0.0192 |
| hits_8 | 80000 | thr=0.6 | 0.156 | 0.159 | 0.961 | yes | 7485602 | 0.172 | 0.342 | 0.00718 | 0.034 | 0.0313 |
| hits_8 | 120000 | thr=0.6 | 0.176 | 0.182 | 0.942 | yes | 7485602 | 0.198 | 0.384 | 0.00622 | 0.0305 | 0.0306 |
| occfrac_128 | 40000 | occ_snap=0.45 | 0.1 | 0.0628 | 3.91 | no | 7485602 | 0.254 | 0.505 | 0.00423 | 0.0211 | 0.0826 |
| occfrac_128 | 80000 | occ_snap=0.45 | 0.121 | 0.0789 | 3.25 | no | 7485602 | 0.268 | 0.524 | 0.00404 | 0.019 | 0.0805 |
| occfrac_128 | 120000 | occ_snap=0.45 | 0.128 | 0.0845 | 3.1 | no | 7485602 | 0.277 | 0.537 | 0.0039 | 0.0181 | 0.0769 |
| points_strong | 40000 | iso=0.5 | 0.178 | 0.191 | 0.873 | yes | 7485602 | 0.223 | 0.403 | 0.00615 | 0.0466 | 0.0019 |
| points_strong | 80000 | iso=0.5 | 0.219 | 0.24 | 0.846 | yes | 7485602 | 0.272 | 0.469 | 0.0048 | 0.0397 | 0.00208 |
| points_strong | 120000 | iso=0.5 | 0.232 | 0.248 | 0.877 | yes | 7485602 | 0.291 | 0.492 | 0.00443 | 0.0352 | 0.00291 |
| occfrac_128 (grid) | 40000 | binarize | 0.154 | 0.164 | 0.884 | yes | 7485602 | 0.192 | 0.362 | 0.00702 | 0.0488 | 0.00206 |
| occfrac_128 (grid) | 80000 | binarize | 0.195 | 0.206 | 0.9 | yes | 7485602 | 0.244 | 0.437 | 0.00535 | 0.0373 | 0.00284 |
| occfrac_128 (grid) | 120000 | binarize | 0.21 | 0.22 | 0.908 | yes | 7485602 | 0.264 | 0.461 | 0.00494 | 0.0352 | 0.00338 |

## Thin bins, hits_8: thin recall at 0.002 (n) per L/h bin

| L/h bin | it 40000 | it 80000 | it 120000 |
|---|---|---|---|
| [0,0.125) | 0.0291 (26933) | 0.0369 (26933) | 0.0362 (26933) |
| [0.125,0.25) | 0.0684 (33718) | 0.0877 (33718) | 0.0854 (33718) |
| [0.25,0.5) | 0.122 (495361) | 0.154 (495361) | 0.151 (495361) |
| [0.5,1) | 0.0941 (861490) | 0.12 (861490) | 0.122 (861490) |
| [1,2) | 0.0844 (1058120) | 0.106 (1058120) | 0.11 (1058120) |
| [2,4) | 0.0758 (1034922) | 0.0893 (1034922) | 0.0978 (1034922) |

## Thin bins, points_strong: thin recall at 0.002 (n) per L/h bin

| L/h bin | it 40000 | it 80000 | it 120000 |
|---|---|---|---|
| [0,0.125) | 0.0217 (26933) | 0.0251 (26933) | 0.0288 (26933) |
| [0.125,0.25) | 0.0491 (33718) | 0.054 (33718) | 0.0606 (33718) |
| [0.25,0.5) | 0.0526 (495361) | 0.0576 (495361) | 0.0664 (495361) |
| [0.5,1) | 0.0454 (861490) | 0.0512 (861490) | 0.0582 (861490) |
| [1,2) | 0.0405 (1058120) | 0.0452 (1058120) | 0.0514 (1058120) |
| [2,4) | 0.056 (1034922) | 0.0664 (1034922) | 0.0738 (1034922) |

## Thin bins, occfrac_128, grid-selected setting (binarize): thin recall at 0.002 (n) per L/h bin

| L/h bin | it 40000 | it 80000 | it 120000 |
|---|---|---|---|
| [0,0.125) | 0.0196 (26933) | 0.0245 (26933) | 0.025 (26933) |
| [0.125,0.25) | 0.0451 (33718) | 0.0527 (33718) | 0.0569 (33718) |
| [0.25,0.5) | 0.0537 (495361) | 0.063 (495361) | 0.0695 (495361) |
| [0.5,1) | 0.043 (861490) | 0.0542 (861490) | 0.0566 (861490) |
| [1,2) | 0.0406 (1058120) | 0.0532 (1058120) | 0.056 (1058120) |
| [2,4) | 0.0523 (1034922) | 0.0678 (1034922) | 0.0732 (1034922) |

## Thin bins, occfrac_128, occ_snap=0.45: out of band, not comparable: thin recall at 0.002 (n) per L/h bin

| L/h bin | it 40000 | it 80000 | it 120000 |
|---|---|---|---|
| [0,0.125) | 0.105 (26933) | 0.107 (26933) | 0.104 (26933) |
| [0.125,0.25) | 0.14 (33718) | 0.143 (33718) | 0.137 (33718) |
| [0.25,0.5) | 0.213 (495361) | 0.212 (495361) | 0.211 (495361) |
| [0.5,1) | 0.204 (861490) | 0.214 (861490) | 0.21 (861490) |
| [1,2) | 0.24 (1058120) | 0.246 (1058120) | 0.242 (1058120) |
| [2,4) | 0.25 (1034922) | 0.251 (1034922) | 0.252 (1034922) |

## Thin bins, hits_8: thin recall at h/2 (descriptive, not the criterion) (n) per L/h bin

| L/h bin | it 40000 | it 80000 | it 120000 |
|---|---|---|---|
| [0,0.125) | 0.0582 (26933) | 0.0718 (26933) | 0.0708 (26933) |
| [0.125,0.25) | 0.108 (33718) | 0.145 (33718) | 0.136 (33718) |
| [0.25,0.5) | 0.219 (495361) | 0.276 (495361) | 0.269 (495361) |
| [0.5,1) | 0.196 (861490) | 0.248 (861490) | 0.251 (861490) |
| [1,2) | 0.178 (1058120) | 0.221 (1058120) | 0.227 (1058120) |
| [2,4) | 0.159 (1034922) | 0.188 (1034922) | 0.203 (1034922) |

## Thin bins, points_strong: thin recall at h/2 (descriptive, not the criterion) (n) per L/h bin

| L/h bin | it 40000 | it 80000 | it 120000 |
|---|---|---|---|
| [0,0.125) | 0.0401 (26933) | 0.0437 (26933) | 0.0546 (26933) |
| [0.125,0.25) | 0.0716 (33718) | 0.0783 (33718) | 0.0904 (33718) |
| [0.25,0.5) | 0.0917 (495361) | 0.0999 (495361) | 0.114 (495361) |
| [0.5,1) | 0.0934 (861490) | 0.103 (861490) | 0.116 (861490) |
| [1,2) | 0.0841 (1058120) | 0.0928 (1058120) | 0.106 (1058120) |
| [2,4) | 0.114 (1034922) | 0.134 (1034922) | 0.147 (1034922) |

## Thin bins, occfrac_128, grid-selected setting (binarize): thin recall at h/2 (descriptive, not the criterion) (n) per L/h bin

| L/h bin | it 40000 | it 80000 | it 120000 |
|---|---|---|---|
| [0,0.125) | 0.0375 (26933) | 0.0472 (26933) | 0.0476 (26933) |
| [0.125,0.25) | 0.0679 (33718) | 0.0792 (33718) | 0.0816 (33718) |
| [0.25,0.5) | 0.0928 (495361) | 0.11 (495361) | 0.119 (495361) |
| [0.5,1) | 0.0919 (861490) | 0.114 (861490) | 0.12 (861490) |
| [1,2) | 0.0865 (1058120) | 0.112 (1058120) | 0.119 (1058120) |
| [2,4) | 0.109 (1034922) | 0.14 (1034922) | 0.151 (1034922) |

## Thin bins, occfrac_128, occ_snap=0.45: out of band, not comparable: thin recall at h/2 (descriptive, not the criterion) (n) per L/h bin

| L/h bin | it 40000 | it 80000 | it 120000 |
|---|---|---|---|
| [0,0.125) | 0.213 (26933) | 0.212 (26933) | 0.204 (26933) |
| [0.125,0.25) | 0.244 (33718) | 0.246 (33718) | 0.232 (33718) |
| [0.25,0.5) | 0.39 (495361) | 0.387 (495361) | 0.383 (495361) |
| [0.5,1) | 0.425 (861490) | 0.441 (861490) | 0.433 (861490) |
| [1,2) | 0.499 (1058120) | 0.508 (1058120) | 0.5 (1058120) |
| [2,4) | 0.518 (1034922) | 0.517 (1034922) | 0.518 (1034922) |
