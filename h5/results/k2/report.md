# H5-K2-2 report: ray-interval query points (memorisation, 32 training shapes)

Evaluation: all rays of the training shapes, the band selection over the iso grid (decoded/GT in [0.8, 1.25], best F1 at 0.002; otherwise the closest ratio). Weighted thin recall L/h < 1: per-hit thin recall at 0.002, population-weighted over the four L/h < 1 bins (the Q0 measure). M: walls and gaps with both ends matched, pooled over L/h < 1.

Same training shapes in all three runs: yes.

## Rule (fixed on 24. 11., at iteration 60000)

**STOP**: meets the thresholds: none of points_rayint_a / points_rayint_c (evaluated: points_rayint_a, points_rayint_c; weighted thin recall >= 0.297, ordinary recall >= 0.702, decoded/GT in band).
points_rayint_b does not meet the same thresholds.

## The same rule against the re-evaluated Q0_points_strong (descriptive only; the fixed numbers decide)

**STOP**: meets the thresholds: none of points_rayint_a / points_rayint_c (evaluated: points_rayint_a, points_rayint_c; weighted thin recall >= 0.2973, ordinary recall >= 0.7024, decoded/GT in band).
points_rayint_b does not meet the same thresholds.

## Iteration 60000

| run | setting | ordinary recall 0.002 | weighted thin recall L/h < 1 | M | precision | decoded/GT | in band | F1 |
|---|---|---|---|---|---|---|---|---|
| K2_points_rayint_a | iso=0.5 | 0.593 | 0.429 | 0.353 | 0.528 | 1.04 | yes | 0.537 |
| K2_points_rayint_b | iso=0.5 | 0.655 | 0.0688 | 0.0108 | 0.597 | 0.867 | yes | 0.554 |
| K2_points_rayint_c | iso=0.5 | 0.483 | 0.348 | 0.23 | 0.408 | 1.09 | yes | 0.426 |
| Q0_points_strong | iso=0.5 | 0.722 | 0.247 | 0.184 | 0.6 | 1.02 | yes | 0.606 |

## Thin recall per L/h bin at 0.002, iteration 60000 (n)

| L/h bin | K2_points_rayint_a | K2_points_rayint_b | K2_points_rayint_c | Q0_points_strong |
|---|---|---|---|---|
| [0,0.125) | 0.127 (3928) | 0.0262 (3928) | 0.0784 (3928) | 0.0591 (3928) |
| [0.125,0.25) | 0.164 (4048) | 0.0371 (4048) | 0.0791 (4048) | 0.084 (4048) |
| [0.25,0.5) | 0.524 (100649) | 0.0799 (100649) | 0.446 (100649) | 0.279 (100649) |
| [0.5,1) | 0.339 (81472) | 0.0588 (81472) | 0.254 (81472) | 0.225 (81472) |
| [1,2) | 0.464 (126373) | 0.218 (126373) | 0.387 (126373) | 0.431 (126373) |
| [2,4) | 0.39 (145027) | 0.253 (145027) | 0.33 (145027) | 0.352 (145027) |

## Thin recall per L/h bin at h/2 (descriptive), iteration 60000 (n)

| L/h bin | K2_points_rayint_a | K2_points_rayint_b | K2_points_rayint_c | Q0_points_strong |
|---|---|---|---|---|
| [0,0.125) | 0.277 (3928) | 0.0522 (3928) | 0.209 (3928) | 0.106 (3928) |
| [0.125,0.25) | 0.321 (4048) | 0.0615 (4048) | 0.219 (4048) | 0.14 (4048) |
| [0.25,0.5) | 0.739 (100649) | 0.151 (100649) | 0.705 (100649) | 0.353 (100649) |
| [0.5,1) | 0.591 (81472) | 0.127 (81472) | 0.479 (81472) | 0.339 (81472) |
| [1,2) | 0.675 (126373) | 0.369 (126373) | 0.601 (126373) | 0.563 (126373) |
| [2,4) | 0.621 (145027) | 0.441 (145027) | 0.553 (145027) | 0.531 (145027) |

## K2 runs per checkpoint

| run | iteration | setting | ordinary recall 0.002 | weighted thin recall L/h < 1 | M | precision | decoded/GT | in band | F1 |
|---|---|---|---|---|---|---|---|---|---|
| K2_points_rayint_a | 5000 | iso=0.5 | 0.137 | 0.0518 | 0.00406 | 0.104 | 1.08 | yes | 0.108 |
| K2_points_rayint_a | 10000 | iso=0.5 | 0.202 | 0.0882 | 0.0135 | 0.156 | 1.1 | yes | 0.164 |
| K2_points_rayint_a | 15000 | iso=0.5 | 0.282 | 0.129 | 0.0224 | 0.224 | 1.08 | yes | 0.233 |
| K2_points_rayint_a | 20000 | iso=0.5 | 0.315 | 0.176 | 0.0661 | 0.263 | 1.06 | yes | 0.27 |
| K2_points_rayint_a | 25000 | iso=0.5 | 0.377 | 0.219 | 0.0996 | 0.318 | 1.05 | yes | 0.326 |
| K2_points_rayint_a | 30000 | iso=0.5 | 0.396 | 0.244 | 0.135 | 0.339 | 1.05 | yes | 0.347 |
| K2_points_rayint_a | 35000 | iso=0.5 | 0.425 | 0.267 | 0.15 | 0.364 | 1.05 | yes | 0.372 |
| K2_points_rayint_a | 40000 | iso=0.5 | 0.398 | 0.314 | 0.202 | 0.356 | 1.04 | yes | 0.363 |
| K2_points_rayint_a | 45000 | iso=0.5 | 0.48 | 0.308 | 0.205 | 0.418 | 1.04 | yes | 0.425 |
| K2_points_rayint_a | 50000 | iso=0.5 | 0.487 | 0.329 | 0.218 | 0.432 | 1.03 | yes | 0.438 |
| K2_points_rayint_a | 55000 | iso=0.5 | 0.572 | 0.422 | 0.348 | 0.509 | 1.04 | yes | 0.518 |
| K2_points_rayint_a | 60000 | iso=0.5 | 0.593 | 0.429 | 0.353 | 0.528 | 1.04 | yes | 0.537 |
| K2_points_rayint_b | 5000 | iso=0.2 | 0.109 | 0.0343 | 0.000313 | 0.11 | 0.779 | no | 0.0968 |
| K2_points_rayint_b | 10000 | iso=0.2 | 0.148 | 0.0431 | 0.000565 | 0.149 | 0.785 | no | 0.131 |
| K2_points_rayint_b | 15000 | iso=0.2 | 0.21 | 0.0522 | 0.000769 | 0.207 | 0.789 | no | 0.183 |
| K2_points_rayint_b | 20000 | iso=0.2 | 0.22 | 0.057 | 0.00121 | 0.215 | 0.8 | no | 0.192 |
| K2_points_rayint_b | 25000 | iso=0.5 | 0.401 | 0.054 | 0.000998 | 0.378 | 0.809 | yes | 0.338 |
| K2_points_rayint_b | 30000 | iso=0.5 | 0.413 | 0.0521 | 0.000793 | 0.393 | 0.803 | yes | 0.35 |
| K2_points_rayint_b | 35000 | iso=0.5 | 0.454 | 0.0565 | 0.00139 | 0.425 | 0.825 | yes | 0.384 |
| K2_points_rayint_b | 40000 | iso=0.5 | 0.472 | 0.061 | 0.0018 | 0.442 | 0.827 | yes | 0.4 |
| K2_points_rayint_b | 45000 | iso=0.5 | 0.497 | 0.0633 | 0.0037 | 0.454 | 0.856 | yes | 0.418 |
| K2_points_rayint_b | 50000 | iso=0.5 | 0.528 | 0.0692 | 0.00465 | 0.487 | 0.856 | yes | 0.449 |
| K2_points_rayint_b | 55000 | iso=0.5 | 0.647 | 0.0712 | 0.0113 | 0.586 | 0.872 | yes | 0.546 |
| K2_points_rayint_b | 60000 | iso=0.5 | 0.655 | 0.0688 | 0.0108 | 0.597 | 0.867 | yes | 0.554 |
| K2_points_rayint_c | 5000 | iso=0.2 | 0.048 | 0.0149 | 0.000457 | 0.035 | 1.12 | yes | 0.037 |
| K2_points_rayint_c | 10000 | iso=0.2 | 0.131 | 0.0325 | 0.00299 | 0.0856 | 1.29 | no | 0.0965 |
| K2_points_rayint_c | 15000 | iso=0.2 | 0.158 | 0.0892 | 0.0259 | 0.112 | 1.28 | no | 0.126 |
| K2_points_rayint_c | 20000 | iso=0.5 | 0.219 | 0.122 | 0.0158 | 0.157 | 1.25 | yes | 0.174 |
| K2_points_rayint_c | 25000 | iso=0.5 | 0.283 | 0.157 | 0.0253 | 0.21 | 1.19 | yes | 0.228 |
| K2_points_rayint_c | 30000 | iso=0.5 | 0.316 | 0.177 | 0.0393 | 0.24 | 1.17 | yes | 0.258 |
| K2_points_rayint_c | 35000 | iso=0.5 | 0.326 | 0.203 | 0.0673 | 0.255 | 1.15 | yes | 0.273 |
| K2_points_rayint_c | 40000 | iso=0.5 | 0.328 | 0.205 | 0.0777 | 0.254 | 1.15 | yes | 0.272 |
| K2_points_rayint_c | 45000 | iso=0.5 | 0.374 | 0.228 | 0.0958 | 0.296 | 1.13 | yes | 0.314 |
| K2_points_rayint_c | 50000 | iso=0.5 | 0.41 | 0.275 | 0.129 | 0.335 | 1.11 | yes | 0.352 |
| K2_points_rayint_c | 55000 | iso=0.5 | 0.476 | 0.327 | 0.205 | 0.397 | 1.1 | yes | 0.415 |
| K2_points_rayint_c | 60000 | iso=0.5 | 0.483 | 0.348 | 0.23 | 0.408 | 1.09 | yes | 0.426 |
