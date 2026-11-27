# H5-K2' report: mixture query points (half near-surface, half ray-interval) (memorisation, 32 training shapes)

Evaluation: all rays of the training shapes, the band selection over the iso grid (decoded/GT in [0.8, 1.25], best F1 at 0.002; otherwise the closest ratio). Weighted thin recall L/h < 1: per-hit thin recall at 0.002, population-weighted over the four L/h < 1 bins (the Q0 measure). M: walls and gaps with both ends matched, pooled over L/h < 1.

Same training shapes in all 6 runs: yes.

## Rule (fixed on 27. 11. 2026, at iteration 60000)

**CONTINUE**: meets the thresholds: points_mix_a (evaluated: points_mix_a; weighted thin recall >= 0.297, ordinary recall >= 0.702, decoded/GT in band).
Control points_mix_b (descriptive): M(points_mix_a) - M(points_mix_b) = +0.2949; weighted thin recall L/h < 1, a - b = +0.2689; points_mix_b does not meet the thresholds.

## The same rule against the re-evaluated Q0_points_strong (descriptive only; the fixed numbers decide)

**CONTINUE**: meets the thresholds: points_mix_a (evaluated: points_mix_a; weighted thin recall >= 0.2973, ordinary recall >= 0.7024, decoded/GT in band).
Control points_mix_b (descriptive): M(points_mix_a) - M(points_mix_b) = +0.2949; weighted thin recall L/h < 1, a - b = +0.2689; points_mix_b does not meet the thresholds.

## Iteration 60000

| run | setting | ordinary recall 0.002 | weighted thin recall L/h < 1 | M | precision | decoded/GT | in band | F1 |
|---|---|---|---|---|---|---|---|---|
| K2p_points_mix_a | iso=0.5 | 0.707 | 0.464 | 0.423 | 0.598 | 1.06 | yes | 0.616 |
| K2p_points_mix_b | iso=0.5 | 0.717 | 0.195 | 0.128 | 0.605 | 0.987 | yes | 0.601 |
| K2_points_rayint_a | iso=0.5 | 0.593 | 0.429 | 0.353 | 0.528 | 1.04 | yes | 0.537 |
| K2_points_rayint_b | iso=0.5 | 0.655 | 0.0688 | 0.0108 | 0.597 | 0.867 | yes | 0.554 |
| K2_points_rayint_c | iso=0.5 | 0.483 | 0.348 | 0.23 | 0.408 | 1.09 | yes | 0.426 |
| Q0_points_strong | iso=0.5 | 0.722 | 0.247 | 0.184 | 0.6 | 1.02 | yes | 0.606 |

## Thin recall per L/h bin at 0.002, iteration 60000 (n)

| L/h bin | K2p_points_mix_a | K2p_points_mix_b | K2_points_rayint_a | K2_points_rayint_b | K2_points_rayint_c | Q0_points_strong |
|---|---|---|---|---|---|---|
| [0,0.125) | 0.086 (3928) | 0.0509 (3928) | 0.127 (3928) | 0.0262 (3928) | 0.0784 (3928) | 0.0591 (3928) |
| [0.125,0.25) | 0.106 (4048) | 0.0699 (4048) | 0.164 (4048) | 0.0371 (4048) | 0.0791 (4048) | 0.084 (4048) |
| [0.25,0.5) | 0.533 (100649) | 0.221 (100649) | 0.524 (100649) | 0.0799 (100649) | 0.446 (100649) | 0.279 (100649) |
| [0.5,1) | 0.414 (81472) | 0.176 (81472) | 0.339 (81472) | 0.0588 (81472) | 0.254 (81472) | 0.225 (81472) |
| [1,2) | 0.496 (126373) | 0.391 (126373) | 0.464 (126373) | 0.218 (126373) | 0.387 (126373) | 0.431 (126373) |
| [2,4) | 0.388 (145027) | 0.334 (145027) | 0.39 (145027) | 0.253 (145027) | 0.33 (145027) | 0.352 (145027) |

## Thin recall per L/h bin at h/2 (descriptive), iteration 60000 (n)

| L/h bin | K2p_points_mix_a | K2p_points_mix_b | K2_points_rayint_a | K2_points_rayint_b | K2_points_rayint_c | Q0_points_strong |
|---|---|---|---|---|---|---|
| [0,0.125) | 0.151 (3928) | 0.0947 (3928) | 0.277 (3928) | 0.0522 (3928) | 0.209 (3928) | 0.106 (3928) |
| [0.125,0.25) | 0.182 (4048) | 0.119 (4048) | 0.321 (4048) | 0.0615 (4048) | 0.219 (4048) | 0.14 (4048) |
| [0.25,0.5) | 0.644 (100649) | 0.298 (100649) | 0.739 (100649) | 0.151 (100649) | 0.705 (100649) | 0.353 (100649) |
| [0.5,1) | 0.573 (81472) | 0.275 (81472) | 0.591 (81472) | 0.127 (81472) | 0.479 (81472) | 0.339 (81472) |
| [1,2) | 0.647 (126373) | 0.531 (126373) | 0.675 (126373) | 0.369 (126373) | 0.601 (126373) | 0.563 (126373) |
| [2,4) | 0.578 (145027) | 0.511 (145027) | 0.621 (145027) | 0.441 (145027) | 0.553 (145027) | 0.531 (145027) |

## K2' runs per checkpoint

| run | iteration | setting | ordinary recall 0.002 | weighted thin recall L/h < 1 | M | precision | decoded/GT | in band | F1 |
|---|---|---|---|---|---|---|---|---|---|
| K2p_points_mix_a | 5000 | iso=0.5 | 0.218 | 0.0501 | 0.00201 | 0.151 | 1.13 | yes | 0.16 |
| K2p_points_mix_a | 10000 | iso=0.5 | 0.356 | 0.0782 | 0.00665 | 0.27 | 1.04 | yes | 0.276 |
| K2p_points_mix_a | 15000 | iso=0.5 | 0.43 | 0.104 | 0.0158 | 0.326 | 1.06 | yes | 0.335 |
| K2p_points_mix_a | 20000 | iso=0.5 | 0.485 | 0.149 | 0.0458 | 0.379 | 1.05 | yes | 0.388 |
| K2p_points_mix_a | 25000 | iso=0.5 | 0.518 | 0.202 | 0.104 | 0.405 | 1.07 | yes | 0.419 |
| K2p_points_mix_a | 30000 | iso=0.5 | 0.561 | 0.254 | 0.161 | 0.445 | 1.07 | yes | 0.461 |
| K2p_points_mix_a | 35000 | iso=0.5 | 0.582 | 0.301 | 0.217 | 0.474 | 1.06 | yes | 0.489 |
| K2p_points_mix_a | 40000 | iso=0.5 | 0.552 | 0.308 | 0.22 | 0.457 | 1.06 | yes | 0.471 |
| K2p_points_mix_a | 45000 | iso=0.5 | 0.569 | 0.349 | 0.266 | 0.469 | 1.07 | yes | 0.484 |
| K2p_points_mix_a | 50000 | iso=0.5 | 0.656 | 0.417 | 0.367 | 0.56 | 1.05 | yes | 0.573 |
| K2p_points_mix_a | 55000 | iso=0.5 | 0.702 | 0.45 | 0.404 | 0.588 | 1.07 | yes | 0.607 |
| K2p_points_mix_a | 60000 | iso=0.5 | 0.707 | 0.464 | 0.423 | 0.598 | 1.06 | yes | 0.616 |
| K2p_points_mix_b | 5000 | iso=0.5 | 0.195 | 0.0433 | 0.000745 | 0.159 | 0.948 | yes | 0.155 |
| K2p_points_mix_b | 10000 | iso=0.5 | 0.326 | 0.0508 | 0.000757 | 0.279 | 0.888 | yes | 0.263 |
| K2p_points_mix_b | 15000 | iso=0.5 | 0.417 | 0.0545 | 0.00174 | 0.359 | 0.886 | yes | 0.338 |
| K2p_points_mix_b | 20000 | iso=0.5 | 0.478 | 0.0742 | 0.00823 | 0.389 | 0.96 | yes | 0.381 |
| K2p_points_mix_b | 25000 | iso=0.5 | 0.525 | 0.0886 | 0.0152 | 0.428 | 0.971 | yes | 0.422 |
| K2p_points_mix_b | 30000 | iso=0.5 | 0.506 | 0.0959 | 0.0202 | 0.419 | 0.968 | yes | 0.412 |
| K2p_points_mix_b | 35000 | iso=0.5 | 0.601 | 0.109 | 0.0348 | 0.502 | 0.965 | yes | 0.493 |
| K2p_points_mix_b | 40000 | iso=0.5 | 0.553 | 0.125 | 0.0447 | 0.451 | 0.998 | yes | 0.451 |
| K2p_points_mix_b | 45000 | iso=0.5 | 0.638 | 0.137 | 0.0654 | 0.542 | 0.958 | yes | 0.53 |
| K2p_points_mix_b | 50000 | iso=0.5 | 0.656 | 0.174 | 0.094 | 0.543 | 0.997 | yes | 0.543 |
| K2p_points_mix_b | 55000 | iso=0.5 | 0.713 | 0.185 | 0.116 | 0.603 | 0.981 | yes | 0.597 |
| K2p_points_mix_b | 60000 | iso=0.5 | 0.717 | 0.195 | 0.128 | 0.605 | 0.987 | yes | 0.601 |
