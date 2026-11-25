# H5 pilot report

**Verdict: REDESIGN ENCODER.** point check 0.001 < 0.3.

- Point check (points_strong at it 20000, recall of walls and gaps, L/h in [0.25, 1)): **0.001** (REDESIGN ENCODER below 0.3).
- P1 gap (train subset, all rays, final it 10000): M(hits_8) 0.018 - M(occfrac_128) 0.190 = **-0.172** (STOP needs <= 0.02).
- P2 gap (validation, all rays): it 5000: **-0.212**, it 10000: **-0.199**, it 15000: **-0.192**, it 20000: **-0.181** (CONTINUE needs >= 0.05 at the last iteration, and >= the previous one).
- Speed condition only: **PASS** (points N=512 / hits_8 = 5.38; points N=512 / hits_8_plane = 25.53; PASS needs >= 10 for hits_8 or hits_8_plane, trained timings)

## Selected decoding settings

| run | iteration | setting |
|---|---|---|
| hits_8 | 5000 | thr=0.3 |
| hits_8 | 10000 | thr=0.4 |
| hits_8 | 15000 | thr=0.5 |
| hits_8 | 20000 | thr=0.4 |
| hits_8_plane | 5000 | thr=0.5 |
| hits_8_plane | 10000 | thr=0.6 |
| hits_8_plane | 15000 | thr=0.7 |
| hits_8_plane | 20000 | thr=0.5 |
| occfrac_128 | 5000 | occ_snap=0.1 |
| occfrac_128 | 10000 | occ_snap=0.1 |
| occfrac_128 | 15000 | occ_snap=0.1 |
| occfrac_128 | 20000 | occ_snap=0.1 |
| points_strong | 5000 | iso=0.5 |
| points_strong | 10000 | iso=0.5 |
| points_strong | 15000 | iso=0.5 |
| points_strong | 20000 | iso=0.5 |
## Cost indication (not a decision)

Indication: the full criterion needs the point recall at N < 512. Trained timings at 20k; recall of walls and gaps (delta 0.002, validation, all rays) pooled over L/h >= 0.25, against points_strong N = 512 at the same iteration.

| method | speed ratio points N=512 / method | recall L/h >= 0.25 | points N=512 recall L/h >= 0.25 | difference |
|---|---|---|---|---|
| hits_8 | 5.38 | 0.0117 | 0.0014 | 0.0103 |
| hits_8_plane | 25.5 | 0.00316 | 0.0014 | 0.00176 |


## Metric M (walls + gaps recall, delta 0.002, L/h < 1) and F1 of all hits

| run | iteration | M | F1 |
|---|---|---|---|
| hits_8 | 5000 | 0.004039 | 0.04705 |
| hits_8 | 10000 | 0.007196 | 0.07234 |
| hits_8 | 15000 | 0.01124 | 0.08119 |
| hits_8 | 20000 | 0.0135 | 0.09329 |
| hits_8_plane | 5000 | 0.001066 | 0.03389 |
| hits_8_plane | 10000 | 0.001498 | 0.04166 |
| hits_8_plane | 15000 | 0.001201 | 0.04304 |
| hits_8_plane | 20000 | 0.002537 | 0.05599 |
| occfrac_128 | 5000 | 0.2164 | 0.02755 |
| occfrac_128 | 10000 | 0.2059 | 0.03374 |
| occfrac_128 | 15000 | 0.2029 | 0.03753 |
| occfrac_128 | 20000 | 0.1947 | 0.04084 |
| points_strong | 5000 | 0.000978 | 0.06202 |
| points_strong | 10000 | 0.0009767 | 0.0836 |
| points_strong | 15000 | 0.001202 | 0.1073 |
| points_strong | 20000 | 0.001128 | 0.1313 |

Ceilings on the same shapes and rays (occ_snap 0.1): hits_8 M 0.8718, occfrac_128 M 0.3878.

## Recall of walls and gaps per L/h bin (delta 0.002, validation, all rays)

### hits_8 at it 20000

| [0,0.125) | [0.125,0.25) | [0.25,0.5) | [0.5,1) | [1,2) | [2,4) |
|---|---|---|---|---|---|
| 0.000348 | 0.00103 | 0.0138 | 0.0138 | 0.0116 | 0.00774 |

### hits_8_plane at it 20000

| [0,0.125) | [0.125,0.25) | [0.25,0.5) | [0.5,1) | [1,2) | [2,4) |
|---|---|---|---|---|---|
| 0.000174 | 0.000564 | 0.00191 | 0.00307 | 0.00406 | 0.00308 |

### occfrac_128 at it 20000

| [0,0.125) | [0.125,0.25) | [0.25,0.5) | [0.5,1) | [1,2) | [2,4) |
|---|---|---|---|---|---|
| 0.111 | 0.147 | 0.223 | 0.177 | 0.194 | 0.166 |

### points_strong at it 20000

| [0,0.125) | [0.125,0.25) | [0.25,0.5) | [0.5,1) | [1,2) | [2,4) |
|---|---|---|---|---|---|
| 0.000348 | 0.000845 | 0.00095 | 0.00127 | 0.000966 | 0.00247 |

## Thin-feature recall per bin (walls, gaps, tangent chords) at the last iteration

### hits_8

| class | [0,0.125) | [0.125,0.25) | [0.25,0.5) | [0.5,1) | [1,2) | [2,4) |
|---|---|---|---|---|---|---|
| wall | 0 | 0.000952 | 0.00952 | 0.0102 | 0.0106 | 0.0104 |
| gap | 0.0012 | 0.00115 | 0.0178 | 0.0179 | 0.0127 | 0.00365 |
| tangent chord | 0.000259 | 0.000522 | 0.000647 | 0.00298 | 0.00296 | 0.0029 |

### hits_8_plane

| class | [0,0.125) | [0.125,0.25) | [0.25,0.5) | [0.5,1) | [1,2) | [2,4) |
|---|---|---|---|---|---|---|
| wall | 0.000246 | 0.000476 | 0.00301 | 0.00332 | 0.00467 | 0.00395 |
| gap | 0 | 0.00069 | 0.000881 | 0.00278 | 0.00333 | 0.00175 |
| tangent chord | 0.000259 | 0.00104 | 0.000941 | 0.00169 | 0.00284 | 0.00358 |

### occfrac_128

| class | [0,0.125) | [0.125,0.25) | [0.25,0.5) | [0.5,1) | [1,2) | [2,4) |
|---|---|---|---|---|---|---|
| wall | 0.131 | 0.175 | 0.215 | 0.177 | 0.194 | 0.169 |
| gap | 0.0606 | 0.107 | 0.231 | 0.176 | 0.195 | 0.16 |
| tangent chord | 0.12 | 0.206 | 0.289 | 0.178 | 0.187 | 0.186 |

### points_strong

| class | [0,0.125) | [0.125,0.25) | [0.25,0.5) | [0.5,1) | [1,2) | [2,4) |
|---|---|---|---|---|---|---|
| wall | 0.000491 | 0.00143 | 0.00125 | 0.00171 | 0.00132 | 0.00376 |
| gap | 0 | 0 | 0.000674 | 0.000762 | 0.000534 | 0.000495 |
| tangent chord | 0.000129 | 0.000913 | 0.000823 | 0.000765 | 0.000917 | 0.00148 |

## P2 gap per iteration

| iteration | M(hits_8) | M(occfrac_128) | gap |
|---|---|---|---|
| 5000 | 0.004039 | 0.2164 | -0.2124 |
| 10000 | 0.007196 | 0.2059 | -0.1987 |
| 15000 | 0.01124 | 0.2029 | -0.1917 |
| 20000 | 0.0135 | 0.1947 | -0.1812 |

## Verdict rule

1. REDESIGN ENCODER if point check < 0.3; 2. STOP if P1 gap <= 0.02 and P2 gap (last) <= 0.02; 3. CONTINUE if P2 gap (last) >= 0.05 and >= P2 gap (previous); 4. UNCLEAR otherwise.
