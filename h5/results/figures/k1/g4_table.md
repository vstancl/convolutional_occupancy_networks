1 run per decoder, 200 val shapes, all rays

| row | decoding setting | F1 0.002 (up) | precision (up) | decoded/GT (target 1) | in band | ord recall 0.002 (up) | ord recall h/2 (up) | ord median (down) | M walls+gaps L/h<1 (up) | output values per shape (down) | training s/it |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hits_8 | thr=0.6 | 0.176 | 0.182 | 0.942 | yes | 0.198 | 0.384 | 0.00622 | 0.0306 | 589824 | 0.209 |
| occfrac_128 | occ_snap=0.45 | 0.128 | 0.0845 | 3.1 | no | 0.277 | 0.537 | 0.0039 | 0.0769 | 6291456 | 0.197 |
| points_strong | iso=0.5 | 0.232 | 0.248 | 0.877 | yes | 0.291 | 0.492 | 0.00443 | 0.00291 | n/a | 0.104 |
| occfrac_128 (grid) | binarize | 0.21 | 0.22 | 0.908 | yes | 0.264 | 0.461 | 0.00494 | 0.00338 | 6291456 | 0.197 |
| binary_128 (grid) | thr=0.5 | 0.166 | 0.163 | 1.05 | yes | 0.215 | 0.425 | 0.00535 | 0.00427 | 6291456 | 0.195 |
| hits_8_plane (grid) | thr=0.7 | 0.116 | 0.12 | 0.939 | yes | 0.132 | 0.274 | 0.00879 | 0.0117 | 589824 | 0.103 |
| points_std (grid) | iso=0.5 | 0.208 | 0.234 | 0.801 | yes | 0.251 | 0.457 | 0.00494 | 0.00188 | n/a | 0.105 |
| tsdf_128 (grid) | default | 0.211 | 0.23 | 0.849 | yes | 0.257 | 0.461 | 0.00491 | 0.00226 | 6291456 | 0.196 |
| ceiling hits_8 (= hits_8_plane) | gate B | 0.987 | 1 | 0.975 | n/a | n/a | n/a | n/a | 0.864 | 589824 | n/a |
| ceiling occfrac_128 | gate B | 0.91 | 0.885 | 1.06 | n/a | n/a | n/a | n/a | 0.405 | 6291456 | n/a |
| ceiling tsdf_128 | gate B | 0.941 | 0.972 | 0.938 | n/a | n/a | n/a | n/a | 0.182 | 6291456 | n/a |
| ceiling binary_128 | gate B | 0.455 | 0.471 | 0.938 | n/a | n/a | n/a | n/a | 0.112 | 6291456 | n/a |
| ceiling points_512 | gate B | 0.999 | 1 | 0.997 | n/a | n/a | n/a | n/a | 0.99 | n/a | n/a |
| ceiling occfrac_128 binarize | binarize | 0.928 | 0.965 | 0.926 | n/a | n/a | n/a | n/a | 0.113 | 6291456 | n/a |

Network rows: last checkpoint, band-selected setting (the (grid) rows select over the whole decoding grid); an out-of-band row is not comparable. Ceiling rows: gate B (csgbox gate_v2, 200 gate shapes, not the evaluation shapes), no network; the binarize ceiling is computed on the same shapes (h5/results/k1_1/occfrac_ceiling.json). Output values per shape: hits = 3 x 128^2 x 12 (4K hit slots + 2K mask logits per ray, K = 2), dense = 3 x 128^2 x 128, point decoders n/a. Training s/it: median of the training log.
