# H5-K2' sampling shares

Train shapes: 100 (every 100th), 20 draws each (the items of k2_thin_share.py), 2048 query points per item; h = h(128) = 0.00859383. Means over the items.

Surface distance, the same definition for every sampler: per axis, the distance along that axis from the point to the nearest GT hit of the axis ray through the point's cell (dexels_128.npz); the minimum over the three axes. Thin share: the length of the GT segment containing the point (as k2_thin_share.py; the near-surface half of a mixture as points_strong, the ray half along its ray); for the other targets see h5/results/k2/thin_share.md.

| target | surface distance < 0.002 | surface distance < h/2 | segment < h | segment < 4 h | items |
|---|---|---|---|---|---|
| points_strong | 0.2236 | 0.3267 | n/a | n/a | 2000 |
| points_rayint_a | 0.0814 | 0.1239 | n/a | n/a | 2000 |
| points_rayint_b | 0.0135 | 0.0255 | n/a | n/a | 2000 |
| points_rayint_c | 0.2631 | 0.3389 | n/a | n/a | 2000 |
| points_mix_a | 0.2600 | 0.3813 | 0.0389 | 0.1304 | 2000 |
| points_mix_b | 0.2259 | 0.3320 | 0.0223 | 0.0840 | 2000 |
