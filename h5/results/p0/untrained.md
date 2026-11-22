# P0: decoding time per shape (untrained)

**Untrained, preliminary.** Untrained networks give outputs near 0.5, so the decoded-hit counts and the conversion times are not those of trained decoders.

- Shapes: first 100 of test.lst; warm-up 5 (not timed); 95 timed; median per shape, batch 1, fp32, NVIDIA GeForce RTX 4070.
- Decoder = network over all 3 * 128^2 rays (points: sampling of N points per ray); conversion = CPU decode to RayHits (points: bisection of the sign changes, which calls the network, plus assembly).
- Untrained methods: hits_8, hits_8_plane, occfrac_128, points_strong.

| method | source | chunk | encoder ms | decoder ms | conversion ms | total ms | total, deterministic ms | sign changes | kept slots | decoded hits |
|---|---|---|---|---|---|---|---|---|---|---|
| hits_8 | untrained | 16384 | 5.9 | 115.0 | 19.7 | 140.6 | 152.4 |  | 83545 | 91528 |
| hits_8_plane | untrained | 49152 | 5.7 | 1.8 | 18.9 | 26.4 | 28.5 |  | 0 | 0 |
| occfrac_128 | untrained | 16384 | 6.1 | 117.8 | 1586.5 | 1711.4 | 1715.8 |  |  | 12472158 |
| points_strong N=64 | untrained | 1024 | 5.6 | 90.6 | 0.7 | 96.9 | 102.3 | 0 |  | 0 |
| points_strong N=128 | untrained | 1024 | 5.7 | 187.1 | 0.8 | 193.6 | 199.4 | 0 |  | 0 |
| points_strong N=256 | untrained | 256 | 5.8 | 377.1 | 1.0 | 383.9 | 399.2 | 0 |  | 0 |
| points_strong N=512 | untrained | 256 | 5.8 | 740.5 | 1.3 | 747.7 | 764.2 | 0 |  | 0 |

## Ratio total(points N) / total(method)

| N | hits_8 | hits_8_plane | occfrac_128 |
|---|---|---|---|
| 64 | 0.69 | 3.66 | 0.06 |
| 128 | 1.38 | 7.32 | 0.11 |
| 256 | 2.73 | 14.52 | 0.22 |
| 512 | 5.32 | 28.28 | 0.44 |

Points: zero sign changes at iso 0.5 for the points rows marked untrained: the untrained point decoder gives a near-constant probability, so the protocol's conversion (bisection) costs almost nothing here. A trained decoder has a sign change at every surface crossing, so its conversion time is larger; trained counts are in the pilot's timing.

hits_8_plane: zero kept slots (untrained mask logits are negative), so its conversion is the bare decoder cost.

Chunk candidates (decoder seconds on the warm-up shapes) are in the JSON.
