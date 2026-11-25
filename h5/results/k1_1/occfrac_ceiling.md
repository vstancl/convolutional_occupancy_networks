# occfrac_128 ceiling on the gate shapes (200 shapes)

Data: `/home/wish/PhD/data/csgbox/gate_v1`, N = 128. `occfrac_128` is the csgbox `decode_occfrac` (the row must match gate_v2.md); `occfrac_128 binarize` is the K1-1b rule (h5/occ_binarize.py) on the same GT fractions. Thin features: material and gap intervals shorter than 4 h, pooled over the classes per L/h bin.

## delta 0.002

| representation | precision | recall | F1 | |dt| mean | |dt| p95 | count err | recall of L/h < 1 (weighted) | recall [0,0.125) | recall [0.125,0.25) | recall [0.25,0.5) | recall [0.5,1) | recall [1,2) | recall [2,4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfrac_128 (decode_occfrac) | 0.8848 | 0.9373 | 0.9103 | 8.014e-05 | 0.0007586 | 0.07653 | 0.4268 | 0.3973 | 0.408 | 0.3554 | 0.4524 | 0.9737 | 0.973 |
| occfrac_128 binarize | 0.9655 | 0.894 | 0.9284 | 3.645e-05 | 1.023e-09 | 0.08626 | 0.1188 | 0 | 0 | 6.801e-05 | 0.1648 | 0.8506 | 0.9507 |

Thin population per bin (pooled over the classes): [0,0.125) 14431, [0.125,0.25) 15386, [0.25,0.5) 235262, [0.5,1) 683896, [1,2) 657212, [2,4) 757717

## delta h/2

| representation | precision | recall | F1 | |dt| mean | |dt| p95 | count err | recall of L/h < 1 (weighted) | recall [0,0.125) | recall [0.125,0.25) | recall [0.25,0.5) | recall [0.5,1) | recall [1,2) | recall [2,4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| occfrac_128 (decode_occfrac) | 0.9236 | 0.9784 | 0.9502 | 0.0002298 | 0.002089 | 0.07653 | 0.6342 | 0.8209 | 0.7676 | 0.6308 | 0.6285 | 0.9938 | 0.9955 |
| occfrac_128 binarize | 1 | 0.926 | 0.9616 | 0.0001424 | 0.001122 | 0.08626 | 0.3586 | 0.0001386 | 0.000195 | 0.007596 | 0.4949 | 0.9714 | 0.9643 |

Thin population per bin (pooled over the classes): [0,0.125) 14431, [0.125,0.25) 15386, [0.25,0.5) 235262, [0.5,1) 683896, [1,2) 657212, [2,4) 757717

