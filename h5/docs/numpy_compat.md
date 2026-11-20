# numpy compatibility of csgbox.ceilings (h5 step H5-0b)

## Question

Do `csgbox.ceilings` and its decoders give the same results under the ConvONet venv
(numpy 1.26.4) and the csgbox venv (numpy 2.5.5)?

## Setup

- Environment A: `~/PhD/convonet/.venv/bin/python`, numpy **1.26.4**, Python 3.12.15.
- Environment B: `uv run --project ~/PhD/csgbox python`, numpy **2.5.5**, Python 3.12.
- csgbox and dexio are put on `sys.path` by `h5/_paths.py:add_external_paths()`. No dependency
  was added or changed; `csgbox/` and `dexelize/` were not modified (`git status` clean after both runs).
- The import path loads no mesh library: `add_external_paths()` raises if `manifold3d` or
  `manifold` is in `sys.modules` (it was not).
- Shapes: 20 test shapes from `data/csgbox/v1/csg/test.lst`, chosen with
  `numpy.random.Generator(PCG64(0))` (`h5/scripts/check_ceilings_numpy.py`).
- Per shape and axis (x, y, z): float64 GT from `tree.json` (`gt_hits`, N = 128). Then:
  - `hits_8` = `decode_hits(gt, 8)`;
  - `tsdf_128` = `tsdf_values` (float32), decoded with `decode_tsdf`;
  - `binary_128` = `binary_occupancy` (bool), decoded with `decode_binary`;
  - `occfrac_128` = `occ_fraction` cast to float32, decoded with `decode_occfrac`;
  - `evaluate_axis` metrics of each decoded representation against the GT, delta 0.002.
- Outputs: `/tmp/h5_np1.npz` (numpy 1.26.4) and `/tmp/h5_np2.npz` (numpy 2.5.5), 3486 arrays each.
- Comparison: `h5/scripts/compare_ceilings_npz.py`. Full output: `/tmp/h5_compare.txt`.

## Comparison

Integer and boolean arrays are compared for exact equality. Float arrays are compared by max abs diff.

| group | arrays | identical | max abs diff |
|---|---|---|---|
| encodings: `hits_8` offsets, t, enter | 60 each | 60/60 | 0 |
| encodings: `tsdf_128` values | 60 | 60/60 | 0 |
| encodings: `binary_128` occupancy | 60 | 60/60 | 0 |
| encodings: `occfrac_128` fraction (float32) | 60 | 60/60 | 0 |
| decoded hits, all four targets (offsets, t, enter) | 60 each, 12 groups | all identical | 0 |
| metrics: `gt_hits`, `dec_hits`, `rays`, `count_err_sum`, `thin_total`, `matched`, `dt_n`, `dt_hist`, `thin_recalled` | 600 per group in total | identical | 0 |
| metrics: `dt_sum` | 240 (all nonzero) | 191/240 | 0 in 191, else ≤ 3.4e-16 relative |

The metric groups (`*/m`, 600 arrays each) differ only in `dt_sum`: 3 (hits_8), 3 (binary_128),
20 (occfrac_128) and 23 (tsdf_128) arrays, 49 in total. (The comparison script labels a group by the
kind of the last array it saw, so the metric groups show `exact` in its table; the differing
values are float sums, see below.)

Every metric except `dt_sum` is bit-identical. `dt_sum` is `m.dt.sum()` in `evaluate_axis`
(`csgbox/csgbox/ceilings.py:398`). Its values differ in 49 of 240 nonzero cases per array set, by at most
1.8e-15 absolute and 3.4e-16 relative (median |dt_sum| about 0.18). The `dt` values themselves
are not stored, so the cause is not checked directly. The most likely cause is a different
summation order of `np.sum` between numpy 1.26 and 2.5 (pairwise and SIMD reductions). This is a
hypothesis; it was not verified.

`dt_n` and the `dt_hist` log histograms, which are used for the p95 |dt|, are identical.

## Timing

Median over shapes of the median over 5 runs of the full-shape encoding of each target (3 axes,
GT already computed), `timing__*` keys:

| target | numpy 1.26.4 | numpy 2.5.5 |
|---|---|---|
| hits_8 (`decode_hits`) | 0.76 ms | 0.64 ms |
| tsdf_128 (`tsdf_values`) | 189.0 ms | 213.9 ms |
| binary_128 (`binary_occupancy`) | 165.1 ms | 193.6 ms |
| occfrac_128 (`occ_fraction`) | 29.6 ms | 21.9 ms |

Single machine, single run of the 20 shapes, no CPU pinning, and no repeat of the whole run, so
the run-to-run noise was not measured. The numbers only say the two environments are of the same
order of magnitude; no performance conclusion should be drawn from them.

## Verdict

**Encodings, decoded hits and matching: OK.** All of them are bit-identical between the two numpy
versions, for all 20 shapes and 3 axes, with 0 difference in every array.

**Metrics: not bit-identical.** One metric, `dt_sum` in `csgbox.ceilings.evaluate_axis`, differs by
float roundoff (≤ 3.4e-16 relative) in some cases. Under the strict rule "metrics must be
identical", this is **not OK**, and the function that differs is `evaluate_axis` (its `dt_sum`
reduction). Under a tolerance of, say, 1e-12 relative, the metrics are OK.

Options (not chosen here, since they touch csgbox or the rule):
1. Accept a relative tolerance of 1e-12 for `dt_sum` in the h5 comparisons (no csgbox change).
2. Change the sum in `evaluate_axis` to an order-independent one (e.g. `math.fsum`). That is a
   change to csgbox, which needs approval.

The h5 code can use either option without further changes to the data path.
