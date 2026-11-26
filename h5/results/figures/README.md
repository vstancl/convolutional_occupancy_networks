# H5-G1 figures for the supervisor meeting

Made on the CPU from files the runs and reports already wrote; nothing was trained. The only new evaluation is the
H5-G1b re-evaluation of the K1 checkpoints at 120000 with distance histograms (for G3; `h5/scripts/eval_hist.py`,
at the setting the combined report selected, F1 and M reproduce the report exactly). Every figure is
a PNG (dpi 200) plus an SVG of the same name. Each decoder has one colour in every figure (`h5/scripts/fig_style.py`);
grey marks an out-of-band row.

Regenerate:

```
.venv/bin/python h5/scripts/plot_runs.py --root ~/PhD/runs/k1_1 --root ~/PhD/runs/k1_4 --group k1 --report h5/results/k1/report.json --title "K1: first round, full train split, 120000 iterations"
.venv/bin/python h5/scripts/plot_runs.py --root ~/PhD/runs/h5_q0 --group q0 --report h5/results/q0/report.json --title "Q0: memorisation of 32 training shapes"
.venv/bin/python h5/scripts/plot_runs.py --root ~/PhD/runs/h5_pilot --group pilot --title "Pilot: P1 (subset, dashed) and P2 runs"
.venv/bin/python h5/scripts/plot_results.py --report h5/results/k1/report.json --group k1 --title "K1: first round, full train split"
.venv/bin/python h5/scripts/plot_results.py --report h5/results/q0/report.json --group q0 --runs-root ~/PhD/runs/h5_q0 --title "Q0: memorisation of 32 training shapes" --header "Memorisation of 32 training shapes: 1 run per decoder, the 32 training shapes, all rays"
.venv/bin/python h5/scripts/eval_hist.py            # GPU, once: writes ~/PhD/runs/k1_*/eval/*_it120000_hist.json
.venv/bin/python h5/scripts/plot_cdf.py --report h5/results/k1/report.json --group k1
```

The training losses of different targets are different quantities (different losses and scales), so they are not
compared in one panel: `curves_all` shows only the validation F1, and each loss is in its own panel of
`curves_per_run`.

## k1: first round, full train split (runs k1_1 and k1_4, report h5/results/k1/report.json)

- `k1/curves_all`: validation F1 at 0.002 of all seven K1 runs, with the learning-rate drop at 100000; hollow markers at 120000 where the final evaluation used another decoding (occfrac_128 binarize, hits_8_plane thr=0.7).
- `k1/curves_per_run`: training loss (EMA solid, raw faint, log y) and validation F1 per run, one panel each, with the same markers.
- `k1/g2_recall_by_thickness_d0002`: walls+gaps recall (both ends matched) at 0.002 per L/h bin, (a) the representation ceilings of gate B without a network, including occfrac_128 binarize, (b) the in-band networks at iteration 120000 (grid rows); the out-of-band occfrac_128 occ_snap=0.45 row is only named in the caption.
- `k1/g2_recall_by_thickness_half_h`: the same at delta h/2 (descriptive).
- `k1/g2c_ceiling_vs_network_d0002`: one log panel (1e-3 to 1): per representation its ceiling dashed and its network solid in one colour (hits_8 and hits_8_plane share the hits ceiling, points_strong and points_std share points_512, occfrac_128 uses the binarize ceiling); values below 1e-3 at 1e-3 with an open marker.
- `k1/g2c_ceiling_vs_network_half_h`: the same at delta h/2.
- `k1/g3_error_cdf`: share of GT hits with distance to the nearest decoded same-type hit <= x, (a) ordinary hits, (b) thin wall/gap hits with L/h < 1, all seven runs at 120000, log x, lines at 0.002 and h/2.
- `k1/g4_table.md`, `k1/g4_table.csv`: one row per network at 120000 plus the ceiling rows: setting, F1, precision, decoded/GT, ordinary recall and median, M, output values per shape, training s/it.

## q0: memorisation of 32 training shapes (runs h5_q0, report h5/results/q0/report.json)

- `q0/curves_all`: F1 on the 32 training shapes of the three Q0 runs, learning-rate drop at 50000 (no hollow markers: every final evaluation used the validation decoding).
- `q0/curves_per_run`: training loss and F1 per run.
- `q0/g2_recall_by_thickness_d0002`: walls+gaps recall per L/h bin at 0.002, (a) the gate-B ceilings, (b) the in-band Q0 runs at 60000 on the 32 training shapes (occfrac_128 occ_snap=0.4, decoded/GT 1.51, out of band, named in the caption).
- `q0/g2_recall_by_thickness_half_h`: the same at h/2 (descriptive).
- `q0/g2c_ceiling_vs_network_d0002`, `q0/g2c_ceiling_vs_network_half_h`: ceiling against network per representation (log), for the in-band Q0 runs.
- `q0/g4_table.md`, `q0/g4_table.csv`: the G4 table of the Q0 runs at 60000 plus the ceiling rows.

## pilot (runs h5_pilot)

- `pilot/curves_all`: training loss and validation F1 of the P1 (dashed) and P2 pilot runs (no learning-rate drop).
- `pilot/curves_per_run`: the same per run.

## Notes

- The ceilings are gate B (csgbox `docs/results/gate_v2.md`, JSON `~/PhD/data/csgbox/gate_v1/gate_v2.json`): 200 gate shapes, not the val or training shapes of the networks, so the n per bin differs between panels (a) and (b). Walls+gaps per bin is derived from the stored per-class recalled and total counts, with the same definition as the network evaluation.
- The occfrac_128 binarize ceiling (`h5/results/k1_1/occfrac_ceiling.json`, same 200 gate shapes) has walls+gaps fields since H5-G1b (`h5/scripts/occ_ceiling_binarize.py --add-walls-gaps`: the stored fields were reproduced exactly and kept; its occfrac_128 row reproduces gate B's walls+gaps).
- Output values per shape: hits = 3 x 128^2 x 12 (8 hit slots and 4 mask logits per ray), dense = 3 x 128^2 x 128, point decoders n/a.
- G3 uses the distance histograms that `h5/eval_rays.py` stores since H5-G1b (`summary.diag.dist_hist`: 200 log-spaced bins from 1e-5 to 1 plus underflow, overflow and inf, for the ordinary hits and for the thin wall/gap hits with L/h < 1). Only `*_it120000_hist.json` of the K1 runs have them.
