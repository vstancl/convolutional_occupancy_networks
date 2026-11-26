# H5-G1 figures for the supervisor meeting

Made on the CPU from files the runs and reports already wrote; nothing was trained or re-evaluated. Every figure is
a PNG (dpi 200) plus an SVG of the same name. Each decoder has one colour in every figure (`h5/scripts/fig_style.py`);
grey marks an out-of-band row.

Regenerate:

```
.venv/bin/python h5/scripts/plot_runs.py --root ~/PhD/runs/k1_1 --root ~/PhD/runs/k1_4 --group k1 --title "K1: first round, full train split, 120000 iterations"
.venv/bin/python h5/scripts/plot_runs.py --root ~/PhD/runs/h5_q0 --group q0 --title "Q0: memorisation of 32 training shapes"
.venv/bin/python h5/scripts/plot_runs.py --root ~/PhD/runs/h5_pilot --group pilot --title "Pilot: P1 (subset, dashed) and P2 runs"
.venv/bin/python h5/scripts/plot_results.py --report h5/results/k1/report.json --group k1 --title "K1: first round, full train split"
.venv/bin/python h5/scripts/plot_results.py --report h5/results/q0/report.json --group q0 --runs-root ~/PhD/runs/h5_q0 --title "Q0: memorisation of 32 training shapes" --header "Memorisation of 32 training shapes: 1 run per decoder, the 32 training shapes, all rays"
```

## k1: first round, full train split (runs k1_1 and k1_4, report h5/results/k1/report.json)

- `k1/curves_all`: training loss (EMA solid, raw faint, log y) and validation F1 at 0.002 of all seven K1 runs, with the learning-rate drop at 100000.
- `k1/curves_per_run`: the same per run, one panel each (loss left axis, F1 right axis).
- `k1/g2_recall_by_thickness_d0002`: walls+gaps recall (both ends matched) at 0.002 per L/h bin, (a) the representation ceilings of gate B without a network, (b) the networks at iteration 120000 (grid rows; old occfrac_128 occ_snap=0.45 grey, out of band).
- `k1/g2_recall_by_thickness_half_h`: the same at delta h/2 (descriptive).
- `k1/g4_table.md`, `k1/g4_table.csv`: one row per network at 120000 plus the ceiling rows: setting, F1, precision, decoded/GT, ordinary recall and median, M, output values per shape, training s/it.

## q0: memorisation of 32 training shapes (runs h5_q0, report h5/results/q0/report.json)

- `q0/curves_all`: training loss and F1 on the 32 training shapes of the three Q0 runs, learning-rate drop at 50000.
- `q0/curves_per_run`: the same per run.
- `q0/g2_recall_by_thickness_d0002`: walls+gaps recall per L/h bin at 0.002, (a) the gate-B ceilings, (b) the Q0 runs at 60000 on the 32 training shapes (occfrac_128 occ_snap=0.4, decoded/GT 1.51, grey, out of band).
- `q0/g2_recall_by_thickness_half_h`: the same at h/2 (descriptive).
- `q0/g4_table.md`, `q0/g4_table.csv`: the G4 table of the Q0 runs at 60000 plus the ceiling rows.

## pilot (runs h5_pilot)

- `pilot/curves_all`: training loss and validation F1 of the P1 (dashed) and P2 pilot runs (no learning-rate drop).
- `pilot/curves_per_run`: the same per run.

## Notes

- The ceilings are gate B (csgbox `docs/results/gate_v2.md`, JSON `~/PhD/data/csgbox/gate_v1/gate_v2.json`): 200 gate shapes, not the val or training shapes of the networks, so the n per bin differs between panels (a) and (b). Walls+gaps per bin is derived from the stored per-class recalled and total counts, with the same definition as the network evaluation.
- The occfrac_128 binarize ceiling (`h5/results/k1_1/occfrac_ceiling.json`) stores only the recall of all thin hits (walls, gaps and grazing together), not walls+gaps; it is drawn dotted with that label and its M is n/a.
- Output values per shape: hits = 3 x 128^2 x 12 (8 hit slots and 4 mask logits per ray), dense = 3 x 128^2 x 128, point decoders n/a.
- **G3 (error CDF) is missing:** the evaluation files store no distance histograms. `h5/eval_rays.py` builds them but writes only ord_n, ord_median, ord_p90, ord_within and ord_recall (at 0.002 and h/2) and the same per thin bin (thin_n, thin_median, thin_p90, thin_within, thin_recall). A CDF needs the histograms stored, which means re-running the evaluation.
