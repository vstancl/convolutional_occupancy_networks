"""[phd] Report plots and summary table from existing logs and results (CPU only).

Writes PNG + SVG to phd_work/report/ and phd_work/report/summary.md.
Add a run by adding an entry to VARIANTS (training runs) or INPUT_ROWS / PRETRAINED.

Plots:
  1 val IoU vs iteration (one line per training run; resumed runs joined in order)
  2 val IoU vs wall-clock hours (time field of each log, plus the final time of earlier logs)
  3 test mesh IoU per class (chair, table, lamp, mean), one marker per run
  4 test mesh IoU vs L: network (dexel L2, L32), input row (L 1, 2, 4, 32), voxel baseline band
  5 histogram of hits per ray on the 3-class test split (recomputed with dexel_hits_hist)
  6 per-object IoU(L32) vs IoU(L2) on the test split, colour = rays with > 2 hits (3 axes)

Usage:
    uv run python phd_work/report_plots.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PHD = os.path.join(REPO, 'phd_work')
REPORT = os.path.join(PHD, 'report')
sys.path.insert(0, PHD)
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from speed_stats import stats as speed_stats  # noqa: E402
import dexel_hits_hist as dhh  # noqa: E402

CLASSES = ['chair', 'table', 'lamp']
CLASS_IDS = {'chair': '03001627', 'table': '04379243', 'lamp': '03636649'}

# ---- variant dict -------------------------------------------------------------
# logs: train.py logs in the order they were run (resumed runs: earlier first).
# test_csv: class-level eval CSV in phd_work/results. pkl: per-object eval (eval_meshes_full.pkl).
VARIANTS = {
    'voxel 3-plane, 3-class (run1+run2)': {
        'logs': ['3cls_run1.log', '3cls_run2.log'],
        'test_csv': 'voxel_3plane_3cls_best.csv',
        'pkl': 'out/phd_3cls/generation_3cls_best/eval_meshes_full.pkl',
        'val_log': None, 'L': None, 'run': 'phd_work/logs/3cls_run1.log + 3cls_run2.log; cfg voxel_3plane_3cls.yaml'},
    'voxel 3-plane, 3-class (s2)': {
        'logs': ['3cls_s2.log'],
        'test_csv': 'voxel_3plane_3cls_s2_best.csv',
        'pkl': 'out/phd_3cls_s2/generation_s2_best/eval_meshes_full.pkl',
        'val_log': None, 'L': None, 'run': 'phd_work/logs/3cls_s2.log; cfg voxel_3plane_3cls_s2.yaml'},
    'dexel L=32': {
        'logs': ['dexel_L32_run.log'],
        'test_csv': 'dexel_3plane_3cls_L32_best.csv',
        'pkl': 'out/phd_dexel_L32/generation_best/eval_meshes_full.pkl',
        'val_log': None, 'L': 32, 'run': 'phd_work/logs/dexel_L32_run.log; cfg dexel_3plane_3cls_L32.yaml'},
    'dexel L=2': {
        'logs': ['dexel_L2_run.log'],
        'test_csv': 'dexel_3plane_3cls_L2_best.csv',
        'pkl': 'out/phd_dexel_L2/generation_best/eval_meshes_full.pkl',
        'val_log': None, 'L': 2, 'run': 'phd_work/logs/dexel_L2_run.log; cfg dexel_3plane_3cls_L2.yaml'},
}

INPUT_ROWS = {L: {'test_csv': 'input_row_L%d.csv' % L,
                  'pkl': 'out/phd_input_L%d/generation/eval_meshes_full.pkl' % L}
              for L in (1, 2, 4, 32)}

# No training: test CSV and val IoU from the val_eval log only (no curves).
PRETRAINED = {
    'pretrained 3-plane (13-class model, 3-class rows)': {
        'test_csv': 'voxel_3plane_pretrained_full.csv', 'val_log': 'val_eval_pretrained.log',
        'run': 'phd_work/results/voxel_3plane_pretrained_full.csv (chair/table/lamp rows)'},
    'pretrained 1-plane': {
        'test_csv': 'pretrained_1plane_3cls.csv', 'val_log': 'val_eval_pretrained_1plane_3cls.log',
        'run': 'phd_work/logs/val_eval_pretrained_1plane_3cls.log'},
    'pretrained grid32': {
        'test_csv': 'pretrained_grid32_3cls.csv', 'val_log': 'val_eval_pretrained_grid32_3cls.log',
        'run': 'phd_work/logs/val_eval_pretrained_grid32_3cls.log'},
}

VOXEL_BASELINE = ['voxel 3-plane, 3-class (run1+run2)', 'voxel 3-plane, 3-class (s2)']

# ---- parsing ----------------------------------------------------------------------
IT_RE = re.compile(r'it=(\d+), loss=[^,]*, time: ([0-9.]+)s')
VAL_RE = re.compile(r'Validation metric \(iou\): ([0-9.]+)')


def parse_log(path):
    """Return (val_points, final_time): val_points = [(it, val_iou, time_s)], time_s is the
    last 'time:' value printed before the validation line (process-local)."""
    vals = []
    last_t = None
    final_t = None
    with open(path, errors='replace') as f:
        for line in f:
            for chunk in line.split('\r'):
                m = IT_RE.search(chunk)
                if m:
                    last_t = (int(m.group(1)), float(m.group(2)))
                    final_t = last_t[1]
                    continue
                m = VAL_RE.search(chunk)
                if m and last_t is not None:
                    vals.append((last_t[0], float(m.group(1)), last_t[1]))
    return vals, final_t


def training_series(logs):
    """Concatenate validation points over logs, adding the final time of earlier logs as offset."""
    pts = []
    offset = 0.0
    total = 0.0
    for name in logs:
        path = os.path.join(PHD, 'logs', name)
        vals, final_t = parse_log(path)
        pts += [(it, v, offset + t) for it, v, t in vals]
        offset += final_t or 0.0
        total = offset
    # later logs win on a repeated iteration (as in val_curve.py)
    d = {}
    for it, v, t in pts:
        d[it] = (v, t)
    its = sorted(d)
    return its, [d[i][0] for i in its], [d[i][1] for i in its], total


def speed_median(logs):
    rates = []
    for name in logs:
        path = os.path.join(PHD, 'logs', name)
        pts = []
        with open(path, errors='replace') as f:
            for line in f:
                for chunk in line.split('\r'):
                    m = IT_RE.search(chunk)
                    if m:
                        pts.append((int(m.group(1)), float(m.group(2))))
        pts = sorted(set(pts))
        for (i0, t0), (i1, t1) in zip(pts, pts[1:]):
            if i0 >= 300 and i1 > i0:
                rates.append((t1 - t0) / (i1 - i0))
    return float(np.median(rates)) if rates else float('nan')


def test_metrics(csv_path):
    d = pd.read_csv(csv_path).set_index('class name')
    out = {}
    for key, col, scale in [('iou', 'iou (mesh)', 1.0), ('chamfer', 'chamfer-L1 (mesh)', 10.0),
                            ('nc', 'normals (mesh)', 1.0), ('f', 'f-score (mesh)', 1.0)]:
        out[key] = {c: float(d.loc[c, col]) * scale for c in CLASSES}
        out[key]['mean'] = float(np.mean([out[key][c] for c in CLASSES]))
    return out


def res(name):
    return os.path.join(PHD, 'results', name)


# ---- plots ----------------------------------------------------------------------
def save(fig, stem):
    os.makedirs(REPORT, exist_ok=True)
    fig.savefig(os.path.join(REPORT, stem + '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(REPORT, stem + '.svg'), bbox_inches='tight')
    plt.close(fig)


def plot_val(series, by_hours, title_suffix=''):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (its, vals, hrs) in series.items():
        x = np.array(hrs) if by_hours else np.array(its)
        ax.plot(x, vals, marker='.', lw=1, label=name)
    ax.set_xlabel('wall-clock training time (h)' if by_hours else 'iteration')
    ax.set_ylabel('val IoU (3-class val split)')
    ax.set_title('Validation IoU vs %s%s' % ('wall-clock time' if by_hours else 'iteration', title_suffix))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_test_per_class(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = CLASSES + ['mean']
    n = len(rows)
    width = 0.8 / max(n, 1)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    for i, (name, m) in enumerate(rows.items()):
        xs = np.arange(len(cats)) + (i - (n - 1) / 2) * width
        ys = [m['iou'][c] for c in cats]
        ax.scatter(xs, ys, marker=markers[i % len(markers)], s=50, label=name, zorder=3)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel('test mesh IoU')
    ax.set_title('Test mesh IoU per class (3-class test split)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=7, loc='lower left')
    return fig


def plot_iou_vs_L(net, inp, voxel_vals):
    fig, ax = plt.subplots(figsize=(7, 5))
    Ls = sorted(net)
    ax.plot(Ls, [net[L] for L in Ls], marker='o', label='network (dexel L)')
    Li = sorted(inp)
    ax.plot(Li, [inp[L] for L in Li], marker='s', ls='--', label='input row (occupancy superset)')
    if voxel_vals:
        lo, hi = min(voxel_vals), max(voxel_vals)
        ax.axhspan(lo, hi, color='grey', alpha=0.25, label='voxel baseline (min..max of runs)')
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(['1', '2', '4', '8', '16', '32'])
    ax.set_xlabel('L (slots per ray)')
    ax.set_ylabel('test mesh IoU, class mean (chair, table, lamp)')
    ax.set_title('Test IoU vs L')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_hits_hist(counts_all, counts_nonzero):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    vals = np.arange(len(counts_all))
    for ax, c, title in [(axes[0], counts_all, 'all rays'), (axes[1], counts_nonzero, 'rays with >= 1 hit')]:
        mask = c > 0
        ax.bar(vals[mask], c[mask], width=0.8)
        ax.set_yscale('log')
        ax.set_xlabel('hits per ray (3 axes pooled)')
        ax.set_ylabel('number of rays (test split)')
        ax.set_title(title)
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle('Hits per ray on the 3-class test split (dexel_hits_hist)')
    return fig


def plot_scatter(iou32, iou2, rays):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(iou32, iou2, c=rays, cmap='viridis', s=8, alpha=0.7)
    ax.plot([0, 1], [0, 1], color='k', lw=0.8, ls='--')
    ax.set_xlabel('IoU, dexel L=32')
    ax.set_ylabel('IoU, dexel L=2')
    ax.set_title('Per-object test IoU (colour: rays with > 2 hits, 3 axes)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('rays with > 2 hits')
    return fig


# ---- main -----------------------------------------------------------------------
def main():
    os.makedirs(REPORT, exist_ok=True)
    # 1, 2: validation curves
    series = {}
    trained_meta = {}
    for name, v in VARIANTS.items():
        its, vals, hrs, total = training_series(v['logs'])
        series[name] = (its, vals, hrs)
        trained_meta[name] = {'its': its, 'vals': vals, 'hours': total / 3600.0}
    save(plot_val({k: (s[0], s[1], s[2]) for k, s in series.items()}, by_hours=False), 'plot1_val_iou_vs_iteration')
    save(plot_val({k: (s[0], s[1], [t / 3600.0 for t in s[2]]) for k, s in series.items()}, by_hours=True),
         'plot2_val_iou_vs_hours')

    # 3: test per class
    rows = {}
    test = {}
    for name, v in VARIANTS.items():
        test[name] = test_metrics(res(v['test_csv']))
        rows[name] = test[name]
    for L, v in INPUT_ROWS.items():
        test['input row L=%d' % L] = test_metrics(res(v['test_csv']))
    for name, v in PRETRAINED.items():
        test[name] = test_metrics(res(v['test_csv']))
    plot_rows = {k: test[k] for k in list(VARIANTS) + ['pretrained 3-plane (13-class model, 3-class rows)']}
    save(plot_test_per_class(plot_rows), 'plot3_test_iou_per_class')

    # 4: test IoU vs L
    net = {VARIANTS[k]['L']: test[k]['iou']['mean'] for k in VARIANTS if VARIANTS[k]['L'] is not None}
    inp = {L: test['input row L=%d' % L]['iou']['mean'] for L in INPUT_ROWS}
    voxel_vals = [test[k]['iou']['mean'] for k in VOXEL_BASELINE]
    save(plot_iou_vs_L(net, inp, voxel_vals), 'plot4_test_iou_vs_L')

    # 5: hits-per-ray histogram on the test split (recomputed)
    root = os.path.join(REPO, 'data', 'ShapeNet')
    paths = []
    ids = []
    for c in CLASSES:
        for m in dhh.read_split(root, CLASS_IDS[c], 'test'):
            paths.append(os.path.join(root, CLASS_IDS[c], m, 'model.binvox'))
            ids.append((CLASS_IDS[c], m))
    hist = np.zeros(33, dtype=np.int64)
    rays_gt2 = {}
    for p, key in zip(paths, ids):
        occ = dhh.load_occupancy(p)
        per_obj = 0
        for a in range(3):
            h = dhh.hits_per_ray(occ, a)
            hist += np.bincount(h.ravel(), minlength=33)[:33]
            per_obj += int((h > 2).sum())
        rays_gt2[key] = per_obj
    hist_nonzero = hist.copy()
    hist_nonzero[0] = 0
    save(plot_hits_hist(hist, hist_nonzero), 'plot5_hits_per_ray_hist')

    # 6: per-object IoU L32 vs L2 on the test split
    p32 = os.path.join(REPO, VARIANTS['dexel L=32']['pkl'])
    p2 = os.path.join(REPO, VARIANTS['dexel L=2']['pkl'])
    d32 = pd.read_pickle(p32).set_index(['class id', 'modelname'])
    d2 = pd.read_pickle(p2).set_index(['class id', 'modelname'])
    joined = d32[['iou (mesh)']].join(d2[['iou (mesh)']], lsuffix='_L32', rsuffix='_L2', how='inner')
    joined['rays_gt2'] = [rays_gt2.get(k, np.nan) for k in joined.index]
    joined = joined.dropna(subset=['rays_gt2'])
    save(plot_scatter(joined['iou (mesh)_L32'].values, joined['iou (mesh)_L2'].values,
                      joined['rays_gt2'].values), 'plot6_per_object_iou_L32_vs_L2')
    scatter_stats = {'objects': len(joined),
                     'mean_L32': float(joined['iou (mesh)_L32'].mean()),
                     'mean_L2': float(joined['iou (mesh)_L2'].mean()),
                     'frac_L32_better': float((joined['iou (mesh)_L32'] > joined['iou (mesh)_L2']).mean())}

    # summary.md
    lines = ['# Report summary', '',
             'Generated by `phd_work/report_plots.py` from `phd_work/logs` and `phd_work/results`.',
             'Test metrics: class mean over chair, table, lamp (Chamfer-L1 ×10). Val IoU: last-10 validation mean ± SD.',
             'Training wall time: sum of the final `time:` of each log in the run. s/it: median, it ≥ 300 (`speed_stats`).',
             '',
             '| variant | source run | test IoU | Chamfer-L1 ×10 | NC | F-Score | val IoU last-10 mean (SD) | s/it median | wall time (h) |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, v in VARIANTS.items():
        m = test[name]
        meta = trained_meta[name]
        last = np.array(meta['vals'][-10:])
        sd = last.std(ddof=1) if len(last) > 1 else float('nan')
        sp = speed_median(v['logs'])
        lines.append('| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f (%.4f) | %.4f | %.2f |'
                     % (name, v['run'], m['iou']['mean'], m['chamfer']['mean'], m['nc']['mean'], m['f']['mean'],
                        last.mean(), sd, sp, meta['hours']))
    for L in sorted(INPUT_ROWS):
        m = test['input row L=%d' % L]
        lines.append('| input row L=%d | %s | %.4f | %.4f | %.4f | %.4f | n/a | n/a | n/a |'
                     % (L, 'phd_work/results/input_row_L%d.csv (no training)' % L,
                        m['iou']['mean'], m['chamfer']['mean'], m['nc']['mean'], m['f']['mean']))
    for name, v in PRETRAINED.items():
        m = test[name]
        val_line = [l for l in open(os.path.join(PHD, 'logs', v['val_log'])) if 'Validation metric (iou)' in l]
        val = float(VAL_RE.search(val_line[-1]).group(1)) if val_line else float('nan')
        lines.append('| %s | %s (no training; val IoU %.4f) | %.4f | %.4f | %.4f | %.4f | n/a | n/a | n/a |'
                     % (name, v['run'], val, m['iou']['mean'], m['chamfer']['mean'], m['nc']['mean'], m['f']['mean']))
    lines += ['', 'Per-object comparison (test split, dexel L=32 vs L=2): %d objects; mean IoU L32 %.4f, L2 %.4f; '
              'fraction of objects where L=32 is better %.3f.'
              % (scatter_stats['objects'], scatter_stats['mean_L32'], scatter_stats['mean_L2'],
                 scatter_stats['frac_L32_better']),
              '', 'Plot 4 uses class-mean test IoU; the voxel band spans the two voxel runs.']
    with open(os.path.join(REPORT, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
