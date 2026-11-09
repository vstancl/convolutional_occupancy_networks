"""[phd] Report plots and summary table from existing logs and results (CPU only).

Writes PNG + SVG to phd_work/report/ and phd_work/report/summary.md.
Any run whose files are missing is skipped (its row/line is left out and listed as missing),
so the script runs before the weekend chain has finished.

Add a run: an entry in VARIANTS (training runs), PRETRAINED, or INPUT_ROWS.
Weekend variants (rounds 1 and 2) are generated from WEEKEND_VARIANTS.

Plots:
  1  val IoU vs iteration (one line per training run)
  2  val IoU vs wall-clock hours (time field of each log, plus earlier logs' final time)
  3  test mesh IoU per class (chair, table, lamp, mean), one marker per run
  4  test mesh IoU vs L: network (3-plane and 1-plane, main and weekend rounds), input row
     (L 1, 2, 4, 32), voxel baseline band (min..max of its runs)
  5  histogram of hits per ray on the 3-class test split (dexel_hits_hist functions)
  6  per-object IoU(L32) vs IoU(L2) on the test split, colour = rays with > 2 hits (3 axes)
  6b per-object IoU(L32) - IoU(L2) vs rays with > 2 hits (log x), scatter + binned means
     with the object count per bin (one scatter per available L32/L2 pair)

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
REPORT = os.environ.get('PHD_REPORT_DIR') or os.path.join(PHD, 'report')  # dry runs set PHD_REPORT_DIR
sys.path.insert(0, PHD)
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from speed_stats import stats as _speed_stats  # noqa: E402,F401  (kept importable; speed_median below)
import dexel_hits_hist as dhh  # noqa: E402

CLASSES = ['chair', 'table', 'lamp']
CLASS_IDS = {'chair': '03001627', 'table': '04379243', 'lamp': '03636649'}

# ---- variants -----------------------------------------------------------------
# logs: train.py logs in the order they were run (resumed runs: earlier first).
VARIANTS = {
    'voxel 3-plane, 3-class (run1+run2)': {
        'logs': ['3cls_run1.log', '3cls_run2.log'], 'test_csv': 'voxel_3plane_3cls_best.csv',
        'pkl': 'out/phd_3cls/generation_3cls_best/eval_meshes_full.pkl', 'L': None, 'planes': 3,
        'run': 'phd_work/logs/3cls_run1.log + 3cls_run2.log; cfg voxel_3plane_3cls.yaml'},
    'voxel 3-plane, 3-class (s2)': {
        'logs': ['3cls_s2.log'], 'test_csv': 'voxel_3plane_3cls_s2_best.csv',
        'pkl': 'out/phd_3cls_s2/generation_s2_best/eval_meshes_full.pkl', 'L': None, 'planes': 3,
        'run': 'phd_work/logs/3cls_s2.log; cfg voxel_3plane_3cls_s2.yaml'},
    'dexel 3-plane L=32': {
        'logs': ['dexel_L32_run.log'], 'test_csv': 'dexel_3plane_3cls_L32_best.csv',
        'pkl': 'out/phd_dexel_L32/generation_best/eval_meshes_full.pkl', 'L': 32, 'planes': 3,
        'run': 'phd_work/logs/dexel_L32_run.log; cfg dexel_3plane_3cls_L32.yaml'},
    'dexel 3-plane L=2': {
        'logs': ['dexel_L2_run.log'], 'test_csv': 'dexel_3plane_3cls_L2_best.csv',
        'pkl': 'out/phd_dexel_L2/generation_best/eval_meshes_full.pkl', 'L': 2, 'planes': 3,
        'run': 'phd_work/logs/dexel_L2_run.log; cfg dexel_3plane_3cls_L2.yaml'},
}

# weekend variants: key -> (label, L or None, planes: 3, 1 or 'grid')
WEEKEND_VARIANTS = {
    'voxel3p': ('voxel 3-plane', None, 3),
    'dexel3p_L32': ('dexel 3-plane L=32', 32, 3),
    'dexel3p_L2': ('dexel 3-plane L=2', 2, 3),
    'dexel1p_L32': ('dexel 1-plane L=32', 32, 1),
    'dexel1p_L2': ('dexel 1-plane L=2', 2, 1),
    'voxel1p': ('voxel 1-plane', None, 1),
    'dexel3p_L4': ('dexel 3-plane L=4', 4, 3),
    'grid32': ('voxel grid32', None, 'grid'),
    'dexel3p_L1': ('dexel 3-plane L=1', 1, 3),
}
WEEKEND_ROUNDS = (1, 2)
ROUND2_EXCLUDED = {'dexel3p_L4', 'grid32', 'dexel3p_L1'}  # not in round 2 (chain_weekend.sh)
for _R in WEEKEND_ROUNDS:
    for _v, (_label, _L, _planes) in WEEKEND_VARIANTS.items():
        if _R == 2 and _v in ROUND2_EXCLUDED:
            continue
        VARIANTS['w%d %s' % (_R, _label)] = {
            'logs': ['w%d_%s.log' % (_R, _v)],
            'test_csv': 'w%d_%s_best.csv' % (_R, _v),
            'pkl': 'out/phd_w%d_%s/generation_best/eval_meshes_full.pkl' % (_R, _v),
            'L': _L, 'planes': _planes,
            'run': 'phd_work/logs/w%d_%s.log; phd_work/results/w%d_%s_best.csv' % (_R, _v, _R, _v)}

INPUT_ROWS = {L: {'test_csv': 'input_row_L%d.csv' % L,
                  'pkl': 'out/phd_input_L%d/generation/eval_meshes_full.pkl' % L}
              for L in (1, 2, 4, 32)}

# No training in this script: test CSV and val IoU from the val_eval log only (no curves).
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
MAIN_PAIR = ('dexel 3-plane L=32', 'dexel 3-plane L=2')

# ---- parsing ----------------------------------------------------------------------
IT_RE = re.compile(r'it=(\d+), loss=[^,]*, time: ([0-9.]+)s')
VAL_RE = re.compile(r'Validation metric \(iou\): ([0-9.]+)')


def log_path(name):
    return os.path.join(PHD, 'logs', name)


def have_all(paths):
    return all(os.path.exists(p) for p in paths)


def parse_log(path):
    """(val_points, final_time): val_points = [(it, val_iou, time_s)]; time_s is the last
    'time:' value printed before the validation line (process-local)."""
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
    """Validation points over logs, with the final time of earlier logs as offset.
    None if any log is missing."""
    paths = [log_path(n) for n in logs]
    if not have_all(paths):
        return None
    pts = []
    offset = 0.0
    for path in paths:
        vals, final_t = parse_log(path)
        pts += [(it, v, offset + t) for it, v, t in vals]
        offset += final_t or 0.0
    d = {}
    for it, v, t in pts:
        d[it] = (v, t)  # later logs win on a repeated iteration (as in val_curve.py)
    its = sorted(d)
    return {'its': its, 'vals': [d[i][0] for i in its], 'secs': [d[i][1] for i in its], 'total_s': offset}


def speed_median(logs):
    rates = []
    for name in logs:
        pts = []
        with open(log_path(name), errors='replace') as f:
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
    if not os.path.exists(csv_path):
        return None
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


def plot_val(series, by_hours):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, s in series.items():
        x = [t / 3600.0 for t in s['secs']] if by_hours else s['its']
        ax.plot(x, s['vals'], marker='.', lw=1, label=name)
    ax.set_xlabel('wall-clock training time (h)' if by_hours else 'iteration')
    ax.set_ylabel('val IoU (3-class val split)')
    ax.set_title('Validation IoU vs %s' % ('wall-clock time' if by_hours else 'iteration'))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return fig


def plot_test_per_class(rows):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cats = CLASSES + ['mean']
    n = len(rows)
    width = 0.8 / max(n, 1)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h', '*', 'p', 'H', '8']
    for i, (name, m) in enumerate(rows.items()):
        xs = np.arange(len(cats)) + (i - (n - 1) / 2) * width
        ys = [m['iou'][c] for c in cats]
        ax.scatter(xs, ys, marker=markers[i % len(markers)], s=40, label=name, zorder=3)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel('test mesh IoU')
    ax.set_title('Test mesh IoU per class (3-class test split)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    return fig


def plot_iou_vs_L(net_series, inp, voxel_vals):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, pts in net_series.items():
        Ls = sorted(pts)
        ax.plot(Ls, [pts[L] for L in Ls], marker='o', label=name)
    Li = sorted(inp)
    if Li:
        ax.plot(Li, [inp[L] for L in Li], marker='s', ls='--', color='k', label='input row (occupancy superset)')
    if voxel_vals:
        ax.axhspan(min(voxel_vals), max(voxel_vals), color='grey', alpha=0.25,
                   label='voxel 3-plane baseline (min..max of runs)')
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(['1', '2', '4', '8', '16', '32'])
    ax.set_xlabel('L (slots per ray)')
    ax.set_ylabel('test mesh IoU, class mean (chair, table, lamp)')
    ax.set_title('Test IoU vs L')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
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


def bin_means(x, y, n_bins=10):
    """Log-spaced bins on x (x >= 1). Returns [(bin_centre, mean_y, count)] for non-empty bins."""
    lo = max(1.0, float(x.min()))
    hi = max(float(x.max()), lo * 1.0001)
    edges = np.geomspace(lo, hi * 1.0001, n_bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.any():
            out.append((float(np.sqrt(a * b)), float(y[m].mean()), int(m.sum())))
    return out


def plot_diff_vs_rays(pairs):
    """pairs: [(label, rays array, diff array)]. One scatter per pair plus binned means."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (label, rays, diff) in enumerate(pairs):
        c = colours[i % len(colours)]
        xr = np.maximum(rays, 1.0)
        ax.scatter(xr, diff, s=5, alpha=0.25, color=c)
        bins = bin_means(xr, diff)
        if bins:
            ax.plot([b[0] for b in bins], [b[1] for b in bins], marker='o', color=c, lw=2,
                    label='%s (binned mean)' % label)
            for xc, ym, cnt in bins:
                ax.annotate('n=%d' % cnt, (xc, ym), textcoords='offset points', xytext=(0, 7),
                            ha='center', fontsize=6, color=c)
    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('rays with > 2 hits per object (3 axes; zero plotted at 1)')
    ax.set_ylabel('IoU(L=32) - IoU(L=2)')
    ax.set_title('Per-object IoU gain of L=32 over L=2 vs number of rays with > 2 hits')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return fig


# ---- main -----------------------------------------------------------------------
def pair_frame(entry32, entry2):
    """Per-object IoU frame joined on (class id, modelname), or None if a pkl is missing."""
    p32 = os.path.join(REPO, entry32['pkl'])
    p2 = os.path.join(REPO, entry2['pkl'])
    if not have_all([p32, p2]):
        return None
    d32 = pd.read_pickle(p32).set_index(['class id', 'modelname'])
    d2 = pd.read_pickle(p2).set_index(['class id', 'modelname'])
    return d32[['iou (mesh)']].join(d2[['iou (mesh)']], lsuffix='_L32', rsuffix='_L2', how='inner')


def main():
    os.makedirs(REPORT, exist_ok=True)
    missing = []

    # 1, 2: validation curves
    series = {}
    trained_meta = {}
    for name, v in VARIANTS.items():
        s = training_series(v['logs'])
        if s is None:
            missing.append(name + ' (training logs)')
            continue
        series[name] = s
        trained_meta[name] = s
    if series:
        save(plot_val(series, by_hours=False), 'plot1_val_iou_vs_iteration')
        save(plot_val(series, by_hours=True), 'plot2_val_iou_vs_hours')

    # test metrics
    test = {}
    for name, v in VARIANTS.items():
        m = test_metrics(res(v['test_csv']))
        if m is None:
            missing.append(name + ' (test CSV)')
        else:
            test[name] = m
    for L, v in INPUT_ROWS.items():
        m = test_metrics(res(v['test_csv']))
        if m is not None:
            test['input row L=%d' % L] = m
    for name, v in PRETRAINED.items():
        m = test_metrics(res(v['test_csv']))
        if m is not None:
            test[name] = m

    # 3: test per class
    plot_rows = {k: test[k] for k in list(VARIANTS) + ['pretrained 3-plane (13-class model, 3-class rows)']
                 if k in test}
    if plot_rows:
        save(plot_test_per_class(plot_rows), 'plot3_test_iou_per_class')

    # 4: test IoU vs L; network series grouped by (main|round, planes)
    net_series = {}
    for name, v in VARIANTS.items():
        if v['L'] is None or name not in test:
            continue
        group = name.split(' ')[0] if name.startswith('w') else 'main'
        key = '%s network, %d-plane' % (group, v['planes'])
        net_series.setdefault(key, {})[v['L']] = test[name]['iou']['mean']
    inp = {L: test['input row L=%d' % L]['iou']['mean'] for L in INPUT_ROWS if 'input row L=%d' % L in test}
    voxel_vals = [test[k]['iou']['mean'] for k in VOXEL_BASELINE if k in test]
    if net_series or inp:
        save(plot_iou_vs_L(net_series, inp, voxel_vals), 'plot4_test_iou_vs_L')

    # 5: hits per ray histogram (recomputed on the test split)
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

    # 6: per-object IoU L32 vs L2 (main pair)
    joined = pair_frame(VARIANTS[MAIN_PAIR[0]], VARIANTS[MAIN_PAIR[1]])
    scatter_stats = None
    if joined is not None:
        joined['rays_gt2'] = [rays_gt2.get(k, np.nan) for k in joined.index]
        joined = joined.dropna(subset=['rays_gt2'])
        save(plot_scatter(joined['iou (mesh)_L32'].values, joined['iou (mesh)_L2'].values,
                          joined['rays_gt2'].values), 'plot6_per_object_iou_L32_vs_L2')
        scatter_stats = {'objects': len(joined),
                         'mean_L32': float(joined['iou (mesh)_L32'].mean()),
                         'mean_L2': float(joined['iou (mesh)_L2'].mean()),
                         'frac_L32_better': float((joined['iou (mesh)_L32'] > joined['iou (mesh)_L2']).mean())}
    else:
        missing.append('plot 6 (per-object pkl for main L32/L2)')

    # 6b: IoU gain vs rays, one scatter per available L32/L2 pair (main, then each weekend round)
    pair_names = [('main', MAIN_PAIR[0], MAIN_PAIR[1])]
    for R in WEEKEND_ROUNDS:
        pair_names.append(('w%d' % R, 'w%d dexel 3-plane L=32' % R, 'w%d dexel 3-plane L=2' % R))
    diff_pairs = []
    pair_bins = {}
    for label, n32, n2 in pair_names:
        if n32 not in VARIANTS or n2 not in VARIANTS:
            continue
        j = pair_frame(VARIANTS[n32], VARIANTS[n2])
        if j is None or j.empty:
            missing.append('plot 6b pair %s (per-object pkl)' % label)
            continue
        rays = np.array([rays_gt2.get(k, np.nan) for k in j.index], dtype=float)
        diff = (j['iou (mesh)_L32'] - j['iou (mesh)_L2']).values
        ok = ~np.isnan(rays)
        diff_pairs.append((label, rays[ok], diff[ok]))
        pair_bins[label] = bin_means(np.maximum(rays[ok], 1.0), diff[ok])
    if diff_pairs:
        save(plot_diff_vs_rays(diff_pairs), 'plot6b_iou_gain_vs_rays_gt2')

    # summary.md
    lines = ['# Report summary', '',
             'Generated by `phd_work/report_plots.py` from `phd_work/logs` and `phd_work/results`.',
             'Test metrics: class mean over chair, table, lamp (Chamfer-L1 ×10). Val IoU: last-10 validation mean (SD).',
             'Training wall time: sum of the final `time:` of each log in the run. s/it: median, it ≥ 300.',
             '',
             '| variant | source | test IoU | Chamfer-L1 ×10 | NC | F-Score | val IoU last-10 (SD) | s/it median | wall time (h) |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, v in VARIANTS.items():
        if name not in test:
            lines.append('| %s | %s | not available yet | | | | | | |' % (name, v['run']))
            continue
        m = test[name]
        meta = trained_meta.get(name)
        if meta is not None and meta['vals']:
            last = np.array(meta['vals'][-10:])
            sd = last.std(ddof=1) if len(last) > 1 else float('nan')
            val_txt = '%.4f (%.4f)' % (last.mean(), sd)
            tail = '%.4f | %.2f' % (speed_median(v['logs']), meta['total_s'] / 3600.0)
        else:
            val_txt, tail = 'n/a', 'n/a | n/a'
        lines.append('| %s | %s | %.4f | %.4f | %.4f | %.4f | %s | %s |'
                     % (name, v['run'], m['iou']['mean'], m['chamfer']['mean'], m['nc']['mean'], m['f']['mean'],
                        val_txt, tail))
    for L in sorted(INPUT_ROWS):
        key = 'input row L=%d' % L
        if key not in test:
            lines.append('| %s | input_row_L%d.csv | not available yet | | | | n/a | n/a | n/a |' % (key, L))
            continue
        m = test[key]
        lines.append('| %s | phd_work/results/input_row_L%d.csv (no training) | %.4f | %.4f | %.4f | %.4f | n/a | n/a | n/a |'
                     % (key, L, m['iou']['mean'], m['chamfer']['mean'], m['nc']['mean'], m['f']['mean']))
    for name, v in PRETRAINED.items():
        if name not in test:
            continue
        m = test[name]
        val_path = log_path(v['val_log'])
        val_line = [l for l in open(val_path) if 'Validation metric (iou)' in l] if os.path.exists(val_path) else []
        val = float(VAL_RE.search(val_line[-1]).group(1)) if val_line else float('nan')
        lines.append('| %s | %s (no training; val IoU %.4f) | %.4f | %.4f | %.4f | %.4f | n/a | n/a | n/a |'
                     % (name, v['run'], val, m['iou']['mean'], m['chamfer']['mean'], m['nc']['mean'], m['f']['mean']))
    lines.append('')
    if scatter_stats is not None:
        lines.append('Per-object comparison (test split, dexel L=32 vs L=2): %d objects; mean IoU L32 %.4f, L2 %.4f; '
                     'fraction of objects where L=32 is better %.3f.'
                     % (scatter_stats['objects'], scatter_stats['mean_L32'], scatter_stats['mean_L2'],
                        scatter_stats['frac_L32_better']))
    for label, bins in pair_bins.items():
        if bins:
            lines.append('Plot 6b bins (%s; rays>2 bin centre: mean IoU L32 - L2 (n objects)): %s'
                         % (label, '; '.join('%.0f: %+.4f (n=%d)' % b for b in bins)))
    lines.append('')
    lines.append('Missing (skipped): %s' % ('none' if not missing else '; '.join(missing)))
    with open(os.path.join(REPORT, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
