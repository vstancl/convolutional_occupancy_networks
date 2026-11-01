"""[phd] Parse validation IoU from one or more train.py logs into CSV + PNG, and fit a trend.

Usage:
    uv run python phd_work/val_curve.py <log> [<log> ...] [--out-prefix PREFIX]
        [--window 20000] [--slope-window 20000]

Logs are read in the order given (e.g. run1 then run2 after a resume). Each
"Validation metric (iou): X" line is printed right after the "[Epoch ..] it=N" line for
the iteration N that triggered validation (validate_every). The iteration is therefore the
last "it=" value seen before the validation line.

If a later log repeats an iteration already seen (a resume that re-ran it), the later
value wins and the duplicate is reported.

Outputs: PREFIX.csv (it, iou) and PREFIX.png (curve, best marker, fitted trend).

Stdout reports:
  - best IoU and its iteration;
  - the old stopping statistic: best-IoU gain over the last --window iterations
    (kept for comparison; it is sensitive to single spikes, see port-log);
  - the least-squares slope of IoU over the last --slope-window iterations, in IoU per
    20k iterations, with the residual SD (ddof=2). The window is it >= last_it - slope_window.
"""
import argparse
import csv
import math
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

IT_RE = re.compile(r'it=(\d+)')
VAL_RE = re.compile(r'Validation metric \(iou\): ([0-9.]+)')


def parse(log_path):
    """Return list of (it, iou) for every validation line in one log."""
    points = []
    last_it = None
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            # tqdm writes carriage returns; the validation line is a plain line.
            for chunk in line.split('\r'):
                m = VAL_RE.search(chunk)
                if m is not None:
                    if last_it is None:
                        raise ValueError('validation line before any it= line in %s' % log_path)
                    points.append((last_it, float(m.group(1))))
                    continue
                m = IT_RE.search(chunk)
                if m is not None:
                    last_it = int(m.group(1))
    return points


def parse_many(log_paths):
    """Concatenate validation points from several logs, in order. Later logs win on duplicate it."""
    by_it = {}
    for path in log_paths:
        pts = parse(path)
        if not pts:
            raise SystemExit('no "Validation metric (iou)" lines found in %s' % path)
        for it, iou in pts:
            if it in by_it:
                print('note: it=%d appears again in %s (old %.4f, new %.4f); keeping the new value'
                      % (it, path, by_it[it], iou))
            by_it[it] = iou
    return sorted(by_it.items())


def fit_line(xs, ys):
    """Ordinary least squares y = a + b*x. Returns (slope, intercept, residual SD with ddof=2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / (n - 2)) if n > 2 else float('nan')
    return b, a, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--out-prefix', default=None,
                    help='Output prefix. Default: first log path without its extension.')
    ap.add_argument('--window', type=int, default=20000,
                    help='Window for the old max-gain statistic (kept for comparison).')
    ap.add_argument('--slope-window', type=int, default=20000,
                    help='Window (iterations, counted back from the last validation) for the slope.')
    args = ap.parse_args()
    prefix = args.out_prefix or args.logs[0].rsplit('.', 1)[0]

    points = parse_many(args.logs)

    with open(prefix + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['it', 'iou'])
        w.writerows(points)

    its = [p[0] for p in points]
    ious = [p[1] for p in points]
    best_it, best_iou = max(points, key=lambda p: p[1])
    last_it = its[-1]

    # Old statistic: best IoU in the last window minus best IoU at or before (last_it - window).
    base_pts = [p for p in points if p[0] <= last_it - args.window]
    if base_pts:
        base_iou = max(p[1] for p in base_pts)
        base_it = max(base_pts, key=lambda p: p[1])[0]
        gain = best_iou - base_iou
        gain_str = ('%.4f (best %.4f at it=%d vs best %.4f at it<=%d)'
                    % (gain, best_iou, best_it, base_iou, base_it))
    else:
        gain_str = 'n/a: run shorter than the window (%d iterations)' % args.window

    # Slope over the last slope_window iterations (inclusive lower bound).
    win_pts = [p for p in points if p[0] >= last_it - args.slope_window]
    slope = sd = intercept = None
    if len(win_pts) >= 3:
        xs = [p[0] / 20000.0 for p in win_pts]  # x in units of 20k iterations
        ys = [p[1] for p in win_pts]
        slope, intercept, sd = fit_line(xs, ys)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(its, ious, marker='o', label='val IoU')
    ax.axvline(best_it, color='gray', linestyle='--', linewidth=0.8, label='best')
    if slope is not None:
        x0, x1 = win_pts[0][0], win_pts[-1][0]
        ax.plot([x0, x1], [intercept + slope * x0 / 20000.0, intercept + slope * x1 / 20000.0],
                color='C3', linewidth=1.2, label='trend (last %d it)' % args.slope_window)
    ax.set_xlabel('iteration')
    ax.set_ylabel('validation IoU')
    ax.set_title('3-class voxel 3-plane baseline (%s)'
                 % ', '.join(p.split('/')[-1] for p in args.logs))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix + '.png', dpi=120)

    print('logs: %s' % ' + '.join(args.logs))
    print('validations: %d (it %d..%d)' % (len(points), its[0], last_it))
    print('best val IoU: %.4f at it=%d' % (best_iou, best_it))
    print('old statistic, best-IoU gain over last %d iterations: %s' % (args.window, gain_str))
    if slope is None:
        print('slope: n/a (fewer than 3 validations in the last %d iterations)' % args.slope_window)
    else:
        print('slope over it >= %d (n=%d validations): %+.5f IoU per 20k iterations; residual SD %.5f'
              % (last_it - args.slope_window, len(win_pts), slope, sd))
    print('wrote %s.csv and %s.png' % (prefix, prefix))


if __name__ == '__main__':
    main()
