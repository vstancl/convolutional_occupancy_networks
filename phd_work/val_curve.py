"""[phd] Parse validation IoU from a train.py log into CSV + PNG.

Usage:
    uv run python phd_work/val_curve.py <log> [--out-prefix PREFIX] [--window 20000]

Each "Validation metric (iou): X" line is printed right after the "[Epoch ..] it=N" line for
the iteration N that triggered validation (validate_every). The iteration is therefore the
last "it=" value seen before the validation line.

Outputs: PREFIX.csv (it, iou) and PREFIX.png; stdout reports best IoU, its iteration, and the
best-IoU gain over the last --window iterations.
"""
import argparse
import csv
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

IT_RE = re.compile(r'it=(\d+)')
VAL_RE = re.compile(r'Validation metric \(iou\): ([0-9.]+)')


def parse(log_path):
    """Return list of (it, iou) for every validation line in the log."""
    points = []
    last_it = None
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            # tqdm writes carriage returns; the validation line is a plain line.
            for chunk in line.split('\r'):
                m = VAL_RE.search(chunk)
                if m is not None:
                    if last_it is None:
                        raise ValueError('validation line before any it= line')
                    points.append((last_it, float(m.group(1))))
                    continue
                m = IT_RE.search(chunk)
                if m is not None:
                    last_it = int(m.group(1))
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--out-prefix', default=None)
    ap.add_argument('--window', type=int, default=20000)
    args = ap.parse_args()
    prefix = args.out_prefix or args.log.rsplit('.', 1)[0]

    points = parse(args.log)
    if not points:
        raise SystemExit('no "Validation metric (iou)" lines found in %s' % args.log)

    with open(prefix + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['it', 'iou'])
        w.writerows(points)

    its = [p[0] for p in points]
    ious = [p[1] for p in points]
    best_it, best_iou = max(points, key=lambda p: p[1])
    last_it = its[-1]

    # Best IoU among validations at or before (last_it - window), the baseline for the gain.
    base_pts = [p for p in points if p[0] <= last_it - args.window]
    if base_pts:
        base_iou = max(p[1] for p in base_pts)
        base_it = max(base_pts, key=lambda p: p[1])[0]
        gain = best_iou - base_iou
        gain_str = ('%.4f (best %.4f at it=%d vs best %.4f at it<=%d)'
                    % (gain, best_iou, best_it, base_iou, base_it))
    else:
        gain_str = 'n/a: run shorter than the window (%d iterations)' % args.window

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(its, ious, marker='o', label='val IoU')
    ax.axvline(best_it, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('iteration')
    ax.set_ylabel('validation IoU')
    ax.set_title('3-class voxel 3-plane baseline (%s)' % args.log.split('/')[-1])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix + '.png', dpi=120)

    print('validations: %d (it %d..%d)' % (len(points), its[0], last_it))
    print('best val IoU: %.4f at it=%d' % (best_iou, best_it))
    print('best-IoU gain over last %d iterations: %s' % (args.window, gain_str))
    print('wrote %s.csv and %s.png' % (prefix, prefix))


if __name__ == '__main__':
    main()
