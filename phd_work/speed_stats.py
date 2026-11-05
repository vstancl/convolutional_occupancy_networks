"""[phd] Median seconds per iteration from a train.py log.

Reads the "[Epoch ..] it=N, loss=..., time: T s" lines (one per print_every iterations),
takes the time difference between consecutive lines with it >= SKIP, divides by the
iteration difference, and reports the median over those intervals.

Usage:
    uv run python phd_work/speed_stats.py <log> [<log> ...] [--skip 300]
"""
import argparse
import re
import statistics

PAT = re.compile(r'it=(\d+), loss=[^,]*, time: ([0-9.]+)s')


def stats(path, skip):
    pts = []
    with open(path, errors='replace') as f:
        for line in f:
            for chunk in line.split('\r'):
                m = PAT.search(chunk)
                if m:
                    pts.append((int(m.group(1)), float(m.group(2))))
    pts = sorted(set(pts))
    rates = []
    for (i0, t0), (i1, t1) in zip(pts, pts[1:]):
        if i0 >= skip and i1 > i0:
            rates.append((t1 - t0) / (i1 - i0))
    return len(rates), statistics.median(rates), pts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--skip', type=int, default=300)
    args = ap.parse_args()
    for path in args.logs:
        n, med, last = stats(path, args.skip)
        print('%s: intervals=%d median s/it=%.4f last (it, time)=%s' % (path, n, med, last))


if __name__ == '__main__':
    main()
