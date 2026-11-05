"""[phd] Hits per ray for voxel-derived dexels (CPU only, no training).

For every object of the config's classes and split, load model.binvox the way
src/data/fields.py VoxelsField does (binvox_rw.read_as_3d_array, .data, > 0.5).
For each of the 3 axes and every 1D column of RES voxels along that axis (a ray),
count the runs of consecutive occupied voxels; hits = 2 * runs.

Reports, per split and per axis plus over all axes:
  - histogram of hits per ray (all rays, and rays with >= 1 hit),
  - max hits, and the 99th / 99.9th / 99.99th percentile over all rays,
  - for L in 2, 4, 6, 8, 12, 16: fraction of rays with hits > L, and fraction of
    objects with at least one ray with hits > L,
  - run time.

Percentiles are exact for these integer counts: the value at index floor(p*(N-1)) of
the sorted hits (the same as numpy.percentile with method='lower').

Usage:
    uv run python phd_work/dexel_hits_hist.py <config> [--split train val test]

Splits are read from <data.path>/<class>/<split>.lst the same way
src/data/core.py Shapes3dDataset does (split on newlines, drop empty strings).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src import config  # noqa: E402
from src.utils import binvox_rw  # noqa: E402

RES = 32
AXIS_NAMES = {0: 'axis 0 (rays along dim 0)', 1: 'axis 1 (rays along dim 1)',
              2: 'axis 2 (rays along dim 2)'}
THRESHOLDS = [2, 4, 6, 8, 12, 16]
NBIN = RES + 2  # hits are 0..2*(RES//2) = 0..32 for RES=32; one spare bin for safety
PERCENTILES = [99.0, 99.9, 99.99]


def read_split(dataset_root, category, split):
    split_file = os.path.join(dataset_root, category, split + '.lst')
    with open(split_file, 'r') as f:
        models = f.read().split('\n')
    return [m for m in models if m != '']


def load_occupancy(model_path):
    """Same as VoxelsField.load, but returns a boolean array."""
    with open(model_path, 'rb') as f:
        voxels = binvox_rw.read_as_3d_array(f)
    occ = voxels.data.astype(np.float32) > 0.5
    assert occ.shape == (RES, RES, RES), occ.shape
    return occ


def hits_per_ray(occ, axis):
    """Hits for every ray along `axis`: 2 * number of runs of consecutive occupied voxels."""
    x = np.moveaxis(occ, axis, -1)  # rays are the last-axis columns, shape (RES, RES, RES)
    starts = x[..., 0].astype(np.int64)
    starts = starts + np.count_nonzero(x[..., 1:] & ~x[..., :-1], axis=-1)
    return 2 * starts  # shape (RES, RES)


def quantile_from_hist(counts, p):
    """Value at index floor(p/100 * (N-1)) of the sorted sample, from its histogram."""
    n = int(counts.sum())
    k = int(np.floor(p / 100.0 * (n - 1)))
    cum = np.cumsum(counts)
    return int(np.searchsorted(cum, k, side='right'))


def fmt_hist(counts, total, label):
    lines = ['  %s (N=%d):' % (label, total)]
    for v in np.nonzero(counts)[0]:
        lines.append('    hits=%2d  count=%12d  frac=%.6f' % (v, counts[v], counts[v] / float(total)))
    return lines


def report_block(lines, name, bins, obj_max, n_objects):
    """bins: (NBIN,) histogram of rays; obj_max: (n_objects,) max hits per object."""
    n_rays = int(bins.sum())
    lines.append('%s' % name)
    lines.append('  rays=%d objects=%d' % (n_rays, n_objects))
    lines.append('  max hits = %d' % int(np.nonzero(bins)[0].max()))
    for p in PERCENTILES:
        lines.append('  p%s = %d' % (('%g' % p), quantile_from_hist(bins, p)))
    lines += fmt_hist(bins, n_rays, 'histogram, all rays')
    nonzero = bins.copy()
    nonzero[0] = 0
    lines += fmt_hist(nonzero, int(nonzero.sum()), 'histogram, rays with >= 1 hit')
    for L in THRESHOLDS:
        frac_rays = int(bins[L + 1:].sum()) / float(n_rays)
        frac_obj = float(np.mean(obj_max > L))
        lines.append('  L=%2d: frac rays hits>L = %.6f   frac objects with >=1 ray hits>L = %.6f'
                     % (L, frac_rays, frac_obj))
    lines.append('')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--split', nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'])
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/default.yaml')
    dataset_root = cfg['data']['path']
    categories = cfg['data']['classes']
    voxels_file = cfg['data']['voxels_file']

    lines = ['config: %s' % args.config,
             'dataset: %s' % dataset_root,
             'classes: %s' % categories,
             'voxels_file: %s' % voxels_file,
             'resolution: %d' % RES,
             '']
    total_start = time.time()
    for split in args.split:
        t0 = time.time()
        paths = []
        for c in categories:
            for m in read_split(dataset_root, c, split):
                paths.append(os.path.join(dataset_root, c, m, voxels_file))
        n_obj = len(paths)
        bins = np.zeros((3, NBIN), dtype=np.int64)
        obj_max = np.zeros((n_obj, 3), dtype=np.int64)
        for i, p in enumerate(paths):
            occ = load_occupancy(p)
            for a in range(3):
                h = hits_per_ray(occ, a)
                bins[a] += np.bincount(h.ravel(), minlength=NBIN)[:NBIN]
                obj_max[i, a] = h.max()
        elapsed = time.time() - t0
        lines.append('=' * 70)
        lines.append('split: %s   objects: %d   time: %.1f s (%.1f ms/object)'
                     % (split, n_obj, elapsed, 1000.0 * elapsed / max(n_obj, 1)))
        lines.append('')
        for a in range(3):
            report_block(lines, AXIS_NAMES[a], bins[a], obj_max[:, a], n_obj)
        report_block(lines, 'all axes', bins.sum(0), obj_max.max(axis=1), n_obj)
        print('split %s done: %d objects, %.1f s' % (split, n_obj, elapsed), file=sys.stderr)

    lines.append('total time: %.1f s' % (time.time() - total_start))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
