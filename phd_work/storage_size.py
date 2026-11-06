"""[phd] Bytes per object: binvox vs dense grid vs sparse dexels vs dense slot tensors.

For every object of the 3-class test split (or --split), reports:
  (a) model.binvox file size;
  (b) dense bit grid 32^3 (4096 bytes);
  (c) single-axis sparse dexel for each of the 3 ray axes, CSR-like: the hit boundary
      indices as uint8 (0..32) for all hits, plus one uint8 count per ray (1024 rays);
      bytes = nnz_hits + 1024;
  (d) tri-dexel = sum of the 3 single-axis dexels;
  (e) the same with float32 depths instead of uint8 (mesh-derived dexels):
      bytes = nnz_hits * 4 + 1024 per axis, and the tri-dexel sum;
  (f) dense slot tensors as fed to the network, float32, shape (3, 2L, 32, 32):
      bytes = 3 * 2L * 1024 * 4 for L = 2 and L = 32.
For each quantity: mean, median, p95, max over objects, and mean / mean(a).

Boundary convention: a run of occupied voxels i0..i1 gives hits at boundary index i0 (entry)
and i1 + 1 (exit), so every hit is in 0..32 and fits in uint8.

Usage:
    uv run python phd_work/storage_size.py [--split test] [--out phd_work/results/storage_size.txt]

The output file is appended to (the prediction header is written before the run, see the
file itself), so the first lines are the prediction.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.config import load_config  # noqa: E402
from src.utils import binvox_rw  # noqa: E402
from phd_work.dexels import RAY_AXIS, RES  # noqa: E402

N_RAYS = RES * RES
DENSE_BYTES = RES ** 3 // 8  # 4096


def ray_hit_boundaries(vox, plane):
    """Per-ray hit boundary indices (uint8-range ints) and counts for one ray axis.

    Returns (flat, counts): flat is the 1D array of all hit boundary indices in ray order
    (row-major over rays), counts has one entry per ray (number of hits, 2 per run).
    """
    x = np.moveaxis(vox, RAY_AXIS[plane], -1).reshape(-1, RES)
    prev = np.zeros_like(x)
    prev[:, 1:] = x[:, :-1]
    nxt = np.zeros_like(x)
    nxt[:, :-1] = x[:, 1:]
    starts = x & ~prev
    ends = x & ~nxt
    run_idx = np.cumsum(starts, axis=1) - 1
    hits = np.zeros((x.shape[0], RES), dtype=np.int64)
    rows, cols = np.nonzero(starts)
    hits[rows, 2 * run_idx[rows, cols]] = cols          # entry at i0
    rows, cols = np.nonzero(ends)
    hits[rows, 2 * run_idx[rows, cols] + 1] = cols + 1  # exit at i1 + 1
    counts = 2 * starts.sum(axis=1)
    valid = np.arange(RES)[None, :] < counts[:, None]
    flat = hits[valid]
    return flat, counts


def object_bytes(vox, binvox_size):
    rec = {'a_binvox': float(binvox_size), 'b_dense_bits': float(DENSE_BYTES)}
    tri_u8 = 0.0
    tri_f32 = 0.0
    for p in range(3):
        flat, counts = ray_hit_boundaries(vox, p)
        nnz = flat.size
        assert counts.size == N_RAYS and (counts <= 32).all()
        b_u8 = nnz * 1 + N_RAYS * 1
        b_f32 = nnz * 4 + N_RAYS * 1
        rec['c_axis%d_u8' % p] = float(b_u8)
        rec['e_axis%d_f32' % p] = float(b_f32)
        tri_u8 += b_u8
        tri_f32 += b_f32
    rec['d_tri_u8'] = tri_u8
    rec['e_tri_f32'] = tri_f32
    for L in (2, 32):
        rec['f_slots_L%d' % L] = float(3 * 2 * L * N_RAYS * 4)
    return rec


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return v.mean(), np.median(v), np.percentile(v, 95), v.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--config', default=os.path.join(REPO, 'phd_work', 'configs', 'voxel_3plane_3cls.yaml'))
    ap.add_argument('--out', default=os.path.join(REPO, 'phd_work', 'results', 'storage_size.txt'))
    args = ap.parse_args()

    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        cfg = load_config(args.config, 'configs/default.yaml')
    finally:
        os.chdir(cwd)
    root = os.path.join(REPO, cfg['data']['path'])
    voxels_file = cfg['data']['voxels_file']

    items = []
    for c in cfg['data']['classes']:
        for m in open(os.path.join(root, c, args.split + '.lst')).read().split('\n'):
            if m:
                items.append(os.path.join(root, c, m, voxels_file))

    recs = {}
    for path in items:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            vox = binvox_rw.read_as_3d_array(f).data > 0.5
        rec = object_bytes(vox, size)
        for k, v in rec.items():
            recs.setdefault(k, []).append(v)

    labels = {
        'a_binvox': '(a) model.binvox file',
        'b_dense_bits': '(b) dense bit grid 32^3',
        'c_axis0_u8': '(c) single-axis dexel, axis 0 (uint8 hits, CSR)',
        'c_axis1_u8': '(c) single-axis dexel, axis 1 (uint8 hits, CSR)',
        'c_axis2_u8': '(c) single-axis dexel, axis 2 (uint8 hits, CSR)',
        'd_tri_u8': '(d) tri-dexel (sum of 3 axes, uint8)',
        'e_axis0_f32': '(e) single-axis dexel, axis 0 (float32 depths)',
        'e_axis1_f32': '(e) single-axis dexel, axis 1 (float32 depths)',
        'e_axis2_f32': '(e) single-axis dexel, axis 2 (float32 depths)',
        'e_tri_f32': '(e) tri-dexel (sum of 3 axes, float32 depths)',
        'f_slots_L2': '(f) dense slot tensor L=2, float32 (3,4,32,32)',
        'f_slots_L32': '(f) dense slot tensor L=32, float32 (3,64,32,32)',
    }
    base = np.mean(recs['a_binvox'])
    lines = ['', '## %s split, %d objects, measured %s' % (args.split, len(items), os.path.basename(args.out)),
             'ratio = mean / mean(a). All values in bytes.',
             '%-52s %10s %10s %10s %10s %8s' % ('quantity', 'mean', 'median', 'p95', 'max', 'ratio')]
    for k in labels:
        mean, med, p95, mx = stats(recs[k])
        lines.append('%-52s %10.1f %10.1f %10.1f %10.1f %8.3f'
                     % (labels[k], mean, med, p95, mx, mean / base))
    tri = np.mean(recs['d_tri_u8']) / base
    sing = np.mean(recs['c_axis0_u8']) / base
    lines.append('')
    lines.append('prediction check: single-axis (c, mean over axes) / (a) = %.3f; tri-dexel (d) / (a) = %.3f'
                 % (np.mean([np.mean(recs['c_axis%d_u8' % p]) for p in range(3)]) / base, tri))
    text = '\n'.join(lines) + '\n'
    with open(args.out, 'a') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
