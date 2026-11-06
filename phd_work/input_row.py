"""[phd] Input row for the dexel variants: occupancy superset from L dexel slots, as meshes.

For every object of the 3-class config's split:
  1. load model.binvox as VoxelsField does; vox = data > 0.5 (bool 32^3);
  2. slots = voxels_to_dexel_slots(vox, L);
  3. per plane p (ray axis RAY_AXIS[p]), per ray, the superset of the occupied voxels
     along that ray, from the kept slots:
       - ray with n <= L hits (not truncated): union of the [entry, exit] intervals;
       - truncated ray (n > L, kept first ceil(L/2) and last floor(L/2) hits):
           kept intervals that are entry-exit pairs in the real hit sequence,
           plus the dropped middle, i.e. the gap between the last kept hit of the
           first half and the first kept hit of the second half;
       - L = 1: from the first hit to the far end of the ray;
  4. occupancy = AND over the three planes (on the 32^3 voxel centres);
  5. mesh = VoxelGrid(occ).to_mesh(), written as generate.py writes voxel meshes:
     <out_dir>/generation/meshes/<class>/<model>.off  (the layout eval_meshes.py reads).

Truncation: the slots alone cannot tell n == L from n > L (both have all L masks set).
The hit count n of each ray is therefore taken from the voxel grid itself (the same
rule as phd_work/dexel_hits_hist.py). This is one bit per ray, not voxel information.

Checks (per object, counted over the split):
  - occupancy contains the original voxels (superset);
  - for L = 32, occupancy equals the original voxels (exact).
Reports the mean fraction of extra voxels: (occ & ~vox).sum() / vox.sum().

Usage:
    uv run python phd_work/input_row.py --L 4 --split test [--out-dir out/phd_input_L4]
Exit code 0 if the checks pass, 1 otherwise (meshes are still written).
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.config import load_config  # noqa: E402
from src.utils import binvox_rw  # noqa: E402
from src.utils.voxels import VoxelGrid  # noqa: E402
from phd_work.dexels import CENTRES, RAY_AXIS, RES, _ray_hits, voxels_to_dexel_slots  # noqa: E402


def plane_superset(vox, slots, plane, L):
    """Bool (RES, RES, RES) superset of the occupied voxels along one plane's rays."""
    hits, count = _ray_hits(vox, plane)  # count: true hit count per ray (for truncation)
    trunc = count > L
    n_rays = RES * RES
    depth = slots[plane, :L].astype(np.float64).reshape(L, n_rays).T  # (rays, L)
    mask = slots[plane, L:].reshape(L, n_rays).T > 0.5
    sup = np.zeros((n_rays, RES), dtype=bool)
    F = (L + 1) // 2  # kept hits in the first half
    Lh = L // 2       # kept hits in the second half

    for i in range(L - 1):
        valid = mask[:, i] & mask[:, i + 1]
        if i < F - 1:
            inside_trunc = (i % 2 == 0)           # real pair within the first half
        elif i == F - 1:
            inside_trunc = True                   # dropped middle
        else:
            k = i - F                             # interval inside the second half
            inside_trunc = ((Lh + k) % 2 == 0)    # real index n-Lh+k is even, n is even
        inside = np.where(trunc, inside_trunc, (i % 2 == 0))
        sel = valid & inside
        sup |= sel[:, None] & (CENTRES[None, :] >= depth[:, i:i + 1]) & (CENTRES[None, :] <= depth[:, i + 1:i + 2])

    if L == 1:
        # from the first hit to the far end of the ray
        sup |= (trunc & mask[:, 0])[:, None] & (CENTRES[None, :] >= depth[:, 0:1])

    sup = sup.reshape(RES, RES, RES)  # axes (u, v, ray)
    return np.moveaxis(sup, -1, RAY_AXIS[plane])


PLANE_BY_NAME = {'xz': 0, 'xy': 1, 'yz': 2}


def superset_occupancy(vox, L, planes=(0, 1, 2)):
    """AND over the given planes (default: all three). One plane: that plane's superset only."""
    slots = voxels_to_dexel_slots(vox, L)
    occ = None
    for p in planes:
        s = plane_superset(vox, slots, p, L)
        occ = s if occ is None else (occ & s)
    return occ


def load_vox(path):
    with open(path, 'rb') as f:
        return binvox_rw.read_as_3d_array(f).data.astype(np.float32) > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--planes', default='xz,xy,yz',
                    help='comma-separated planes to AND (default xz,xy,yz; e.g. xz for one plane)')
    ap.add_argument('--config', default=os.path.join(REPO, 'phd_work', 'configs', 'voxel_3plane_3cls.yaml'))
    args = ap.parse_args()
    L = args.L
    if L < 1 or L > RES or (L > 1 and L % 2):
        raise ValueError('L must be 1 or even, <= %d' % RES)
    planes = []
    for name in args.planes.split(','):
        name = name.strip()
        if name not in PLANE_BY_NAME:
            raise ValueError('--planes entries must be xz, xy or yz, got %r' % name)
        planes.append(PLANE_BY_NAME[name])
    planes = tuple(planes)
    tag = '' if planes == (0, 1, 2) else '_' + '_'.join(sorted(args.planes.split(','), key=['xz','xy','yz'].index))
    out_dir = args.out_dir or os.path.join(REPO, 'out', 'phd_input%s_L%d' % (tag, L))

    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        cfg = load_config(args.config, 'configs/default.yaml')
    finally:
        os.chdir(cwd)
    root = cfg['data']['path']
    if not os.path.isabs(root):
        root = os.path.join(REPO, root)
    classes = cfg['data']['classes']
    voxels_file = cfg['data']['voxels_file']
    mesh_root = os.path.join(out_dir, 'generation', 'meshes')

    items = []
    for c in classes:
        lst = os.path.join(root, c, args.split + '.lst')
        for m in open(lst).read().split('\n'):
            if m:
                items.append((c, m))

    n = 0
    n_superset_fail = 0
    n_exact_fail = 0
    extra_fracs = []
    t_convert = 0.0
    t0 = time.time()
    for c, m in items:
        vox = load_vox(os.path.join(root, c, m, voxels_file))
        ts = time.time()
        occ = superset_occupancy(vox, L, planes)
        t_convert += time.time() - ts
        n += 1
        if not np.all(occ[vox]):
            n_superset_fail += 1
        if L == RES and not np.array_equal(occ, vox):
            n_exact_fail += 1
        n_orig = int(vox.sum())
        if n_orig:
            extra_fracs.append(float((occ & ~vox).sum()) / n_orig)
        mesh = VoxelGrid(occ).to_mesh()
        mesh_dir = os.path.join(mesh_root, c)
        os.makedirs(mesh_dir, exist_ok=True)
        mesh.export(os.path.join(mesh_dir, '%s.off' % m))

    print('L=%d split=%s planes=%s objects=%d out_dir=%s' % (L, args.split, args.planes, n, out_dir))
    print('superset check (occ contains original voxels): failures=%d' % n_superset_fail)
    if L == RES:
        print('exact check (L=32 occ == original voxels): failures=%d' % n_exact_fail)
    print('mean extra-voxel fraction (extra / original occupied): %.4f' % np.mean(extra_fracs))
    print('mean superset time per object: %.2f ms (voxel conversion + plane supersets, excl. meshing)'
          % (1000.0 * t_convert / max(n, 1)))
    print('total time: %.1f s' % (time.time() - t0))
    ok = n_superset_fail == 0 and n_exact_fail == 0
    print('CHECKS %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
