"""[phd] Voxel -> dexel slot conversion for the dexel encoder (numpy only, CPU).

A 32^3 occupancy grid is turned into three ray-slot images, one per ConvONet plane:

    plane 0 'xz': rays run along voxel axis 1; image indices are voxel axes (0, 2)
    plane 1 'xy': rays run along voxel axis 2; image indices are voxel axes (0, 1)
    plane 2 'yz': rays run along voxel axis 0; image indices are voxel axes (1, 2)

Coordinates follow src/encoder/voxels.py: voxel centres c_i = linspace(-0.5, 0.5, 32)[i],
cell size h = 1/31. A run of occupied voxels i0..i1 on a ray gives an entry hit at
c_i0 - h/2 and an exit hit at c_i1 + h/2. Each run therefore contributes two hits, and
hits are sorted by depth along the ray.

Slot layout for one plane, for a given L (even, or 1):
    slots[plane, s, u, v]         for s < L:   depth of slot s (0 if empty)
    slots[plane, L + s, u, v]     for s < L:   mask of slot s (1 = slot holds a hit)
If a ray has more than L hits, the first ceil(L/2) and the last floor(L/2) are kept
(L=1: only the first hit). Otherwise all hits are kept, in order, in the first slots.
Only with L = 32 (the maximum hit count of a 32-voxel ray) is the conversion lossless.
"""
import numpy as np

RES = 32
H = 1.0 / (RES - 1)
CENTRES = np.linspace(-0.5, 0.5, RES)
# plane index -> voxel axis the rays run along (ConvONet planes xz, xy, yz)
RAY_AXIS = (1, 2, 0)


def _check_L(L):
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)) or not 1 <= L <= RES:
        raise ValueError('L must be an integer in [1, %d], got %r' % (RES, L))
    if L > 1 and L % 2:
        raise ValueError('L must be 1 or even, got %d' % L)


def _ray_hits(occ, plane):
    """Sorted hit depths per ray for one plane.

    Returns (hits, count): hits has shape (RES*RES, RES) float64, row u*RES+v holds the
    depths of ray (u, v) in order (entry, exit, entry, exit, ...); count has shape
    (RES*RES,) with the number of hits per ray.
    """
    x = np.moveaxis(occ, RAY_AXIS[plane], -1)  # axes (u, v, k): k runs along the ray
    occ_rays = x.reshape(-1, RES)
    prev = np.zeros_like(occ_rays)
    prev[:, 1:] = occ_rays[:, :-1]
    nxt = np.zeros_like(occ_rays)
    nxt[:, :-1] = occ_rays[:, 1:]
    starts = occ_rays & ~prev
    ends = occ_rays & ~nxt
    run_idx = np.cumsum(starts, axis=1) - 1  # run containing each occupied voxel

    n_rays = occ_rays.shape[0]
    hits = np.zeros((n_rays, RES), dtype=np.float64)
    rows, cols = np.nonzero(starts)
    hits[rows, 2 * run_idx[rows, cols]] = CENTRES[cols] - H / 2
    rows, cols = np.nonzero(ends)
    hits[rows, 2 * run_idx[rows, cols] + 1] = CENTRES[cols] + H / 2
    count = 2 * starts.sum(axis=1)
    return hits, count


def _select(hits, count, L):
    """Keep the first ceil(L/2) and last floor(L/2) hits of rays with more than L hits."""
    n_rays = hits.shape[0]
    first = (L + 1) // 2
    depth = np.zeros((n_rays, L), dtype=np.float64)
    mask = np.zeros((n_rays, L), dtype=bool)
    rows = np.arange(n_rays)
    for s in range(L):
        src = np.where(count <= L, s, np.where(s < first, s, count - L + s))
        valid = s < count
        src = np.clip(src, 0, RES - 1)
        depth[:, s] = np.where(valid, hits[rows, src], 0.0)
        mask[:, s] = valid
    return depth, mask


def voxels_to_dexel_slots(vox, L):
    """Convert a 32^3 occupancy grid to dexel slots of shape (3, 2*L, 32, 32), float32.

    Args:
        vox: bool or 0/1 array of shape (32, 32, 32), as loaded by VoxelsField.
        L: slots per ray; 1 or even, 1 <= L <= 32.
    """
    _check_L(L)
    vox = np.asarray(vox)
    if vox.shape != (RES, RES, RES):
        raise ValueError('vox must have shape (%d, %d, %d), got %s' % (RES, RES, RES, vox.shape))
    occ = vox > 0.5
    out = np.zeros((3, 2 * L, RES, RES), dtype=np.float32)
    for p in range(3):
        hits, count = _ray_hits(occ, p)
        depth, mask = _select(hits, count, L)
        out[p, :L] = depth.T.reshape(L, RES, RES)
        out[p, L:] = mask.T.reshape(L, RES, RES)
    return out


def dexel_slots_to_voxels(slots, plane=0):
    """Reconstruct the occupancy grid from one plane of dexel slots (testing only).

    Pairs slots (2j, 2j+1) as (entry, exit) and marks every voxel whose centre lies in
    [entry, exit]. Exact only when no hits were dropped, i.e. slots built with L = 32.

    Args:
        slots: array of shape (3, 2*L, 32, 32) with L even.
        plane: 0 ('xz'), 1 ('xy') or 2 ('yz').
    Returns:
        bool array of shape (32, 32, 32) in the original voxel axis order.
    """
    slots = np.asarray(slots)
    if plane not in (0, 1, 2):
        raise ValueError('plane must be 0, 1 or 2, got %r' % (plane,))
    if slots.ndim != 4 or slots.shape[0] != 3 or slots.shape[2:] != (RES, RES):
        raise ValueError('slots must have shape (3, 2*L, %d, %d)' % (RES, RES))
    if slots.shape[1] % 2 or slots.shape[1] < 2:
        raise ValueError('slots channel count must be 2*L with L even')
    L = slots.shape[1] // 2
    if L % 2:
        raise ValueError('reconstruction needs even L, got %d' % L)
    depth = slots[plane, :L].astype(np.float64)
    mask = slots[plane, L:] > 0.5

    out = np.zeros((RES, RES, RES), dtype=bool)  # axes (u, v, k)
    centres = CENTRES[None, None, :]
    for j in range(L // 2):
        ok = mask[2 * j] & mask[2 * j + 1]
        lo = depth[2 * j][..., None]
        hi = depth[2 * j + 1][..., None]
        out |= (centres >= lo) & (centres <= hi) & ok[..., None]
    return np.moveaxis(out, -1, RAY_AXIS[plane])
