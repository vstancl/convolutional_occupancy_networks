"""Ray protocol for the point decoders (H5-5a): occupancy samples along each ray, sign changes bisected.

A point decoder gives an occupancy probability at any point. For a set of rays (origin, unit direction
along the ray axis, z = coordinate along the axis):

1. ``N`` uniform samples per ray at ``z_k = BOX_LO + (k + 1/2) (BOX_HI - BOX_LO) / N``;
   the occupancy is ``probability > iso``;
2. every sign change between consecutive samples ``k, k+1`` is refined by ``BISECT_STEPS`` bisection steps
   on ``[z_k, z_{k+1}]`` (the side that keeps the state change is kept); the hit is the midpoint of the final
   bracket; ``enter`` is True for an outside-to-inside change;
3. boundary hits (a convention of this protocol, not of the gate): if the first sample is inside, an enter
   hit at ``BOX_LO`` opens the ray's first interval; if the last sample is inside, an exit hit at ``BOX_HI``
   closes the last one. Without them the hit list would start with an exit.

Everything runs in torch on the device of the probability function, in ray chunks. The sampling is done once
and shared by all iso values; the bisection is per iso. The output is one ``RayHits`` for all rays, in the
input order of the rays.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from _paths import add_external_paths

add_external_paths()

from csgbox.ceilings import RayHits  # noqa: E402
from csgbox.constants import BOX_HI, BOX_LO  # noqa: E402
from csgbox.grid import cell_size, remaining_axes  # noqa: E402

PROTOCOL_N: int = 512  # samples per ray (validation; the gate's points_512)
BISECT_STEPS: int = 8  # bisection steps per sign change (task spec)
RAY_CHUNK: int = 1024  # rays per sampling chunk (bounds the points in memory: 1024 * 512)
BISECT_CHUNK: int = 1 << 18  # sign changes per bisection batch
GRID_N: int = 128  # the dexel grid of the rays (cells per axis); ray origins are its cell centres

# ProbFn: points (1, T, 3) -> occupancy probabilities (1, T)
ProbFn = Callable[[torch.Tensor], torch.Tensor]


def ray_geometry(ray_axis: np.ndarray, ray_index: np.ndarray, n: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
    """Origins (R, 3) float64 and directions (R, 3) of rays given by axis and ``ray_index`` (= i_col + n i_row).

    The origin has axis coordinate 0 and the two other coordinates at the cell centres of the n x n grid
    (``BOX_LO + (k + 1/2) h``), the same as ``csgbox.grid.ray_grid``. A point of the ray is
    ``origin + z * dir``, so ``z`` is the axis coordinate.
    """
    ray_axis = np.asarray(ray_axis, dtype=np.int64)
    ray_index = np.asarray(ray_index, dtype=np.int64)
    h = cell_size(n)
    centres = BOX_LO + (np.arange(n, dtype=np.float64) + 0.5) * h
    col_i = ray_index % n
    row_i = ray_index // n
    origins = np.zeros((ray_axis.size, 3))
    dirs = np.zeros((ray_axis.size, 3))
    for a in range(3):
        rows = np.flatnonzero(ray_axis == a)
        col, row = remaining_axes(a)
        origins[rows, col] = centres[col_i[rows]]
        origins[rows, row] = centres[row_i[rows]]
        dirs[rows, a] = 1.0
    return origins, dirs


def network_prob_fn(model, planes: dict) -> ProbFn:
    """Occupancy probability of a ``PointModel`` for the plane features of one shape (batch 1)."""

    def fn(points: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(model.decode(planes, points.float()))

    return fn


def _sample_probs(prob_fn: ProbFn, origins: np.ndarray, dirs: np.ndarray, n: int, dtype: torch.dtype,
                  device: torch.device, chunk: int = RAY_CHUNK) -> torch.Tensor:
    """Probabilities (R, n) float32 at the N uniform samples of every ray."""
    R = origins.shape[0]
    h = (BOX_HI - BOX_LO) / n
    z = torch.as_tensor(BOX_LO + (np.arange(n, dtype=np.float64) + 0.5) * h, dtype=dtype, device=device)
    probs = torch.empty((R, n), dtype=torch.float32, device=device)
    with torch.no_grad():
        for c0 in range(0, R, chunk):
            c1 = min(c0 + chunk, R)
            o = torch.as_tensor(origins[c0:c1], dtype=dtype, device=device)
            d = torch.as_tensor(dirs[c0:c1], dtype=dtype, device=device)
            pts = o[:, None, :] + z[None, :, None] * d[:, None, :]  # (c, n, 3)
            p = prob_fn(pts.reshape(1, -1, 3)).reshape(c1 - c0, n)
            probs[c0:c1] = p.float()
    return probs


def _decode_iso(probs: torch.Tensor, prob_fn: ProbFn, origins: np.ndarray, dirs: np.ndarray, iso: float,
                n: int, dtype: torch.dtype, device: torch.device, bisect: int) -> RayHits:
    R = probs.shape[0]
    occ = probs > iso
    h = (BOX_HI - BOX_LO) / n
    z = torch.as_tensor(BOX_LO + (np.arange(n, dtype=np.float64) + 0.5) * h, dtype=dtype, device=device)
    r, k = torch.nonzero(occ[:, :-1] != occ[:, 1:], as_tuple=True)
    state_lo = occ[r, k]  # inside (True) or outside before the change
    lo_all = z[k]
    hi_all = z[k + 1]
    o_all = torch.as_tensor(origins, dtype=dtype, device=device)
    d_all = torch.as_tensor(dirs, dtype=dtype, device=device)
    ts = []
    with torch.no_grad():
        for c0 in range(0, r.numel(), BISECT_CHUNK):
            sl = slice(c0, c0 + BISECT_CHUNK)
            rr, st = r[sl], state_lo[sl]
            lo, hi = lo_all[sl].clone(), hi_all[sl].clone()
            o, d = o_all[rr], d_all[rr]
            for _ in range(bisect):
                mid = 0.5 * (lo + hi)
                inside = prob_fn((o + mid[:, None] * d)[None])[0] > iso
                same = inside == st
                lo = torch.where(same, mid, lo)
                hi = torch.where(same, hi, mid)
            ts.append(0.5 * (lo + hi))
    t_change = torch.cat(ts) if ts else torch.zeros(0, dtype=dtype, device=device)
    enter_change = ~state_lo

    first_in = torch.nonzero(occ[:, 0], as_tuple=True)[0]
    last_in = torch.nonzero(occ[:, -1], as_tuple=True)[0]
    ray = torch.cat([r, first_in, last_in])
    key = torch.cat([k, torch.full_like(first_in, -1), torch.full_like(last_in, n + 1)])
    t = torch.cat([t_change, torch.full((first_in.numel(),), BOX_LO, dtype=dtype, device=device),
                   torch.full((last_in.numel(),), BOX_HI, dtype=dtype, device=device)])
    enter = torch.cat([enter_change, torch.ones(first_in.numel(), dtype=torch.bool, device=device),
                       torch.zeros(last_in.numel(), dtype=torch.bool, device=device)])
    # ray-major, then along the ray: keys are -1 (first boundary), 0..n-1 (changes), n+1 (last boundary);
    # key + 1 lies in [0, n + 2], so the multiplier is n + 3 (a multiplier of n + 2 let keys of neighbouring rays collide)
    order = torch.argsort(ray * (n + 3) + (key + 1))
    ray, t, enter = ray[order], t[order], enter[order]
    counts = torch.bincount(ray, minlength=R)
    offsets = np.zeros(R + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts.cpu().numpy())
    return RayHits(offsets=offsets, t=t.double().cpu().numpy(), enter=enter.cpu().numpy().astype(bool), normal=None)


def ray_protocol(prob_fn: ProbFn, origins: np.ndarray, dirs: np.ndarray, isos: list[float], n: int = PROTOCOL_N,
                 bisect: int = BISECT_STEPS, dtype: torch.dtype = torch.float32,
                 device: torch.device | None = None, chunk: int = RAY_CHUNK) -> list[RayHits]:
    """One ``RayHits`` (all rays, input order) per iso value, from one sampling pass."""
    device = device or torch.device("cpu")
    probs = _sample_probs(prob_fn, origins, dirs, n, dtype, device, chunk)
    return [_decode_iso(probs, prob_fn, origins, dirs, iso, n, dtype, device, bisect) for iso in isos]
