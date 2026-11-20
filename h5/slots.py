"""Two-sided interval slots for the h5 ray decoders: encoding of GT hits, and decoding of predictions.

A ray with ``n`` hits has ``k = n // 2`` intervals ``I_0 .. I_{k-1}`` (interval i = hits 2i, 2i+1).
With ``K`` slots per side:

- front slot ``j`` holds ``I_j`` if ``j < k``;
- back slot ``j`` holds ``I_{k-1-j}`` if ``j < k``.

The two sides are filled independently, so with ``k < 2K`` the same interval can fill a front and a
back slot (duplicates are allowed). Layout of the outputs:

- ``t`` (R, 4K): ``[F0.enter, F0.exit, ..., F_{K-1}.enter, F_{K-1}.exit,``
  ``B_{K-1}.enter, B_{K-1}.exit, ..., B_0.enter, B_0.exit]``;
- ``mask`` (R, 2K): ``[F0, ..., F_{K-1}, B_{K-1}, ..., B_0]``.

Missing slots have ``t = 0`` and ``mask = False``.
"""

from __future__ import annotations

import numpy as np

from _paths import add_external_paths

add_external_paths()
from csgbox.ceilings import RayHits as _RayHits  # noqa: E402


def encode_slots(gt: _RayHits, K: int) -> tuple[np.ndarray, np.ndarray]:
    """``(t, mask)`` of the two-sided slots of every ray of ``gt`` (see the module docstring)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    R = gt.n_rays
    counts = np.diff(gt.offsets)
    k = counts // 2
    j = np.arange(K)
    valid_f = j[None, :] < k[:, None]  # (R, K): front slot j valid
    valid_b = valid_f  # back slot j valid under the same condition
    iv_b = k[:, None] - 1 - j[None, :]  # interval index held by back slot j
    base = gt.offsets[:-1][:, None]

    t = np.zeros((R, 4 * K), dtype=np.float64)
    mask = np.zeros((R, 2 * K), dtype=bool)
    if gt.t.size == 0:
        return t, mask
    last = gt.t.size - 1

    def gather(iv: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = base + 2 * np.where(valid, iv, 0)
        e = np.minimum(e, last)
        x = np.minimum(e + 1, last)
        return np.where(valid, gt.t[e], 0.0), np.where(valid, gt.t[x], 0.0)

    fe, fx = gather(np.broadcast_to(j, (R, K)), valid_f)
    be, bx = gather(iv_b, valid_b)
    t[:, 0:2 * K:2] = fe
    t[:, 1:2 * K:2] = fx
    bcols = 2 * K + 2 * (K - 1 - j)  # enter column of back slot j
    t[:, bcols] = be
    t[:, bcols + 1] = bx
    mask[:, :K] = valid_f
    mask[:, K + (K - 1 - j)] = valid_b
    return t, mask


def _interval_union(E: np.ndarray, X: np.ndarray, keep: np.ndarray) -> _RayHits:
    """Union of the intervals ``[E, X]`` where ``keep`` (R, C), per ray, as a CSR ``RayHits``.

    Touching intervals (``E <= running max of exits``) are merged.
    """
    R, C = E.shape
    E = np.where(keep, E, np.inf)  # invalid intervals sort last
    X = np.where(keep, X, -np.inf)
    order = np.argsort(E, axis=1, kind="stable")
    Es = np.take_along_axis(E, order, axis=1)
    Xs = np.take_along_axis(X, order, axis=1)
    Vs = np.isfinite(Es)
    RM = np.maximum.accumulate(np.where(Vs, Xs, -np.inf), axis=1)
    prev = np.concatenate([np.full((R, 1), -np.inf), RM[:, :-1]], axis=1)
    start = Vs & (Es > prev)
    nxt_start = np.concatenate([start[:, 1:], np.zeros((R, 1), bool)], axis=1)
    nxt_valid = np.concatenate([Vs[:, 1:], np.zeros((R, 1), bool)], axis=1)
    end = Vs & (~nxt_valid | nxt_start)
    s_r, _ = np.nonzero(start)  # row-major: starts and ends pair up in order
    e_val = Es[start]
    x_val = RM[end]
    counts = np.bincount(s_r, minlength=R).astype(np.int64)
    offsets = np.zeros(R + 1, dtype=np.int64)
    np.cumsum(2 * counts, out=offsets[1:])
    t = np.empty(int(offsets[-1]), dtype=np.float64)
    enter = np.zeros(int(offsets[-1]), dtype=bool)
    t[0::2] = e_val
    t[1::2] = x_val
    enter[0::2] = True
    return _RayHits(offsets=offsets, t=t, enter=enter, normal=None)


def decode_slots(t: np.ndarray, mask_prob: np.ndarray, K: int, thr: float = 0.5, merge_iou: float = 0.5) -> _RayHits:
    """Predicted slots (``t`` (R, 4K), ``mask_prob`` (R, 2K)) to ``RayHits``.

    1. Keep an interval if its slot probability is above ``thr`` and ``enter < exit``.
    2. A front and a back interval with 1-D IoU >= ``merge_iou`` are replaced by one interval
       whose endpoints are the averages of the two. Matching is greedy by IoU, one interval
       in at most one pair, over all ray-vectorized rounds (at most K).
    3. The remaining intervals are unioned per ray (overlapping or touching intervals merge).
    """
    t = np.asarray(t, dtype=np.float64)
    mask_prob = np.asarray(mask_prob, dtype=np.float64)
    R = t.shape[0]
    if t.shape != (R, 4 * K) or mask_prob.shape != (R, 2 * K):
        raise ValueError("t must be (R, 4K) and mask_prob (R, 2K)")
    j = np.arange(K)
    bcols = 2 * K + 2 * (K - 1 - j)
    mcols_b = K + (K - 1 - j)
    fe = t[:, 0:2 * K:2].copy()
    fx = t[:, 1:2 * K:2].copy()
    be = t[:, bcols].copy()
    bx = t[:, bcols + 1].copy()
    keep_f = (mask_prob[:, :K] > thr) & (fe < fx)
    keep_b = (mask_prob[:, mcols_b] > thr) & (be < bx)
    used_f = np.zeros((R, K), dtype=bool)
    used_b = np.zeros((R, K), dtype=bool)

    # 1-D IoU of every front and back interval of a ray
    inter = np.clip(np.minimum(fx[:, :, None], bx[:, None, :]) - np.maximum(fe[:, :, None], be[:, None, :]), 0.0, None)
    union = (fx - fe)[:, :, None] + (bx - be)[:, None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    pair_ok = keep_f[:, :, None] & keep_b[:, None, :]

    for _ in range(K):
        cand = np.where(pair_ok & ~used_f[:, :, None] & ~used_b[:, None, :], iou, -1.0).reshape(R, K * K)
        best = cand.argmax(axis=1)
        best_v = cand.max(axis=1)
        rows = np.flatnonzero(best_v >= merge_iou)
        if rows.size == 0:
            break
        j1 = best[rows] // K
        j2 = best[rows] % K
        fe[rows, j1] = 0.5 * (fe[rows, j1] + be[rows, j2])
        fx[rows, j1] = 0.5 * (fx[rows, j1] + bx[rows, j2])
        used_f[rows, j1] = True
        used_b[rows, j2] = True
        keep_b[rows, j2] = False  # merged into the front interval

    E = np.concatenate([fe, be], axis=1)
    X = np.concatenate([fx, bx], axis=1)
    keep = np.concatenate([keep_f, keep_b], axis=1)
    return _interval_union(E, X, keep)
