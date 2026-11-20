"""Decoding of a binary occupancy column to hits (h5), the array version of csgbox's decode_binary.

``decode_binary_array(occ)`` is numpy-only apart from the ``RayHits`` container. It gives the same
hits, bit for bit, as ``csgbox.ceilings.decode_binary(gt, M)`` for ``occ = binary_occupancy(gt, M)``:
a transition between samples k and k+1 is a hit at their midpoint, ``enter`` is the transition
from outside to inside.
"""

from __future__ import annotations

import numpy as np

from _paths import add_external_paths

add_external_paths()
from csgbox.ceilings import RayHits  # noqa: E402

# Same literals as csgbox.constants.BOX_LO / BOX_HI; the test checks that they agree.
BOX_LO: float = -0.550005
BOX_HI: float = 0.550005


def decode_binary_array(occ: np.ndarray) -> RayHits:
    """Hits of an occupancy column ``occ`` (R, M) bool, sampled at the M cell centres of the box."""
    occ = np.asarray(occ, dtype=bool)
    if occ.ndim != 2:
        raise ValueError("occ must be (R, M)")
    R, M = occ.shape
    h = (BOX_HI - BOX_LO) / M  # same operation order as csgbox.grid.cell_size
    z = BOX_LO + (np.arange(M) + 0.5) * h  # same as csgbox.ceilings._sample_positions
    r, k = np.nonzero(occ[:, :-1] != occ[:, 1:])
    t = 0.5 * (z[k] + z[k + 1])
    enter = ~occ[r, k]
    counts = np.bincount(r, minlength=R).astype(np.int64)
    offsets = np.zeros(R + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return RayHits(offsets=offsets, t=t, enter=enter, normal=None)
