"""Binarize decoding of occupancy fractions (H5-K1-1b): a fair decoder for ``occfrac_M`` outputs.

Rule (chosen for K1-1b, documented in h5/docs/k1-1b.md):

- a cell is occupied iff its fraction ``f >= OCC_BIN_THRESHOLD`` (0.5); the box ends count as empty;
- at every change between neighbouring samples ``a`` (before) and ``b`` (after), along the ray, one hit is
  placed, with ``end_a`` the upper end of cell ``a``, ``h`` the cell size and ``f_a``, ``f_b`` the fractions:

  - outside -> inside (enter):  ``t = end_a + (1 - f_a - f_b) h``
  - inside -> outside (exit):   ``t = end_a + (f_a + f_b - 1) h``

  clipped to ``[start_a, end_b]`` (the span of the two cells). For a single plane inside a cell this places the
  hit exactly on the plane (tests/test_occ_binarize.py), so one surface gives an enter and an exit at the right
  places. A pair of cells with fractions exactly 0.5 can give two hits at the same ``t`` (a zero-length interval).

The output is a ``RayHits`` (csgbox), in the same layout as ``csgbox.ceilings.decode_occfrac``.
"""

from __future__ import annotations

import numpy as np

from csgbox.ceilings import RayHits, _csr
from csgbox.constants import BOX_LO
from csgbox.grid import cell_size

OCC_BIN_THRESHOLD: float = 0.5  # a cell is occupied iff f >= this (the K1-1b rule)


def decode_occfrac_binarize(f: np.ndarray) -> RayHits:
    """Binarize decoding of the fractions ``f`` (R, M) of ``occfrac_M``. See the module docstring."""
    f = np.asarray(f, dtype=np.float64)
    R, M = f.shape
    h = cell_size(M)
    # padded index q covers cell q - 1, which spans [BOX_LO + (q - 1) h, BOX_LO + q h]; the pads are empty
    fp = np.zeros((R, M + 2), dtype=np.float64)
    fp[:, 1:-1] = f
    s = fp >= OCC_BIN_THRESHOLD
    r, p = np.nonzero(s[:, :-1] != s[:, 1:])  # change between padded p and p + 1
    fa = fp[r, p]
    fb = fp[r, p + 1]
    end_a = BOX_LO + p * h
    enter = s[r, p + 1]  # outside -> inside
    t = np.where(enter, end_a + (1.0 - fa - fb) * h, end_a + (fa + fb - 1.0) * h)
    t = np.clip(t, end_a - h, end_a + h)  # [start_a, end_b]
    order = np.lexsort((t, r))
    return _csr(r[order], t[order], enter[order], R)
