"""Tests of the binarize decoding of occfrac (H5-K1-1b, h5/occ_binarize.py).

    cd ~/PhD/convonet && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest h5/tests/test_occ_binarize.py -q

a. a cell boundary: f = 1 on cells 3..4 of 8, 0 elsewhere: enter at the lower boundary of cell 3, exit at the upper
   boundary of cell 4 (exact).
b. a plane inside a cell, material on its upper side with fraction alpha: the enter (alpha >= 0.5 gives the hit
   between the empty and the partial cell) or the exit (alpha < 0.5 gives it between the partial and the full cell)
   is exactly the plane z0 = upper end - alpha h, for alpha on both sides of 0.5.
c. partial pair clipped to the span of the two cells, and enter/exit order along the ray.
d. an empty ray gives no hit; a ray full to both box ends gives an enter at the lower and an exit at the upper end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.constants import BOX_LO  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402

from occ_binarize import decode_occfrac_binarize  # noqa: E402

TOL = 1e-12


def _hits(dec, ray=0):
    a, b = int(dec.offsets[ray]), int(dec.offsets[ray + 1])
    return dec.t[a:b], dec.enter[a:b]


def test_a_block_of_full_cells_gives_its_boundaries():
    M = 8
    h = cell_size(M)
    f = np.zeros((1, M))
    f[0, 3:5] = 1.0
    t, enter = _hits(decode_occfrac_binarize(f))
    np.testing.assert_allclose(t, [BOX_LO + 3 * h, BOX_LO + 5 * h], atol=TOL)
    assert enter.tolist() == [True, False]


@pytest.mark.parametrize("alpha", [0.2, 0.45, 0.5, 0.6, 0.9])
def test_b_a_plane_inside_a_cell_is_recovered_exactly(alpha):
    # material is the upper part of cell k = 3 (fraction alpha); the plane is at upper end - alpha h
    M = 8
    h = cell_size(M)
    k = 3
    f = np.zeros((1, M))
    f[0, k] = alpha
    f[0, k + 1:] = 1.0  # material above the plane, to the end of the box
    t, enter = _hits(decode_occfrac_binarize(f))
    plane = BOX_LO + (k + 1) * h - alpha * h
    # enter on the plane, exit at the upper box end (the material runs to it)
    assert enter.tolist() == [True, False]
    assert t[0] == pytest.approx(plane, abs=TOL)
    assert t[1] == pytest.approx(BOX_LO + M * h, abs=TOL)


def test_c_partial_pair_is_clipped_and_ordered():
    M = 8
    h = cell_size(M)
    f = np.zeros((1, M))
    f[0, 2] = 0.3   # empty by the rule (0.3 < 0.5)
    f[0, 3] = 0.9   # occupied
    f[0, 4] = 0.0   # then empty again
    t, enter = _hits(decode_occfrac_binarize(f))
    # enter: between cells 2 and 3, end_a = BOX_LO + 3h, t = end_a + (1 - 0.3 - 0.9) h = end_a - 0.2 h
    # exit: between cells 3 and 4, end_a = BOX_LO + 4h, t = end_a + (0.9 + 0 - 1) h = end_a - 0.1 h
    np.testing.assert_allclose(t, [BOX_LO + 3 * h - 0.2 * h, BOX_LO + 4 * h - 0.1 * h], atol=TOL)
    assert enter.tolist() == [True, False]
    assert np.all(np.diff(t) > 0)
    # both hits lie in the span of their pair of cells
    assert BOX_LO + 2 * h <= t[0] <= BOX_LO + 4 * h
    assert BOX_LO + 3 * h <= t[1] <= BOX_LO + 5 * h


def test_d_empty_and_full_rays():
    M = 8
    h = cell_size(M)
    f = np.zeros((2, M))
    f[1, :] = 1.0  # full to both box ends
    dec = decode_occfrac_binarize(f)
    t0, e0 = _hits(dec, 0)
    t1, e1 = _hits(dec, 1)
    assert t0.size == 0
    # a full ray is one interval from the bottom to the top of the box
    np.testing.assert_allclose(t1, [BOX_LO, BOX_LO + M * h], atol=TOL)
    assert e1.tolist() == [True, False]
