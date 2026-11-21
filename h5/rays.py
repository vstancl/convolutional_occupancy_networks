"""Ray utilities for h5: load one axis of a dexel file as RayHits, and gather a subset of rays.

Importing this module calls ``_paths.add_external_paths()`` (csgbox and dexio on sys.path).
Every h5 module expects the ``h5/`` directory itself to be on ``sys.path``.
"""

from __future__ import annotations

import numpy as np

from _paths import add_external_paths

add_external_paths()

import dexio  # noqa: E402
from csgbox.ceilings import RayHits  # noqa: E402

AXIS_LETTERS: tuple[str, str, str] = ("x", "y", "z")


def rayhits_from_arrays(arrays) -> RayHits:
    """``RayHits`` of one ``dexio`` axis (``DexelFile.axes[letter]``), with float64 ``t`` and normals."""
    normal = None if arrays.normal is None else arrays.normal.astype(np.float64)
    return RayHits(
        offsets=arrays.offsets.astype(np.int64),
        t=arrays.t.astype(np.float64),
        enter=arrays.enter.astype(bool),
        normal=normal,
    )


def load_axis_rayhits(npz_path, axis: int) -> RayHits:
    """Hits of one ray axis of ``dexels_128.npz`` as a ``RayHits``.

    ``t`` is float32 in the file and is returned as float64, ``enter`` (uint8) as bool, and the
    normals (float32) as float64. Ray order is the file's (``r = i_col + N * i_row``).
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
    dex = dexio.read(npz_path)
    return rayhits_from_arrays(dex.axes[AXIS_LETTERS[axis]])


def subset_rays(gt: RayHits, rays: np.ndarray) -> RayHits:
    """The rays ``rays`` of ``gt`` as a new ``RayHits``, in the order given by ``rays``.

    A ray may appear more than once. Vectorized: one gather of the hit indices, no loop over rays.
    """
    rays = np.asarray(rays, dtype=np.int64)
    if rays.ndim != 1:
        raise ValueError("rays must be a 1-D index array")
    if rays.size and (rays.min() < 0 or rays.max() >= gt.n_rays):
        raise ValueError("ray index out of range")
    starts = gt.offsets[:-1][rays]
    counts = gt.offsets[1:][rays] - starts
    offsets = np.zeros(rays.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    within = np.arange(int(offsets[-1])) - np.repeat(offsets[:-1], counts)
    idx = np.repeat(starts, counts) + within
    normal = None if gt.normal is None else gt.normal[idx]
    return RayHits(offsets=offsets, t=gt.t[idx], enter=gt.enter[idx], normal=normal)
