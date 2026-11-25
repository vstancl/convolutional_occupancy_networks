"""Torch Dataset of one shape for the ray decoders (h5 step H5-1b).

One item is one shape: ``n_points`` noisy surface points (the encoder input) and ``n_rays`` sampled
rays with their targets. Only the requested target is built. Targets come from the dexel file
``dexels_128.npz`` (float32 ``t``) through the ``csgbox.ceilings`` encoders, applied to the sampled
rays only (``rays.subset_rays``).

Ray sampling (per item): each of the ``n_rays`` rays is, with probability 1/2, uniform over all
``3 * 128**2`` rays, and otherwise uniform over the rays with at least two hits (the three axes
pooled). A ray may be drawn twice.

Randomness: one ``numpy.random.Generator`` per item. With ``seed`` given it is seeded by
``SeedSequence([seed, index])``, so a fixed seed gives identical items. With ``seed=None`` it is
seeded by ``torch.initial_seed()``, which a DataLoader worker sets per worker and per epoch, so the
items are then not reproducible across runs.

Input points are plain numpy (not ConvONet's ``PointCloudField``): that field adds noise from the
global RNG, and the h5 dataset needs the explicit Generator.

Importing this module calls ``_paths.add_external_paths()`` (csgbox and dexio on sys.path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from _paths import add_external_paths

add_external_paths()

import dexio  # noqa: E402
from csgbox import ceilings as ceil  # noqa: E402
from csgbox.constants import BOX_HI, BOX_LO, GATE_TSDF_TAU_CELLS  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402
from csgbox.ceilings import RayHits  # noqa: E402

from rays import AXIS_LETTERS, rayhits_from_arrays, subset_rays  # noqa: E402
from slots import encode_slots  # noqa: E402

DATA_ROOT: Path = Path.home() / "PhD" / "data" / "csgbox" / "v1"
POINTCLOUD_FILE: str = "pointcloud.npz"
DEXELS_FILE: str = "dexels_128.npz"

M: int = 128  # samples per ray of the dense targets, and dexel resolution
N_AXIS: int = M * M  # rays per axis
N_ALL: int = 3 * N_AXIS  # rays of a shape, axes pooled

# hits targets: K intervals per side (4K hits), see h5/slots.py
HITS_K: dict[str, int] = {"hits_8": 2, "hits_16": 4}
DENSE_TARGETS: tuple[str, ...] = ("occfrac_128", "tsdf_128", "binary_128")
TARGETS: tuple[str, ...] = tuple(HITS_K) + DENSE_TARGETS

# Probability that a sampled ray is drawn from all rays (else from the rays with >= 2 hits).
P_UNIFORM: float = 0.5
# The second half of the sampler draws from the rays with at least this many hits.
MULTI_HIT_MIN: int = 2


def _check_target(target: str) -> None:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")


def read_shape_axes(shape_dir: Path) -> tuple[RayHits, RayHits, RayHits]:
    """The three axes of ``dexels_128.npz`` as float64 ``RayHits``; the file is read once."""
    dex = dexio.read(Path(shape_dir) / DEXELS_FILE)
    return tuple(rayhits_from_arrays(dex.axes[letter]) for letter in AXIS_LETTERS)  # type: ignore[return-value]


def ray_counts(axes) -> np.ndarray:
    """Hits per ray, (3 * N^2,) int64, axes concatenated in order x, y, z."""
    return np.concatenate([np.diff(a.offsets) for a in axes]).astype(np.int64)


def sample_global_rays(counts: np.ndarray, rng: np.random.Generator, n_rays: int) -> np.ndarray:
    """Global ray ids ``g = axis * N^2 + r`` drawn by the mixture sampler (see the module docstring)."""
    multi = np.flatnonzero(counts >= MULTI_HIT_MIN)
    if multi.size == 0:
        raise ValueError("shape has no ray with at least two hits")
    uniform = rng.integers(0, counts.size, size=n_rays)
    from_multi = multi[rng.integers(0, multi.size, size=n_rays)]
    use_uniform = rng.random(n_rays) < P_UNIFORM
    return np.where(use_uniform, uniform, from_multi).astype(np.int64)


def ray_fields(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ray_axis, ray_index, ray_origin_uv)`` of global ray ids ``g``.

    ``ray_index = i_col + N * i_row``. The (col, row) axes are (y, z) for x-rays, (x, z) for y-rays
    and (x, y) for z-rays (dexelize SPEC section 2). ``ray_origin_uv`` is the centre of the cell
    ``(col, row)`` on those two axes, ``lo + (k + 1/2) h``, float32.
    """
    h = cell_size(M)
    ray_axis = g // N_AXIS
    ray_index = g % N_AXIS
    col = ray_index % M
    row = ray_index // M
    uv = np.stack([BOX_LO + (col + 0.5) * h, BOX_LO + (row + 0.5) * h], axis=1).astype(np.float32)
    return ray_axis.astype(np.int64), ray_index.astype(np.int64), uv


def _per_axis(axes, ray_axis: np.ndarray, ray_index: np.ndarray):
    """Yield ``(axis, rows, subset)``: the sampled rays of one axis as a ``RayHits`` (row order kept)."""
    for a in range(3):
        rows = np.flatnonzero(ray_axis == a)
        if rows.size:
            yield a, rows, subset_rays(axes[a], ray_index[rows])


def hits_targets(axes, ray_axis, ray_index, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Slots of the sampled rays: ``s`` (n, 4K) float32 in cell units, ``mask`` (n, 2K) bool.

    ``s = (t - BOX_LO) / h`` with ``h = cell_size(128)``. For a masked-out slot ``t = 0``, so its
    ``s`` is the value of ``t = 0`` and must be ignored (the mask is the only valid information).
    """
    h = cell_size(M)
    if K not in HITS_K.values():
        raise ValueError(f"K must be one of {sorted(HITS_K.values())}, got {K}")
    n = ray_axis.size
    s = np.zeros((n, 4 * K), dtype=np.float64)
    mask = np.zeros((n, 2 * K), dtype=bool)
    for _a, rows, sub in _per_axis(axes, ray_axis, ray_index):
        t, m = encode_slots(sub, K)
        s[rows] = (t - BOX_LO) / h
        mask[rows] = m
    return s.astype(np.float32), mask


def dense_values(axes, ray_axis, ray_index, target: str) -> np.ndarray:
    """Dense column of the sampled rays: float32 (n, 128) for occfrac/tsdf, bool for binary."""
    n = ray_axis.size
    dtype = np.bool_ if target == "binary_128" else np.float32
    y = np.zeros((n, M), dtype=dtype)
    for _a, rows, sub in _per_axis(axes, ray_axis, ray_index):
        if target == "occfrac_128":
            vals = ceil.occ_fraction(sub, M)
        elif target == "tsdf_128":
            vals, _z, _h = ceil.tsdf_values(sub, M)
        else:
            vals = ceil.binary_occupancy(sub, M)
        y[rows] = vals
    return y


def informative(y: np.ndarray, target: str) -> np.ndarray:
    """Bool (n, 128): the cells that carry the information of the target (the loss is weighted on them).

    occfrac: ``0 < y < 1``. binary: the cells next to a change of occupancy (both sides).
    tsdf: ``|y| < 2.5 h`` (not clamped; the clamp value is ``float32(2.5 h)`` exactly).
    Computed on the stored float32 values, so the rule is the one the loss sees.
    """
    if target == "occfrac_128":
        return (y > 0) & (y < 1)
    if target == "tsdf_128":
        thr = np.float32(GATE_TSDF_TAU_CELLS * cell_size(M))
        return np.abs(y) < thr
    if target == "binary_128":
        n = y.shape[0]
        trans = y[:, 1:] != y[:, :-1]  # (n, 127): change between cell k and k+1
        tp = np.zeros((n, M + 1), dtype=bool)
        tp[:, 1:M] = trans  # tp[:, k] is the change on the left of cell k
        return tp[:, :-1] | tp[:, 1:]
    raise ValueError(f"no informative rule for {target!r}")


def build_item(shape_dir: Path, rng: np.random.Generator, target: str, n_rays: int,
               n_points: int, noise: float) -> dict[str, np.ndarray]:
    """One item of the dataset from ``shape_dir``, with all randomness drawn from ``rng``."""
    _check_target(target)
    if noise < 0:
        raise ValueError("noise must be >= 0")
    with np.load(Path(shape_dir) / POINTCLOUD_FILE) as pc:
        points = pc["points"]
        idx = rng.choice(points.shape[0], size=n_points, replace=False)
        pts = points[idx].astype(np.float64)
    pts = pts + rng.normal(0.0, noise, size=pts.shape)
    axes = read_shape_axes(shape_dir)
    g = sample_global_rays(ray_counts(axes), rng, n_rays)
    ray_axis, ray_index, uv = ray_fields(g)
    item: dict[str, np.ndarray] = {
        "inputs": pts.astype(np.float32),
        "ray_axis": ray_axis,
        "ray_index": ray_index,
        "ray_origin_uv": uv,
    }
    if target in HITS_K:
        s, mask = hits_targets(axes, ray_axis, ray_index, HITS_K[target])
        item["hits_s"] = s
        item["hits_mask"] = mask
    else:
        y = dense_values(axes, ray_axis, ray_index, target)
        item["y"] = y
        item["informative"] = informative(y, target)
    return item


def fixed_val_rays(shape_index: int, n: int = 4096, split: str = "val",
                   root: Path = DATA_ROOT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` fixed rays of validation shape ``shape_index`` (position in ``split.lst``).

    The same mixture sampler, with ``Generator(seed=shape_index)``, so validation always sees the
    same rays. Returns ``(ray_axis, ray_index, ray_origin_uv)``.
    """
    shape_dir = shape_dir_of(split, shape_index, root)
    axes = read_shape_axes(shape_dir)
    rng = np.random.default_rng(shape_index)
    g = sample_global_rays(ray_counts(axes), rng, n)
    return ray_fields(g)


def shape_ids(split: str, root: Path = DATA_ROOT) -> list[str]:
    """Model ids of ``csg/{split}.lst`` in file order."""
    return (Path(root) / "csg" / f"{split}.lst").read_text(encoding="utf-8").split()


def shape_dir_of(split: str, index: int, root: Path = DATA_ROOT) -> Path:
    return Path(root) / "csg" / shape_ids(split, root)[index]


class RayShapeDataset(Dataset):
    """One item per shape of ``csg/{split}.lst``; see the module docstring.

    Items are dicts of numpy arrays (torch's default collate turns them into tensors):
    ``inputs`` (n_points, 3) float32; ``ray_axis`` (n_rays,) int64; ``ray_index`` (n_rays,) int64;
    ``ray_origin_uv`` (n_rays, 2) float32; and for ``hits_K``: ``hits_s`` (n_rays, 4K) float32 and
    ``hits_mask`` (n_rays, 2K) bool; for the dense targets: ``y`` (n_rays, 128) float32 (bool for
    binary) and ``informative`` (n_rays, 128) bool.
    """

    def __init__(self, split: str, target: str, n_rays: int = 1024, n_points: int = 10000,
                 noise: float = 0.001, seed: int | None = None, root: Path = DATA_ROOT):
        _check_target(target)
        if seed is not None and seed < 0:
            raise ValueError("seed must be >= 0")
        self.split = split
        self.target = target
        self.n_rays = n_rays
        self.n_points = n_points
        self.noise = noise
        self.seed = seed
        self.root = Path(root)
        self.ids = shape_ids(split, self.root)

    def __len__(self) -> int:
        return len(self.ids)

    def rng_for(self, index: int) -> np.random.Generator:
        """The Generator of item ``index`` (see the module docstring)."""
        if self.seed is not None:
            return np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        return np.random.default_rng(np.random.SeedSequence([int(torch.initial_seed()) % (2**63), index]))

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        shape_dir = self.root / "csg" / self.ids[index]
        return build_item(shape_dir, self.rng_for(index), self.target, self.n_rays, self.n_points, self.noise)


# ---------------------------------------------------------------------------------------------------
# Point decoders (H5-5a): query points with occupancy, for ConvONet's point-wise decoder.
# ---------------------------------------------------------------------------------------------------

POINT_TARGETS: tuple[str, ...] = ("points_std", "points_strong")
POINTS_FILE: str = "points.npz"  # uniform in the box, packed occupancies (ConvONet's points.npz)
SURFACE_POINTS_FILE: str = "points_surface.npz"  # near the surface; sigma 0.002 and 0.01, half each
N_QUERY: int = 2048  # query points per shape (ConvONet points_subsample)


def read_occupancy_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``points`` (P, 3) float32 and ``occ`` (P,) float32 in {0, 1} of a ConvONet-style point file.

    The occupancies are packed bits (``np.packbits``, big bit order); unpacked as ConvONet does
    (``np.unpackbits(...)[:P]``).
    """
    with np.load(path) as z:
        points = z["points"].astype(np.float32)
        occ = np.unpackbits(z["occupancies"])[: points.shape[0]].astype(np.float32)
    return points, occ


def build_point_item(shape_dir: Path, rng: np.random.Generator, target: str, n_points: int, noise: float,
                     n_query: int = N_QUERY) -> dict[str, np.ndarray]:
    """One point-decoder item: the encoder input (as ``build_item``) and ``n_query`` query points with occupancy.

    ``points_std``: all query points from ``points.npz``. ``points_strong``: ``n_query // 2`` from
    ``points.npz`` and ``n_query // 2`` from ``points_surface.npz`` (both sigma halves of the stored file).
    ``from_surface`` (bool) marks the points drawn from the surface file (all False for ``points_std``).
    """
    if target not in POINT_TARGETS:
        raise ValueError(f"target must be one of {POINT_TARGETS}, got {target!r}")
    with np.load(Path(shape_dir) / POINTCLOUD_FILE) as pc:
        cloud = pc["points"]
        idx = rng.choice(cloud.shape[0], size=n_points, replace=False)
        pts = cloud[idx].astype(np.float64)
    pts = pts + rng.normal(0.0, noise, size=pts.shape)
    uni_p, uni_o = read_occupancy_points(Path(shape_dir) / POINTS_FILE)
    if target == "points_std":
        idx = rng.choice(uni_p.shape[0], size=n_query, replace=False)
        query, occ = uni_p[idx], uni_o[idx]
        from_surface = np.zeros(n_query, dtype=bool)
    else:
        half = n_query // 2
        i_u = rng.choice(uni_p.shape[0], size=n_query - half, replace=False)
        sur_p, sur_o = read_occupancy_points(Path(shape_dir) / SURFACE_POINTS_FILE)
        i_s = rng.choice(sur_p.shape[0], size=half, replace=False)
        query = np.concatenate([uni_p[i_u], sur_p[i_s]])
        occ = np.concatenate([uni_o[i_u], sur_o[i_s]])
        from_surface = np.concatenate([np.zeros(n_query - half, dtype=bool), np.ones(half, dtype=bool)])
    return {"inputs": pts.astype(np.float32), "query": query.astype(np.float32),
            "occ": occ.astype(np.float32), "from_surface": from_surface}


# ---------------------------------------------------------------------------------------------------
# Ray-interval query points (H5-K2-1a): query points for the point decoder from the GT intervals of rays.
# ---------------------------------------------------------------------------------------------------

RAYINT_TARGETS: tuple[str, ...] = ("points_rayint_a", "points_rayint_b")
RAYINT_N_RAYS: int = 512  # sampled rays per item
RAYINT_MIN_SEGMENT: float = 1e-9  # segments shorter than this are dropped


def rayint_segments(axes, start_inside: tuple[np.ndarray, np.ndarray, np.ndarray],
                    g: np.ndarray) -> dict[str, np.ndarray]:
    """The segments of the rays ``g``: the box ``[BOX_LO, BOX_HI]`` cut at the hits ``t_0 < ... < t_{k-1}``.

    Occupancy alternates from outside: the first segment of a ray is outside, then every hit toggles. This needs
    ``start_inside == 0`` for every sampled ray, which is asserted (it holds in csgbox dataset v1). Segments
    shorter than ``RAYINT_MIN_SEGMENT`` are dropped. Returns float64 ``start``, ``end``, bool ``occ`` and int64
    ``row`` (the position of the ray in ``g``) per segment, rays in the order of ``g``.
    """
    ray_axis, ray_index, _uv = ray_fields(g)
    starts, ends, occs, rows = [], [], [], []
    for a in range(3):
        sel = np.flatnonzero(ray_axis == a)
        if not sel.size:
            continue
        r = ray_index[sel]
        if np.any(start_inside[a][r] != 0):
            raise AssertionError("a sampled ray has start_inside != 0; the alternation from outside does not hold")
        sub = subset_rays(axes[a], r)
        n_seg = np.diff(sub.offsets) + 1  # k hits cut the box into k + 1 segments
        ray_of_seg = np.repeat(np.arange(sel.size), n_seg)
        first_seg = np.concatenate([[0], np.cumsum(n_seg)[:-1]])
        j = np.arange(ray_of_seg.size) - first_seg[ray_of_seg]  # index of the segment within its ray
        lo = np.insert(sub.t, sub.offsets[:-1], BOX_LO)  # per ray: BOX_LO, t_0, ..., t_{k-1}
        hi = np.insert(sub.t, sub.offsets[1:], BOX_HI)  # per ray: t_0, ..., t_{k-1}, BOX_HI
        starts.append(lo)
        ends.append(hi)
        occs.append(j % 2 == 1)
        rows.append(sel[ray_of_seg])
    start, end = np.concatenate(starts), np.concatenate(ends)
    occ, row = np.concatenate(occs), np.concatenate(rows)
    order = np.argsort(row, kind="stable")  # rays in the order of g, segments in t order within a ray
    start, end, occ, row = start[order], end[order], occ[order], row[order]
    keep = (end - start) >= RAYINT_MIN_SEGMENT
    return {"start": start[keep], "end": end[keep], "occ": occ[keep], "row": row[keep]}


def _rayint_sample(shape_dir: Path, rng: np.random.Generator, target: str, n_points: int, noise: float,
                   n_query: int, n_rays: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``build_rayint_item`` and, for the tests, the sampled rays, their segments and the segment of every point."""
    if target not in RAYINT_TARGETS:
        raise ValueError(f"target must be one of {RAYINT_TARGETS}, got {target!r}")
    with np.load(Path(shape_dir) / POINTCLOUD_FILE) as pc:  # encoder input: as build_point_item, rng first
        cloud = pc["points"]
        idx = rng.choice(cloud.shape[0], size=n_points, replace=False)
        pts = cloud[idx].astype(np.float64)
    pts = pts + rng.normal(0.0, noise, size=pts.shape)
    dex = dexio.read(Path(shape_dir) / DEXELS_FILE)
    axes = tuple(rayhits_from_arrays(dex.axes[letter]) for letter in AXIS_LETTERS)
    start_inside = tuple(np.asarray(dex.axes[letter].start_inside) for letter in AXIS_LETTERS)
    g = sample_global_rays(ray_counts(axes), rng, n_rays)
    seg = rayint_segments(axes, start_inside, g)
    length = seg["end"] - seg["start"]
    n_seg = length.size
    if target == "points_rayint_a":
        pick = rng.integers(0, n_seg, size=n_query)  # every segment equally likely
    else:
        pick = rng.choice(n_seg, size=n_query, replace=True, p=length / length.sum())  # proportional to length
    t = rng.uniform(seg["start"][pick], seg["end"][pick])
    ray_axis, _ray_index, uv = ray_fields(g)
    row = seg["row"][pick]
    a = ray_axis[row]
    query = np.empty((n_query, 3), dtype=np.float64)
    for axis in range(3):
        col_axis, row_axis = [b for b in range(3) if b != axis]  # (y, z) for x-rays, (x, z) for y, (x, y) for z
        m = a == axis
        query[m, axis] = t[m]
        query[m, col_axis] = uv[row[m], 0]
        query[m, row_axis] = uv[row[m], 1]
    item = {"inputs": pts.astype(np.float32), "query": query.astype(np.float32),
            "occ": seg["occ"][pick].astype(np.float32), "from_surface": np.zeros(n_query, dtype=bool)}
    extra = {"g": g, "segments": seg, "pick": pick, "t": t}
    return item, extra


def build_rayint_item(shape_dir: Path, rng: np.random.Generator, target: str, n_points: int, noise: float,
                      n_query: int = N_QUERY, n_rays: int = RAYINT_N_RAYS) -> dict[str, np.ndarray]:
    """One point-decoder item with query points from the GT intervals of sampled rays (H5-K2-1a).

    Same keys as ``build_point_item``: ``inputs`` (n_points, 3) float32, ``query`` (n_query, 3) float32, ``occ``
    (n_query,) float32 in {0, 1}, ``from_surface`` (n_query,) bool, all False. Order of the draws from ``rng``:

    1. the encoder input exactly as ``build_point_item`` (subset of ``pointcloud.npz``, then the noise);
    2. ``n_rays`` rays with the mixture sampler (``sample_global_rays``) over ``dexels_128.npz``;
    3. ``n_query`` segments with replacement (``rayint_segments``): ``points_rayint_a`` uniform over all segments of
       all sampled rays, ``points_rayint_b`` with probability proportional to the length (points uniform along the
       sampled rays, the control);
    4. ``t`` uniform in each chosen segment.

    The point is the cell centre of its ray on the two transverse axes (``ray_fields``) and ``t`` on the ray axis;
    ``occ`` is the occupancy of the segment.
    """
    return _rayint_sample(shape_dir, rng, target, n_points, noise, n_query, n_rays)[0]


class PointShapeDataset(Dataset):
    """One item per shape of ``csg/{split}.lst``, as ``RayShapeDataset``, for the point decoders.

    ``POINT_TARGETS`` use ``build_point_item``, ``RAYINT_TARGETS`` (H5-K2-1a) ``build_rayint_item``.

    Same encoder input and the same per-item seeding (``seed`` gives ``SeedSequence([seed, index])``).
    Items: ``inputs`` (n_points, 3), ``query`` (n_query, 3), ``occ`` (n_query,), ``from_surface`` (n_query,) bool.
    """

    def __init__(self, split: str, target: str, n_points: int = 10000, noise: float = 0.001,
                 seed: int | None = None, root: Path = DATA_ROOT):
        if target not in POINT_TARGETS + RAYINT_TARGETS:
            raise ValueError(f"target must be one of {POINT_TARGETS + RAYINT_TARGETS}, got {target!r}")
        self.split, self.target, self.n_points, self.noise, self.seed = split, target, n_points, noise, seed
        self.root = Path(root)
        self.ids = shape_ids(split, self.root)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if self.seed is not None:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        else:
            rng = np.random.default_rng(np.random.SeedSequence([int(torch.initial_seed()) % (2**63), index]))
        build = build_rayint_item if self.target in RAYINT_TARGETS else build_point_item
        return build(self.root / "csg" / self.ids[index], rng, self.target, self.n_points, self.noise)
