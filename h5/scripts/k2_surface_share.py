"""H5-K2'-1 sampling diagnostic (CPU): how close the query points of each sampler lie to the GT surface, and (for the
two mixtures) how many lie in a thin GT segment.

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/k2_surface_share.py [--n-shapes 100 --step 100 --draws 20] \
        [--out h5/results/k2p/sampling_share.json]

Items: exactly those of ``k2_thin_share.py``: train shapes 0, step, 2 step, ... (``--n-shapes``), ``--draws`` items
per shape and target, item ``d`` of shape ``i`` drawn with ``Generator(PCG64(SeedSequence([i, d])))``, 10 000 input
points, noise 0.001, 2048 query points, 512 rays per item. Targets: points_strong, points_rayint_a/b/c,
points_mix_a/b.

Surface distance of a query point (the same definition for every sampler): for each axis, the ray of that axis
through the point's cell (the 128^3 grid; the two transverse cell indices of the point, clipped to the grid), and the
distance along the axis from the point's coordinate to the nearest GT hit of that ray (``dexels_128.npz``; inf if the
ray has no hit); the minimum over the three axes. For points drawn on a ray (rayint and the ray half of the mixtures)
the point lies on the cell centre of its own ray, so its own axis gives the exact distance along it; points that are
not on a cell centre are measured along the rays through their cell centre.

Thin share (mixtures only; the other four targets are in ``h5/results/k2/thin_share.md``): the length of the GT
segment containing the point, as ``k2_thin_share.segment_lengths``: the near-surface half as points_strong (the
shortest of the three axes), the ray half along the axis of its ray.

Per item: the share of query points with surface distance < 0.002 and < h/2, and (mixtures) with segment length < h
and < 4 h. The tables give the means over all items.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

import dexio  # noqa: E402
from csgbox.constants import BOX_LO  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402
from csgbox.tree import from_json  # noqa: E402

from data import (  # noqa: E402
    AXIS_LETTERS,
    DEXELS_FILE,
    M,
    MIX_TARGETS,
    N_QUERY,
    _rayint_sample,
    build_point_item,
    ray_fields,
    shape_dir_of,
    shape_ids,
)
from k2_thin_share import N_POINTS, NOISE, segment_lengths  # noqa: E402
from rays import rayhits_from_arrays  # noqa: E402

TARGETS: tuple[str, ...] = ("points_strong", "points_rayint_a", "points_rayint_b", "points_rayint_c",
                            "points_mix_a", "points_mix_b")
H = cell_size(M)
LIMITS = {"lt_0002": 0.002, "lt_half_h": H / 2.0}
N_RAYS = 512
KEY_SPAN = 10.0  # |t| <= 0.56 < 5: the rays do not overlap in the key ray * KEY_SPAN + t


class SurfaceIndex:
    """The GT hits of one shape per axis, keyed for a nearest-hit search along the ray through a cell."""

    def __init__(self, shape_dir: Path):
        dex = dexio.read(shape_dir / DEXELS_FILE)
        self.keys = []
        for letter in AXIS_LETTERS:
            hits = rayhits_from_arrays(dex.axes[letter])
            ray = np.repeat(np.arange(hits.offsets.size - 1), np.diff(hits.offsets))
            self.keys.append((ray, hits.t, ray * KEY_SPAN + hits.t))  # sorted: rays in order, t increasing

    def distance(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        best = np.full(p.shape[0], np.inf)
        for a in range(3):
            col_axis, row_axis = [b for b in range(3) if b != a]
            col = np.clip(np.floor((p[:, col_axis] - BOX_LO) / H).astype(np.int64), 0, M - 1)
            row = np.clip(np.floor((p[:, row_axis] - BOX_LO) / H).astype(np.int64), 0, M - 1)
            ray_q = col + M * row
            ray, t, key = self.keys[a]
            if not t.size:
                continue
            k = np.searchsorted(key, ray_q * KEY_SPAN + p[:, a], side="left")
            succ, pred = np.minimum(k, t.size - 1), np.maximum(k - 1, 0)
            d_succ = np.where((k < t.size) & (ray[succ] == ray_q), np.abs(t[succ] - p[:, a]), np.inf)
            d_pred = np.where((k >= 1) & (ray[pred] == ray_q), np.abs(t[pred] - p[:, a]), np.inf)
            best = np.minimum(best, np.minimum(d_succ, d_pred))
        return best


def draw(target: str, index: int, draw_i: int) -> tuple[dict, dict | None]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([index, draw_i])))
    shape_dir = shape_dir_of("train", index)
    if target == "points_strong":
        return build_point_item(shape_dir, rng, target, N_POINTS, NOISE), None
    return _rayint_sample(shape_dir, rng, target, N_POINTS, NOISE, N_QUERY, N_RAYS)


def mix_lengths(tree, item: dict, extra: dict) -> np.ndarray:
    n_s = extra["n_surface"]
    q = item["query"]
    sur = np.min(np.stack([segment_lengths(tree, q[:n_s], np.full(n_s, a)) for a in range(3)]), axis=0)
    axis = ray_fields(extra["g"])[0][extra["segments"]["row"][extra["pick"]]]
    return np.concatenate([sur, segment_lengths(tree, q[n_s:], axis)])


def run(n_shapes: int, step: int, draws: int) -> dict:
    n_all = len(shape_ids("train"))
    shapes = [i * step for i in range(n_shapes) if i * step < n_all]
    acc = {t: {k: [] for k in (*LIMITS, "thin_lt_h", "thin_lt_4h")} for t in TARGETS}
    t0 = time.perf_counter()
    for index in shapes:
        surf = SurfaceIndex(shape_dir_of("train", index))
        tree = from_json((shape_dir_of("train", index) / "tree.json").read_text(encoding="utf-8"))
        for d in range(draws):
            for target in TARGETS:
                item, extra = draw(target, index, d)
                dist = surf.distance(item["query"])
                for k, lim in LIMITS.items():
                    acc[target][k].append(float(np.mean(dist < lim)))
                if target in MIX_TARGETS:
                    length = mix_lengths(tree, item, extra)
                    acc[target]["thin_lt_h"].append(float(np.mean(length < H)))
                    acc[target]["thin_lt_4h"].append(float(np.mean(length < 4 * H)))
    rows = {}
    for t, v in acc.items():
        rows[t] = {f"share_{k}": (float(np.mean(x)) if x else None) for k, x in v.items()}
        rows[t]["items"] = len(v["lt_0002"])
    return {"shapes": [shape_ids("train")[i] for i in shapes], "n_shapes": len(shapes), "step": step, "draws": draws,
            "h": H, "limits": LIMITS, "seconds": time.perf_counter() - t0, "rows": rows}


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def markdown(doc: dict) -> str:
    L = ["# H5-K2' sampling shares", "",
         f"Train shapes: {doc['n_shapes']} (every {doc['step']}th), {doc['draws']} draws each (the items of "
         f"k2_thin_share.py), {N_QUERY} query points per item; h = h(128) = {doc['h']:.6g}. Means over the items.", "",
         "Surface distance, the same definition for every sampler: per axis, the distance along that axis from the "
         "point to the nearest GT hit of the axis ray through the point's cell (dexels_128.npz); the minimum over the "
         "three axes. Thin share: the length of the GT segment containing the point (as k2_thin_share.py; the "
         "near-surface half of a mixture as points_strong, the ray half along its ray); for the other targets see "
         "h5/results/k2/thin_share.md.", "",
         "| target | surface distance < 0.002 | surface distance < h/2 | segment < h | segment < 4 h | items |",
         "|---|---|---|---|---|---|"]
    for t, r in doc["rows"].items():
        L.append(f"| {t} | {_f(r['share_lt_0002'])} | {_f(r['share_lt_half_h'])} | {_f(r['share_thin_lt_h'])} | "
                 f"{_f(r['share_thin_lt_4h'])} | {r['items']} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-shapes", type=int, default=100)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "k2p" / "sampling_share.json")
    args = p.parse_args(argv)
    doc = run(args.n_shapes, args.step, args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    md = markdown(doc)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md + f"({doc['seconds']:.0f} s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
