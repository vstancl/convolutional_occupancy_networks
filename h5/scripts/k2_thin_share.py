"""H5-K2-1b thin-share diagnostic (CPU): how many query points of a point target lie in a thin GT segment.

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/k2_thin_share.py [--n-shapes 100 --step 100 --draws 20] \
        [--out h5/results/k2/thin_share.json]

Shapes: train shapes 0, step, 2 step, ... (``--n-shapes`` of them). Per shape ``--draws`` items of each target, item
``d`` of shape ``i`` drawn with ``Generator(PCG64(SeedSequence([i, d])))`` and the builder of the target
(``build_point_item`` for points_strong, ``build_rayint_item`` for points_rayint_a/b), 10 000 input points, noise 0.001.

Segment of a query point: along an axis, the line through the point parallel to that axis is cut at the analytic
interval ends of ``tree.json`` (``csgbox.tree.eval_rays``, the full line, ends clipped to the box), and the box
``[BOX_LO, BOX_HI]`` is cut at those ends. This is the same segment definition as ``data.rayint_segments``: material,
gaps and the outer segments to the box faces. The point's segment is the one containing its coordinate on the axis. A
point outside the box on that axis has no segment (counted as not thin).

- points_rayint_a/b: the axis of the ray the point was drawn from;
- points_strong: all three axes through the point, and the shortest of the three segments.

Per item: the share of the ``n_query`` points whose segment is shorter than h(128), and shorter than 4 h(128). The
table gives the mean over all items (shapes x draws).
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

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.constants import BOX_HI, BOX_LO  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402
from csgbox.tree import eval_rays, from_json  # noqa: E402

from data import N_QUERY, _rayint_sample, build_point_item, ray_fields, shape_dir_of, shape_ids  # noqa: E402

TARGETS: tuple[str, ...] = ("points_strong", "points_rayint_a", "points_rayint_b")
N_POINTS = 10000
NOISE = 0.001
H = cell_size(128)


def segment_lengths(tree, points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Length of the segment containing each point along its ``axis`` (inf when outside the box on that axis)."""
    p = np.asarray(points, dtype=np.float64)
    out = np.full(p.shape[0], np.inf)
    for a in range(3):
        sel = np.flatnonzero(axis == a)
        if not sel.size:
            continue
        origins = p[sel].copy()
        t = origins[:, a].copy()
        origins[:, a] = 0.0
        dirs = np.zeros_like(origins)
        dirs[:, a] = 1.0
        iv = eval_rays(tree, origins, dirs)
        ray = np.repeat(np.arange(sel.size), np.diff(iv.offsets))
        ends = np.clip(np.stack([iv.t0, iv.t1], axis=1).ravel(), BOX_LO, BOX_HI)
        ends_ray = np.repeat(ray, 2)
        # per ray: boundaries BOX_LO, the interval ends, BOX_HI; find the two around t with one sorted search
        key_b = ends_ray * 10.0 + ends  # |ends| <= 0.56 < 5, so rays do not overlap in the key
        order = np.argsort(key_b, kind="stable")
        key_b, ends_s, ends_ray_s = key_b[order], ends[order], ends_ray[order]
        inside = (t >= BOX_LO) & (t <= BOX_HI)
        key_t = np.arange(sel.size) * 10.0 + t
        k = np.searchsorted(key_b, key_t, side="right")  # first boundary after t (in the flat sorted list)
        lo = np.full(sel.size, BOX_LO)
        hi = np.full(sel.size, BOX_HI)
        has_lo = (k > 0) & (ends_ray_s[np.maximum(k - 1, 0)] == np.arange(sel.size))
        lo[has_lo] = ends_s[k[has_lo] - 1]
        has_hi = (k < ends_s.size) & (ends_ray_s[np.minimum(k, ends_s.size - 1)] == np.arange(sel.size))
        hi[has_hi] = ends_s[k[has_hi]]
        length = hi - lo
        out[sel] = np.where(inside, length, np.inf)
    return out


def item_lengths(target: str, index: int, draw: int, tree) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([index, draw])))
    shape_dir = shape_dir_of("train", index)
    if target == "points_strong":
        q = build_point_item(shape_dir, rng, target, N_POINTS, NOISE)["query"]
        per_axis = [segment_lengths(tree, q, np.full(q.shape[0], a)) for a in range(3)]
        return np.min(np.stack(per_axis), axis=0)
    item, extra = _rayint_sample(shape_dir, rng, target, N_POINTS, NOISE, N_QUERY, 512)
    axis = ray_fields(extra["g"])[0][extra["segments"]["row"][extra["pick"]]]
    return segment_lengths(tree, item["query"], axis)


def run(n_shapes: int, step: int, draws: int) -> dict:
    n_all = len(shape_ids("train"))
    shapes = [i * step for i in range(n_shapes) if i * step < n_all]
    acc = {t: {"lt_h": [], "lt_4h": []} for t in TARGETS}
    t0 = time.perf_counter()
    for index in shapes:
        tree = from_json((shape_dir_of("train", index) / "tree.json").read_text(encoding="utf-8"))
        for draw in range(draws):
            for target in TARGETS:
                length = item_lengths(target, index, draw, tree)
                acc[target]["lt_h"].append(float(np.mean(length < H)))
                acc[target]["lt_4h"].append(float(np.mean(length < 4 * H)))
    rows = {t: {"share_lt_h": float(np.mean(v["lt_h"])), "share_lt_4h": float(np.mean(v["lt_4h"])),
                "items": len(v["lt_h"])} for t, v in acc.items()}
    return {"shapes": [shape_ids("train")[i] for i in shapes], "n_shapes": len(shapes), "step": step,
            "draws": draws, "h": H, "seconds": time.perf_counter() - t0, "rows": rows}


def markdown(doc: dict) -> str:
    L = [f"Train shapes: {doc['n_shapes']} (every {doc['step']}th), {doc['draws']} draws each, {N_QUERY} query points "
         f"per item; h = h(128) = {doc['h']:.6g}. Mean over the items of the share of query points whose GT segment "
         "is shorter than the limit.", "",
         "| target | share < h | share < 4 h | items |", "|---|---|---|---|"]
    for t, r in doc["rows"].items():
        L.append(f"| {t} | {r['share_lt_h']:.4f} | {r['share_lt_4h']:.4f} | {r['items']} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-shapes", type=int, default=100)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--out", type=Path, default=H5_DIR / "results" / "k2" / "thin_share.json")
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
