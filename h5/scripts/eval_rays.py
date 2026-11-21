"""Ray-space evaluation of one ray decoder (H5-3): summary JSON and markdown table.

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/eval_rays.py --target hits_8 --untrained --split val \
        --n-shapes 200 --out ~/PhD/data/csgbox/v1/h5_eval/hits_8_untrained.json

    ... --checkpoint PATH           a file written by torch.save({"target": T, "state_dict": ...})
    ... --all-rays                  every ray of every shape (default: the fixed 4096 validation rays)
    ... --occ-snap 1e-5             occupancy snapping for occfrac_128 (default: off, the literal rule)

Writes ``OUT.json`` (the aggregate summary) and ``OUT.md`` (the table of ``eval_rays.to_markdown``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from data import TARGETS, fixed_val_rays, shape_ids  # noqa: E402
from eval_rays import EVAL_CHUNK_RAYS, EVAL_THR, aggregate, evaluate_shape, to_markdown  # noqa: E402
from models import H5RayModel  # noqa: E402


def load_model(target: str, checkpoint: str | None, seed: int, device: torch.device) -> H5RayModel:
    if checkpoint is None:
        torch.manual_seed(seed)
        return H5RayModel(target).to(device)
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    if blob.get("target") != target:
        raise ValueError(f"checkpoint is for {blob.get('target')!r}, not {target!r}")
    model = H5RayModel(target).to(device)
    model.load_state_dict(blob["state_dict"])
    return model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, choices=sorted(set(TARGETS) | {"hits_8_plane"}))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str)
    src.add_argument("--untrained", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="torch seed of an untrained model")
    p.add_argument("--split", default="val")
    p.add_argument("--n-shapes", type=int, default=200)
    p.add_argument("--n-rays", type=int, default=4096, help="rays per shape when not --all-rays")
    p.add_argument("--all-rays", action="store_true")
    p.add_argument("--thr", type=float, default=EVAL_THR)
    p.add_argument("--occ-snap", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    model = load_model(args.target, args.checkpoint, args.seed, device)
    ids = shape_ids(args.split)[: args.n_shapes]
    t0 = time.perf_counter()
    results: list[dict] = []
    for i, sid in enumerate(ids):
        if args.all_rays:
            rays = "all"
        else:
            ray_axis, ray_index, _uv = fixed_val_rays(i, n=args.n_rays, split=args.split)
            rays = (ray_axis, ray_index)
        results.extend(evaluate_shape(model, sid, args.split, rays=rays, thr=args.thr, device=device,
                                      chunk_rays=EVAL_CHUNK_RAYS, occ_snap=args.occ_snap))
    wall = time.perf_counter() - t0
    summary, _md = aggregate(results)
    summary.update({"target": args.target, "split": args.split, "n_shapes": len(ids),
                    "rays": "all" if args.all_rays else f"fixed {args.n_rays} per shape",
                    "checkpoint": args.checkpoint or "untrained", "seed": args.seed, "thr": args.thr,
                    "occ_snap": args.occ_snap, "wall_seconds": wall})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    title = f"Ray-space evaluation: {args.target} ({args.split}, {len(ids)} shapes, {summary['rays']})"
    args.out.with_suffix(".md").write_text(to_markdown(summary, title=title), encoding="utf-8")
    print(f"{args.target}: {len(ids)} shapes, {wall:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
