"""P0: decoding time per shape of the ray decoders and of the point decoder with the ray protocol (H5-5b).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/time_decode.py --untrained --tag untrained
    .venv/bin/python h5/scripts/time_decode.py --checkpoint-dir DIR --tag pilot20k     # DIR/<method>.pt
    .venv/bin/python h5/scripts/time_decode.py --ckpt hits_8=PATH --ckpt points_strong=PATH

Per shape (batch 1, fp32, GPU): encoder; decoder (network, all 3 * 128^2 rays, in chunks); conversion (the
CPU decode to RayHits: ``decode_slots`` / ``decode_occfrac`` for the ray decoders; for points the protocol's
bisection, which calls the network, and the assembly of the hits). Every timed part is bracketed by
``torch.cuda.synchronize``. Methods without a checkpoint are untrained (seed 0) and labelled so.

Warm-up: the first ``WARMUP`` shapes of ``test.lst`` are not timed. The chunk size of each method is the fastest of
its candidates on the warm-up shapes. The timed shapes are the rest of the first ``--n-shapes``; the reported
numbers are medians over them. Each method is timed in the default mode and once more in deterministic mode.

Output: ``h5/results/p0/<tag>.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from data import DATA_ROOT, ray_fields, shape_ids  # noqa: E402
from eval_points import BISECT_STEPS, _decode_iso, _sample_probs, network_prob_fn, ray_geometry  # noqa: E402
from eval_rays import EVAL_THR, decode_by_axis, deterministic_kernels, shape_points  # noqa: E402
from models import build_model, load_trained  # noqa: E402

RAY_METHODS: tuple[str, ...] = ("hits_8", "hits_8_plane", "occfrac_128")
POINT_METHOD: str = "points_strong"
METHODS: tuple[str, ...] = RAY_METHODS + (POINT_METHOD,)
RAY_CHUNKS: tuple[int, ...] = (4096, 16384, 49152)  # rays per decoder call
POINT_CHUNKS: tuple[int, ...] = (256, 1024, 4096)  # rays per sampling chunk of the protocol
POINT_NS: tuple[int, ...] = (64, 128, 256, 512)  # samples per ray
WARMUP: int = 5  # shapes not timed (and used to pick the chunk size)
ISO: float = 0.5  # point protocol iso of the timing
OCC_SNAP_DEFAULT: float = 1e-2  # occfrac setting when no checkpoint setting is available (for information)
N_ALL: int = 3 * 128 * 128
RESULTS: Path = H5_DIR / "results" / "p0"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timer() -> float:
    _sync()
    return time.perf_counter()


def load_method(method: str, ckpt: Path | None, device: torch.device) -> tuple[torch.nn.Module, dict, str]:
    """Model, decoding setting, and the source label (checkpoint path or 'untrained')."""
    target = POINT_METHOD if method == POINT_METHOD else method
    if ckpt is None:
        torch.manual_seed(0)
        model = build_model(target).to(device)
        setting = {"iso": ISO} if method == POINT_METHOD else (
            {"thr": EVAL_THR} if method in ("hits_8", "hits_8_plane") else {"occ_snap": OCC_SNAP_DEFAULT})
        model.eval()
        return model, setting, "untrained"
    model, blob = load_trained(ckpt, device, target)  # with the run's model config (H5-K2-4)
    side = ckpt.with_suffix(".json")
    if side.exists():
        setting = json.loads(side.read_text(encoding="utf-8"))["setting"]
    else:
        setting = {"iso": ISO} if method == POINT_METHOD else {"thr": EVAL_THR}
    return model, setting, str(ckpt)


def time_ray(model, sid: str, chunk: int, setting: dict, device: torch.device, det: bool) -> dict:
    """Encoder, network decoder over all rays in chunks, and the conversion to RayHits, for one shape."""
    ray_axis, ray_index, uv = ray_fields(np.arange(N_ALL, dtype=np.int64))
    inputs = torch.as_tensor(shape_points(sid), device=device)[None]
    ctx = deterministic_kernels() if det else _nullcontext()
    with ctx, torch.no_grad():
        t0 = _timer()
        planes = model.encode(inputs)
        t1 = _timer()
        chunks: dict[str, list] = {}
        for c0 in range(0, N_ALL, chunk):
            sl = slice(c0, c0 + chunk)
            ax_t = torch.as_tensor(ray_axis[sl], device=device)[None]
            uv_t = torch.as_tensor(uv[sl], device=device)[None]
            out = model.decode(planes, ax_t, uv_t)
            for k, v in out.items():
                chunks.setdefault(k, []).append(v[0].cpu().numpy())
        t2 = _timer()
    outputs = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    thr = float(setting.get("thr", EVAL_THR))
    dec = decode_by_axis(model.target, outputs, ray_axis, setting)
    t3 = _timer()
    kept = int((1.0 / (1.0 + np.exp(-outputs["mask_logits"])) > thr).sum()) if "mask_logits" in outputs else None
    return {"encoder": t1 - t0, "decoder": t2 - t1, "conversion": t3 - t2, "total": t3 - t0,
            "kept_slots": kept, "decoded_hits": int(sum(int(d.offsets[-1]) for d in dec))}


def time_points(model, sid: str, n: int, chunk: int, device: torch.device, det: bool) -> dict:
    """Encoder, network sampling of N points per ray (all rays), and the bisection and assembly of the protocol."""
    ray_axis, ray_index, _uv = ray_fields(np.arange(N_ALL, dtype=np.int64))
    origins, dirs = ray_geometry(ray_axis, ray_index)
    inputs = torch.as_tensor(shape_points(sid), device=device)[None]
    ctx = deterministic_kernels() if det else _nullcontext()
    with ctx, torch.no_grad():
        t0 = _timer()
        planes = model.encode(inputs)
        t1 = _timer()
        prob = network_prob_fn(model, planes)
        probs = _sample_probs(prob, origins, dirs, n, torch.float32, device, chunk)
        t2 = _timer()
        hits = _decode_iso(probs, prob, origins, dirs, ISO, n, torch.float32, device, BISECT_STEPS)
        t3 = _timer()
    occ = probs > ISO
    changes = int((occ[:, :-1] != occ[:, 1:]).sum())
    return {"encoder": t1 - t0, "decoder": t2 - t1, "conversion": t3 - t2, "total": t3 - t0,
            "sign_changes": changes, "decoded_hits": int(hits.offsets[-1])}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _pick_chunk(run_fn, candidates: tuple[int, ...], warm: list[str], key: str) -> tuple[int, dict]:
    """Fastest candidate by the mean of ``key`` over the warm-up shapes (after one untimed call of each)."""
    timings = {}
    for c in candidates:
        run_fn(warm[0], c)  # untimed: kernels and allocations
        timings[c] = statistics.fmean(run_fn(s, c)[key] for s in warm)
    best = min(candidates, key=lambda c: timings[c])
    return best, {str(c): v for c, v in timings.items()}


def measure(method_label: str, method: str, model, setting: dict, device, shapes: list[str], which: str,
            n: int | None = None) -> dict:
    warm, timed = shapes[:WARMUP], shapes[WARMUP:]
    if which == "points":
        run = lambda sid, c, det=False: time_points(model, sid, n, c, device, det)  # noqa: E731
        candidates = POINT_CHUNKS
    else:
        run = lambda sid, c, det=False: time_ray(model, sid, c, setting, device, det)  # noqa: E731
        candidates = RAY_CHUNKS
    chunk, cand_timings = _pick_chunk(lambda sid, c: run(sid, c), candidates, warm, "decoder")
    rows = [run(sid, chunk) for sid in timed]
    rows_det = [run(sid, chunk, True) for sid in timed]
    out = {"label": method_label, "method": method, "chunk": chunk, "chunk_candidates_decoder_s": cand_timings,
           "shapes_timed": len(timed)}
    for key in ("encoder", "decoder", "conversion", "total"):
        out[key] = statistics.median(r[key] for r in rows)
    out["total_deterministic"] = statistics.median(r["total"] for r in rows_det)
    for key in ("kept_slots", "decoded_hits", "sign_changes"):
        if key in rows[0] and rows[0][key] is not None:
            out[key] = statistics.median(r[key] for r in rows)
    return out


def to_markdown(doc: dict) -> str:
    ms = lambda x: f"{1000 * x:.1f}"  # noqa: E731
    lines = [f"# P0: decoding time per shape ({doc['tag']})", ""]
    if doc["tag"] == "untrained" or doc["untrained_methods"]:
        lines += ["**Untrained, preliminary.** Untrained networks give outputs near 0.5, so the decoded-hit counts and "
                  "the conversion times are not those of trained decoders.", ""]
    lines += [f"- Shapes: first {doc['n_shapes']} of test.lst; warm-up {WARMUP} (not timed); "
              f"{doc['shapes_timed']} timed; median per shape, batch 1, fp32, "
              f"{doc['gpu']}.",
              "- Decoder = network over all 3 * 128^2 rays (points: sampling of N points per ray); conversion = "
              "CPU decode to RayHits (points: bisection of the sign changes, which calls the network, plus assembly).",
              f"- Untrained methods: {', '.join(doc['untrained_methods']) or 'none'}.", "",
              "| method | source | chunk | encoder ms | decoder ms | conversion ms | total ms | total, deterministic ms | "
              "sign changes | kept slots | decoded hits |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        lines.append(f"| {r['label']} | {r['source']} | {r['chunk']} | {ms(r['encoder'])} | {ms(r['decoder'])} | "
                     f"{ms(r['conversion'])} | {ms(r['total'])} | {ms(r['total_deterministic'])} | "
                     f"{r.get('sign_changes', '')} | {r.get('kept_slots', '')} | {r.get('decoded_hits', '')} |")
    lines += ["", "## Ratio total(points N) / total(method)", "",
              "| N | " + " | ".join(doc["ratio_methods"]) + " |", "|---|" + "---|" * len(doc["ratio_methods"])]
    for n in POINT_NS:
        row = next(r for r in doc["rows"] if r["label"] == f"points_strong N={n}")
        cells = []
        for m in doc["ratio_methods"]:
            other = next(r for r in doc["rows"] if r["label"] == m)
            cells.append(f"{row['total'] / other['total']:.2f}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    if any(r.get("sign_changes") == 0 for r in doc["rows"] if "sign_changes" in r):
        lines += ["", "Points: zero sign changes at iso 0.5 for the points rows marked untrained: the untrained point "
                  "decoder gives a near-constant probability, so the protocol's conversion (bisection) costs almost "
                  "nothing here. A trained decoder has a sign change at every surface crossing, so its conversion "
                  "time is larger; trained counts are in the pilot's timing."]
    if any(r.get("kept_slots") == 0 for r in doc["rows"] if r.get("kept_slots") is not None):
        lines += ["", "hits_8_plane: zero kept slots (untrained mask logits are negative), so its conversion is the "
                  "bare decoder cost."]
    lines += ["", "Chunk candidates (decoder seconds on the warm-up shapes) are in the JSON."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--untrained", action="store_true")
    src.add_argument("--checkpoint-dir", type=Path)
    src.add_argument("--ckpt", action="append", metavar="METHOD=PATH", default=None)
    p.add_argument("--n-shapes", type=int, default=100)
    p.add_argument("--tag", default=None)
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    args = p.parse_args(argv)
    device = torch.device("cuda")
    tag = args.tag or ("untrained" if args.untrained else "checkpoints")
    ckpts: dict[str, Path | None] = {m: None for m in METHODS}
    if args.checkpoint_dir is not None:
        for m in METHODS:
            f = args.checkpoint_dir / f"{m}.pt"
            if f.exists():
                ckpts[m] = f
    for item in args.ckpt or []:
        m, _, path = item.partition("=")
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}")
        ckpts[m] = Path(path).expanduser()

    shapes = shape_ids("test", DATA_ROOT)[: args.n_shapes]
    rows: list[dict] = []
    untrained: list[str] = []
    for m in RAY_METHODS:
        model, setting, source = load_method(m, ckpts[m], device)
        if source == "untrained":
            untrained.append(m)
        r = measure(m, m, model, setting, device, shapes, "ray")
        r["source"] = source
        rows.append(r)
        print(f"{m}: total {1000 * r['total']:.1f} ms, chunk {r['chunk']}", flush=True)
    model, setting, source = load_method(POINT_METHOD, ckpts[POINT_METHOD], device)
    if source == "untrained":
        untrained.append(POINT_METHOD)
    for n in POINT_NS:
        r = measure(f"points_strong N={n}", POINT_METHOD, model, setting, device, shapes, "points", n)
        r["source"] = source
        rows.append(r)
        print(f"points N={n}: total {1000 * r['total']:.1f} ms, chunk {r['chunk']}", flush=True)
    doc = {"tag": tag, "n_shapes": len(shapes), "warmup": WARMUP, "shapes_timed": len(shapes) - WARMUP,
           "gpu": torch.cuda.get_device_name(0), "rows": rows, "untrained_methods": untrained,
           "ratio_methods": list(RAY_METHODS)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{tag}.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (args.out_dir / f"{tag}.md").write_text(to_markdown(doc), encoding="utf-8")
    print(f"-> {args.out_dir / (tag + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
