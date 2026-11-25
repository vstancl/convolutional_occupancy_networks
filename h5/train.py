"""Training loop of the ray decoders (H5-4).

    cd ~/PhD/convonet && .venv/bin/python h5/train.py h5/configs/hits_8.yaml [--resume] [--max-it N]

Configuration: ``h5/configs/default.yaml`` is read first, then the target file overrides it (see the
config keys in ``h5/docs/h5-4.md``). Nothing is random except through explicit seeds:

- sample ``g`` (global index) of the run is shape ``perm_epoch[g mod n]`` with ``perm_epoch`` a
  ``Generator(SeedSequence([seed, 1, epoch]))`` permutation of the training shapes, and its rays and points
  come from ``Generator(SeedSequence([seed, 2, g]))``. Iteration ``i`` uses samples ``i B .. i B + B - 1``.
  So a resumed run draws the same batches as an uninterrupted one, and the number of DataLoader workers
  does not change the data.

Decoding setting of a validation (``_select``, H5-K1-1a): best F1 at delta 0.002 among the settings whose decoded/GT
ratio is in ``RATIO_BAND``; if none is, the setting whose ratio is closest to the band. Each validation records the
chosen setting, its ratio and whether it is in the band.

Files in ``out_dir``: ``config.json``; ``log.jsonl`` (every ``log_every`` iterations: loss parts, s/it, peak
GPU memory); ``val.jsonl`` (every validation: the summary of every decoding setting); ``model_last.pt`` (model,
optimizer, iteration, best F1 and RNG state, written at every validation, used by ``--resume``);
``model_best.pt`` (the best validation F1 over the grid, with its decoding setting); ``model_it{N}.pt`` and
``model_it{N}.json`` (model weights and the selected setting and its summary, at every validation).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

H5_DIR = Path(__file__).resolve().parent
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from data import (  # noqa: E402
    DATA_ROOT, POINT_TARGETS, RAYINT_TARGETS, TARGETS, build_item, build_point_item, build_rayint_item, shape_ids,
)
from eval_rays import RATIO_BAND, decoding_grid, dec_gt_ratio, grid_summaries, prepare_all, prepare_fixed  # noqa: E402
from losses import POS_WEIGHT_K2, dense_loss, hits_loss  # noqa: E402
from models import POINT_TARGET_NAMES, build_model  # noqa: E402

CONFIG_DIR: Path = H5_DIR / "configs"
ALL_TARGETS: tuple[str, ...] = tuple(TARGETS) + ("hits_8_plane",) + tuple(POINT_TARGETS) + tuple(RAYINT_TARGETS)
HITS_TARGETS: tuple[str, ...] = ("hits_8", "hits_16", "hits_8_plane")
SELECTION_DELTA: str = "0.002"  # the validation metric is F1 at this delta


def load_config(path: Path, overrides: dict | None = None) -> dict:
    """``default.yaml`` (same directory) overridden by ``path``, then by ``overrides``. Unknown keys are errors."""
    path = Path(path)
    base = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text(encoding="utf-8")) or {}
    own = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(own) - set(base)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    cfg = {**base, **own, **(overrides or {})}
    if cfg["target"] not in ALL_TARGETS:
        raise ValueError(f"target must be one of {ALL_TARGETS}, got {cfg['target']!r}")
    if cfg["out_dir"] is None:
        raise ValueError("out_dir must be set")
    for key in ("batch_size", "n_rays", "n_points", "log_every", "val_every", "val_shapes", "val_rays", "max_it"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be > 0")
    cfg["out_dir"] = str(Path(cfg["out_dir"]).expanduser())
    for step in cfg["lr_steps"] or []:
        if len(step) != 2 or int(step[0]) <= 0 or float(step[1]) <= 0:
            raise ValueError(f"lr_steps entries are [iteration > 0, lr > 0], got {step!r}")
    return cfg


def lr_at(base_lr: float, steps, k: int) -> float:
    """Learning rate of iteration k (1-based, as logged): the lr of the last step with iteration <= k."""
    lr = float(base_lr)
    for it, new in sorted(steps or [], key=lambda s: int(s[0])):
        if k >= int(it):
            lr = float(new)
    return lr


def choose_train_shapes(cfg: dict) -> list[str]:
    """All training ids, or the ``train_subset`` ids chosen with ``Generator(PCG64(seed))``."""
    ids = shape_ids(cfg["train_split"], DATA_ROOT)
    n = cfg["train_subset"]
    if n is None:
        return ids
    rng = np.random.Generator(np.random.PCG64(int(cfg["seed"])))
    idx = rng.choice(len(ids), size=int(n), replace=False)
    return [ids[i] for i in idx]


class TrainSamples(Dataset):
    """Map-style dataset indexed by the global sample id ``g`` (see the module docstring)."""

    def __init__(self, shape_ids_: list[str], target: str, n_rays: int, n_points: int, noise: float, seed: int):
        self.dirs = [DATA_ROOT / "csg" / s for s in shape_ids_]
        self.target = target
        self.n_rays = n_rays
        self.n_points = n_points
        self.noise = noise
        self.seed = seed
        self._perm: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.dirs)

    def _order(self, epoch: int) -> np.ndarray:
        if epoch not in self._perm:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1, epoch]))
            self._perm[epoch] = rng.permutation(len(self.dirs))
        return self._perm[epoch]

    def shape_of(self, g: int) -> Path:
        epoch, pos = divmod(int(g), len(self.dirs))
        return self.dirs[int(self._order(epoch)[pos])]

    def __getitem__(self, g: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2, int(g)]))
        if self.target in RAYINT_TARGETS:  # point decoder, query points from the GT intervals of rays (H5-K2-1a)
            # n_rays of the config is the number of sampled rays per item (default 512 = data.RAYINT_N_RAYS)
            return build_rayint_item(self.shape_of(g), rng, self.target, self.n_points, self.noise,
                                     n_rays=self.n_rays)
        if self.target in POINT_TARGET_NAMES:  # point decoders: query points, the n_rays setting is not used
            return build_point_item(self.shape_of(g), rng, self.target, self.n_points, self.noise)
        # hits_8_plane has the hits_8 targets (its head differs, not its data)
        data_target = "hits_8" if self.target == "hits_8_plane" else self.target
        return build_item(self.shape_of(g), rng, data_target, self.n_rays, self.n_points, self.noise)


def make_loader(cfg: dict, shapes: list[str], g_start: int, g_end: int) -> DataLoader:
    ds = TrainSamples(shapes, cfg["target"], int(cfg["n_rays"]), int(cfg["n_points"]), float(cfg["noise"]),
                      int(cfg["seed"]))
    return DataLoader(ds, batch_size=int(cfg["batch_size"]), sampler=range(g_start, g_end), drop_last=True,
                      num_workers=int(cfg["num_workers"]))


def compute_loss(model, batch: dict, target: str, cfg: dict) -> dict:
    if target in POINT_TARGET_NAMES:
        # ConvONet's point loss: BCE with logits per query point, summed over points, mean over shapes
        logits = model(batch["inputs"], batch["query"])["logits"]
        bce = F.binary_cross_entropy_with_logits(logits, batch["occ"], reduction="none")
        return {"total": bce.sum(-1).mean()}
    out = model(batch["inputs"], batch["ray_axis"], batch["ray_origin_uv"])
    if target in HITS_TARGETS:
        pw = cfg["pos_weight"] if cfg["pos_weight"] is not None else POS_WEIGHT_K2
        pw_t = torch.as_tensor(pw, dtype=torch.float32, device=out["s_pred"].device)
        d = hits_loss(out["s_pred"], out["mask_logits"], batch["hits_s"], batch["hits_mask"], pw_t,
                      lam=float(cfg["lam"]))
        return {"total": d["total"], "L_t": d["L_t"], "L_mask": d["L_mask"]}
    d = dense_loss(out["logits"], batch["y"], batch["informative"], target)
    return {"total": d["total"], "informative": d["informative"], "other": d["other"],
            "entropy_floor": d["entropy_floor"], "excess": d["excess"]}


def _json_line(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=float) + "\n")


def _f1(summary: dict) -> float:
    """F1 at delta 0.002 of a summary (NaN counts as -inf)."""
    v = summary["deltas"][SELECTION_DELTA]["f1"]
    return -math.inf if v != v else v


def _band_distance(ratio: float) -> float:
    """Distance of a decoded/GT ratio to ``RATIO_BAND``: 0 inside, distance to the nearer edge outside, inf if NaN."""
    if ratio != ratio:
        return math.inf
    lo, hi = RATIO_BAND
    if lo <= ratio <= hi:
        return 0.0
    return min(abs(ratio - lo), abs(ratio - hi))


def _select(summaries: list[dict]) -> dict:
    """Choice of the decoding setting (H5-K1-1a).

    Among the settings whose decoded/GT ratio is in ``RATIO_BAND``, the one with the best F1 at delta 0.002. If no
    setting is in the band, the one whose ratio is closest to the band (ties: higher F1). Remaining ties go to the
    first setting. Returns ``{"index", "ratio", "in_band"}`` of the chosen setting.
    """
    ratios = [dec_gt_ratio(s) for s in summaries]
    f1 = [_f1(s) for s in summaries]
    dist = [_band_distance(r) for r in ratios]
    in_band = [k for k in range(len(summaries)) if dist[k] == 0.0]
    if in_band:
        k = max(in_band, key=lambda j: f1[j])
    else:
        k = min(range(len(summaries)), key=lambda j: (dist[j], -f1[j]))
    return {"index": k, "ratio": ratios[k], "in_band": dist[k] == 0.0}


def validation(model, cfg: dict, device: torch.device) -> list[dict]:
    """One summary per decoding setting of the target: on the validation shapes, or with ``eval_on_train``
    on the training subset with all rays."""
    settings = decoding_grid(cfg["target"])
    if cfg["eval_on_train"]:
        items = [prepare_all(s, DATA_ROOT) for s in choose_train_shapes(cfg)]
    else:
        n = min(int(cfg["val_shapes"]), len(shape_ids(cfg["val_split"], DATA_ROOT)))
        items = [prepare_fixed(cfg["val_split"], i, int(cfg["val_rays"]), DATA_ROOT) for i in range(n)]
    was_training = model.training
    summaries = grid_summaries(model, items, settings, device, DATA_ROOT)
    model.train(was_training)
    return summaries


def save_checkpoint(path: Path, blob: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)  # atomic: a crash never leaves a half-written model_last.pt


def run(cfg: dict, resume: bool = False, device: torch.device | None = None) -> dict:
    """Train according to ``cfg``. Returns ``{"iteration", "best_f1", "best_iteration"}``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["out_dir"])
    target = cfg["target"]
    out.mkdir(parents=True, exist_ok=True)
    log_path, val_path = out / "log.jsonl", out / "val.jsonl"
    if not resume and (log_path.exists() or (out / "model_last.pt").exists()):
        raise FileExistsError(f"{out} already holds a run; use --resume or another out_dir")

    torch.manual_seed(int(cfg["seed"]))
    model = build_model(target, cfg.get("model") or None).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    iteration, best_f1, best_it = 0, -math.inf, None
    if resume:
        blob = torch.load(out / "model_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        opt.load_state_dict(blob["optimizer"])
        iteration = int(blob["iteration"])
        best_f1, best_it = float(blob["best_f1"]), blob["best_iteration"]
        torch.set_rng_state(blob["rng_cpu"].cpu())
        if device.type == "cuda" and blob.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(blob["rng_cuda"].cpu(), device)
        # the config of this resume (the run's config.json is the first one; max_it or lr_steps may differ now)
        (out / f"config_resumed_at_{iteration}.json").write_text(json.dumps(cfg, indent=1, default=str),
                                                                encoding="utf-8")
    else:
        (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
        if cfg["eval_on_train"] or cfg["train_subset"] is not None:
            (out / "train_subset.json").write_text(json.dumps(choose_train_shapes(cfg)), encoding="utf-8")

    shapes = choose_train_shapes(cfg)
    max_it = int(cfg["max_it"])
    B = int(cfg["batch_size"])
    if iteration >= max_it:
        return {"iteration": iteration, "best_f1": best_f1, "best_iteration": best_it}
    loader = make_loader(cfg, shapes, iteration * B, max_it * B)
    model.train()
    window_t, window_n = time.perf_counter(), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        lr_now = lr_at(cfg["lr"], cfg.get("lr_steps"), iteration + 1)  # from the iteration counter: resume-safe
        for group in opt.param_groups:
            group["lr"] = lr_now
        parts = compute_loss(model, batch, target, cfg)
        opt.zero_grad(set_to_none=True)
        parts["total"].backward()
        opt.step()
        iteration += 1
        window_n += 1
        if iteration % int(cfg["log_every"]) == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            record = {"iteration": iteration, "loss": float(parts["total"].detach()), "lr": lr_now,
                      "parts": {k: float(v.detach()) for k, v in parts.items() if k != "total"},
                      "s_per_it": (now - window_t) / window_n,
                      "peak_gib": (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else None}
            _json_line(log_path, record)
            window_t, window_n = now, 0
        if iteration % int(cfg["val_every"]) == 0 or iteration == max_it:
            t_val = time.perf_counter()
            summaries = validation(model, cfg, device)
            choice = _select(summaries)
            sel = choice["index"]
            f1_all = [s["deltas"][SELECTION_DELTA]["f1"] for s in summaries]
            _json_line(val_path, {"iteration": iteration, "selected": summaries[sel]["setting_name"],
                                  "selected_f1": f1_all[sel], "selected_ratio": choice["ratio"],
                                  "selected_in_band": choice["in_band"], "f1": f1_all, "settings": summaries,
                                  "val_seconds": time.perf_counter() - t_val})
            state = {"target": target, "state_dict": model.state_dict(), "iteration": iteration,
                     "decoding": summaries[sel]["setting"], "f1": f1_all[sel], "ratio": choice["ratio"],
                     "in_band": choice["in_band"]}
            torch.save({**state, "summary": summaries[sel]}, out / f"model_it{iteration}.pt")
            (out / f"model_it{iteration}.json").write_text(json.dumps(
                {"iteration": iteration, "selected_setting": summaries[sel]["setting_name"],
                 "setting": summaries[sel]["setting"], "f1": f1_all[sel], "ratio": choice["ratio"],
                 "in_band": choice["in_band"], "summary": summaries[sel]},
                indent=1, default=float), encoding="utf-8")
            if f1_all[sel] == f1_all[sel] and f1_all[sel] > best_f1:  # NaN never improves
                best_f1, best_it = f1_all[sel], iteration
                torch.save({**state, "best_iteration": best_it}, out / "model_best.pt")
            rng_cuda = torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None
            save_checkpoint(out / "model_last.pt", {
                "target": target, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                "iteration": iteration, "best_f1": best_f1, "best_iteration": best_it,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": rng_cuda, "config": cfg})
            window_t, window_n = time.perf_counter(), 0  # validation time is not training time
            model.train()
    return {"iteration": iteration, "best_f1": best_f1, "best_iteration": best_it}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-it", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key (value parsed as YAML), repeatable")
    args = p.parse_args(argv)
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if not _ or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    if args.max_it is not None:
        overrides["max_it"] = args.max_it
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    cfg = load_config(args.config, overrides)
    res = run(cfg, resume=args.resume)
    print(f"done: iteration {res['iteration']}, best F1 {res['best_f1']:.4g} at iteration {res['best_iteration']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
