"""Forward + backward time and peak GPU memory per ray count (H5-3 step 4; same procedure as test g of H5-2).

    cd ~/PhD/convonet && .venv/bin/python h5/scripts/time_rays.py

Batch 32 shapes x n_rays rays x 10000 points, fp32, forward + backward + loss, one warm-up, median of 3.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

H5_DIR = Path(__file__).resolve().parents[1]
if str(H5_DIR) not in sys.path:
    sys.path.insert(0, str(H5_DIR))

from losses import POS_WEIGHT_K2, dense_loss, hits_loss  # noqa: E402
from models import H5RayModel, n_params  # noqa: E402

BATCH = 32
POINTS = 10000
REPEATS = 3
RAY_COUNTS = (1024, 512)
TARGETS = ("hits_8", "occfrac_128")


def measure(target: str, n_rays: int, gen: torch.Generator, device: torch.device) -> dict:
    model = H5RayModel(target).to(device)
    inputs = (torch.rand(BATCH, POINTS, 3, generator=gen) - 0.5).to(device)
    ray_axis = torch.randint(0, 3, (BATCH, n_rays), generator=gen).to(device)
    uv = ((torch.rand(BATCH, n_rays, 2, generator=gen) - 0.5) * 1.1).float().to(device)
    if target == "hits_8":
        s_t = (torch.rand(BATCH, n_rays, 8, generator=gen) * 128).to(device)
        m_t = (torch.rand(BATCH, n_rays, 4, generator=gen) < 0.5).to(device)
        pw = torch.tensor(POS_WEIGHT_K2, device=device)

        def step():
            out = model(inputs, ray_axis, uv)
            return hits_loss(out["s_pred"], out["mask_logits"], s_t, m_t, pw)["total"]
    else:
        y = torch.rand(BATCH, n_rays, 128, generator=gen).to(device)
        inf = (y > 0) & (y < 1)

        def step():
            out = model(inputs, ray_axis, uv)
            return dense_loss(out["logits"], y, inf, target)["total"]

    def fwd_bwd():
        step().backward()
        model.zero_grad(set_to_none=True)

    fwd_bwd()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd_bwd()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    row = {"target": target, "n_rays": n_rays, "time_s": statistics.median(times), "peak_gib": peak,
           "encoder": n_params(model.encoder), "trunk": n_params(model.trunk), "head": n_params(model.head)}
    del model, inputs, ray_axis, uv
    torch.cuda.empty_cache()
    return row


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    device = torch.device("cuda")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(8)
    rows = [measure(t, n, gen, device) for t in TARGETS for n in RAY_COUNTS]
    print(f"batch {BATCH} shapes x n_rays x {POINTS} points, fp32, {torch.cuda.get_device_name(0)}")
    print(f"{'target':<14}{'n_rays':>8}{'time (s)':>11}{'peak GiB':>10}")
    for r in rows:
        print(f"{r['target']:<14}{r['n_rays']:>8}{r['time_s']:>11.3f}{r['peak_gib']:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
