"""[phd] Profile one training step (GPU): voxel encoder vs dexel encoder.

Usage:
    uv run python phd_work/profile_step.py <config.yaml> [--warmup 20] [--steps 50] [--out-dir out/profile_step]

What it does (no upstream edits; the step is re-assembled from the same modules that
ConvolutionalOccupancyNetwork / Trainer.compute_loss use):
  1. Builds the dataset, model, Adam optimizer (lr 1e-4, as train.py) and trainer; dexel
     configs go through phd_work/dexel_register.register() as in phd_work/run.py.
  2. Takes ONE real training batch from the train loader and moves it to the GPU once.
  3. Checks that the split encoder (first layer, scatter, U-Net) gives the same plane
     features as model.encode_inputs on that batch.
  4. Times, with torch.cuda.Event pairs, one step split into parts:
       a first encoder layer (voxel: conv_in + act; dexel: ray_encoder + act, per plane)
       b scatter to the planes (normalize + coordinate2index + scatter_mean, per plane)
       c U-Net (per plane)
       d decoder (model.decode, incl. the Bernoulli wrapper)
       e loss (BCE with logits, sum over points, mean over batch)
       f backward
       g optimizer step
     Plus the total step (zero_grad to step end) and the remainder (total - sum of parts).
     Warm-up steps are run first; the reported statistic is the median over --steps steps.
     Query points and plane coordinates are constant per batch and are precomputed outside
     the timed regions (the parent recomputes them each step; this is a few small tensors).
  5. Peak GPU memory of one step (torch.cuda.max_memory_allocated, after reset).
  6. torch.profiler over --steps real trainer.train_step calls (CPU + CUDA): top 20 CUDA
     kernels by self CUDA time with call counts, and a chrome trace written to --out-dir.

Plus the header with driver, torch, CUDA and cuDNN versions is printed first.
DataLoader-only timing is a separate run of phd_work/bench_loader.py (see the shell wrapper).
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src import config  # noqa: E402
from src.common import coordinate2index, normalize_coordinate  # noqa: E402
from src.encoder.voxels import scatter_mean  # noqa: E402
from phd_work.dexel_register import register  # noqa: E402
from phd_work.scatter_switch import apply_from_env  # noqa: E402

PLANE_NAMES = ('xz', 'xy', 'yz')
PARTS = ['a_first_layer', 'b_scatter', 'c_unet', 'd_decoder', 'e_loss', 'f_backward', 'g_optimizer']


def versions_header():
    lines = []
    try:
        drv = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout.strip()
        lines.append('gpu, driver: %s' % drv)
    except Exception as e:  # noqa: BLE001
        lines.append('gpu, driver: unavailable (%s)' % e)
    lines.append('torch: %s' % torch.__version__)
    lines.append('cuda (torch build): %s' % torch.version.cuda)
    lines.append('cudnn: %s' % torch.backends.cudnn.version())
    return '\n'.join(lines)


class Timer:
    """Collects (name, start_event, end_event) pairs; one synchronize at the end of a step."""

    def __init__(self):
        self.pairs = []

    def run(self, name, fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        self.pairs.append((name, s, e))
        return out

    def collect(self):
        torch.cuda.synchronize()
        sums = {p: 0.0 for p in PARTS}
        for name, s, e in self.pairs:
            sums[name] += s.elapsed_time(e)
        self.pairs = []
        return sums


# ---- the split step --------------------------------------------------------------
def voxel_points(B, device):
    c = torch.linspace(-0.5, 0.5, 32, device=device)
    p = torch.stack(torch.meshgrid(c, c, c, indexing='ij'), dim=-1).reshape(1, -1, 3)
    return p.expand(B, -1, -1).contiguous()


def scatter(enc, pts, c, plane):
    """generate_plane_features without the U-Net (same ops as the parent)."""
    B = c.size(0)
    xy = normalize_coordinate(pts.clone(), plane=plane, padding=enc.padding)
    index = coordinate2index(xy, enc.reso_plane)
    fea = c.new_zeros(B, enc.c_dim, enc.reso_plane ** 2)
    fea = scatter_mean(c.permute(0, 2, 1), index, out=fea)
    return fea.reshape(B, enc.c_dim, enc.reso_plane, enc.reso_plane)


class Split:
    """Holds the encoder-specific pieces of the step."""

    def __init__(self, model, is_dexel, device, batch):
        self.model = model
        self.enc = model.encoder
        self.is_dexel = is_dexel
        self.planes = [n for n in PLANE_NAMES if n in self.enc.plane_type]
        B = batch['points'].size(0)
        if is_dexel:
            self.pts = {}
            for n in self.planes:
                self.pts[n] = self.enc.ray_points(PLANE_NAMES.index(n), device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
        else:
            pv = voxel_points(B, device)
            self.pts = {n: pv for n in self.planes}

    def first_layer(self, inputs):
        enc = self.enc
        if self.is_dexel:
            feats = {}
            for n in self.planes:
                f = enc.actvn(enc.ray_encoder(inputs[:, PLANE_NAMES.index(n)]))
                feats[n] = f.reshape(f.size(0), enc.c_dim, -1).permute(0, 2, 1)
            return feats
        x = inputs.unsqueeze(1)
        c = enc.actvn(enc.conv_in(x)).view(inputs.size(0), enc.c_dim, -1).permute(0, 2, 1)
        return {n: c for n in self.planes}

    def full_c(self, inputs):
        """Split path, without timers, returns the plane feature dict (for the check)."""
        cs = self.first_layer(inputs)
        out = {}
        for n in self.planes:
            out[n] = self.enc.unet(scatter(self.enc, self.pts[n], cs[n], n))
        return out


def timed_step(split, model, optimizer, timer, batch, device):
    p = batch['points']
    occ = batch['points.occ']
    inputs = batch['inputs']
    optimizer.zero_grad()
    cs = timer.run('a_first_layer', lambda: split.first_layer(inputs))
    fea = {}
    for n in split.planes:
        fea[n] = timer.run('b_scatter', lambda n=n: scatter(split.enc, split.pts[n], cs[n], n))
    c = {}
    for n in split.planes:
        c[n] = timer.run('c_unet', lambda n=n: split.enc.unet(fea[n]))
    logits = timer.run('d_decoder', lambda: model.decode(p, c).logits)
    loss = timer.run('e_loss', lambda: F.binary_cross_entropy_with_logits(logits, occ, reduction='none').sum(-1).mean())
    timer.run('f_backward', lambda: loss.backward())
    timer.run('g_optimizer', lambda: optimizer.step())
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'out', 'profile_step'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        cfg = config.load_config(args.config, 'configs/default.yaml')
    finally:
        os.chdir(cwd)
    is_dexel = cfg['data']['input_type'] == 'dexels'
    if is_dexel:
        register()
    print('scatter implementation: %s (SCATTER env)' % apply_from_env())
    device = torch.device('cuda')
    name = os.path.splitext(os.path.basename(args.config))[0]

    print('=== %s (input_type=%s, encoder=%s)' % (name, cfg['data']['input_type'], cfg['model']['encoder']))
    train_ds = config.get_dataset('train', cfg)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                                         num_workers=cfg['training']['n_workers'], shuffle=True)
    first = next(iter(loader))
    batch = {k: v.to(device) for k, v in first.items() if k in ('points', 'points.occ', 'inputs')}
    print('batch: points %s, inputs %s (float32 %s)' % (tuple(batch['points'].shape),
                                                       tuple(batch['inputs'].shape), batch['inputs'].dtype))

    model = config.get_model(cfg, device=device, dataset=train_ds)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)
    model.train()
    split = Split(model, is_dexel, device, batch)
    timer = Timer()

    # check: split path == model.encode_inputs on this batch
    # The CUDA scatter (scatter_reduce, mean) sums with atomics, so two calls of the SAME
    # model can differ at float rounding level. Report that noise floor next to the check.
    with torch.no_grad():
        ref = model.encode_inputs(batch['inputs'])
        ref2 = model.encode_inputs(batch['inputs'])
        mine = split.full_c(batch['inputs'])
        same_keys = set(ref.keys()) == set(split.planes)
        max_diff = max((ref[n] - mine[n]).abs().max().item() for n in split.planes)
        noise = max((ref[n] - ref2[n]).abs().max().item() for n in split.planes)
        scale = max(ref[n].abs().max().item() for n in split.planes)
    print('check: split encoder vs model.encode_inputs: keys equal %s, max |diff| = %g '
          '(same model called twice: max |diff| = %g; max |feature| = %g)'
          % (same_keys, max_diff, noise, scale))
    assert same_keys and max_diff <= 1e-4 * max(scale, 1.0), 'split encoder differs from model.encode_inputs'

    # warm-up
    for _ in range(args.warmup):
        timed_step(split, model, optimizer, timer, batch, device)
        timer.collect()

    # timed steps
    per_part = {p: [] for p in PARTS}
    totals = []
    overhead = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        timed_step(split, model, optimizer, timer, batch, device)
        t1.record()
        sums = timer.collect()
        total = t0.elapsed_time(t1)
        totals.append(total)
        for p in PARTS:
            per_part[p].append(sums[p])
        overhead.append(total - sum(sums.values()))
    print('timed steps: %d (warm-up %d); median ms per part:' % (args.steps, args.warmup))
    part_med = {}
    for p in PARTS:
        part_med[p] = float(np.median(per_part[p]))
        print('  %-14s %9.3f ms' % (p, part_med[p]))
    print('  %-14s %9.3f ms' % ('sum of parts', sum(part_med.values())))
    print('  %-14s %9.3f ms  (zero_grad, timer overhead, Python)' % ('remainder', float(np.median(overhead))))
    print('  %-14s %9.3f ms  (median of total step, zero_grad to step end)' % ('total', float(np.median(totals))))

    # peak memory of one step
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    timed_step(split, model, optimizer, timer, batch, device)
    timer.collect()
    peak = torch.cuda.max_memory_allocated()
    print('peak GPU memory, one step: max_memory_allocated %.1f MiB; allocated before step %.1f MiB; '
          'transient peak above that %.1f MiB'
          % (peak / 2 ** 20, base / 2 ** 20, (peak - base) / 2 ** 20))

    # torch.profiler: real trainer.train_step on the same GPU batch
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            trainer.train_step(batch)
        torch.cuda.synchronize()
    trace = os.path.join(args.out_dir, '%s_trace.json' % name)
    prof.export_chrome_trace(trace)
    from torch.autograd import DeviceType
    rows = []
    for e in prof.key_averages():
        if getattr(e, 'device_type', None) != DeviceType.CUDA:
            continue
        self_us = getattr(e, 'self_device_time_total', None)
        if self_us is None:
            self_us = getattr(e, 'self_cuda_time_total', 0.0)
        rows.append((float(self_us), int(e.count), e.key))
    rows.sort(reverse=True)
    total_cuda = sum(r[0] for r in rows) or 1.0
    print('torch.profiler, %d real train_step calls: top 20 CUDA kernels by self CUDA time' % args.steps)
    print('  %-4s %-90s %8s %12s %10s %8s' % ('rank', 'kernel', 'calls', 'self CUDA ms', 'ms/call', '% CUDA'))
    for i, (us, cnt, key) in enumerate(rows[:20], 1):
        key_short = key if len(key) <= 90 else key[:87] + '...'
        print('  %-4d %-90s %8d %12.2f %10.4f %7.1f%%' % (i, key_short, cnt, us / 1000.0, us / 1000.0 / max(cnt, 1),
                                                       100.0 * us / total_cuda))
    print('  (sum of self CUDA over all CUDA kernels: %.1f ms over %d steps)' % (total_cuda / 1000.0, args.steps))
    print('chrome trace: %s' % trace)
    print()


if __name__ == '__main__':
    main()
