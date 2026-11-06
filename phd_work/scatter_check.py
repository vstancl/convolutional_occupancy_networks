"""[phd] Encoder forward, native scatter vs torch_scatter scatter, on one real batch (GPU).

Same weights (seeded init), same batch. Reports max |diff| of the plane features between
the two scatter implementations, and between two native calls (the atomic-order noise floor).
Usage: uv run python phd_work/scatter_check.py
"""
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src import config  # noqa: E402
from phd_work.dexel_register import register  # noqa: E402
from phd_work.scatter_switch import use_native, use_torch_scatter  # noqa: E402
import src.encoder.voxels as voxels_mod  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402


def which_scatter(model, inputs, expect):
    """Function object in use + aten scatter operator counts for one encode. Fails if not `expect`."""
    name = voxels_mod.scatter_mean.__qualname__
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.no_grad():
            model.encode_inputs(inputs)
    counts = {e.key: int(e.count) for e in prof.key_averages()}
    n_reduce = counts.get('aten::scatter_reduce_', 0)
    print('  function in use: %s; aten::scatter_reduce_ calls: %d, aten::scatter_add_ calls: %d'
          % (name, n_reduce, counts.get('aten::scatter_add_', 0)))
    if expect == 'torch_scatter' and (n_reduce > 0 or not name.endswith('scatter_mean_upstream')):
        raise RuntimeError('expected torch_scatter, got %s (scatter_reduce_ calls %d)' % (name, n_reduce))
    if expect == 'native' and (n_reduce == 0 or name.endswith('scatter_mean_upstream')):
        raise RuntimeError('expected native, got %s (scatter_reduce_ calls %d)' % (name, n_reduce))

CONFIGS = ['voxel_3plane_3cls.yaml', 'dexel_3plane_3cls_L32.yaml']


def encode(model, inputs):
    with torch.no_grad():
        return {k: v.clone() for k, v in model.encode_inputs(inputs).items()}


def main():
    register()
    device = torch.device('cuda')
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        for name in CONFIGS:
            cfg = config.load_config(os.path.join(REPO, 'phd_work', 'configs', name), 'configs/default.yaml')
            ds = config.get_dataset('train', cfg)
            loader = torch.utils.data.DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False,
                                                 num_workers=0)
            batch = next(iter(loader))
            inputs = batch['inputs'].to(device)
            torch.manual_seed(0)
            model = config.get_model(cfg, device=device, dataset=ds).eval()

            use_native()
            print('  native pass:')
            which_scatter(model, inputs, 'native')
            native_a = encode(model, inputs)
            native_b = encode(model, inputs)
            use_torch_scatter()
            try:
                print('  torch_scatter pass:')
                which_scatter(model, inputs, 'torch_scatter')
                upstream = encode(model, inputs)
            finally:
                use_native()

            print('=== %s: batch inputs %s' % (name, tuple(inputs.shape)))
            for k in sorted(native_a):
                scale = native_a[k].abs().max().item()
                d_ts = (native_a[k] - upstream[k]).abs().max().item()
                d_nn = (native_a[k] - native_b[k]).abs().max().item()
                print('  plane %-3s max|feature| %.4f | native vs torch_scatter max|diff| %.3g | '
                      'native vs native (noise floor) max|diff| %.3g'
                      % (k, scale, d_ts, d_nn))
            print('  planes compared: %s' % sorted(native_a))
            print()
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    main()
