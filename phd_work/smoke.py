'''Smoke test of the pretrained voxel 3-plane ConvONet without any dataset.

Mirrors the model setup in generate.py. Run from the repo root:
    uv run python phd_work/smoke.py [--device {cuda,cpu}]
'''
import argparse
import os
import sys

# Make the repo root importable (`from src import ...`) when run as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

import torch
from src import config
from src.checkpoints import CheckpointIO


parser = argparse.ArgumentParser(description='Smoke test pretrained voxel ConvONet.')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
args = parser.parse_args()

device = torch.device(args.device)

# 1. Environment
print('torch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
if device.type == 'cuda':
    print('device name:', torch.cuda.get_device_name(device))
else:
    print('device name: cpu')

# 2. Config
cfg = config.load_config(
    'configs/voxel/pretrained/shapenet_3plane.yaml', 'configs/default.yaml')

# 3. Model
model = config.get_model(cfg, device=device, dataset=None)
n_params = sum(p.numel() for p in model.parameters())
print('parameter count:', n_params)

# 4. Checkpoint
checkpoint_io = CheckpointIO(cfg['training']['out_dir'], model=model)
scalars = checkpoint_io.load(cfg['test']['model_file'])
# load() returns every non-module entry, including the (huge) optimizer state;
# print only true scalars and summarize the rest.
print('checkpoint scalars:',
      {k: v for k, v in scalars.items() if not isinstance(v, (dict, list))})
print('checkpoint other keys:', [k for k, v in scalars.items() if isinstance(v, (dict, list))])

# 5. Forward pass on random data
model.eval()
with torch.no_grad():
    torch.manual_seed(0)
    voxels = (torch.rand(1, 32, 32, 32) > 0.5).float().to(device)
    p = (torch.rand(1, 100000, 3) * 1.1 - 0.55).to(device)

    c = model.encode_inputs(voxels)
    out = model.decode(p, c)
    logits = out.logits

    print('type(out):', type(out))
    if isinstance(c, dict):
        for name, plane in c.items():
            print(f'plane {name} shape:', tuple(plane.shape))
    else:
        print('c shape:', tuple(c.shape))
    print('logits shape:', tuple(logits.shape))
    print('logits min/mean/max:',
          logits.min().item(), logits.mean().item(), logits.max().item())

    threshold = cfg['test']['threshold']
    frac_occ = (torch.sigmoid(logits) > threshold).float().mean().item()
    print(f'fraction of points with sigmoid(logits) > {threshold}:', frac_occ)

    assert torch.isfinite(logits).all(), 'non-finite logits'
    print('OK: all logits finite')
