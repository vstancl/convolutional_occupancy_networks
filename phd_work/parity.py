'''Parity test for the voxel 3-plane pretrained ConvONet.

Captures the encoder features and decoder logits on fixed, seeded inputs, and
compares a later run against them. Used before/after any change that can alter
numerical outputs (CLAUDE.md).

    uv run python phd_work/parity.py capture   # write reference (run before the change)
    uv run python phd_work/parity.py compare   # check current code against reference

Reference file (not committed, regenerated on demand):
    phd_work/parity/ref_voxel3plane.pt
'''
import argparse
import hashlib
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # config paths in the repo are relative to the root

import torch
from src import config
from src.checkpoints import CheckpointIO

# GPU scatter (atomic adds) and cuDNN are not bit-reproducible by default:
# two identical runs differed by ~0.02 in the encoder features on the RTX 4070.
# Deterministic mode makes the reference comparison meaningful. See port-log.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REF_PATH = os.path.join(REPO_ROOT, 'phd_work', 'parity', 'ref_voxel3plane.pt')
ATOL = 1e-5   # float32 summation-order tolerance for scatter reductions
N_BATCH = 2   # two objects, so batch indexing in the scatter is exercised


def build_model(device):
    cfg = config.load_config(
        'configs/voxel/pretrained/shapenet_3plane.yaml', 'configs/default.yaml')
    model = config.get_model(cfg, device=device, dataset=None)
    CheckpointIO(cfg['training']['out_dir'], model=model).load(cfg['test']['model_file'])
    model.eval()
    return model


def make_inputs(device):
    g = torch.Generator().manual_seed(0)
    voxels = (torch.rand(N_BATCH, 32, 32, 32, generator=g) > 0.5).float().to(device)
    p = (torch.rand(N_BATCH, 20000, 3, generator=g) * 1.1 - 0.55).to(device)
    return voxels, p


@torch.no_grad()
def run(device):
    model = build_model(device)
    voxels, p = make_inputs(device)
    c = model.encode_inputs(voxels)
    logits = model.decode(p, c).logits
    planes = {k: v.detach().float().cpu() for k, v in c.items()} if isinstance(c, dict) \
        else {'c': c.detach().float().cpu()}
    return planes, logits.detach().float().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['capture', 'compare'])
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    device = torch.device(args.device)

    planes, logits = run(device)

    if args.mode == 'capture':
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        torch.save({'planes': planes, 'logits': logits, 'torch': str(torch.__version__),
                    'device': args.device}, REF_PATH)
        digest = hashlib.sha256(open(REF_PATH, 'rb').read()).hexdigest()
        print(f'captured reference -> {REF_PATH}')
        print('sha256:', digest)
        print('logits shape:', tuple(logits.shape), 'planes:', {k: tuple(v.shape) for k, v in planes.items()})
        return

    ref = torch.load(REF_PATH)
    if ref['device'] != args.device:
        sys.exit(f"reference was captured on {ref['device']}; compare with --device {ref['device']}. "
                 f"(cuda and cpu differ by ~0.03 in logits, so they are not interchangeable)")
    ok = True
    for k in ref['planes']:
        d = (planes[k] - ref['planes'][k]).abs().max().item()
        print(f'encoder {k}: max|diff| = {d:.3e}')
        ok &= d <= ATOL
    d = (logits - ref['logits']).abs().max().item()
    print(f'logits: max|diff| = {d:.3e}')
    ok &= d <= ATOL
    print('PARITY OK' if ok else f'PARITY FAIL (atol={ATOL})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
