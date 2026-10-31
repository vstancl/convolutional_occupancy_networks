'''Parity test for the voxel 3-plane pretrained ConvONet.

Captures encoder features and decoder logits (and, with --grad, parameter
gradients) on fixed seeded inputs, and compares a later run against them.
Used before/after any change that can alter numerical outputs (CLAUDE.md).

    uv run python phd_work/parity.py capture [--grad]   # write reference
    uv run python phd_work/parity.py compare [--grad]   # check current code

Provenance: the reference records the git commit it was captured from. `capture`
refuses to run on a tree whose model code (src/, configs/) has uncommitted changes,
so a reference always corresponds to a commit. The harness script itself is not
part of that check; its sha256 is recorded instead.

Reference file: phd_work/parity/ref_voxel3plane.pt (committed with `git add -f`).
'''
import argparse
import hashlib
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # config paths in the repo are relative to the root

# GPU scatter (atomic adds) and cuDNN are not bit-reproducible by default:
# two identical runs differed by ~0.02 in the encoder features on the RTX 4070.
# Deterministic mode makes the reference comparison meaningful. See port-log.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn.functional as F
from src import config
from src.checkpoints import CheckpointIO

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REF_PATH = os.path.join(REPO_ROOT, 'phd_work', 'parity', 'ref_voxel3plane.pt')
# Gradients: the CUDA backward of grid_sampler_2d has no deterministic implementation
# (torch raises under deterministic mode), so the --grad reference is CPU-only.
REF_GRAD_PATH = os.path.join(REPO_ROOT, 'phd_work', 'parity', 'ref_voxel3plane_grad_cpu.pt')
ATOL = 1e-5   # float32 summation-order tolerance for scatter reductions
N_BATCH = 2   # two objects, so batch indexing in the scatter is exercised
GRAD_KEYS = {
    'encoder.conv_in.weight': lambda m: m.encoder.conv_in.weight,
    'encoder.conv_in.bias': lambda m: m.encoder.conv_in.bias,
    'decoder.fc_p.weight': lambda m: m.decoder.fc_p.weight,
    'decoder.fc_p.bias': lambda m: m.decoder.fc_p.bias,
}


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def provenance():
    commit = git('rev-parse', 'HEAD').strip()
    dirty = git('status', '--porcelain', '--untracked-files=no', '--', 'src', 'configs').strip()
    return {
        'commit': commit,
        'model_code_dirty': bool(dirty),
        'dirty_paths': dirty.splitlines(),
        'harness_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
    }


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
    g2 = torch.Generator().manual_seed(1)
    targets = (torch.rand(N_BATCH, 20000, generator=g2) > 0.5).float().to(device)
    return voxels, p, targets


@torch.no_grad()
def run_forward(model, voxels, p):
    c = model.encode_inputs(voxels)
    logits = model.decode(p, c).logits
    planes = {k: v.detach().float().cpu() for k, v in c.items()} if isinstance(c, dict) \
        else {'c': c.detach().float().cpu()}
    return planes, logits.detach().float().cpu()


def run_grad(model, voxels, p, targets):
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        c = model.encode_inputs(voxels)
        logits = model.decode(p, c).logits
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
    grads = {k: get(model).grad.detach().float().cpu().clone() for k, get in GRAD_KEYS.items()}
    return loss.detach().float().cpu().item(), grads


def run_all(device, want_grad):
    model = build_model(device)
    voxels, p, targets = make_inputs(device)
    planes, logits = run_forward(model, voxels, p)
    out = {'planes': planes, 'logits': logits}
    if want_grad:
        loss, grads = run_grad(model, voxels, p, targets)
        out.update(loss=loss, grads=grads)
    return out


def report(name, cur, ref):
    d = (cur - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f'{name}: max|diff| = {d:.3e}  (max|ref| = {scale:.3e})')
    return d <= ATOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['capture', 'compare'])
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--grad', action='store_true', help='also check gradients (encoder conv_in, decoder fc_p)')
    args = parser.parse_args()
    device = torch.device(args.device)

    prov = provenance()
    if args.grad and args.device != 'cpu':
        sys.exit('--grad runs on CPU only (CUDA backward is not deterministic); use --device cpu')
    ref_path = REF_GRAD_PATH if args.grad else REF_PATH

    if args.mode == 'capture':
        if prov['model_code_dirty']:
            sys.exit(f"refusing to capture: uncommitted changes in model code: {prov['dirty_paths']}")
        out = run_all(device, args.grad)
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        torch.save({**out, 'meta': {**prov, 'torch': str(torch.__version__),
                                    'device': args.device, 'grad': args.grad}}, ref_path)
        print(f"captured reference from commit {prov['commit']} -> {ref_path}")
        print('sha256:', hashlib.sha256(open(ref_path, 'rb').read()).hexdigest())
        return

    ref = torch.load(ref_path)
    meta = ref['meta']
    if meta['device'] != args.device:
        sys.exit(f"reference was captured on {meta['device']}; compare with --device {meta['device']}. "
                 f"(cuda and cpu differ by ~0.03 in logits, so they are not interchangeable)")
    if args.grad and not meta['grad']:
        sys.exit('reference has no gradients; capture with --grad')
    if not args.grad and meta['grad']:
        sys.exit('this reference is the --grad (CPU) one; compare with --grad')
    print(f"reference: commit {meta['commit']} (dirty={meta['model_code_dirty']}), torch {meta['torch']}")
    print(f"current:   commit {prov['commit']} (dirty={prov['model_code_dirty']})")

    cur = run_all(device, args.grad)
    ok = True
    for k in ref['planes']:
        ok &= report(f'encoder {k}', cur['planes'][k], ref['planes'][k])
    ok &= report('logits', cur['logits'], ref['logits'])
    if args.grad:
        print(f"loss: {cur['loss']:.9g} vs ref {ref['loss']:.9g} (diff {abs(cur['loss'] - ref['loss']):.3e})")
        for k in GRAD_KEYS:
            ok &= report(f'grad {k}', cur['grads'][k], ref['grads'][k])
    print('PARITY OK' if ok else f'PARITY FAIL (atol={ATOL})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
