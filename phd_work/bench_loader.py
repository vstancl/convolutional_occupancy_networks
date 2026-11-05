"""[phd] CPU data-loading time per batch, train loader as in train.py.

Usage:
    uv run python phd_work/bench_loader.py <config.yaml> [--batches 50] [--warmup 5]

Builds the train dataset with config.get_dataset (dexel registration active, as in
phd_work/run.py), then a DataLoader with the config's batch_size, n_workers and shuffle=True
(as in train.py). Times `warmup` batches (worker start-up, not reported), then `batches`
batches.
"""
import argparse
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'phd_work'))

import torch  # noqa: E402
from src import config  # noqa: E402
from src.config import load_config  # noqa: E402
from dexel_register import register  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--batches', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=5)
    args = ap.parse_args()

    register()
    cfg = load_config(args.config, os.path.join(REPO, 'configs', 'default.yaml'))
    bs = cfg['training']['batch_size']
    nw = cfg['training']['n_workers']
    ds = config.get_dataset('train', cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=nw, shuffle=True)

    it = iter(loader)
    for _ in range(args.warmup):
        next(it)
    t0 = time.perf_counter()
    shape = None
    for _ in range(args.batches):
        batch = next(it)
        shape = tuple(batch['inputs'].shape)
    dt = time.perf_counter() - t0
    print('config=%s input_type=%s batch_size=%d n_workers=%d batches=%d inputs_shape=%s'
          % (os.path.basename(args.config), cfg['data']['input_type'], bs, nw, args.batches, shape))
    print('time per batch: %.1f ms  (total %.2f s for %d batches)'
          % (1000 * dt / args.batches, dt, args.batches))


if __name__ == '__main__':
    main()
