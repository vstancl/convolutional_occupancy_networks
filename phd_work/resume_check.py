"""[phd] Check that a checkpoint contains restorable training state.

Loads a checkpoint through src.checkpoints.CheckpointIO with no modules registered,
so every entry is returned (model and optimizer state included), prints the scalar
entries, and checks the Adam per-parameter 'step' counters.

After training for `it` iterations with one optimizer.step() per iteration, every
Adam step must equal `it`. That only holds after a resume if the optimizer state was
restored (a fresh Adam would restart its step count at 1).

Usage:
    uv run python phd_work/resume_check.py <checkpoint> [--min-it N]

Exit 0 on pass, 1 on fail.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.checkpoints import CheckpointIO  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--min-it', type=int, default=None,
                        help='Fail unless it > MIN_IT (proves the run resumed past an earlier point).')
    args = parser.parse_args()

    path = os.path.abspath(args.checkpoint)
    if not os.path.exists(path):
        print('FAIL: checkpoint not found: %s' % path)
        return 1

    io = CheckpointIO(os.path.dirname(path))
    entries = io.load(path)

    print('checkpoint: %s' % path)
    print('entries: %s' % sorted(entries.keys()))
    for k, v in sorted(entries.items()):
        if isinstance(v, (int, float)):
            print('  %s = %r' % (k, v))
    lvb = entries.get('loss_val_best')
    print('  loss_val_best type: %s' % (type(lvb).__name__ if lvb is not None else None))

    it = entries.get('it')
    epoch_it = entries.get('epoch_it')
    print('it = %s, epoch_it = %s' % (it, epoch_it))

    ok = True
    if it is None:
        print('FAIL: checkpoint has no "it" entry')
        return 1

    opt = entries.get('optimizer')
    steps = set()
    if opt is None:
        print('FAIL: checkpoint has no optimizer state')
        ok = False
    else:
        for idx, st in opt.get('state', {}).items():
            if 'step' in st:
                steps.add(float(st['step']))
    print('Adam per-parameter step values: %s' % (sorted(steps) if steps else None))
    if steps and steps != {float(it)}:
        print('FAIL: Adam steps %s != it %s' % (sorted(steps), it))
        ok = False
    elif not steps and opt is not None:
        print('FAIL: optimizer state has no step counters')
        ok = False

    if args.min_it is not None:
        if it > args.min_it:
            print('OK: it %d > --min-it %d' % (it, args.min_it))
        else:
            print('FAIL: it %d is not > --min-it %d' % (it, args.min_it))
            ok = False

    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
