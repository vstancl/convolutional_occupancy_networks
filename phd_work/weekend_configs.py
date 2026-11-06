"""[phd] Write the weekend configs for one round (train, test, and the input-row eval configs).

Usage:
    uv run python phd_work/weekend_configs.py --round 1 --cfg-dir phd_work/configs/weekend
        [--out-prefix out/phd_w] [--dry]

--dry: max_it 200, validate_every 100, checkpoint_every 100, test_split test_smoke, out-prefix out/dry_w.
Every config inherits a committed base config; only the settings listed here are overridden.
Train configs: <cfg-dir>/w{R}_{variant}.yaml; test configs: <cfg-dir>/w{R}_{variant}_test.yaml.
"""
import argparse
import os

import yaml

BASE = {
    'voxel3p': 'phd_work/configs/voxel_3plane_3cls.yaml',
    'dexel3p_L32': 'phd_work/configs/dexel_3plane_3cls_L32.yaml',
    'dexel3p_L2': 'phd_work/configs/dexel_3plane_3cls_L2.yaml',
    'dexel1p_L32': 'phd_work/configs/dexel_1plane_3cls_L32.yaml',
    'dexel1p_L2': 'phd_work/configs/dexel_1plane_3cls_L2.yaml',
    'voxel1p': 'phd_work/configs/voxel_1plane_3cls.yaml',
    'dexel3p_L4': 'phd_work/configs/dexel_3plane_3cls_L4.yaml',
    'grid32': 'phd_work/configs/voxel_grid32_3cls.yaml',
    'dexel3p_L1': 'phd_work/configs/dexel_3plane_3cls_L1.yaml',
}
ROUND1 = ['voxel3p', 'dexel3p_L32', 'dexel3p_L2', 'dexel1p_L32', 'dexel1p_L2', 'voxel1p', 'dexel3p_L4',
          'grid32', 'dexel3p_L1']
ROUND2 = ['voxel3p', 'dexel3p_L32', 'dexel3p_L2', 'dexel1p_L32', 'dexel1p_L2', 'voxel1p']
ROUNDS = {1: ROUND1, 2: ROUND2}


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True, choices=sorted(ROUNDS))
    ap.add_argument('--cfg-dir', required=True)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()
    R = args.round
    for v in ROUNDS[R]:
        name = 'w%d_%s' % (R, v)
        train = {'inherit_from': BASE[v], 'training': {
            'out_dir': ('out/dry_w%d_%s' % (R, v)) if args.dry else ('out/phd_w%d_%s' % (R, v)),
            'max_it': 200 if args.dry else 120000}}
        if args.dry:
            train['training'].update(validate_every=100, checkpoint_every=100)
        else:
            train['training']['validate_every'] = 2000
        train_path = os.path.join(args.cfg_dir, name + '.yaml')
        write(train_path, train)
        test = {'inherit_from': train_path, 'generation': {'generation_dir': 'generation_best'},
                'test': {'model_file': 'model_best.pt'}}
        if args.dry:
            test['data'] = {'test_split': 'test_smoke'}
        write(os.path.join(args.cfg_dir, name + '_test.yaml'), test)
    # input-row (xz, one plane) eval configs for the CPU steps after the trainings
    for L in (2, 32):
        prefix = 'out/dry_' if args.dry else 'out/phd_'
        cfg = {'inherit_from': 'phd_work/configs/input_row_L%d.yaml' % L,
               'training': {'out_dir': prefix + 'input_xz_L%d' % L}}
        if args.dry:
            cfg['data'] = {'test_split': 'test_smoke'}
        write(os.path.join(args.cfg_dir, 'input_xz_L%d.yaml' % L), cfg)
    print('wrote %d round-%d configs to %s' % (len(ROUNDS[R]) * 2 + 2, R, args.cfg_dir))


if __name__ == '__main__':
    main()
