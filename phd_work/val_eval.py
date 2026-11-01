"""[phd] Evaluate a checkpoint on the validation split exactly as train.py does.

Builds model, trainer and the val dataset the same way train.py does
(config.get_dataset('val', cfg, return_idx=True), batch_size=1, shuffle=False,
same collate_remove_none / worker_init_fn), loads the weights, and prints
trainer.evaluate(val_loader). The model-selection metric is printed on its own line.

Usage (from the repo root):
    uv run python phd_work/val_eval.py <config> [--model PATH_OR_URL]

Without --model, the weights come from <out_dir>/model_best.pt, as train.py would
load them. With --model, it may be a local path or a http(s) URL (loaded via
CheckpointIO, like generate.py does for test.model_file).
"""
import argparse
import os
import sys

import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src import config, data  # noqa: E402
from src.checkpoints import CheckpointIO  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--model', type=str, default=None,
                        help='Checkpoint path or http(s) URL. Default: <out_dir>/model_best.pt')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device('cuda' if is_cuda else 'cpu')
    out_dir = cfg['training']['out_dir']
    model_selection_metric = cfg['training']['model_selection_metric']

    # Datasets: same construction as train.py (train dataset is needed by get_model).
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg, return_idx=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
        collate_fn=data.collate_remove_none, worker_init_fn=data.worker_init_fn)

    # Model and trainer, as in train.py. The optimizer is only needed to build the trainer.
    model = config.get_model(cfg, device=device, dataset=train_dataset)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    if args.model is None:
        checkpoint_io = CheckpointIO(out_dir, model=model)
        checkpoint_io.load('model_best.pt')
        source = os.path.join(out_dir, 'model_best.pt')
    else:
        checkpoint_io = CheckpointIO(os.path.dirname(os.path.abspath(args.model)) or '.', model=model)
        checkpoint_io.load(args.model)
        source = args.model

    print('config: %s' % args.config)
    print('weights: %s' % source)
    print('val samples: %d' % len(val_dataset))
    eval_dict = trainer.evaluate(val_loader)
    for k in sorted(eval_dict):
        print('%s: %.6f' % (k, eval_dict[k]))
    print('Validation metric (%s): %.4f' % (model_selection_metric, eval_dict[model_selection_metric]))
    print('Validation metric exact (%s): %r' % (model_selection_metric, float(eval_dict[model_selection_metric])))


if __name__ == '__main__':
    main()
