"""[phd] Register the dexel encoder and dexel input field with the upstream config code.

No upstream file is edited. register():
  - adds 'dexel_local' to src.encoder.encoder_dict in place (src/conv_onet/config.py
    imported that same dict object, so it sees the entry);
  - replaces src.config.get_inputs_field with a wrapper: input_type 'dexels' gives a
    DexelSlotsField, every other input type goes to the original function unchanged.
    get_dataset looks get_inputs_field up as a module global, so the wrapper is used.

register() is idempotent. unregister() restores the original state (used by tests).
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src import config as upstream_config  # noqa: E402
from src.data.fields import VoxelsField  # noqa: E402
from src.encoder import encoder_dict  # noqa: E402
from src.utils import binvox_rw  # noqa: E402
from phd_work.dexel_encoder import DexelEncoder  # noqa: E402
from phd_work.column_encoder import VoxelColumnEncoder  # noqa: E402
from phd_work.dexels import voxels_to_dexel_slots  # noqa: E402

ENCODERS = {'dexel_local': DexelEncoder, 'voxel_column': VoxelColumnEncoder}  # control: P10

_saved = {}


class DexelSlotsField(VoxelsField):
    ''' Reads model.binvox like VoxelsField, returns dexel slots (3, 2*L, 32, 32) float32. '''

    def __init__(self, file_name, L, transform=None):
        super().__init__(file_name, transform=transform)
        self.L = L

    def load(self, model_path, idx, category):
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels_to_dexel_slots(voxels, self.L)


def _get_inputs_field(mode, cfg):
    if cfg['data']['input_type'] == 'dexels':
        return DexelSlotsField(cfg['data']['voxels_file'],
                               L=cfg['model']['encoder_kwargs']['L'])
    return _saved['orig_get_inputs_field'](mode, cfg)


def register():
    if 'orig_get_inputs_field' not in _saved:
        _saved['orig_get_inputs_field'] = upstream_config.get_inputs_field
    for name, cls in ENCODERS.items():
        _saved.setdefault('had_' + name, name in encoder_dict)
        encoder_dict[name] = cls
    upstream_config.get_inputs_field = _get_inputs_field


def unregister():
    if 'orig_get_inputs_field' in _saved:
        upstream_config.get_inputs_field = _saved.pop('orig_get_inputs_field')
    for name in ENCODERS:
        if 'had_' + name in _saved and not _saved.pop('had_' + name):
            encoder_dict.pop(name, None)
