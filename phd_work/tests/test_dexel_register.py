"""[phd] Tests for phd_work/dexel_register.py and phd_work/run.py (CPU)."""
import copy
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHD_WORK = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(PHD_WORK, '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, PHD_WORK)

from src import config  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.fields import VoxelsField  # noqa: E402
from src.encoder import encoder_dict  # noqa: E402
from src.utils import binvox_rw  # noqa: E402
from phd_work.dexel_encoder import DexelEncoder  # noqa: E402
from phd_work.dexel_register import DexelSlotsField, register, unregister  # noqa: E402
from phd_work.dexels import RES, voxels_to_dexel_slots  # noqa: E402

DATA = os.path.join(REPO, 'data', 'ShapeNet')
requires_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, '03001627', 'train.lst')),
    reason='ShapeNet data not present at %s' % DATA)


def load_cfg(name):
    cwd = os.getcwd()
    os.chdir(REPO)  # inherit_from paths are relative to the repo root
    try:
        return load_config(os.path.join(PHD_WORK, 'configs', name), 'configs/default.yaml')
    finally:
        os.chdir(cwd)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def ray_encoder_params(L, c_dim=32, hidden=64):
    return (2 * L * hidden + hidden) + (hidden * c_dim + c_dim)


@pytest.fixture
def registered():
    register()
    yield
    register()  # leave registered; register() is idempotent


def read_vox(path):
    with open(path, 'rb') as f:
        return binvox_rw.read_as_3d_array(f).data.astype(np.float32)


def test_register_adds_encoder_and_field_wrapper(registered):
    assert encoder_dict['dexel_local'] is DexelEncoder
    cfg = load_cfg('dexel_3plane_3cls_L32.yaml')
    field = config.get_inputs_field('train', cfg)
    assert isinstance(field, DexelSlotsField) and field.L == 32
    cfg_v = load_cfg('voxel_3plane_3cls.yaml')
    assert type(config.get_inputs_field('train', cfg_v)) is VoxelsField


@requires_data
def test_dataset_item_equals_conversion_of_binvox(registered):
    cfg = load_cfg('dexel_3plane_3cls_L32.yaml')
    ds = config.get_dataset('train', cfg)
    item = ds[0]
    inputs = item['inputs']
    assert inputs.shape == (3, 64, RES, RES)
    assert inputs.dtype == np.float32
    entry = ds.models[0]
    path = os.path.join(cfg['data']['path'], entry['category'], entry['model'], cfg['data']['voxels_file'])
    expected = voxels_to_dexel_slots(read_vox(path), 32)
    assert np.array_equal(inputs, expected)


@requires_data
@pytest.mark.parametrize('L', [32, 2])
def test_get_model_builds_dexel_encoder(registered, L):
    cfg = load_cfg('dexel_3plane_3cls_L%d.yaml' % L)
    model = config.get_model(cfg, device='cpu')
    enc = model.encoder
    assert isinstance(enc, DexelEncoder)
    assert enc.L == L and enc.hidden_dim == 64
    assert enc.unet is not None
    assert n_params(enc.ray_encoder) == ray_encoder_params(L)
    assert n_params(enc) == ray_encoder_params(L) + n_params(enc.unet)


@requires_data
def test_dexel_model_forward_on_dataset_item(registered):
    cfg = load_cfg('dexel_3plane_3cls_L2.yaml')
    ds = config.get_dataset('train', cfg)
    model = config.get_model(cfg, device='cpu')
    inputs = torch.from_numpy(ds[0]['inputs']).unsqueeze(0)
    p = torch.rand(1, 256, 3) - 0.5
    with torch.no_grad():
        logits = model(p, inputs).logits  # forward returns a Bernoulli distribution
    assert logits.shape == (1, 256)
    assert torch.isfinite(logits).all()


@requires_data
def test_transparency_for_voxel_config():
    cfg = load_cfg('voxel_3plane_3cls.yaml')
    g = torch.Generator().manual_seed(5)
    vox = (torch.rand(2, RES, RES, RES, generator=g) > 0.7).float()
    p = torch.rand(2, 512, 3, generator=g) - 0.5

    def build_and_run():
        torch.manual_seed(0)
        model = config.get_model(cfg, device='cpu')
        field = config.get_inputs_field('train', cfg)
        with torch.no_grad():
            out = model(p, vox).logits  # forward returns a Bernoulli distribution
        return type(field), type(model), n_params(model), out

    unregister()
    try:
        before = build_and_run()
        assert 'dexel_local' not in encoder_dict
    finally:
        register()
    after = build_and_run()

    assert before[0] is after[0] is VoxelsField
    assert before[1] is after[1]
    assert before[2] == after[2]
    max_diff = (before[3] - after[3]).abs().max().item()
    assert max_diff == 0.0, 'max |diff| = %g' % max_diff


def test_unregister_restores_original_state():
    from src import config as cfgmod
    unregister()
    original = cfgmod.get_inputs_field
    assert 'dexel_local' not in encoder_dict
    register()
    assert cfgmod.get_inputs_field is not original
    assert encoder_dict['dexel_local'] is DexelEncoder
    unregister()
    assert cfgmod.get_inputs_field is original
    assert 'dexel_local' not in encoder_dict
    register()


def test_run_py_registers_and_passes_argv(tmp_path):
    script = tmp_path / 'probe.py'
    script.write_text(textwrap.dedent('''
        import sys
        from src.encoder import encoder_dict
        print('ARGV', sys.argv)
        print('REGISTERED', 'dexel_local' in encoder_dict)
    '''))
    out = subprocess.run(
        [sys.executable, os.path.join(PHD_WORK, 'run.py'), str(script), 'a', '--b'],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ARGV ['%s', 'a', '--b']" % str(script) in out.stdout
    assert 'REGISTERED True' in out.stdout
