"""[phd] Tests for phd_work/dexel_encoder.py (CPU)."""
import copy
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHD_WORK = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(PHD_WORK, '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, PHD_WORK)

from src.common import coordinate2index, normalize_coordinate  # noqa: E402
from src.config import load_config  # noqa: E402
from src.encoder.voxels import LocalVoxelEncoder  # noqa: E402
from dexel_encoder import DexelEncoder  # noqa: E402
from dexels import RAY_AXIS, RES, voxels_to_dexel_slots  # noqa: E402

PLANE_NAMES = ('xz', 'xy', 'yz')


def make_kwargs():
    """Encoder kwargs and c_dim from the 3-class config (inherits configs/voxel/shapenet_3plane.yaml)."""
    cwd = os.getcwd()
    os.chdir(REPO)  # load_config resolves inherit_from relative to the repo root
    try:
        cfg = load_config(os.path.join(PHD_WORK, 'configs', 'voxel_3plane_3cls.yaml'),
                          'configs/default.yaml')
    finally:
        os.chdir(cwd)
    kw = copy.deepcopy(cfg['model']['encoder_kwargs'])
    kw.update(dim=cfg['data']['dim'], padding=cfg['data']['padding'])
    return kw, cfg['model']['c_dim']


KW, C_DIM = make_kwargs()


def cell_index(points, plane, padding=KW['padding'], reso=KW['plane_resolution']):
    ''' Flat plane-cell index of each point, as in generate_plane_features. points: (N, 3). '''
    xy = normalize_coordinate(points.unsqueeze(0).clone(), plane=plane, padding=padding)
    return coordinate2index(xy, reso)[0, 0]  # coordinate2index returns (B, 1, N)


def voxel_point_set():
    ''' Points of LocalVoxelEncoder.forward, in its ordering (i, j, k) -> i*1024 + j*32 + k. '''
    c = torch.linspace(-0.5, 0.5, RES)
    p = torch.stack(torch.meshgrid(c, c, c, indexing='ij'), dim=-1)
    return p.reshape(-1, 3)


def random_slots(batch, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth = torch.randn(batch, 3, L, RES, RES, generator=g)
    mask = (torch.rand(batch, 3, L, RES, RES, generator=g) > 0.5).float()
    return torch.cat([depth, mask], dim=2)


def test_output_keys_and_shapes_match_local_voxel_encoder():
    torch.manual_seed(0)
    local = LocalVoxelEncoder(c_dim=C_DIM, **KW).eval()
    dex = DexelEncoder(L=4, c_dim=C_DIM, **KW).eval()
    vox = (torch.rand(2, RES, RES, RES) > 0.7).float()
    slots = random_slots(2, 4)
    with torch.no_grad():
        out_l = local(vox)
        out_d = dex(slots)
    assert set(out_d.keys()) == set(out_l.keys()) == set(KW['plane_type'])
    for k in out_l:
        assert out_d[k].shape == out_l[k].shape == (2, C_DIM, KW['plane_resolution'], KW['plane_resolution'])
        assert out_d[k].dtype == out_l[k].dtype


def test_unet_config_matches_yaml():
    # the kwargs used everywhere else in this file are the shapenet_3plane ones
    assert KW['plane_type'] == ['xz', 'xy', 'yz']
    assert KW['plane_resolution'] == 64
    assert KW['unet'] is True
    assert KW['unet_kwargs']['depth'] == 4
    assert KW['unet_kwargs']['start_filts'] == 32
    assert C_DIM == 32


@pytest.mark.parametrize('plane', PLANE_NAMES)
def test_placement_matches_voxel_columns(plane):
    p_idx = PLANE_NAMES.index(plane)
    dex = DexelEncoder(L=2, c_dim=C_DIM, **KW)
    ray_cells = cell_index(dex.ray_points(p_idx), plane).numpy()

    vox_cells = cell_index(voxel_point_set(), plane).numpy()
    ids = np.arange(RES ** 3).reshape(RES, RES, RES)
    ray_axis = RAY_AXIS[p_idx]
    moved = np.moveaxis(ids, ray_axis, -1)  # (u, v, ray) -> voxel ids of the column
    for u in range(RES):
        for v in range(RES):
            column = moved[u, v, :]
            cells = vox_cells[column]
            assert (cells == ray_cells[u * RES + v]).all(), (plane, u, v)
    assert len(set(ray_cells.tolist())) == RES * RES


def test_single_ray_feature_lands_in_its_cell_only():
    dex = DexelEncoder(L=2, c_dim=C_DIM, **KW)
    unet, dex.unet = dex.unet, None  # generate_plane_features without the U-Net
    try:
        for plane in PLANE_NAMES:
            p_idx = PLANE_NAMES.index(plane)
            pts = dex.ray_points(p_idx)
            u, v = 5, 20
            ray = u * RES + v
            c = torch.zeros(1, RES * RES, C_DIM)
            c[0, ray, :] = 1.0
            out = dex.generate_plane_features(pts.unsqueeze(0), c, plane=plane)
            assert out.shape == (1, C_DIM, KW['plane_resolution'], KW['plane_resolution'])
            nonzero = out[0].abs().sum(0).reshape(-1).nonzero().flatten().tolist()
            expected_cell = int(cell_index(pts, plane)[ray])
            assert nonzero == [expected_cell], plane
            assert torch.all(out[0].reshape(C_DIM, -1)[:, expected_cell] == 1.0)
    finally:
        dex.unet = unet


def test_backward_gives_nonzero_ray_encoder_grads():
    torch.manual_seed(1)
    dex = DexelEncoder(L=4, c_dim=C_DIM, **KW)
    slots = random_slots(2, 4, seed=1)
    feats = dex(slots)
    loss = sum(f.sum() for f in feats.values())
    loss.backward()
    params = list(dex.ray_encoder.parameters())
    assert len(params) == 4  # two convs, weight and bias each
    for prm in params:
        assert prm.grad is not None
        assert prm.grad.abs().sum().item() > 0


def test_end_to_end_from_voxels_runs():
    torch.manual_seed(2)
    vox = np.zeros((RES, RES, RES), dtype=bool)
    vox[4:20, 6:10, 8:12] = True  # a block
    vox[0, 0, :] = True
    slots = torch.from_numpy(voxels_to_dexel_slots(vox, 4)).unsqueeze(0)
    dex = DexelEncoder(L=4, c_dim=C_DIM, **KW)
    feats = dex(slots)
    for name in PLANE_NAMES:
        assert feats[name].shape == (1, C_DIM, KW['plane_resolution'], KW['plane_resolution'])
        assert torch.isfinite(feats[name]).all()


def test_param_counts_consistent():
    local = LocalVoxelEncoder(c_dim=C_DIM, **KW)
    dex = DexelEncoder(L=4, c_dim=C_DIM, **KW)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    unet_local = count(local.unet)
    assert count(dex.unet) == unet_local
    assert count(dex) == count(dex.ray_encoder) + count(dex.unet)
    assert count(local) == count(local.conv_in) + unet_local


def test_rejects_grid_plane_type_and_bad_shape():
    kw = dict(KW)
    kw['plane_type'] = ['grid']
    with pytest.raises(ValueError):
        DexelEncoder(L=4, c_dim=C_DIM, **kw)
    dex = DexelEncoder(L=4, c_dim=C_DIM, **KW)
    with pytest.raises(ValueError):
        dex(torch.zeros(1, 3, 6, RES, RES))  # wrong channel count for L=4
