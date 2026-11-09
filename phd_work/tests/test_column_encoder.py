"""[phd] Tests for phd_work/column_encoder.py (voxel-column control encoder, CPU)."""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHD_WORK = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(PHD_WORK, '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, PHD_WORK)

from src.common import coordinate2index, normalize_coordinate  # noqa: E402
from src.config import load_config  # noqa: E402
from src.encoder.voxels import LocalVoxelEncoder  # noqa: E402
from phd_work.column_encoder import VoxelColumnEncoder  # noqa: E402  (package path: one module instance)
from phd_work.dexel_encoder import DexelEncoder  # noqa: E402
from phd_work.dexels import RES  # noqa: E402

PLANE_NAMES = ('xz', 'xy', 'yz')


def make_kwargs():
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        cfg = load_config(os.path.join(PHD_WORK, 'configs', 'voxel_3plane_3cls.yaml'), 'configs/default.yaml')
    finally:
        os.chdir(cwd)
    kw = copy.deepcopy(cfg['model']['encoder_kwargs'])
    kw.update(dim=cfg['data']['dim'], padding=cfg['data']['padding'])
    return kw, cfg['model']['c_dim']


KW, C_DIM = make_kwargs()


def cell(points, plane, padding=KW['padding'], reso=KW['plane_resolution']):
    xy = normalize_coordinate(points.unsqueeze(0).clone(), plane=plane, padding=padding)
    return coordinate2index(xy, reso)[0, 0]


def test_column_order_equals_voxel_order_along_ray():
    g = torch.Generator().manual_seed(0)
    vox = (torch.rand(2, RES, RES, RES, generator=g) > 0.6).float()
    enc = VoxelColumnEncoder(c_dim=C_DIM, **KW)
    for plane, idx in [(0, lambda k, u, v: (u, k, v)),   # xz: rays along axis 1, image (0, 2)
                       (1, lambda k, u, v: (u, v, k)),   # xy: rays along axis 2, image (0, 1)
                       (2, lambda k, u, v: (k, u, v))]:  # yz: rays along axis 0, image (1, 2)
        cols = enc.ray_columns(vox, plane)
        assert cols.shape == (2, RES, RES, RES)
        for u, v, k in [(0, 0, 0), (3, 17, 5), (31, 2, 30), (5, 5, 31)]:
            for b in range(2):
                i, j, l = idx(k, u, v)
                # channel k of image (u, v) is the voxel at position k along the ray
                assert cols[b, k, u, v].item() == vox[b, i, j, l].item(), (plane, u, v, k)


def test_ray_cells_match_dexel_encoder():
    dex = DexelEncoder(L=32, c_dim=C_DIM, **KW)
    col = VoxelColumnEncoder(c_dim=C_DIM, **KW)
    for plane in PLANE_NAMES:
        a = cell(dex.ray_points(PLANE_NAMES.index(plane)), plane).numpy()
        b = cell(col.ray_points(PLANE_NAMES.index(plane)), plane).numpy()
        assert np.array_equal(a, b), plane
        assert len(set(b.tolist())) == RES * RES


def test_single_ray_feature_lands_in_its_cell_only():
    enc = VoxelColumnEncoder(c_dim=C_DIM, **KW)
    unet, enc.unet = enc.unet, None
    try:
        for plane in PLANE_NAMES:
            p_idx = PLANE_NAMES.index(plane)
            pts = enc.ray_points(p_idx)
            ray = 5 * RES + 20
            c = torch.zeros(1, RES * RES, C_DIM)
            c[0, ray, :] = 1.0
            out = enc.generate_plane_features(pts.unsqueeze(0), c, plane=plane)
            nz = out[0].abs().sum(0).reshape(-1).nonzero().flatten().tolist()
            assert nz == [int(cell(pts, plane)[ray])], plane
    finally:
        enc.unet = unet


def test_forward_shapes_and_backward():
    torch.manual_seed(0)
    enc = VoxelColumnEncoder(c_dim=C_DIM, **KW)
    vox = (torch.rand(2, RES, RES, RES) > 0.7).float()
    out = enc(vox)
    assert set(out) == set(KW['plane_type'])
    for k, v in out.items():
        assert v.shape == (2, C_DIM, KW['plane_resolution'], KW['plane_resolution'])
    sum(v.sum() for v in out.values()).backward()
    for prm in enc.ray_encoder.parameters():
        assert prm.grad is not None and prm.grad.abs().sum().item() > 0


def test_parameter_counts():
    enc = VoxelColumnEncoder(c_dim=C_DIM, **KW)
    voxel = LocalVoxelEncoder(c_dim=C_DIM, **KW)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(enc.ray_encoder) == (32 * 64 + 64) + (64 * 32 + 32) == 4192
    assert count(enc.unet) == count(voxel.unet)
    assert count(enc) == count(enc.ray_encoder) + count(enc.unet)
    assert count(voxel.conv_in) == 896


def test_registered_as_voxel_column():
    from phd_work.dexel_register import register
    from src.encoder import encoder_dict
    register()
    try:
        assert encoder_dict['voxel_column'] is VoxelColumnEncoder
        assert encoder_dict['dexel_local'] is DexelEncoder
    finally:
        register()
