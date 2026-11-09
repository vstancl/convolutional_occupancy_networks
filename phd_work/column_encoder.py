"""[phd] Control encoder: the raw voxel column along each ray, as channels.

Same ray encoder and plane placement as DexelEncoder (phd_work/dexel_encoder.py), but the per-ray
input is the occupancy column itself: 32 voxels as 32 channels, in voxel order along the ray.
The depth parametrisation (dexel slots) is removed, so this separates "per-ray vector encoding"
from "depth parametrisation".

Input: the voxel grid as loaded by VoxelsField, (B, 32, 32, 32) (input_type voxels, no new data field).
For plane p, the ray axis and image axes are those of phd_work/dexels.py:
    xz (p=0): rays along voxel axis 1, image axes (0, 2)
    xy (p=1): rays along voxel axis 2, image axes (0, 1)
    yz (p=2): rays along voxel axis 0, image axes (1, 2)
The ray encoder is Conv2d(32, 64, 1) -> ReLU -> Conv2d(64, 32, 1) -> actvn (the DexelEncoder's
ray encoder with 2L = 32 input channels), then the parent's scatter and U-Net, unchanged.
"""
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from phd_work.dexel_encoder import DexelEncoder, PLANE_INDEX, PLANE_NAMES  # noqa: E402
from phd_work.dexels import RAY_AXIS, RES  # noqa: E402


class VoxelColumnEncoder(DexelEncoder):
    ''' Voxel-column control encoder.

    Args:
        c_dim (int): output feature dimension (32).
        hidden_dim (int): hidden width of the ray encoder (64).
        **kwargs: the LocalVoxelEncoder arguments (dim, unet, unet_kwargs, plane_resolution,
            plane_type, kernel_size, padding, ...).
    '''

    def __init__(self, c_dim=32, hidden_dim=64, **kwargs):
        # DexelEncoder builds Conv2d(2L, hidden, 1); 2L = 32 = RES input channels, so this is
        # the column ray encoder. L is only used to size that layer here.
        super().__init__(L=RES // 2, c_dim=c_dim, hidden_dim=hidden_dim, **kwargs)

    def ray_columns(self, voxels, plane):
        ''' (B, 32, 32, 32): channel k = occupancy at position k along the ray axis; (u, v) image
        indices as in dexels.py. Channel k of image (u, v) is voxel (u, k, v) for xz, (u, v, k) for xy,
        (k, u, v) for yz. '''
        x = torch.movedim(voxels, 1 + RAY_AXIS[plane], -1)  # (B, a, b, k): image axes a, b in order
        return x.permute(0, 3, 1, 2).contiguous()           # (B, k, a, b)

    def forward(self, voxels):
        if voxels.dim() != 4 or tuple(voxels.shape[1:]) != (RES, RES, RES):
            raise ValueError('voxels must have shape (B, %d, %d, %d), got %s'
                             % (RES, RES, RES, tuple(voxels.shape)))
        voxels = voxels.float()
        batch_size = voxels.size(0)
        fea = {}
        for name in PLANE_NAMES:
            if name not in self.plane_type:
                continue
            p_idx = PLANE_INDEX[name]
            feat = self.actvn(self.ray_encoder(self.ray_columns(voxels, p_idx)))  # (B, c_dim, RES, RES)
            c = feat.reshape(batch_size, self.c_dim, RES * RES).permute(0, 2, 1)
            pts = self.ray_points(p_idx, device=voxels.device).unsqueeze(0).expand(batch_size, -1, -1)
            fea[name] = self.generate_plane_features(pts, c, plane=name)
        return fea
