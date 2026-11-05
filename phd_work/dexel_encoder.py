"""[phd] Dexel encoder: ray features from voxel dexel slots, scattered onto ConvONet planes.

Subclass of LocalVoxelEncoder (src/encoder/voxels.py). The parent's voxel conv (conv_in)
is replaced by a 2D ray encoder shared across the three planes. Input slots come from
phd_work/dexels.py voxels_to_dexel_slots: shape (B, 3, 2L, 32, 32), plane index 0 'xz',
1 'xy', 2 'yz'. Each ray (u, v) of plane p gives one feature, placed at the 3D point
whose two image coordinates are the voxel centres of (u, v) and whose ray-axis
coordinate is 0. The parent's generate_plane_features (scatter to the plane grid, then
U-Net) is used unchanged.
"""
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.encoder.voxels import LocalVoxelEncoder  # noqa: E402
from phd_work.dexels import RAY_AXIS, RES  # noqa: E402

PLANE_INDEX = {'xz': 0, 'xy': 1, 'yz': 2}
PLANE_NAMES = ('xz', 'xy', 'yz')


class DexelEncoder(LocalVoxelEncoder):
    ''' Dexel encoder for voxel-derived ray slots.

    Args:
        L (int): slots per ray (even, or 1); input has 2*L channels per plane.
        c_dim (int): output feature dimension.
        hidden_dim (int): hidden width of the ray encoder.
        **kwargs: same arguments as LocalVoxelEncoder (dim, unet, unet_kwargs,
            plane_resolution, plane_type, kernel_size, padding, ...).
    '''

    def __init__(self, L, c_dim=32, hidden_dim=64, **kwargs):
        super().__init__(c_dim=c_dim, **kwargs)
        if 'grid' in self.plane_type or not any(p in self.plane_type for p in PLANE_NAMES):
            raise ValueError('DexelEncoder needs plane_type with xz/xy/yz, got %r' % (self.plane_type,))
        self.L = L
        self.hidden_dim = hidden_dim
        # replace the parent's voxel conv with the shared 2D ray encoder
        del self.conv_in
        self.ray_encoder = nn.Sequential(
            nn.Conv2d(2 * L, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, c_dim, 1),
        )

    def ray_points(self, plane, device=None):
        ''' Points of the rays of one plane, shape (RES*RES, 3), row u*RES + v.

        Image coordinates are the voxel centres of (u, v), in the two axes of the plane
        (increasing order); the ray-axis coordinate is 0.
        '''
        centres = torch.linspace(-0.5, 0.5, RES, dtype=torch.float32, device=device)
        ray_axis = RAY_AXIS[plane]
        img_axes = [a for a in range(3) if a != ray_axis]
        pts = torch.zeros(RES, RES, 3, dtype=torch.float32, device=device)
        pts[..., img_axes[0]] = centres.view(-1, 1)
        pts[..., img_axes[1]] = centres.view(1, -1)
        return pts.reshape(-1, 3)

    def forward(self, slots):
        ''' slots: (B, 3, 2L, 32, 32) float tensor. Returns dict like LocalVoxelEncoder. '''
        if slots.dim() != 5 or slots.shape[1:] != (3, 2 * self.L, RES, RES):
            raise ValueError('slots must have shape (B, 3, %d, %d, %d), got %s'
                             % (2 * self.L, RES, RES, tuple(slots.shape)))
        batch_size = slots.size(0)
        fea = {}
        for name in PLANE_NAMES:
            if name not in self.plane_type:
                continue
            p_idx = PLANE_INDEX[name]
            feat = self.actvn(self.ray_encoder(slots[:, p_idx]))  # (B, c_dim, RES, RES)
            c = feat.reshape(batch_size, self.c_dim, RES * RES).permute(0, 2, 1)
            pts = self.ray_points(p_idx, device=slots.device)
            pts = pts.unsqueeze(0).expand(batch_size, -1, -1)
            fea[name] = self.generate_plane_features(pts, c, plane=name)
        return fea
