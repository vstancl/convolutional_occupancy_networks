"""Ray decoder models for the h5 steps (H5-2).

``H5RayModel(target, cfg)`` is a ConvONet encoder (``pointnet_local_pool`` with three 64^2 planes and
a U-Net, as in ``configs/pointcloud/shapenet_3plane.yaml``) followed by one of three decoders:

- ``hits_8``, ``hits_16`` (and the dense targets): a **ray trunk** that samples the planes at 64
  points per ray, then 5 dilated residual blocks along the ray; then a head.
- ``hits_8_plane``: no trunk. Only the plane perpendicular to the ray, sampled at the ray centre,
  goes to an MLP.

Batch layout: ``inputs`` (B, N, 3) float32 points; ``ray_axis`` (B, R) int64 with 0 = x, 1 = y,
2 = z; ``ray_origin_uv`` (B, R, 2) float32 with the cell centre on the two perpendicular axes, in
(col, row) order. The (col, row) axes are (y, z) for x-rays, (x, z) for y-rays and (x, y) for z-rays.

Outputs (dict):

- hits targets: ``s_pred`` (B, R, 4K) float32 in cell units, ``mask_logits`` (B, R, 2K).
  Slot order as in ``h5/slots.py``: ``s`` follows the t order, ``mask_logits`` the mask order.
- dense targets: ``logits`` (B, R, 128); cell ``2m + i`` is channel ``i`` of sample ``m``.

The ConvONet modules are used as they are: ``LocalPoolPointnet`` for the encoder and
``LocalDecoder.sample_plane_feature`` for the plane sampling (with ConvONet's padding and sample
mode). Nothing from ``h5/`` is imported by ConvONet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

CONVONET_ROOT: Path = Path(__file__).resolve().parents[1]
if str(CONVONET_ROOT) not in sys.path:
    sys.path.append(str(CONVONET_ROOT))

from src.conv_onet.models.decoder import LocalDecoder  # noqa: E402
from src.encoder.pointnet import LocalPoolPointnet  # noqa: E402

from _paths import add_external_paths  # noqa: E402

add_external_paths()

from csgbox.constants import BOX_LO  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402

M: int = 128  # cells per axis of the dense targets and of the hit coordinate s
H: float = cell_size(M)  # cell size over the box, h = (BOX_HI - BOX_LO) / 128
N_SAMPLES: int = 64  # trunk samples per ray; sample m covers cells 2m and 2m+1

# ConvONet's defaults and shapenet_3plane.yaml values.
PLANES: tuple[str, ...] = ("xz", "xy", "yz")  # plane_type order of the config; sum order of LocalDecoder
PADDING: float = 0.1  # ConvONet's padding: the box +-0.5 maps to +-0.55 on the planes
SAMPLE_MODE: str = "bilinear"  # ConvONet's sample_mode of LocalDecoder
ENC_C_DIM: int = 32  # encoder c_dim (= decoder c_dim)
ENC_HIDDEN: int = 32  # encoder hidden_dim
PLANE_RES: int = 64  # plane_resolution
UNET_KWARGS: dict = {"depth": 4, "merge_mode": "concat", "start_filts": 32}

TRUNK_HIDDEN: int = 32  # trunk width
TRUNK_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 16)  # one residual block per dilation
PLANE_MLP_HIDDEN: int = 128  # hidden width of the PlaneHitsHead MLP

# Ray axis -> (col axis, row axis, perpendicular plane name) for x-, y-, z-rays.
COL_AXIS: tuple[int, ...] = (1, 0, 0)
ROW_AXIS: tuple[int, ...] = (2, 2, 1)
PLANE_OF_AXIS: tuple[str, ...] = ("yz", "xz", "xy")
PLANE_INDEX_OF_AXIS: tuple[int, ...] = tuple(PLANES.index(p) for p in PLANE_OF_AXIS)

# Target -> head kind and K (K intervals per side, 4K hits; see h5/slots.py).
TARGET_KIND: dict[str, tuple[str, int]] = {
    "hits_8": ("hits", 2),
    "hits_16": ("hits", 4),
    "hits_8_plane": ("plane", 2),
    "occfrac_128": ("dense", 0),
    "tsdf_128": ("dense", 0),
    "binary_128": ("dense", 0),
}

# A namespace with the two attributes LocalDecoder.sample_plane_feature reads. The function is
# called unbound, so no decoder weights are created.
_SAMPLER = SimpleNamespace(padding=PADDING, sample_mode=SAMPLE_MODE)


def _ray_points(ray_axis: torch.Tensor, ray_origin_uv: torch.Tensor, axis_coords: torch.Tensor) -> torch.Tensor:
    """World points ``(B, R, S, 3)`` float32: the ray's (col, row) = ``ray_origin_uv`` and the axis
    coordinate ``axis_coords`` (S,) float64 along the ray's own axis."""
    dev = ray_origin_uv.device
    uv = ray_origin_uv.to(torch.float64)
    col = torch.tensor(COL_AXIS, device=dev)[ray_axis]
    row = torch.tensor(ROW_AXIS, device=dev)[ray_axis]
    base = uv[..., 0:1] * F.one_hot(col, 3).to(torch.float64) + uv[..., 1:2] * F.one_hot(row, 3).to(torch.float64)
    along = F.one_hot(ray_axis, 3).to(torch.float64)  # (B, R, 3)
    pts = base[:, :, None, :] + axis_coords.to(dev)[None, None, :, None] * along[:, :, None, :]
    return pts.to(torch.float32)


def ray_sample_points(ray_axis: torch.Tensor, ray_origin_uv: torch.Tensor) -> torch.Tensor:
    """Sample points of every ray, ``(B, R, 64, 3)`` float32. Sample m sits at the axis coordinate
    ``BOX_LO + (2m + 1) h``, the centre of cells 2m and 2m+1."""
    m = torch.arange(N_SAMPLES, dtype=torch.float64)
    world = BOX_LO + (2.0 * m + 1.0) * H
    return _ray_points(ray_axis, ray_origin_uv, world)


def summed_plane_feature(p: torch.Tensor, planes: dict) -> torch.Tensor:
    """``c`` (B, T, c_dim): the sum over the three planes of ConvONet's ``sample_plane_feature``.

    Same order and same function as ``LocalDecoder.forward``, which sums ``xz``, ``xy``, ``yz``.
    """
    c = 0
    for name in PLANES:
        c = c + LocalDecoder.sample_plane_feature(_SAMPLER, p, planes[name], plane=name)
    return c.transpose(1, 2)


class _ResBlock1d(nn.Module):
    """``x + Conv1x1(ReLU(Conv_k3_dil_d(ReLU(x))))`` along the sample axis; the last conv is
    zero-initialised, as in ConvONet's ResNet blocks."""

    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.conv_k3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv1d(hidden, hidden, kernel_size=1)
        nn.init.zeros_(self.conv_1.weight)
        nn.init.zeros_(self.conv_1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_1(F.relu(self.conv_k3(F.relu(x))))


class RayTrunk(nn.Module):
    """Per-sample features along each ray: ``fc_p(p) + fc_c(c)``, then residual dilated blocks."""

    def __init__(self, c_dim: int = ENC_C_DIM, hidden: int = TRUNK_HIDDEN,
                 dilations: tuple[int, ...] = TRUNK_DILATIONS):
        super().__init__()
        self.hidden = hidden
        self.fc_p = nn.Linear(3, hidden)
        self.fc_c = nn.Linear(c_dim, hidden)
        self.blocks = nn.ModuleList([_ResBlock1d(hidden, d) for d in dilations])

    def forward(self, pts: torch.Tensor, planes: dict) -> torch.Tensor:
        """``pts`` (B, R, S, 3) to features (B, R, hidden, S)."""
        B, R, S, _ = pts.shape
        p = pts.reshape(B, R * S, 3)
        c = summed_plane_feature(p, planes)  # (B, R*S, c_dim)
        x = self.fc_p(p.float()) + self.fc_c(c)  # (B, R*S, hidden)
        x = x.reshape(B, R, S, self.hidden).permute(0, 1, 3, 2).reshape(B * R, self.hidden, S)
        for blk in self.blocks:
            x = blk(x)
        return x.reshape(B, R, self.hidden, S)


class DenseHead(nn.Module):
    """``Conv1d(hidden, 2, 1)`` per sample: cell ``2m + i`` = channel ``i`` of sample ``m``."""

    def __init__(self, hidden: int = TRUNK_HIDDEN):
        super().__init__()
        self.conv = nn.Conv1d(hidden, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, R, C, S = x.shape
        y = self.conv(x.reshape(B * R, C, S))  # (B*R, 2, S)
        return y.permute(0, 2, 1).reshape(B, R, 2 * S)


class HitsHead(nn.Module):
    """Attention over the samples for each hit slot, plus a bounded residual; and a mask head.

    ``s_pred[slot] = sum_m p_m (2m + 1) + sum_m p_m * 2 tanh(r_m)`` with ``p = softmax_m(a)``, in
    cell units, so ``s_pred`` lies in ``[-1, 129]``. The slot order is the t order of ``h5/slots.py``.
    The mask logits come from max- and mean-pooled trunk features, in the mask order of slots.py.
    """

    def __init__(self, K: int, hidden: int = TRUNK_HIDDEN):
        super().__init__()
        self.K = K
        self.attn = nn.Conv1d(hidden, 4 * K, kernel_size=1)
        self.resid = nn.Conv1d(hidden, 4 * K, kernel_size=1)
        self.mask_mlp = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 2 * K))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, R, C, S = x.shape
        xf = x.reshape(B * R, C, S)
        a = self.attn(xf)
        r = self.resid(xf)
        with torch.autocast(device_type=x.device.type, enabled=False):
            a = a.float()
            r = r.float()
            cells = 2.0 * torch.arange(S, dtype=torch.float32, device=x.device) + 1.0  # 2m + 1
            p = torch.softmax(a, dim=-1)
            s = (p * cells).sum(-1) + (p * torch.tanh(r) * 2.0).sum(-1)  # (B*R, 4K)
        s = s.reshape(B, R, 4 * self.K)
        pooled = torch.cat([xf.amax(dim=-1), xf.mean(dim=-1)], dim=1)  # (B*R, 2C)
        mask = self.mask_mlp(pooled).reshape(B, R, 2 * self.K)
        return s, mask


class PlaneHitsHead(nn.Module):
    """Hits from the plane perpendicular to the ray, sampled at the ray centre (no trunk).

    ``s_pred = 64 (tanh(o) + 1)`` in cell units, in ``[0, 128]``; the mask logits are the last 2K outputs.
    """

    def __init__(self, c_dim: int = ENC_C_DIM, K: int = 2):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(c_dim, PLANE_MLP_HIDDEN), nn.ReLU(),
            nn.Linear(PLANE_MLP_HIDDEN, PLANE_MLP_HIDDEN), nn.ReLU(),
            nn.Linear(PLANE_MLP_HIDDEN, 6 * K),
        )

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.mlp(c)
        with torch.autocast(device_type=c.device.type, enabled=False):
            o = o.float()
            s = 64.0 * (torch.tanh(o[..., : 4 * self.K]) + 1.0)
            mask = o[..., 4 * self.K:]
        return s, mask


def perpendicular_plane_feature(ray_axis: torch.Tensor, ray_origin_uv: torch.Tensor, planes: dict) -> torch.Tensor:
    """``c`` (B, R, c_dim) from the plane perpendicular to each ray, at the ray centre (axis coordinate 0).

    Each ray uses only its own plane. The three planes are sampled at all rays with ConvONet's
    sampler and the perpendicular one is selected per ray. The two other samples are discarded, so
    the result depends on the perpendicular plane only (tested).
    """
    p = _ray_points(ray_axis, ray_origin_uv, torch.zeros(1, dtype=torch.float64))[:, :, 0]  # (B, R, 3)
    feats = torch.stack([LocalDecoder.sample_plane_feature(_SAMPLER, p, planes[name], plane=name).transpose(1, 2)
                         for name in PLANES], dim=2)  # (B, R, 3 planes, c_dim)
    sel = torch.tensor(PLANE_INDEX_OF_AXIS, device=feats.device)[ray_axis]  # (B, R)
    idx = sel[:, :, None, None].expand(-1, -1, 1, feats.shape[-1])
    return feats.gather(2, idx).squeeze(2)


def _encoder() -> LocalPoolPointnet:
    """ConvONet's encoder as in ``shapenet_3plane.yaml``."""
    return LocalPoolPointnet(
        c_dim=ENC_C_DIM, dim=3, hidden_dim=ENC_HIDDEN, plane_resolution=PLANE_RES,
        plane_type=list(PLANES), padding=PADDING, unet=True, unet_kwargs=dict(UNET_KWARGS),
    )


class H5RayModel(nn.Module):
    """Encoder, trunk (absent for ``hits_8_plane``) and head for one target.

    ``cfg`` may override ``hidden`` (trunk width) and ``dilations``; the default is the config.
    """

    def __init__(self, target: str, cfg: dict | None = None):
        super().__init__()
        if target not in TARGET_KIND:
            raise ValueError(f"target must be one of {sorted(TARGET_KIND)}, got {target!r}")
        cfg = dict(cfg or {})
        self.target = target
        self.kind, self.K = TARGET_KIND[target]
        self.encoder = _encoder()
        if self.kind == "plane":
            self.trunk = None
            self.head = PlaneHitsHead(c_dim=ENC_C_DIM, K=self.K)
        else:
            hidden = int(cfg.get("hidden", TRUNK_HIDDEN))
            self.trunk = RayTrunk(c_dim=ENC_C_DIM, hidden=hidden,
                                  dilations=tuple(cfg.get("dilations", TRUNK_DILATIONS)))
            self.head = HitsHead(self.K, hidden=hidden) if self.kind == "hits" else DenseHead(hidden=hidden)

    def encode(self, inputs: torch.Tensor) -> dict:
        """Plane features of one shape batch (run once per shape at evaluation)."""
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor, ray_axis: torch.Tensor, ray_origin_uv: torch.Tensor) -> dict:
        return self.decode(self.encode(inputs), ray_axis, ray_origin_uv)

    def decode(self, planes: dict, ray_axis: torch.Tensor, ray_origin_uv: torch.Tensor) -> dict:
        """Head outputs for the rays given the plane features (any number of rays)."""
        if self.kind == "plane":
            c = perpendicular_plane_feature(ray_axis, ray_origin_uv, planes)
            s, mask = self.head(c)
            return {"s_pred": s, "mask_logits": mask}
        pts = ray_sample_points(ray_axis, ray_origin_uv).to(ray_origin_uv.device)
        feats = self.trunk(pts, planes)
        if self.kind == "hits":
            s, mask = self.head(feats)
            return {"s_pred": s, "mask_logits": mask}
        return {"logits": self.head(feats)}


def n_params(module: nn.Module | None) -> int:
    """Number of parameters of a module (0 for None)."""
    return 0 if module is None else sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------------------------------
# Point decoders (H5-5a): the same encoder, and ConvONet's LocalDecoder as the point-wise decoder.
# ---------------------------------------------------------------------------------------------------

POINT_HIDDEN: int = 32  # shapenet_3plane.yaml decoder_kwargs.hidden_size; n_blocks is LocalDecoder's default (5)


class PointModel(nn.Module):
    """Encoder (as ``H5RayModel``) and ConvONet's ``LocalDecoder`` (simple_local, bilinear, padding 0.1).

    ``forward(inputs, query)`` returns ``{"logits": (B, P)}``: the occupancy logit of each query point.
    The decoder is ConvONet's own module, built with the settings of ``shapenet_3plane.yaml``.
    """

    kind = "points"

    def __init__(self, target: str, cfg: dict | None = None):
        super().__init__()
        if target not in POINT_TARGET_NAMES:
            raise ValueError(f"not a point target: {target!r}")
        self.target = target
        self.encoder = _encoder()
        self.decoder = LocalDecoder(dim=3, c_dim=ENC_C_DIM, hidden_size=POINT_HIDDEN, sample_mode=SAMPLE_MODE,
                                    padding=PADDING)
        self.trunk = None
        self.head = self.decoder

    def encode(self, inputs: torch.Tensor) -> dict:
        return self.encoder(inputs)

    def decode(self, planes: dict, query: torch.Tensor) -> torch.Tensor:
        """Logits (B, P) of the query points (B, P, 3) given the plane features."""
        return self.decoder(query.float(), planes)

    def forward(self, inputs: torch.Tensor, query: torch.Tensor) -> dict:
        return {"logits": self.decode(self.encode(inputs), query)}


# The point-decoder targets: data.POINT_TARGETS + data.RAYINT_TARGETS + data.MIX_TARGETS (H5-K2-1b, K2'-1; a test
# checks that they agree).
POINT_TARGET_NAMES: tuple[str, ...] = ("points_std", "points_strong", "points_rayint_a", "points_rayint_b",
                                     "points_rayint_c", "points_mix_a", "points_mix_b")


def build_model(target: str, cfg: dict | None = None) -> nn.Module:
    """``PointModel`` for the point targets, ``H5RayModel`` for the others."""
    if target in POINT_TARGET_NAMES:
        return PointModel(target, cfg)
    return H5RayModel(target, cfg)
