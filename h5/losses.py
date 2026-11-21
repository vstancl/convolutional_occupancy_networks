"""Losses of the h5 ray decoders (H5-2): interval hits and dense columns.

``hits_loss`` takes ``s_pred`` (..., 4K) and ``mask_logits`` (..., 2K) from ``h5/models.py`` and the
batch fields of ``h5/data.py`` (``hits_s`` in cell units, ``hits_mask``). Hit slot ``h`` belongs to
interval mask ``h // 2``, and the two slot orders of ``h5/slots.py`` agree on that grouping.

``dense_loss`` takes the (..., 128) logits and the dense target of ``h5/data.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from _paths import add_external_paths

add_external_paths()

from csgbox.constants import GATE_TSDF_TAU_CELLS  # noqa: E402
from csgbox.grid import cell_size  # noqa: E402

# Upper bound on every per-slot positive weight of the mask BCE. Rare slots (deep intervals) have
# weights of about 12 to 31 at K = 4; the cap keeps a single slot from dominating the mask loss.
POS_WEIGHT_CAP: float = 10.0

# Default per-slot pos_weight for K = 2, slots [F0, F1, B1, B0] (h5-1a: (1 - p)/p under the sampler).
POS_WEIGHT_K2: tuple[float, ...] = (0.4491, 4.0451, 4.0451, 0.4491)

M: int = 128  # dense columns have 128 cells


def _flat(x: torch.Tensor, width: int) -> torch.Tensor:
    return x.reshape(-1, width)


def hits_loss(s_pred: torch.Tensor, mask_logits: torch.Tensor, s_tgt: torch.Tensor, mask_tgt: torch.Tensor,
              pos_weight: torch.Tensor, lam: float = 1.0) -> dict[str, torch.Tensor]:
    """Hit location loss and interval mask loss.

    ``L_t``: ``|s_pred - s_tgt|`` in cell units for each hit slot, averaged over the rays where the
    slot exists (its interval mask is positive), then averaged over the hit slots that have at least
    one such ray in the batch. Masked slots contribute nothing, whatever ``s_tgt`` holds.

    ``L_mask``: binary cross-entropy with logits on the 2K mask slots, with the per-slot
    ``pos_weight`` clamped to ``POS_WEIGHT_CAP``, averaged over rays and slots.

    Returns ``{"total": L_t + lam * L_mask, "L_t": ..., "L_mask": ...}``.
    """
    width_s = s_pred.shape[-1]
    width_m = mask_logits.shape[-1]
    if width_s != 2 * width_m:
        raise ValueError("s has 4K entries and mask has 2K")
    if pos_weight.numel() != width_m:
        raise ValueError("pos_weight must have 2K entries")
    s = _flat(s_pred, width_s)
    st = _flat(s_tgt, width_s).to(s.dtype)
    lg = _flat(mask_logits, width_m)
    mt = _flat(mask_tgt, width_m)

    hit_exists = mt.repeat_interleave(2, dim=-1).to(s.dtype)  # (n, 4K): hit 2j, 2j+1 use mask j
    count = hit_exists.sum(dim=0)  # rays where each hit slot exists
    err = (s - st).abs()
    per_slot = (err * hit_exists).sum(dim=0) / count.clamp_min(1.0)
    present = count > 0
    if bool(present.any()):
        l_t = per_slot[present].mean()
    else:
        l_t = s.sum() * 0.0

    pw = pos_weight.to(device=lg.device, dtype=lg.dtype).clamp(max=POS_WEIGHT_CAP)
    l_mask = F.binary_cross_entropy_with_logits(lg, mt.to(lg.dtype), pos_weight=pw, reduction="mean")
    return {"total": l_t + lam * l_mask, "L_t": l_t, "L_mask": l_mask}


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over the entries of ``m``; zero (no gradient) if ``m`` is empty."""
    n = m.sum()
    if bool(n == 0):
        return x.sum() * 0.0
    return (x * m).sum() / n


def dense_loss(logits: torch.Tensor, y: torch.Tensor, informative: torch.Tensor, target: str) -> dict[str, torch.Tensor]:
    """Dense column loss, ``0.5 * mean(informative cells) + 0.5 * mean(other cells)``.

    Per target:
    - ``occfrac_128``: BCE with logits against the soft occupancy fraction ``y``.
    - ``binary_128``: BCE with logits against ``y`` (0 or 1).
    - ``tsdf_128``: L1 between ``2.5 h tanh(logits)`` and ``y``, in units of ``h``.

    A mean over an empty set is skipped: its 0.5 term is zero (see decision in ``h5/docs/h5-2.md``).
    Returns ``{"total", "informative", "other", "entropy_floor", "excess"}``. ``entropy_floor`` is the
    same 0.5/0.5 mean of the binary entropy of ``y`` (occfrac only; 0 for binary and tsdf), and
    ``excess = total - entropy_floor``.
    """
    lg = _flat(logits, M)
    yy = _flat(y, M)
    inf = _flat(informative, M).to(torch.bool)
    if target == "occfrac_128" or target == "binary_128":
        cell = F.binary_cross_entropy_with_logits(lg, yy.to(lg.dtype), reduction="none")
    elif target == "tsdf_128":
        h = cell_size(M)
        pred = GATE_TSDF_TAU_CELLS * h * torch.tanh(lg)
        cell = (pred - yy.to(lg.dtype)).abs() / h
    else:
        raise ValueError(f"no dense loss for target {target!r}")
    l_inf = _masked_mean(cell, inf.to(cell.dtype))
    l_oth = _masked_mean(cell, (~inf).to(cell.dtype))
    total = 0.5 * l_inf + 0.5 * l_oth
    # Irreducible part of the loss: the binary entropy of a soft target is the BCE floor. Zero for the
    # binary and tsdf targets, whose entropy floor is not defined here (their loss is not a BCE on soft y).
    if target == "occfrac_128":
        yc = yy.to(lg.dtype).clamp(0.0, 1.0)
        ent = -(torch.xlogy(yc, yc) + torch.xlogy(1.0 - yc, 1.0 - yc))  # 0 log 0 = 0
        e_inf = _masked_mean(ent, inf.to(ent.dtype))
        e_oth = _masked_mean(ent, (~inf).to(ent.dtype))
        floor = 0.5 * e_inf + 0.5 * e_oth
    else:
        floor = total.detach() * 0.0
    return {"total": total, "informative": l_inf, "other": l_oth, "entropy_floor": floor,
            "excess": total - floor}
