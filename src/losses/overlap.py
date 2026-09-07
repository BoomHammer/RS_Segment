"""Enforce agreement across overlapping contexts during training."""

from torch.nn import functional as F


def overlap_consistency_loss(reference, prediction, mask, top, left, border=16):
    height, width = prediction.shape[-2:]
    target = reference.detach()[..., top : top + height, left : left + width].float()
    # Avoid matching artificial crop-edge context, and supervise only valid core.
    valid = mask.clone().bool()
    if border:
        valid[..., :border, :] = False
        valid[..., -border:, :] = False
        valid[..., :, :border] = False
        valid[..., :, -border:] = False
    values = F.kl_div(
        prediction.float().log_softmax(dim=1), target.softmax(dim=1), reduction="none"
    ).sum(dim=1)
    return values[valid].mean() if valid.any() else prediction.sum() * 0.0
