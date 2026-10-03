"""Multi-class focal cross-entropy with ignore-mask and class-weight support."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def focal_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    gamma: float = 2.0,
    weight: Tensor | None = None,
    weight_normalization: str = "weighted_mean",
    ignore_index: int = -1,
) -> Tensor:
    """Return mean focal loss; ``gamma=0`` is weighted cross-entropy."""

    if gamma < 0:
        raise ValueError("focal gamma 必须为非负数")
    if weight_normalization not in {"weighted_mean", "sample_mean"}:
        raise ValueError("weight_normalization 必须为 weighted_mean 或 sample_mean")
    valid = labels.ne(ignore_index)
    if not valid.any():
        return logits.sum() * 0.0
    safe_labels = labels.masked_fill(~valid, 0)
    log_probability = F.log_softmax(logits, dim=1)
    target_log_probability = log_probability.gather(
        1, safe_labels.unsqueeze(1)
    ).squeeze(1)
    target_probability = target_log_probability.exp()
    losses = -(1.0 - target_probability).pow(gamma) * target_log_probability
    if weight is None:
        return losses[valid].mean()
    if weight.ndim != 1 or weight.numel() != logits.shape[1]:
        raise ValueError("类别权重长度必须等于 logits 类别数")
    pixel_weights = weight.to(logits.device)[safe_labels]
    if weight_normalization == "sample_mean":
        # Normalize by sample count so a single point retains its class weight.
        return (losses[valid] * pixel_weights[valid]).mean()
    denominator = pixel_weights[valid].sum().clamp_min(torch.finfo(logits.dtype).eps)
    return (losses[valid] * pixel_weights[valid]).sum() / denominator
