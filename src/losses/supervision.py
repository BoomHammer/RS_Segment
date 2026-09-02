"""Masked supervision utilities for ground-truth and weak labels."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn import functional as F


def masked_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    *,
    ignore_index: int = -1,
) -> Tensor:
    """Calculate cross entropy only on valid label pixels."""

    effective_mask = mask.bool() & labels.ne(ignore_index)
    if not effective_mask.any():
        return logits.sum() * 0.0
    # The data contract uses class IDs 1..N; PyTorch cross entropy uses 0..N-1.
    safe_labels = labels.sub(1).masked_fill(~effective_mask, ignore_index)
    return F.cross_entropy(logits, safe_labels, ignore_index=ignore_index)


def combined_supervision_loss(
    logits: Tensor,
    batch: dict[str, Any],
    *,
    ground_truth_weight: float = 1.0,
    weak_label_weight: float = 0.5,
    ignore_index: int = -1,
) -> dict[str, Tensor]:
    """Use ground truth and weak labels as two masked supervision sources."""

    if logits.ndim != 4:
        raise ValueError("logits 必须是 [B, C, H, W]")
    ground_truth_loss = masked_cross_entropy(
        logits,
        batch["ground_truth"].long(),
        batch["ground_truth_mask"] & batch["valid_mask"],
        ignore_index=ignore_index,
    )
    weak_label_loss = masked_cross_entropy(
        logits,
        batch["weak_label"].long(),
        batch["weak_label_mask"] & batch["valid_mask"],
        ignore_index=ignore_index,
    )
    total = (
        ground_truth_weight * ground_truth_loss + weak_label_weight * weak_label_loss
    )
    return {
        "loss": total,
        "ground_truth_loss": ground_truth_loss,
        "weak_label_loss": weak_label_loss,
    }
