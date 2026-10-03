"""Hierarchical, weakly supervised, and long-tail segmentation losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from losses.focal import focal_cross_entropy


def effective_number_weights(
    counts: Sequence[int] | Tensor, beta: float = 0.9999
) -> Tensor:
    """Return normalized effective-number weights for rare-class compensation."""

    values = torch.as_tensor(counts, dtype=torch.float32).clamp_min(1.0)
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta 必须位于 [0, 1)")
    weights = (1.0 - beta) / (1.0 - beta**values)
    return weights / weights.mean()


def _zero_based(labels: Tensor, mask: Tensor, ignore_index: int) -> Tensor:
    valid = mask.bool() & labels.ne(ignore_index)
    return labels.long().sub(1).masked_fill(~valid, ignore_index)


def _coarse_labels(
    labels: Tensor, mask: Tensor, mapping: Tensor, ignore_index: int
) -> Tensor:
    fine = _zero_based(labels, mask, ignore_index)
    output = torch.full_like(fine, ignore_index)
    valid = fine.ne(ignore_index) & fine.ge(0) & fine.lt(mapping.numel())
    output[valid] = mapping.to(fine.device)[fine[valid]]
    return output


def hierarchical_supervision_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    fine_to_coarse: Sequence[int],
    *,
    ground_truth_weight: float = 1.0,
    weak_label_weight: float = 0.5,
    hierarchy_weight: float = 0.2,
    focal_gamma: float = 0.0,
    class_weights: Tensor | None = None,
    weight_normalization: str = "weighted_mean",
    ignore_index: int = -1,
) -> dict[str, Tensor]:
    """Train joint coarse/fine heads while keeping weak labels lower confidence."""

    mapping = torch.as_tensor(
        fine_to_coarse, dtype=torch.long, device=outputs["fine_logits"].device
    )
    valid_pixels = batch["valid_mask"].bool()
    total_fine = outputs["fine_logits"].sum() * 0.0
    total_coarse = outputs["coarse_logits"].sum() * 0.0
    source_losses: dict[str, Tensor] = {}
    for source, source_weight in (
        ("ground_truth", ground_truth_weight),
        ("weak_label", weak_label_weight),
    ):
        labels = batch[source]
        mask = batch[f"{source}_mask"] & valid_pixels
        fine_labels = _zero_based(labels, mask, ignore_index)
        coarse_labels = _coarse_labels(labels, mask, mapping, ignore_index)
        fine_loss = focal_cross_entropy(
            outputs["fine_logits"],
            fine_labels,
            gamma=focal_gamma,
            weight=class_weights,
            weight_normalization=weight_normalization,
            ignore_index=ignore_index,
        )
        coarse_loss = focal_cross_entropy(
            outputs["coarse_logits"],
            coarse_labels,
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )
        source_losses[f"{source}_fine_loss"] = fine_loss
        source_losses[f"{source}_coarse_loss"] = coarse_loss
        total_fine = total_fine + source_weight * fine_loss
        total_coarse = total_coarse + source_weight * coarse_loss
    coarse_from_fine = outputs["fine_probability"].new_zeros(
        outputs["coarse_logits"].shape
    )
    for fine_index, coarse_index in enumerate(mapping.tolist()):
        coarse_from_fine[:, coarse_index] += outputs["fine_probability"][:, fine_index]
    coarse_from_fine = coarse_from_fine.clamp_min(1e-8)
    coarse_from_fine = coarse_from_fine / coarse_from_fine.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    consistency_per_pixel = F.kl_div(
        outputs["coarse_logits"].log_softmax(dim=1),
        coarse_from_fine.log(),
        reduction="none",
        log_target=True,
    ).sum(dim=1)
    consistency_mask = batch["valid_mask"].bool()
    if consistency_mask.any():
        consistency = consistency_per_pixel[consistency_mask].mean()
    else:
        consistency = consistency_per_pixel.sum() * 0.0
    total = total_fine + total_coarse + hierarchy_weight * consistency
    return {
        "loss": total,
        "fine_loss": total_fine,
        "coarse_loss": total_coarse,
        "hierarchy_loss": consistency,
        **source_losses,
    }
