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
    level_parents: Sequence[Sequence[int]] | None = None,
    ground_truth_weight: float = 1.0,
    weak_label_weight: float = 0.5,
    hierarchy_weight: float = 0.2,
    focal_gamma: float = 0.0,
    class_weights: Tensor | None = None,
    weight_normalization: str = "weighted_mean",
    ignore_index: int = -1,
) -> dict[str, Tensor]:
    """Train joint coarse/fine heads while keeping weak labels lower confidence."""

    if "ground_truth_levels" in batch and "level_0_logits" not in outputs:
        outputs = {
            **outputs,
            "level_0_logits": outputs["coarse_logits"].float().log_softmax(1),
            "level_1_logits": outputs["fine_logits"].float().log_softmax(1),
        }
        level_parents = [list(fine_to_coarse)]
    if "level_0_logits" in outputs:
        return multilevel_supervision_loss(
            outputs,
            batch,
            level_parents,
            ground_truth_weight=ground_truth_weight,
            weak_label_weight=weak_label_weight,
            hierarchy_weight=hierarchy_weight,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            weight_normalization=weight_normalization,
            ignore_index=ignore_index,
        )

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


def multilevel_supervision_loss(
    outputs,
    batch,
    parents,
    *,
    ground_truth_weight,
    weak_label_weight,
    hierarchy_weight,
    focal_gamma,
    class_weights,
    weight_normalization,
    ignore_index,
):
    """Supervise every level using ancestors of the observed leaf, never argmax."""
    logits = [
        outputs[f"level_{i}_logits"] for i in range(3) if f"level_{i}_logits" in outputs
    ]
    if parents is None:
        if len(logits) != 1:
            raise ValueError("多层监督需要 level_parents 映射")
        parents = []
    if len(parents) != len(logits) - 1:
        raise ValueError("level_parents 与模型输出层数不一致")
    edges = [
        torch.as_tensor(edge, device=logits[0].device, dtype=torch.long)
        for edge in parents
    ]
    zero = logits[-1].sum() * 0.0
    totals = [zero for _ in logits]
    result = {}
    valid = batch["valid_mask"].bool()
    if "supervision_split_mask" in batch:
        valid = valid & batch["supervision_split_mask"].bool()
    for source, weight in (
        ("ground_truth", ground_truth_weight),
        ("weak_label", weak_label_weight),
    ):
        leaf = _zero_based(batch[source], batch[f"{source}_mask"] & valid, ignore_index)
        targets = [leaf]
        for edge in reversed(edges):
            child = targets[-1]
            mask = child.ne(ignore_index)
            parent = torch.full_like(child, ignore_index)
            parent[mask] = edge[child[mask]]
            targets.append(parent)
        targets.reverse()
        if source == "ground_truth" and "ground_truth_levels" in batch:
            observed = batch["ground_truth_levels"]
            if observed.shape[1] != len(logits):
                raise ValueError("ground_truth_levels 与分类层数不一致")
            targets = [
                _zero_based(
                    observed[:, depth], observed[:, depth].gt(0) & valid, ignore_index
                )
                for depth in range(len(logits))
            ]
        losses = []
        for depth, (prediction, target) in enumerate(zip(logits, targets, strict=True)):
            loss = focal_cross_entropy(
                prediction,
                target,
                gamma=focal_gamma,
                weight=class_weights if depth == len(logits) - 1 else None,
                weight_normalization=weight_normalization,
                ignore_index=ignore_index,
            )
            losses.append(loss)
            totals[depth] = totals[depth] + weight * loss
            result[f"{source}_level_{depth}_loss"] = loss
        result[f"{source}_fine_loss"] = losses[-1]
        result[f"{source}_coarse_loss"] = sum(losses[:-1], zero)
    consistency = zero
    for depth, edge in enumerate(edges):
        aggregate = torch.zeros_like(logits[depth]).index_add(
            1, edge, logits[depth + 1].exp()
        )
        divergence = F.kl_div(
            logits[depth],
            aggregate.clamp_min(1e-8).log(),
            reduction="none",
            log_target=True,
        ).sum(1)
        consistency = consistency + (
            divergence[valid].mean() if valid.any() else divergence.sum() * 0
        )
    result.update(
        {
            "loss": sum(totals, zero) + hierarchy_weight * consistency,
            "fine_loss": totals[-1],
            "coarse_loss": sum(totals[:-1], zero),
            "hierarchy_loss": consistency,
        }
    )
    return result
