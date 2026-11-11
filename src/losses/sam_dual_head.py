"""E09.1 SAM supervision with two hierarchies and no generated pseudo-labels."""

from math import isfinite

import torch

from losses.experiment import _lovasz_softmax
from losses.supervision import combined_supervision_loss


def sam_dual_head_loss(
    outputs, batch, derived, supervision, settings, *, class_weights=None
):
    """Preserve E01 partial/mixed supervision and separate measured/SAM evidence.

    Both heads learn measured observations (averaged to retain E01's scale).
    Only the expanded head learns SAM, through E01's hierarchical objective
    plus Lovasz over terminal leaf IDs. Consistency uses train-owned core pixels.
    The caller's SAM mask is narrowed for accurate training audit counts; label
    values are never generated or replaced.
    """
    consistency_weight = float(settings.get("consistency_weight", 1.0))
    lovasz_weight = float(settings.get("lovasz_weight", 1.0))
    if any(not isfinite(w) or w < 0 for w in (consistency_weight, lovasz_weight)):
        raise ValueError("E09.1 consistency/lovasz 权重必须非负且有限")
    valid = batch["valid_mask"].bool().clone()
    for key in ("core_mask", "supervision_split_mask"):
        if key in batch:
            valid &= batch[key].bool()

    measured = batch["ground_truth_mask"].bool().clone()
    if "ground_truth_levels" in batch:
        measured |= batch["ground_truth_levels"].gt(0).any(1)
    observations = batch.get("ground_truth_observations")
    if observations is not None and len(observations):
        b, y, x = observations[:, :3].unbind(1)
        measured[b, y, x] = True
    sam_mask = batch["weak_label_mask"].bool() & valid & ~measured
    # Keep source metrics aligned with the pixels actually used by this loss.
    batch["weak_label_mask"] = sam_mask
    expanded_batch = {**batch, "valid_mask": valid}
    base_batch = {
        **expanded_batch,
        "weak_label_mask": torch.zeros_like(sam_mask),
    }
    kwargs = {
        "fine_to_coarse": derived["fine_to_coarse"],
        "level_parents": derived.get("level_parents"),
        "leaf_depths": derived.get("leaf_depths"),
        "level_weights": supervision.get("level_weights"),
        "ground_truth_weight": float(supervision.get("ground_truth_weight", 1.0)),
        "weak_label_weight": float(supervision.get("weak_label_weight", 0.5)),
        "hierarchy_weight": float(supervision.get("hierarchy_weight", 0.2)),
        "focal_gamma": float(supervision.get("focal_gamma", 0.0)),
        "class_weights": class_weights,
        "weight_normalization": supervision.get(
            "weight_normalization", "weighted_mean"
        ),
        "ignore_index": int(supervision.get("ignore_index", -1)),
    }
    base = outputs["base_output"]
    expanded = outputs["expanded_output"]
    base_losses = combined_supervision_loss(base, base_batch, **kwargs)
    expanded_losses = combined_supervision_loss(expanded, expanded_batch, **kwargs)

    def source_loss(losses, source):
        return losses[f"{source}_coarse_loss"] + losses[f"{source}_fine_loss"]

    ground_truth = 0.5 * (
        source_loss(base_losses, "ground_truth")
        + source_loss(expanded_losses, "ground_truth")
    )
    sam_hierarchy = source_loss(expanded_losses, "weak_label")
    base_probability = base["fine_probability"].float()
    expanded_probability = expanded["fine_probability"].float()
    zero = (base_probability.sum() + expanded_probability.sum()) * 0.0
    sam_lovasz = (
        _lovasz_softmax(expanded_probability, batch["weak_label"], sam_mask)
        if lovasz_weight > 0 and sam_mask.any()
        else zero
    )
    consistency = (
        (base_probability - expanded_probability).square().sum(1)[valid].mean()
        if valid.any()
        else zero
    )
    hierarchy = 0.5 * (
        base_losses["hierarchy_loss"] + expanded_losses["hierarchy_loss"]
    )
    weak = sam_hierarchy + lovasz_weight * sam_lovasz
    total = (
        kwargs["ground_truth_weight"] * ground_truth
        + kwargs["weak_label_weight"] * weak
        + kwargs["hierarchy_weight"] * hierarchy
        + consistency_weight * consistency
    )
    if not torch.isfinite(total):
        raise FloatingPointError("E09.1 loss 非有限；已停止，避免更新损坏权重")
    return {
        "loss": total,
        "ground_truth_loss": ground_truth,
        "weak_label_loss": weak,
        "sam_hierarchical_loss": sam_hierarchy,
        "sam_lovasz_loss": sam_lovasz,
        "hierarchy_loss": hierarchy,
        "consistency_loss": consistency,
    }
