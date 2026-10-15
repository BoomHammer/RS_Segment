"""Stream raster windows and aggregate held-out predictions by global pixel."""

import torch
from tqdm import tqdm

from data.label_observations import mixed_level_observations
from precision import resolve_amp_dtype


def collect_point_predictions(
    model: torch.nn.Module,
    loader: object,
    device: torch.device,
    amp_dtype: str = "auto",
    level_predictions: dict | None = None,
    raw_predictions: dict | None = None,
    show_progress: bool = True,
    leaf_depths: list[int] | None = None,
    mixed_predictions: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], int]:
    """Aggregate probabilities by global position, then evaluate every point once."""

    model.eval()
    resolved_dtype = resolve_amp_dtype(device, amp_dtype)
    point_predictions = {}
    by_level = {}
    mixed = {}
    leaf_classes = 0
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="验证",
            unit="batch",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=resolved_dtype or torch.float32,
                enabled=resolved_dtype is not None,
            ):
                output = model(tensor_batch)
            if "core_mask" in tensor_batch:
                tensor_batch["valid_mask"] = (
                    tensor_batch["valid_mask"] & tensor_batch["core_mask"]
                )
            split_mask = tensor_batch.get("supervision_split_mask")
            if (
                split_mask is not None
                and (tensor_batch["ground_truth_mask"] & ~split_mask).any()
            ):
                raise AssertionError("验证批次包含非验证归属的真实标签")
            leaf_classes = output["fine_logits"].shape[1]
            if (
                level_predictions is not None or raw_predictions is not None
            ) and "ground_truth_levels" in tensor_batch:
                observed = tensor_batch["ground_truth_levels"]
                for level in range(observed.shape[1]):
                    level_mask = observed[:, level].gt(0) & tensor_batch["valid_mask"]
                    if leaf_depths is not None:
                        depths = torch.as_tensor(leaf_depths, device=device)
                        leaf = observed[:, -1]
                        level_mask &= leaf.lt(1) | depths[(leaf - 1).clamp_min(0)].gt(
                            level
                        )
                    if split_mask is not None and (level_mask & ~split_mask).any():
                        raise AssertionError(
                            "Partial labels cross validation split ownership"
                        )
                    key = f"level_{level}_logits"
                    if key not in output:
                        key = (
                            "fine_logits"
                            if level == observed.shape[1] - 1
                            else "coarse_logits"
                        )
                    _accumulate_points(
                        by_level.setdefault(level, {}),
                        output[key],
                        observed[:, level] - 1,
                        level_mask,
                        batch,
                    )
                    _accumulate_mixed(
                        mixed.setdefault(str(level), {}),
                        output[key],
                        observed[:, level] - 1,
                        tensor_batch,
                        level,
                        leaf_depths,
                    )
            mask = tensor_batch["ground_truth_mask"] & tensor_batch["valid_mask"]
            target = (tensor_batch["ground_truth"] - 1).masked_fill(~mask, -1)
            _accumulate_points(
                point_predictions, output["fine_logits"], target, mask, batch
            )
            _accumulate_mixed(
                mixed.setdefault("leaf", {}),
                output["fine_logits"],
                target,
                tensor_batch,
                -1,
                None,
            )
            del output, tensor_batch
    if raw_predictions is not None:
        raw_predictions.update(
            {"leaf": point_predictions, "levels": by_level, "mixed": mixed}
        )
    if mixed_predictions is not None:
        mixed_predictions.update(
            {level: _finish_points(points) for level, points in mixed.items() if points}
        )
    if level_predictions is not None:
        level_predictions.update(
            {
                level: _finish_points(points)
                for level, points in by_level.items()
                if points
            }
        )
    if not point_predictions:
        if level_predictions or mixed_predictions or raw_predictions is not None:
            return torch.empty(0, dtype=torch.long), torch.empty(0, leaf_classes), [], 0
        raise ValueError("No valid independent ground-truth points to evaluate")
    return _finish_points(point_predictions)


def _accumulate_mixed(points, logits, target, batch, level, leaf_depths):
    observations = batch.get("ground_truth_observations")
    if observations is None or not len(observations):
        return
    depth = observations.shape[1] - 4 if level == -1 else level
    if leaf_depths is not None:
        depths = torch.as_tensor(leaf_depths, device=logits.device)
        leaf = observations[:, -1]
        observations = observations[
            leaf.lt(1) | depths[(leaf - 1).clamp_min(0)].gt(depth)
        ]
    valid = batch["valid_mask"]
    split = batch.get("supervision_split_mask")
    if split is not None:
        b, y, x = observations[:, :3].unbind(1)
        if (~split[b, y, x]).any():
            raise AssertionError("Mixed labels cross validation split ownership")
        valid = valid & split
    positions, labels, weights, unique, inverse = mixed_level_observations(
        observations, target, valid, depth
    )
    if not len(positions):
        return
    targets = logits.new_zeros((len(unique), logits.shape[1]), dtype=torch.float32)
    targets.index_put_((inverse, labels), weights, accumulate=True)
    b, y, x = unique.unbind(1)
    probabilities = logits[b, :, y, x].float().softmax(1).cpu()
    for position, probability, distribution in zip(
        unique.cpu().tolist(), probabilities, targets.cpu(), strict=True
    ):
        sample, row, column = position
        window = batch["input_window"][sample]
        key = (int(window.row_off) + row, int(window.col_off) + column)
        if key in points:
            previous, count, previous_target = points[key]
            if not torch.equal(previous_target, distribution):
                raise ValueError("Mixed target distribution changed across windows")
            points[key] = (previous + probability, count + 1, distribution)
        else:
            points[key] = (probability, 1, distribution)


def _accumulate_points(points, logits, target, mask, batch):
    if not mask.any():
        return
    if "input_window" not in batch:
        raise ValueError("Unique-point evaluation requires input_window")
    for sample, window in enumerate(batch["input_window"]):
        local_mask = mask[sample]
        positions = local_mask.nonzero().cpu().tolist()
        probabilities = logits[sample, :, local_mask].float().softmax(0).T.cpu()
        targets = target[sample][local_mask].cpu().tolist()
        for position, probability, label in zip(
            positions, probabilities, targets, strict=True
        ):
            key = (int(window.row_off) + position[0], int(window.col_off) + position[1])
            if key in points:
                previous, count, previous_label = points[key]
                if label != previous_label:
                    raise ValueError("Conflicting labels at one evaluation pixel")
                points[key] = (previous + probability, count + 1, label)
            else:
                points[key] = (probability, 1, label)


def _finish_points(points):
    if not points:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 0), [], 0
    return (
        torch.stack([value[2] for value in points.values()])
        if isinstance(next(iter(points.values()))[2], torch.Tensor)
        else torch.tensor([value[2] for value in points.values()]),
        torch.stack([value[0] / value[1] for value in points.values()]),
        list(points),
        sum(value[1] for value in points.values()),
    )


def merge_point_prediction_shards(shards, mixed_predictions=None):
    """Merge raw point accumulators produced by distributed evaluation ranks."""

    leaf = {}
    levels = {}
    mixed = {}
    for shard in shards:
        _merge_point_records(leaf, shard["leaf"])
        for level, points in shard["levels"].items():
            _merge_point_records(levels.setdefault(level, {}), points)
        for level, points in shard.get("mixed", {}).items():
            _merge_point_records(mixed.setdefault(level, {}), points)
    if mixed_predictions is not None:
        mixed_predictions.update(
            {level: _finish_points(points) for level, points in mixed.items() if points}
        )
    return _finish_points(leaf), {
        level: _finish_points(points) for level, points in levels.items() if points
    }


def _merge_point_records(destination, source):
    for key, (probability_sum, count, label) in source.items():
        if key in destination:
            previous, previous_count, previous_label = destination[key]
            if not torch.equal(torch.as_tensor(label), torch.as_tensor(previous_label)):
                raise ValueError("Conflicting labels at one evaluation pixel")
            destination[key] = (
                previous + probability_sum,
                previous_count + count,
                label,
            )
        else:
            destination[key] = (probability_sum, count, label)


def point_metrics(targets, probabilities, occurrences):
    """Summarize predictions after spatial ownership filtering and deduplication."""
    predictions = probabilities.argmax(dim=1)
    classes = probabilities.shape[1]
    confusion = torch.bincount(
        targets * classes + predictions, minlength=classes * classes
    ).reshape(classes, classes)
    support = confusion.sum(dim=1)
    predicted_support = confusion.sum(dim=0)
    true_positive = confusion.diag().float()
    recall = true_positive / support.clamp_min(1)
    f1_denominator = support + predicted_support
    per_class_f1 = 2 * true_positive / f1_denominator.clamp_min(1)
    present = f1_denominator > 0
    loss = -probabilities[torch.arange(len(targets)), targets].clamp_min(1e-8).log()
    diagnostics = {
        "confusion_matrix": confusion.tolist(),
        "class_support": support.tolist(),
        "classes_without_ground_truth": (support == 0)
        .nonzero()
        .flatten()
        .add(1)
        .tolist(),
        "per_class_recall": recall.tolist(),
        "macro_recall": float(recall[support > 0].mean()),
        "macro_f1": float(per_class_f1[present].mean()),
        "majority_baseline": float(support.max() / support.sum()),
        "unique_point_count": len(targets),
        "unique_point_accuracy": float((predictions == targets).float().mean()),
        "window_label_occurrences": occurrences,
    }
    return {
        "loss": float(loss.mean()),
        "accuracy": float((predictions == targets).float().mean()),
        "labeled_pixels": len(targets),
        **diagnostics,
    }


def summarize_point_predictions(
    leaf, levels, selection_level=None, mixed_predictions=None
):
    """Select a configured hierarchy level without discarding leaf diagnostics."""
    targets, probabilities, _, occurrences = leaf
    leaf_report = (
        point_metrics(targets, probabilities, occurrences) if targets.numel() else None
    )
    selected_level = "leaf" if selection_level is None else selection_level
    mixed_predictions = mixed_predictions or {}
    if selection_level is not None:
        if type(selection_level) is not int or (
            selection_level not in levels
            and str(selection_level) not in mixed_predictions
        ):
            raise ValueError("selection_level 必须是有验证标签的层级索引（从 0 开始）")
        targets, probabilities, _, occurrences = levels.get(
            selection_level, mixed_predictions.get(str(selection_level))
        )
    elif not targets.numel():
        if not levels and not mixed_predictions:
            raise ValueError("No valid independent ground-truth points to evaluate")
        selected_level = max(levels) if levels else "leaf"
        targets, probabilities, _, occurrences = (
            levels[selected_level] if levels else mixed_predictions["leaf"]
        )
    report = (
        point_metrics(targets, probabilities, occurrences)
        if targets.ndim == 1
        else _mixed_selection_metrics(targets, probabilities, occurrences)
    )
    report["selection_level"] = selected_level
    report["leaf"] = leaf_report
    report["levels"] = {
        str(level): point_metrics(values[0], values[1], values[3])
        for level, values in levels.items()
    }
    report["mixed_pixels"] = {
        level: mixed_point_metrics(values[0], values[1], values[3])
        for level, values in (mixed_predictions or {}).items()
    }
    return report


def _mixed_selection_metrics(targets, probabilities, occurrences):
    """Allow all-mixed validation while explicitly identifying fractional targets."""
    summary = mixed_point_metrics(targets, probabilities, occurrences)
    predicted = torch.nn.functional.one_hot(
        probabilities.argmax(1), probabilities.shape[1]
    ).float()
    confusion = targets.T @ predicted
    support, predicted_support = confusion.sum(1), confusion.sum(0)
    tp = confusion.diag()
    f1 = 2 * tp / (support + predicted_support).clamp_min(1e-8)
    return {
        "loss": summary["cross_entropy"],
        "accuracy": summary["observed_label_agreement"],
        "unique_point_accuracy": summary["observed_label_agreement"],
        "labeled_pixels": len(targets),
        "macro_f1": float(f1[(support + predicted_support) > 0].mean()),
        "macro_recall": float((tp / support.clamp_min(1e-8))[support > 0].mean()),
        "selection_target_type": "mixed_observation_distribution",
        **summary,
    }


def mixed_point_metrics(targets, probabilities, occurrences):
    """Report agreement with observed label frequencies, not hard accuracy."""
    predictions = probabilities.argmax(1)
    return {
        "pixel_count": len(targets),
        "observed_label_agreement": float(
            targets[torch.arange(len(targets)), predictions].mean()
        ),
        "cross_entropy": float(
            -(targets * probabilities.clamp_min(1e-8).log()).sum(1).mean()
        ),
        "window_label_occurrences": occurrences,
        "protocol": "per_pixel_observed_label_distribution_not_area_fraction",
    }


def evaluate_points(
    model, loader, device, amp_dtype="auto", *, selection_level=None, leaf_depths=None
):
    """Use the same unique-point metric for training and standalone evaluation."""
    levels = {}
    mixed = {}
    leaf = collect_point_predictions(
        model,
        loader,
        device,
        amp_dtype,
        level_predictions=levels,
        leaf_depths=leaf_depths,
        mixed_predictions=mixed,
    )
    return summarize_point_predictions(leaf, levels, selection_level, mixed)
