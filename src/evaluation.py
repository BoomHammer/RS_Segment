"""Stream raster windows and aggregate held-out predictions by global pixel."""

import torch
from tqdm import tqdm

from precision import resolve_amp_dtype


def collect_point_predictions(
    model: torch.nn.Module,
    loader: object,
    device: torch.device,
    amp_dtype: str = "auto",
    level_predictions: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], int]:
    """Aggregate probabilities by global position, then evaluate every point once."""

    model.eval()
    resolved_dtype = resolve_amp_dtype(device, amp_dtype)
    point_predictions = {}
    by_level = {}
    leaf_classes = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="验证", unit="batch", dynamic_ncols=True):
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
            if level_predictions is not None and "ground_truth_levels" in tensor_batch:
                observed = tensor_batch["ground_truth_levels"]
                for level in range(observed.shape[1]):
                    level_mask = observed[:, level].gt(0) & tensor_batch["valid_mask"]
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
            mask = tensor_batch["ground_truth_mask"] & tensor_batch["valid_mask"]
            target = (tensor_batch["ground_truth"] - 1).masked_fill(~mask, -1)
            _accumulate_points(
                point_predictions, output["fine_logits"], target, mask, batch
            )
            del output, tensor_batch
    if level_predictions is not None:
        level_predictions.update(
            {
                level: _finish_points(points)
                for level, points in by_level.items()
                if points
            }
        )
    if not point_predictions:
        if level_predictions:
            return torch.empty(0, dtype=torch.long), torch.empty(0, leaf_classes), [], 0
        raise ValueError("No valid independent ground-truth points to evaluate")
    return _finish_points(point_predictions)


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
    return (
        torch.tensor([value[2] for value in points.values()]),
        torch.stack([value[0] / value[1] for value in points.values()]),
        list(points),
        sum(value[1] for value in points.values()),
    )


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


def evaluate_points(model, loader, device, amp_dtype="auto"):
    """Use the same unique-point metric for training and standalone evaluation."""
    levels = {}
    targets, probabilities, _, occurrences = collect_point_predictions(
        model, loader, device, amp_dtype, level_predictions=levels
    )
    selected_level = "leaf"
    if not targets.numel():
        selected_level = max(levels)
        targets, probabilities, _, occurrences = levels[selected_level]
    report = point_metrics(targets, probabilities, occurrences)
    report["selection_level"] = selected_level
    report["levels"] = {
        str(level): point_metrics(values[0], values[1], values[3])
        for level, values in levels.items()
    }
    return report
