"""Audit point duplication and run a cached fixed-patch fitting diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from config import load_config
from data.sample_index import WindowedSampleDataset, sample_collate_fn
from data.spatial_split import load_spatial_split
from data.training_policy import point_windows
from models.architecture import SegFormerUtae
from models.config import load_model_contract

TARGET_CODES = (5, 8, 11, 14, 17, 20, 23, 25, 26, 27)
CONTROL_CLASS_COUNT = 6
MILESTONES = (0, 20, 50, 100, 200)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的记录: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mapping(run: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    path = next(iter(sorted(run.glob("label_mapping*.json"))), None)
    if path is None:
        raise FileNotFoundError(f"缺少标签映射: {run}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup = {int(item["alliance_code"]): item for item in payload["classes"]}
    return payload, lookup


def _dataset(experiment: Path) -> tuple[WindowedSampleDataset, Path]:
    log = json.loads((experiment / "train_log.json").read_text(encoding="utf-8"))
    run = Path(log["source_run"])
    config = load_config(experiment / "data.yaml")
    stage2 = config.data.stage2
    window = dict(stage2.get("window", {}))
    mapping, _ = _mapping(run)
    statistics = next(iter(sorted(run.glob("raster_stats*.json"))), None)
    dataset = WindowedSampleDataset(
        index=run / "sample_index.json",
        window_size=tuple(log.get("window_size", window.get("size", (256, 256)))),
        stride=tuple(log.get("stride", window.get("stride", (128, 128)))),
        halo=tuple(log.get("halo", window.get("halo", (0, 0)))),
        grid_offset=tuple(log.get("grid_offset", (0, 0))),
        label_columns=config.data.label_columns,
        label_mapping=mapping,
        statistics=statistics,
        nodata=config.data.raster.get("nodata", -9999),
        stage2=stage2,
        use_weak_labels=False,
        transforms=None,
    )
    return dataset, run


def audit_points(
    experiment: Path,
    dataset: WindowedSampleDataset,
    class_lookup: dict[int, dict[str, Any]],
    validation: dict[str, Any],
    output: Path,
) -> None:
    manifest = load_spatial_split(experiment / "spatial_split.json")
    train_windows = set(manifest.splits["train"])
    validation_windows = set(manifest.splits["validation"])
    confusion = validation["confusion_matrix"]
    summaries: dict[int, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []

    per_class: dict[int, dict[str, Any]] = {
        code: {
            "train": set(),
            "validation": set(),
            "intersection": set(),
            "validation_occurrences": 0,
            "validation_max_repeats": 0,
        }
        for code in class_lookup
    }
    for (row, column), code in dataset.ground_truth_pixels.items():
        covering = set(dataset.query_windows_for_pixel(row, column))
        train_count = len(covering & train_windows)
        validation_count = len(covering & validation_windows)
        coordinate = (int(row), int(column))
        if train_count:
            per_class[code]["train"].add(coordinate)
        if validation_count:
            per_class[code]["validation"].add(coordinate)
            per_class[code]["validation_occurrences"] += validation_count
            per_class[code]["validation_max_repeats"] = max(
                per_class[code]["validation_max_repeats"], validation_count
            )
            details.append(
                {
                    "class_code": code,
                    "class_name": class_lookup[code]["alliance"].strip('"'),
                    "raster_row": int(row),
                    "raster_column": int(column),
                    "train_window_count": train_count,
                    "validation_window_count": validation_count,
                    "validation_duplicate_count": max(validation_count - 1, 0),
                    "in_train_validation_intersection": int(train_count > 0),
                }
            )
        if train_count and validation_count:
            per_class[code]["intersection"].add(coordinate)

    for code in sorted(class_lookup):
        values = per_class[code]
        validation_points = len(values["validation"])
        occurrences = int(values["validation_occurrences"])
        confusion_support = int(sum(confusion[code - 1]))
        summaries[code] = {
            "class_code": code,
            "class_name": class_lookup[code]["alliance"].strip('"'),
            "formation_name": class_lookup[code]["formation"],
            "train_independent_points": len(values["train"]),
            "validation_independent_points": validation_points,
            "train_validation_intersection_points": len(values["intersection"]),
            "validation_window_occurrences": occurrences,
            "validation_duplicate_occurrences": occurrences - validation_points,
            "validation_mean_windows_per_point": (
                occurrences / validation_points if validation_points else 0.0
            ),
            "validation_max_windows_per_point": int(values["validation_max_repeats"]),
            "confusion_matrix_support": confusion_support,
            "occurrence_minus_confusion_support": occurrences - confusion_support,
        }

    _write_csv(output / "point_audit_32_classes.csv", list(summaries.values()))
    _write_csv(
        output / "validation_point_repetitions.csv",
        sorted(
            details,
            key=lambda item: (
                int(item["class_code"]),
                int(item["raster_row"]),
                int(item["raster_column"]),
            ),
        ),
    )
    target_rows = [summaries[code] for code in TARGET_CODES]
    _write_csv(output / "point_audit_zero_recall_classes.csv", target_rows)
    summary = {
        "coordinate_basis": "target-grid raster row and column",
        "validation_epoch": int(validation["epoch"]),
        "zero_recall_codes": list(TARGET_CODES),
        "zero_recall_train_independent_points": sum(
            int(item["train_independent_points"]) for item in target_rows
        ),
        "zero_recall_validation_independent_points": sum(
            int(item["validation_independent_points"]) for item in target_rows
        ),
        "zero_recall_train_validation_intersection_points": sum(
            int(item["train_validation_intersection_points"]) for item in target_rows
        ),
        "zero_recall_validation_window_occurrences": sum(
            int(item["validation_window_occurrences"]) for item in target_rows
        ),
        "zero_recall_confusion_support": sum(
            int(item["confusion_matrix_support"]) for item in target_rows
        ),
    }
    (output / "point_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _control_codes(validation: dict[str, Any]) -> tuple[int, ...]:
    confusion = validation["confusion_matrix"]
    counts: Counter[int] = Counter()
    for true_code in TARGET_CODES:
        for predicted_index, count in enumerate(confusion[true_code - 1]):
            predicted_code = predicted_index + 1
            if predicted_code not in TARGET_CODES:
                counts[predicted_code] += int(count)
    return tuple(code for code, _ in counts.most_common(CONTROL_CLASS_COUNT))


def _select_windows(
    dataset: WindowedSampleDataset,
    experiment: Path,
    control_codes: tuple[int, ...],
) -> list[int]:
    manifest = load_spatial_split(experiment / "spatial_split.json")
    train_windows = set(manifest.splits["train"])
    labels_by_window = point_windows(dataset)
    selected: list[int] = []
    requested = (*TARGET_CODES, *control_codes)
    for code in requested:
        if any(labels_by_window[index].get(str(code), 0) for index in selected):
            continue
        candidates = [
            index
            for index in train_windows
            if labels_by_window.get(index, {}).get(str(code), 0)
        ]
        if not candidates:
            raise ValueError(f"训练划分没有类别 {code} 的候选窗口")
        selected.append(
            min(
                candidates,
                key=lambda index: (
                    -labels_by_window[index][str(code)],
                    -sum(labels_by_window[index].values()),
                    index,
                ),
            )
        )
    return selected


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _target(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["ground_truth_mask"] & batch["valid_mask"] & batch["core_mask"]
    target = (batch["ground_truth"] - 1).masked_fill(~mask, -1)
    if not torch.equal(target[~mask], torch.full_like(target[~mask], -1)):
        raise AssertionError("未标注位置没有被 ignore_index 屏蔽")
    return target, mask


def _evaluate_cached(
    model: SegFormerUtae,
    cached: list[dict[str, Any]],
    device: torch.device,
    step: int,
    codes: tuple[int, ...],
    class_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    loss_sums: defaultdict[int, float] = defaultdict(float)
    correct: Counter[int] = Counter()
    support: Counter[int] = Counter()
    model.eval()
    with torch.inference_mode():
        for cpu_batch in cached:
            batch = _to_device(cpu_batch, device)
            target, mask = _target(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch)["fine_logits"]
            losses = F.cross_entropy(
                logits.float(), target, reduction="none", ignore_index=-1
            )
            prediction = logits.argmax(dim=1)
            for code in codes:
                class_mask = mask & target.eq(code - 1)
                count = int(class_mask.sum())
                if not count:
                    continue
                support[code] += count
                correct[code] += int((prediction[class_mask] == code - 1).sum())
                loss_sums[code] += float(losses[class_mask].sum())
    return [
        {
            "step": step,
            "class_code": code,
            "class_name": class_lookup[code]["alliance"].strip('"'),
            "role": "zero_recall" if code in TARGET_CODES else "confusion_control",
            "support": support[code],
            "loss": loss_sums[code] / support[code] if support[code] else "",
            "accuracy": correct[code] / support[code] if support[code] else "",
            "correct": correct[code],
        }
        for code in codes
    ]


def fixed_batch_fit(
    experiment: Path,
    dataset: WindowedSampleDataset,
    run: Path,
    class_lookup: dict[int, dict[str, Any]],
    validation: dict[str, Any],
    output: Path,
    steps: int,
    device: torch.device,
) -> None:
    controls = _control_codes(validation)
    selected = _select_windows(dataset, experiment, controls)
    cached = [
        sample_collate_fn([dataset[index]])
        for index in tqdm(selected, desc="缓存固定诊断 patch", unit="patch")
    ]
    selected_rows = []
    for position, (index, batch) in enumerate(zip(selected, cached, strict=True)):
        _, mask = _target(batch)
        counts = Counter(int(value) for value in batch["ground_truth"][mask].tolist())
        selected_rows.append(
            {
                "cache_position": position,
                "window_index": index,
                "ground_truth_pixels": int(mask.sum()),
                "class_counts": json.dumps(dict(sorted(counts.items()))),
            }
        )
    _write_csv(output / "fixed_patch_selection.csv", selected_rows)

    contract = load_model_contract(experiment / "model.yaml", run)
    model = SegFormerUtae.from_contract(contract).to(device)
    checkpoint_path = experiment / "best_loss.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = experiment / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    codes = (*TARGET_CODES, *controls)
    rows = _evaluate_cached(model, cached, device, 0, codes, class_lookup)
    milestones = set(MILESTONES) | {steps}
    losses = []
    progress = tqdm(range(1, steps + 1), desc="固定 patch 拟合", unit="step")
    for step in progress:
        batch = _to_device(cached[(step - 1) % len(cached)], device)
        target, mask = _target(batch)
        if not mask.any():
            raise ValueError("固定 patch 没有有效真实标签")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(batch)["fine_logits"]
            loss = F.cross_entropy(logits.float(), target, ignore_index=-1)
        loss.backward()
        optimizer.step()
        losses.append({"step": step, "cross_entropy": float(loss.detach())})
        progress.set_postfix(loss=f"{float(loss.detach()):.5f}")
        if step in milestones:
            rows.extend(
                _evaluate_cached(model, cached, device, step, codes, class_lookup)
            )
    _write_csv(output / "fixed_batch_fit_metrics.csv", rows)
    _write_csv(output / "fixed_batch_step_losses.csv", losses)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "validation_epoch": int(validation["epoch"]),
        "steps": steps,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "augmentation": False,
        "dropout_and_drop_path": False,
        "loss": "unweighted fine-label cross entropy on ground truth only",
        "ignore_index": -1,
        "selected_windows": selected,
        "target_codes": list(TARGET_CODES),
        "control_codes": list(controls),
    }
    (output / "fixed_batch_fit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--validation-epoch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    output = experiment / "fixed_batch_diagnostic"
    validation_path = experiment / (
        f"validation_epoch_{args.validation_epoch:03d}.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["epoch"] = args.validation_epoch
    dataset, run = _dataset(experiment)
    _, class_lookup = _mapping(run)
    audit_points(experiment, dataset, class_lookup, validation, output)
    if args.audit_only:
        print(f"样点审计结果: {output}")
        return 0
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    fixed_batch_fit(
        experiment,
        dataset,
        run,
        class_lookup,
        validation,
        output,
        args.steps,
        device,
    )
    print(f"诊断结果: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
