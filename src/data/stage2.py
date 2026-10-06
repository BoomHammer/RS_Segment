"""Stage-2 configuration validation and dataset construction helpers."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import torch
from pyproj import CRS

from .sample_index import SampleIndex, _feature_name, _timestamp, load_sample_index


def _index(value: SampleIndex | str | Path) -> SampleIndex:
    return load_sample_index(value) if isinstance(value, (str, Path)) else value


def _timestamp_date(timestamp: str) -> date:
    return (
        date.fromisoformat(f"{timestamp}-01")
        if len(timestamp) == 7
        else date.fromisoformat(timestamp)
    )


def validate_stage2_config(
    stage2: dict[str, Any],
    sample_index: SampleIndex | str | Path,
    *,
    require_files: bool = True,
) -> dict[str, Any]:
    """Validate stage-2 settings against indexed assets and return a report."""

    index = _index(sample_index)
    features = dict(stage2.get("features", {}))
    dynamic = {
        _feature_name(asset) for asset in index.assets if asset.role == "dynamic"
    }
    static = {asset.name for asset in index.assets if asset.role == "static"}
    selected_dynamic = set(features.get("dynamic", [])) or dynamic
    selected_static = set(features.get("static", [])) or static
    errors: list[str] = []
    try:
        CRS.from_user_input(index.target_grid["crs"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"索引目标网格 CRS 无效: {exc}")
    for key in ("width", "height"):
        if int(index.target_grid.get(key, 0)) < 1:
            errors.append(f"索引目标网格 {key} 必须为正数")
    if require_files:
        errors.extend(
            f"索引资产文件不存在: {asset.path}"
            for asset in index.assets
            if not Path(asset.path).exists()
        )
    for name in selected_dynamic - dynamic:
        errors.append(f"动态特征不存在于索引: {name}")
    for name in selected_static - static:
        errors.append(f"静态特征不存在于索引: {name}")
    order = list(features.get("dynamic_order", []))
    if len(order) != len(set(order)):
        errors.append("features.dynamic_order 不能包含重复特征")
    if order and not set(order).issubset(selected_dynamic):
        errors.append("features.dynamic_order 必须是选定动态特征的子集")
    time_config = dict(stage2.get("time", {}))
    start = date.fromisoformat(str(time_config.get("start", "1900-01-01")))
    end = date.fromisoformat(str(time_config.get("end", "9999-12-31")))
    if start > end:
        errors.append("time.start 不能晚于 time.end")
    timestamps = sorted(
        _timestamp(asset) for asset in index.assets if asset.role == "dynamic"
    )
    selected_timestamps = [
        timestamp
        for timestamp in timestamps
        if start <= _timestamp_date(timestamp) <= end
    ]
    window = dict(stage2.get("window", {}))
    size = tuple(window.get("size", (256, 256)))
    stride = tuple(window.get("stride", size))
    if len(size) != 2 or len(stride) != 2 or min(size) < 1 or min(stride) < 1:
        errors.append("window.size 和 window.stride 必须是两个正整数")
    elif stride[0] * 2 > size[0] or stride[1] * 2 > size[1]:
        errors.append("window.stride 不得超过 window.size 的一半")
    missing = str(time_config.get("missing", "mask_nan"))
    if missing not in {"mask_nan", "drop"}:
        errors.append("time.missing 只能是 mask_nan 或 drop")
    cache = dict(stage2.get("cache", {}))
    if int(cache.get("max_items", 0)) < 0:
        errors.append("cache.max_items 不能为负数")
    training = dict(stage2.get("training", {}))
    if str(training.get("amp_dtype", "bfloat16")) not in {
        "bfloat16",
        "float16",
        "none",
    }:
        errors.append("training.amp_dtype 必须是 bfloat16、float16 或 none")
    if int(training.get("gradient_accumulation_steps", 1)) < 1:
        errors.append("training.gradient_accumulation_steps 必须为正数")
    for key in ("statistics_file", "split_file"):
        value = stage2.get(key)
        if require_files and value is not None and not Path(value).exists():
            errors.append(f"{key} 文件不存在: {value}")
    if errors:
        raise ValueError("阶段2配置校验失败: " + "; ".join(errors))
    return {
        "dynamic_features": sorted(selected_dynamic),
        "static_features": sorted(selected_static),
        "dynamic_order": order or sorted(selected_dynamic),
        "timestamps": sorted(set(selected_timestamps)),
        "window_size": list(size),
        "stride": list(stride),
        "missing_strategy": missing,
    }


def write_stage2_report(report: dict[str, Any], output: str | Path) -> None:
    """Write a validation report without writing any raster data."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def benchmark_dataset(
    dataset: Any,
    *,
    split_file: str | Path | None = None,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    batches: int = 20,
    warmup: int = 3,
    amp_dtype: str = "bfloat16",
    gradient_accumulation_steps: int = 1,
) -> dict[str, Any]:
    """Benchmark bounded reads, host-to-device copies and CUDA peak memory."""

    from .sampling import SpatialWeightedSampler, build_dataloader

    sampler = None
    if split_file is not None and Path(split_file).exists():
        sampler = SpatialWeightedSampler(
            dataset,
            manifest=split_file,
            split=split,
            num_samples=max(1, (warmup + batches) * batch_size),
        )
    loader = build_dataloader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def next_batch(iterator: Any) -> tuple[Any, Any]:
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def move_batch(batch: dict[str, Any]) -> None:
        for key in ("dynamic", "static", "valid_mask"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    for _ in range(warmup):
        batch, iterator = next_batch(iterator)
        move_batch(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(batches):
        batch, iterator = next_batch(iterator)
        move_batch(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "device": str(device),
        "cuda_available": device.type == "cuda",
        "batches": batches,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "amp_dtype": amp_dtype,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "seconds": elapsed,
        "batches_per_second": batches / elapsed if elapsed else 0.0,
        "samples_per_second": batches * batch_size / elapsed if elapsed else 0.0,
        "peak_memory_allocated_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
        "peak_memory_reserved_mb": (
            torch.cuda.max_memory_reserved() / 1024**2
            if device.type == "cuda"
            else None
        ),
    }


def validate_dataset_windows(
    dataset: Any,
    *,
    max_windows: int = 2,
    require_ground_truth: bool = True,
) -> dict[str, Any]:
    """Validate bounded real windows and deterministic repeated reads.

    When requested, candidates are selected from indexed ground-truth points so
    the real-data check cannot pass only on unlabeled background windows.
    """

    if max_windows < 1:
        raise ValueError("max_windows 必须是正整数")
    import torch

    candidate_ids: list[int] = []
    supervision = getattr(dataset, "supervision_pixels", dataset.ground_truth_pixels)
    if require_ground_truth and supervision:
        for row, column in supervision:
            candidate_ids.extend(dataset.query_windows_for_pixel(row, column))
        candidate_ids = list(dict.fromkeys(candidate_ids))
    if not candidate_ids:
        candidate_ids = list(range(len(dataset)))
    checked = []
    for window_id in candidate_ids[:max_windows]:
        sample = dataset[window_id]
        repeated = dataset[window_id]
        if not torch.allclose(sample["dynamic"], repeated["dynamic"], equal_nan=True):
            raise AssertionError(f"窗口 {window_id} 读取不可复现")
        if sample["dynamic"].shape[-2:] != sample["static"].shape[-2:]:
            raise AssertionError(f"窗口 {window_id} 多源空间形状不一致")
        labels = sample["ground_truth"][sample["ground_truth_mask"]]
        if torch.any(labels < 1):
            raise AssertionError(f"窗口 {window_id} 存在非法地面真实值类别编码")
        observed = sample.get("ground_truth_levels", sample["ground_truth"])
        if require_ground_truth and not observed.gt(0).any():
            raise AssertionError(f"窗口 {window_id} 未包含地面真实值标签")
        checked.append(sample["sample_status"])
    if require_ground_truth and not checked:
        raise AssertionError("没有可用于真实地面真实值验收的窗口")
    return {
        "checked_windows": len(checked),
        "required_ground_truth": require_ground_truth,
        "window_status": checked,
    }
