"""Streaming weak-label GeoTIFF generation from point prompts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from data.label_quality import write_label_quality_visualization
from data.labels import LabelRecord
from data.raster_alignment import TargetGrid, aligned_raster, locate_points
from inference.pointsam import (
    PointSAMInferencer,
    build_pointsam_request,
    run_pointsam_with_npc,
)


@dataclass(frozen=True, slots=True)
class WeakLabelGenerationConfig:
    """Memory-bounded settings for one weak-label generation run."""

    window_size: tuple[int, int] = (384, 384)
    label_radius: int = 16
    input_range: tuple[float, float] = (-100.0, 16000.0)
    stretch_percentiles: tuple[float, float] = (2.0, 98.0)
    medium_confidence: float = 0.70
    medium_label_radius: int = 8
    fallback_radius: int = 4
    spectral_distance_threshold: float = 0.12
    logit_threshold: float = 0.5
    reject_boundary_touch: bool = True
    conflict_margin: float = 0.05
    output_nodata: int = -9999
    min_confidence: float = 0.0


def _window_for_point(
    row: int, column: int, grid: TargetGrid, size: tuple[int, int]
) -> Window:
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("window_size 必须是两个正整数")
    left = max(0, min(column - width // 2, grid.width - width))
    top = max(0, min(row - height // 2, grid.height - height))
    return Window(left, top, min(width, grid.width), min(height, grid.height))


def _seed_component(mask: np.ndarray, row: int, column: int) -> np.ndarray:
    """Keep only the connected SAM region containing the positive point."""

    if (
        not (0 <= row < mask.shape[0] and 0 <= column < mask.shape[1])
        or not mask[row, column]
    ):
        return np.zeros_like(mask, dtype=bool)
    selected = np.zeros_like(mask, dtype=bool)
    stack = [(row, column)]
    selected[row, column] = True
    while stack:
        current_row, current_column = stack.pop()
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if not row_offset and not column_offset:
                    continue
                next_row = current_row + row_offset
                next_column = current_column + column_offset
                if (
                    0 <= next_row < mask.shape[0]
                    and 0 <= next_column < mask.shape[1]
                    and mask[next_row, next_column]
                    and not selected[next_row, next_column]
                ):
                    selected[next_row, next_column] = True
                    stack.append((next_row, next_column))
    return selected


def _global_input_ranges(
    datasets: Sequence[Any],
    grid: TargetGrid,
    raw_range: tuple[float, float],
    percentiles: tuple[float, float],
) -> list[tuple[float, float]]:
    """Estimate one global stretch per band from a bounded raster sample."""

    lower, upper = raw_range
    percentile_low, percentile_high = percentiles
    if not lower < upper or not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError("输入范围或拉伸百分位参数无效")
    result = []
    for dataset in datasets:
        values = []
        for row in range(0, grid.height, 512):
            window = Window(0, row, grid.width, min(512, grid.height - row))
            array = dataset.read(1, window=window, masked=True).filled(np.nan)
            sampled = array[::16, ::16]
            valid = sampled[
                np.isfinite(sampled) & (sampled >= lower) & (sampled <= upper)
            ]
            if valid.size:
                values.append(valid.astype(np.float32, copy=False))
        if not values:
            result.append((lower, upper))
            continue
        sample = np.concatenate(values)
        low, high = np.percentile(sample, [percentile_low, percentile_high])
        result.append((float(low), float(high if high > low else low + 1.0)))
    return result


def _stretch_rgb(
    image: np.ndarray, ranges: Sequence[tuple[float, float]]
) -> np.ndarray:
    result = np.empty_like(image, dtype=np.uint8)
    for band, (lower, upper) in enumerate(ranges):
        result[band] = np.clip(
            (image[band].astype(np.float32) - lower) * 255.0 / (upper - lower),
            0,
            255,
        ).astype(np.uint8)
    return result


def _disk_mask(
    shape: tuple[int, int], row: int, column: int, radius: int
) -> np.ndarray:
    if radius < 0:
        raise ValueError("label_radius 必须是非负整数")
    rows, columns = np.ogrid[: shape[0], : shape[1]]
    return (rows - row) ** 2 + (columns - column) ** 2 <= radius**2


def _touches_boundary(mask: np.ndarray, window: Window, grid: TargetGrid) -> bool:
    return bool(
        (int(window.row_off) > 0 and mask[0].any())
        or (int(window.row_off + window.height) < grid.height and mask[-1].any())
        or (int(window.col_off) > 0 and mask[:, 0].any())
        or (int(window.col_off + window.width) < grid.width and mask[:, -1].any())
    )


def _merge_labels(existing: np.ndarray, mask: np.ndarray, code: int) -> np.ndarray:
    if existing.shape != mask.shape:
        raise ValueError("SAM2 mask 与输出窗口形状不匹配")
    result = existing.copy()
    selected = mask.astype(bool, copy=False)
    conflict = selected & (result > 0) & (result != code)
    result[conflict] = -1
    writable = selected & (result == 0)
    result[writable] = code
    return result


def _profile(grid: TargetGrid, nodata: int) -> dict[str, Any]:
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "int32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    if grid.width >= 16 and grid.height >= 16:
        profile.update(
            tiled=True,
            blockxsize=min(256, grid.width),
            blockysize=min(256, grid.height),
        )
    return profile


def _write_initial_nodata(
    dataset: rasterio.io.DatasetWriter, grid: TargetGrid, nodata: int
) -> None:
    for row in range(0, grid.height, 256):
        for column in range(0, grid.width, 256):
            window = Window(
                column, row, min(256, grid.width - column), min(256, grid.height - row)
            )
            shape = (int(window.height), int(window.width))
            dataset.write(np.full(shape, nodata, dtype=np.int32), 1, window=window)


def generate_weak_labels(
    records: Iterable[LabelRecord],
    *,
    grid: TargetGrid,
    image_paths: Sequence[str | Path],
    inferencer: PointSAMInferencer,
    output_path: str | Path,
    config: WeakLabelGenerationConfig | None = None,
    quality_report_path: str | Path | None = None,
    quality_visualization_path: str | Path | None = None,
    alliance_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Generate sparse, point-local labels from high-confidence SAM regions."""

    config = config or WeakLabelGenerationConfig()
    if len(image_paths) != 3:
        raise ValueError("image_paths 必须提供三个单波段影像路径")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_profile = _profile(grid, config.output_nodata)
    with rasterio.open(output, "w", **output_profile) as destination:
        _write_initial_nodata(destination, grid, config.output_nodata)
    located_records = []
    sample_outcomes: list[dict[str, Any]] = []
    for record in records:
        location = locate_points(
            [(record.x, record.y)], point_crs="EPSG:4326", grid=grid
        )[0]
        if location["inside"]:
            outcome_index = len(sample_outcomes)
            located_records.append(
                (
                    record,
                    int(location["row"]),
                    int(location["column"]),
                    outcome_index,
                )
            )
            sample_outcomes.append(
                {
                    "row": int(location["row"]),
                    "column": int(location["column"]),
                    "alliance_code": record.alliance_code,
                    "status": "pending",
                }
            )
        else:
            sample_outcomes.append(
                {
                    "row": int(location["row"]),
                    "column": int(location["column"]),
                    "alliance_code": record.alliance_code,
                    "status": "outside_grid",
                }
            )
    with (
        aligned_raster(image_paths[0], grid) as band_a,
        aligned_raster(image_paths[1], grid) as band_b,
        aligned_raster(image_paths[2], grid) as band_c,
        rasterio.open(output, "r+") as destination,
    ):
        ranges = _global_input_ranges(
            (band_a, band_b, band_c),
            grid,
            config.input_range,
            config.stretch_percentiles,
        )
        score = np.full((grid.height, grid.width), -np.inf, dtype=np.float32)
        labels = np.full(
            (grid.height, grid.width), config.output_nodata, dtype=np.int32
        )
        global_seeds = []
        for record, row, column, outcome_index in tqdm(
            located_records, desc="生成弱标签", unit="sample"
        ):
            outcome = sample_outcomes[outcome_index]
            window = _window_for_point(row, column, grid, config.window_size)
            row_start, column_start = int(window.row_off), int(window.col_off)
            local_row, local_column = row - row_start, column - column_start
            arrays = [
                dataset.read(1, window=window, masked=True)
                for dataset in (band_a, band_b, band_c)
            ]
            valid = np.logical_and.reduce(
                [~np.ma.getmaskarray(array) for array in arrays]
            )
            image = _stretch_rgb(
                np.stack([array.filled(0) for array in arrays]), ranges
            )
            seed = record_to_seed(record, local_row, local_column)
            request = build_pointsam_request(
                image,
                [seed],
                [],
                metadata={
                    "window": (
                        window.col_off,
                        window.row_off,
                        window.width,
                        window.height,
                    )
                },
            )
            prediction = run_pointsam_with_npc(
                inferencer,
                request.image,
                request.positive_points,
                request.negative_points,
                spatial_shape=image.shape[-2:],
                metadata=request.metadata,
            )
            confidence = (
                float(prediction.confidence.flat[0])
                if prediction.confidence is not None
                else 1.0
            )
            global_seeds.append(record_to_seed(record, row, column))
            selected = np.zeros(valid.shape, dtype=bool)
            pixel_score = np.zeros(valid.shape, dtype=np.float32)
            if confidence >= config.medium_confidence:
                component = _seed_component(
                    prediction.mask.astype(bool), local_row, local_column
                )
                if component.any() and not (
                    config.reject_boundary_touch
                    and _touches_boundary(component, window, grid)
                ):
                    radius = (
                        config.label_radius
                        if confidence >= config.min_confidence
                        else config.medium_label_radius
                    )
                    selected = (
                        component
                        & valid
                        & _disk_mask(component.shape, local_row, local_column, radius)
                    )
                    if prediction.mask_logits is not None:
                        pixel_score = 1.0 / (
                            1.0 + np.exp(-prediction.mask_logits.astype(np.float32))
                        )
                    else:
                        pixel_score.fill(confidence)
                    selected &= pixel_score >= config.logit_threshold
            if not selected.any():
                normalized = image.astype(np.float32) / 255.0
                reference = normalized[:, local_row, local_column]
                distance = np.sqrt(
                    np.mean(
                        (normalized - reference[:, None, None]) ** 2,
                        axis=0,
                    )
                )
                selected = (
                    _disk_mask(
                        valid.shape,
                        local_row,
                        local_column,
                        config.fallback_radius,
                    )
                    & valid
                    & (distance <= config.spectral_distance_threshold)
                )
                pixel_score = np.clip(
                    1.0 - distance / config.spectral_distance_threshold,
                    0.0,
                    1.0,
                ).astype(np.float32)
                if selected.any():
                    outcome["status"] = "spectral_fallback_accepted"
                else:
                    if confidence < config.medium_confidence:
                        outcome["status"] = "low_confidence"
                    elif not prediction.mask[local_row, local_column]:
                        outcome["status"] = "mask_missing_seed"
                    elif config.reject_boundary_touch and _touches_boundary(
                        _seed_component(
                            prediction.mask.astype(bool), local_row, local_column
                        ),
                        window,
                        grid,
                    ):
                        outcome["status"] = "boundary_touch"
                    else:
                        outcome["status"] = "no_pixels_after_filters"
                    continue
            elif confidence >= config.min_confidence:
                outcome["status"] = "high_confidence_accepted"
            else:
                outcome["status"] = "medium_confidence_accepted"
            candidate_score = pixel_score * confidence
            global_rows = slice(row_start, row_start + int(window.height))
            global_columns = slice(column_start, column_start + int(window.width))
            current_score = score[global_rows, global_columns]
            current_labels = labels[global_rows, global_columns]
            better = selected & (
                candidate_score > current_score + config.conflict_margin
            )
            tied = (
                selected
                & np.isfinite(current_score)
                & (np.abs(candidate_score - current_score) <= config.conflict_margin)
            )
            current_score[better] = candidate_score[better]
            current_labels[better] = record.alliance_code
            current_labels[tied] = config.output_nodata
            current_score[tied] = -np.inf
        destination.write(labels, 1)

    report = evaluate_label_quality_from_raster(
        output,
        seeds=global_seeds,
        alliance_names=alliance_names,
        sample_outcomes=sample_outcomes,
    )
    if quality_report_path is not None:
        report_path = Path(quality_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if quality_visualization_path is not None:
        with rasterio.open(output) as dataset:
            max_size = 1024
            scale = min(1.0, max_size / max(dataset.width, dataset.height))
            preview = dataset.read(
                1,
                out_shape=(
                    1,
                    max(1, int(dataset.height * scale)),
                    max(1, int(dataset.width * scale)),
                ),
                resampling=rasterio.enums.Resampling.nearest,
            )
            valid = (
                preview != dataset.nodata
                if dataset.nodata is not None
                else np.ones(preview.shape, dtype=bool)
            )
        write_label_quality_visualization(
            preview,
            quality_visualization_path,
            valid_mask=valid,
        )
    return report


def record_to_seed(record: LabelRecord, row: int, column: int) -> Any:
    """Convert an encoded CSV record to the local window coordinate contract."""

    from data.npc import PointSeed

    return PointSeed(row, column, record.formation_code, record.alliance_code)


def evaluate_label_quality_from_raster(
    path: str | Path,
    *,
    seeds: Sequence[Any] = (),
    alliance_names: dict[int, str] | None = None,
    sample_outcomes: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a generated label raster by reading bounded windows."""

    grid_size = (4, 4)
    class_counts: Counter[int] = Counter()
    spatial_grid = [[0 for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    labeled_pixels = invalid_pixels = valid_pixels = total_pixels = 0
    with rasterio.open(path) as dataset:
        total_pixels = dataset.width * dataset.height
        for row in range(0, dataset.height, 256):
            for column in range(0, dataset.width, 256):
                window = Window(
                    column,
                    row,
                    min(256, dataset.width - column),
                    min(256, dataset.height - row),
                )
                labels = dataset.read(1, window=window)
                valid = (
                    labels != dataset.nodata
                    if dataset.nodata is not None
                    else np.ones(labels.shape, dtype=bool)
                )
                positive = (labels > 0) & valid
                valid_pixels += int(valid.sum())
                labeled_pixels += int(positive.sum())
                invalid_pixels += int(((labels < 0) & valid).sum())
                for value in np.unique(labels[positive]):
                    class_counts[int(value)] += int(((labels == value) & valid).sum())
                grid_rows = (
                    (row + np.arange(positive.shape[0]))
                    * grid_size[0]
                    // dataset.height
                )
                grid_columns = (
                    (column + np.arange(positive.shape[1]))
                    * grid_size[1]
                    // dataset.width
                )
                for grid_row in np.unique(grid_rows):
                    for grid_column in np.unique(grid_columns):
                        selected = (
                            positive
                            & (grid_rows[:, None] == grid_row)
                            & (grid_columns[None, :] == grid_column)
                        )
                        spatial_grid[int(grid_row)][int(grid_column)] += int(
                            selected.sum()
                        )
        sample_invalid = 0
        for seed in seeds:
            if not (
                0 <= seed.row < dataset.height and 0 <= seed.column < dataset.width
            ):
                sample_invalid += 1
                continue
            value = dataset.read(1, window=Window(seed.column, seed.row, 1, 1))[0, 0]
            if value != seed.alliance_code:
                sample_invalid += 1
    if sample_outcomes is None:
        sample_quality = {
            "total_samples": len(seeds),
            "invalid_samples": sample_invalid,
            "invalid_sample_ratio": sample_invalid / len(seeds) if seeds else 0.0,
        }
    else:
        outcomes = [dict(item) for item in sample_outcomes]
        accepted_statuses = {
            "high_confidence_accepted",
            "medium_confidence_accepted",
            "spectral_fallback_accepted",
        }
        with rasterio.open(path) as dataset:
            for outcome in outcomes:
                if outcome["status"] not in accepted_statuses:
                    continue
                value = dataset.read(
                    1, window=Window(outcome["column"], outcome["row"], 1, 1)
                )[0, 0]
                if value != outcome["alliance_code"]:
                    outcome["status"] = "conflict_or_removed"
        reason_counts = Counter(
            outcome["status"]
            for outcome in outcomes
            if outcome["status"] not in accepted_statuses
        )
        invalid_count = sum(reason_counts.values())
        in_grid_outcomes = [
            outcome for outcome in outcomes if outcome["status"] != "outside_grid"
        ]
        in_grid_invalid_count = sum(
            outcome["status"] not in accepted_statuses for outcome in in_grid_outcomes
        )
        sample_quality = {
            "total_samples": len(outcomes),
            "in_grid_samples": len(in_grid_outcomes),
            "inference_attempted": sum(
                outcome["status"] not in {"outside_grid", "pending"}
                for outcome in outcomes
            ),
            "accepted_samples": sum(
                outcome["status"] in accepted_statuses for outcome in outcomes
            ),
            "invalid_samples": invalid_count,
            "invalid_sample_ratio": invalid_count / len(outcomes) if outcomes else 0.0,
            "in_grid_invalid_samples": in_grid_invalid_count,
            "in_grid_invalid_ratio": (
                in_grid_invalid_count / len(in_grid_outcomes)
                if in_grid_outcomes
                else 0.0
            ),
            "failure_reasons": dict(sorted(reason_counts.items())),
        }
    return {
        "shape": [dataset.height, dataset.width],
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "labeled_pixels": labeled_pixels,
        "coverage": labeled_pixels / valid_pixels if valid_pixels else 0.0,
        "invalid_pixel_count": invalid_pixels,
        "invalid_pixel_ratio": invalid_pixels / valid_pixels if valid_pixels else 0.0,
        "class_distribution": {
            str(code): {
                "alliance": (
                    alliance_names.get(code, str(code))
                    if alliance_names is not None
                    else str(code)
                ),
                "count": count,
                "ratio_of_labeled": count / labeled_pixels if labeled_pixels else 0.0,
                "ratio_of_valid": count / valid_pixels if valid_pixels else 0.0,
            }
            for code, count in sorted(class_counts.items())
        },
        "spatial_grid": spatial_grid,
        "sample_quality": sample_quality,
    }
