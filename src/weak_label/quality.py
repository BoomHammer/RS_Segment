"""Quality metrics and visual diagnostics for weak labels."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.npc import NegativeCandidate, PointSeed


def _spatial_distribution(mask: np.ndarray) -> dict[str, Any]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return {
            "count": 0,
            "row_min": None,
            "row_max": None,
            "column_min": None,
            "column_max": None,
            "row_mean": None,
            "column_mean": None,
            "quadrants": {},
        }
    row_mid = mask.shape[0] / 2
    column_mid = mask.shape[1] / 2
    quadrants = {
        "top_left": int(((rows < row_mid) & (columns < column_mid)).sum()),
        "top_right": int(((rows < row_mid) & (columns >= column_mid)).sum()),
        "bottom_left": int(((rows >= row_mid) & (columns < column_mid)).sum()),
        "bottom_right": int(((rows >= row_mid) & (columns >= column_mid)).sum()),
    }
    return {
        "count": int(len(rows)),
        "row_min": int(rows.min()),
        "row_max": int(rows.max()),
        "column_min": int(columns.min()),
        "column_max": int(columns.max()),
        "row_mean": float(rows.mean()),
        "column_mean": float(columns.mean()),
        "quadrants": quadrants,
    }


def _sample_quality(
    seeds: Sequence[PointSeed],
    labels: np.ndarray,
    valid_mask: np.ndarray,
    alliance_to_formation: Mapping[int, int] | None,
) -> dict[str, Any]:
    invalid = 0
    out_of_bounds = 0
    nodata = 0
    label_conflicts = 0
    hierarchy_conflicts = 0
    height, width = labels.shape
    for seed in seeds:
        if not (0 <= seed.row < height and 0 <= seed.column < width):
            out_of_bounds += 1
            continue
        if not valid_mask[seed.row, seed.column]:
            nodata += 1
        if labels[seed.row, seed.column] != seed.alliance_code:
            label_conflicts += 1
        if alliance_to_formation is not None and (
            alliance_to_formation.get(seed.alliance_code) != seed.formation_code
        ):
            hierarchy_conflicts += 1
    invalid = out_of_bounds + nodata + label_conflicts + hierarchy_conflicts
    total = len(seeds)
    return {
        "total_samples": total,
        "invalid_samples": invalid,
        "invalid_sample_ratio": invalid / total if total else 0.0,
        "out_of_bounds": out_of_bounds,
        "nodata": nodata,
        "label_conflicts": label_conflicts,
        "hierarchy_conflicts": hierarchy_conflicts,
    }


def evaluate_label_quality(
    labels: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    seeds: Sequence[PointSeed] = (),
    negative_candidates: Sequence[NegativeCandidate] = (),
    alliance_to_formation: Mapping[int, int] | None = None,
    alliance_names: Mapping[int, str] | None = None,
    grid_size: tuple[int, int] = (4, 4),
) -> dict[str, Any]:
    """Return coverage, class distribution, invalid rates, and spatial metrics."""

    if labels.ndim != 2 or min(labels.shape) < 1:
        raise ValueError("labels 必须是非空二维数组")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels 必须是整数类别编码数组")
    if valid_mask is None:
        valid_mask = np.ones(labels.shape, dtype=bool)
    if valid_mask.shape != labels.shape:
        raise ValueError("valid_mask 的形状必须与 labels 相同")
    if grid_size[0] < 1 or grid_size[1] < 1:
        raise ValueError("grid_size 必须是两个正整数")

    valid = valid_mask.astype(bool, copy=False)
    positive = (labels > 0) & valid
    conflicts = (labels < 0) & valid
    valid_count = int(valid.sum())
    class_distribution: dict[str, dict[str, float | int]] = {}
    for class_code in sorted(int(value) for value in np.unique(labels[positive])):
        count = int(((labels == class_code) & valid).sum())
        class_distribution[str(class_code)] = {
            "alliance": (
                alliance_names.get(class_code, str(class_code))
                if alliance_names is not None
                else str(class_code)
            ),
            "count": count,
            "ratio_of_labeled": count / int(positive.sum()) if positive.any() else 0.0,
            "ratio_of_valid": count / valid_count if valid_count else 0.0,
        }

    row_bins = np.linspace(0, labels.shape[0], grid_size[0] + 1, dtype=int)
    column_bins = np.linspace(0, labels.shape[1], grid_size[1] + 1, dtype=int)
    spatial_grid: list[list[int]] = []
    for row_index in range(grid_size[0]):
        row_values = []
        for column_index in range(grid_size[1]):
            window = positive[
                row_bins[row_index] : row_bins[row_index + 1],
                column_bins[column_index] : column_bins[column_index + 1],
            ]
            row_values.append(int(window.sum()))
        spatial_grid.append(row_values)

    result = {
        "shape": list(labels.shape),
        "total_pixels": int(labels.size),
        "valid_pixels": valid_count,
        "labeled_pixels": int(positive.sum()),
        "coverage": int(positive.sum()) / valid_count if valid_count else 0.0,
        "invalid_pixel_count": int(conflicts.sum()),
        "invalid_pixel_ratio": int(conflicts.sum()) / valid_count
        if valid_count
        else 0.0,
        "class_distribution": class_distribution,
        "spatial_distribution": _spatial_distribution(positive),
        "spatial_grid": spatial_grid,
        "sample_quality": _sample_quality(seeds, labels, valid, alliance_to_formation),
        "negative_candidate_count": len(negative_candidates),
    }
    return result


def write_label_quality_report(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a JSON quality report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_label_quality_visualization(
    labels: np.ndarray,
    output_path: str | Path,
    *,
    valid_mask: np.ndarray | None = None,
    seeds: Sequence[PointSeed] = (),
    negative_candidates: Sequence[NegativeCandidate] = (),
) -> Path:
    """Write a PNG diagnostic using an optional visualization dependency."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("可视化需要 matplotlib，请安装项目 visual 依赖") from exc
    if labels.ndim != 2:
        raise ValueError("labels 必须是二维数组")
    if valid_mask is None:
        valid_mask = np.ones(labels.shape, dtype=bool)
    if valid_mask.shape != labels.shape:
        raise ValueError("valid_mask 的形状必须与 labels 相同")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    display = np.ma.masked_where(~valid_mask, labels)
    axes[0].imshow(display, interpolation="nearest")
    axes[0].set_title("Weak-label classes")
    axes[1].imshow(valid_mask & (labels > 0), cmap="gray", interpolation="nearest")
    if seeds:
        axes[1].scatter(
            [seed.column for seed in seeds],
            [seed.row for seed in seeds],
            s=12,
            c="red",
            label="positive",
        )
    if negative_candidates:
        axes[1].scatter(
            [item.column for item in negative_candidates],
            [item.row for item in negative_candidates],
            s=12,
            c="cyan",
            label="negative",
        )
    axes[1].set_title("Coverage and prompts")
    if seeds or negative_candidates:
        axes[1].legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
