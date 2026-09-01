"""PointSAM negative-point calibration (NPC) primitives.

The module operates on pixel coordinates and class probabilities. It does
not read imagery itself, so callers can use it with streamed raster windows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PointSeed:
    """A labelled point in (row, column) pixel coordinates."""

    row: int
    column: int
    formation_code: int
    alliance_code: int


@dataclass(frozen=True, slots=True)
class NegativeCandidate:
    """A confidence-filtered negative point for one positive anchor."""

    row: int
    column: int
    anchor_alliance_code: int
    predicted_formation_code: int
    predicted_alliance_code: int
    confidence: float


def _validate_shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2 or any(size < 1 for size in shape):
        raise ValueError("shape 必须是两个正整数")


def _disk_offsets(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius 必须是非负整数")
    extent = np.arange(-radius, radius + 1)
    rows, columns = np.meshgrid(extent, extent, indexing="ij")
    mask = rows**2 + columns**2 <= radius**2
    return np.stack((rows[mask], columns[mask]), axis=1)


def _check_seed(seed: PointSeed, shape: tuple[int, int]) -> None:
    if not (0 <= seed.row < shape[0] and 0 <= seed.column < shape[1]):
        raise ValueError(f"样点超出栅格范围: {seed}")
    if seed.formation_code < 1 or seed.alliance_code < 1:
        raise ValueError(f"类别编码必须为正整数: {seed}")


def expand_positive_samples(
    seeds: Sequence[PointSeed],
    shape: tuple[int, int],
    *,
    radius: int = 2,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Expand point labels into disks and mark cross-class conflicts as -1.

    The returned int32 array contains alliance IDs, 0 for unlabeled pixels,
    and -1 for ambiguous overlaps. valid_mask prevents expansion into NoData.
    """

    _validate_shape(shape)
    if valid_mask is not None and valid_mask.shape != shape:
        raise ValueError("valid_mask 的形状必须等于 shape")
    labels = np.zeros(shape, dtype=np.int32)
    offsets = _disk_offsets(radius)
    for seed in seeds:
        _check_seed(seed, shape)
        for row_offset, column_offset in offsets:
            row = seed.row + int(row_offset)
            column = seed.column + int(column_offset)
            if not (0 <= row < shape[0] and 0 <= column < shape[1]):
                continue
            if valid_mask is not None and not bool(valid_mask[row, column]):
                continue
            current = labels[row, column]
            if current == 0 or current == seed.alliance_code:
                labels[row, column] = seed.alliance_code
            elif current != -1:
                labels[row, column] = -1
    return labels


def _validate_probabilities(probabilities: np.ndarray) -> tuple[int, int, int]:
    if probabilities.ndim != 3:
        raise ValueError("probabilities 必须是 [类别, 行, 列] 三维数组")
    if probabilities.shape[0] < 1 or probabilities.shape[1] < 1:
        raise ValueError("probabilities 不能为空")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities 不能包含 NaN 或 Inf")
    if (probabilities < 0).any():
        raise ValueError("probabilities 不能为负数")
    return probabilities.shape


def filter_negative_candidates(
    candidates: Sequence[NegativeCandidate],
    *,
    min_confidence: float = 0.8,
    max_candidates_per_anchor: int | None = None,
) -> list[NegativeCandidate]:
    """Apply confidence filtering and deterministic per-anchor top-k pruning."""

    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence 必须位于 [0, 1]")
    if max_candidates_per_anchor is not None and max_candidates_per_anchor < 1:
        raise ValueError("max_candidates_per_anchor 必须是正整数或 None")
    selected = [item for item in candidates if item.confidence >= min_confidence]
    selected.sort(
        key=lambda item: (
            item.anchor_alliance_code,
            -item.confidence,
            item.row,
            item.column,
        )
    )
    if max_candidates_per_anchor is None:
        return selected
    counts: dict[int, int] = {}
    result: list[NegativeCandidate] = []
    for item in selected:
        count = counts.get(item.anchor_alliance_code, 0)
        if count >= max_candidates_per_anchor:
            continue
        counts[item.anchor_alliance_code] = count + 1
        result.append(item)
    return result


def generate_candidate_negatives(
    probabilities: np.ndarray,
    seeds: Sequence[PointSeed],
    positive_labels: np.ndarray,
    *,
    class_ids: Sequence[int],
    alliance_to_formation: Mapping[int, int],
    valid_mask: np.ndarray | None = None,
    protection_radius: int = 3,
    search_radius: int = 12,
    min_confidence: float = 0.8,
    exclude_same_formation: bool = True,
    max_candidates_per_anchor: int | None = 8,
) -> list[NegativeCandidate]:
    """Generate calibrated negative points around positive anchors.

    probabilities is [classes, rows, columns] and class_ids maps its channels
    to alliance IDs. A candidate is accepted only when its predicted class is
    confident, differs from the anchor class, is outside every positive disk,
    and satisfies the hierarchy constraint. By default it must also belong to
    a different formation.
    """

    class_count, height, width = _validate_probabilities(probabilities)
    shape = (height, width)
    if len(class_ids) != class_count or len(set(class_ids)) != class_count:
        raise ValueError("class_ids 必须与概率通道一一对应且不重复")
    if positive_labels.shape != shape:
        raise ValueError("positive_labels 的形状必须等于 probabilities 的空间形状")
    if valid_mask is not None and valid_mask.shape != shape:
        raise ValueError("valid_mask 的形状必须等于 probabilities 的空间形状")
    if search_radius <= protection_radius or protection_radius < 0:
        raise ValueError("search_radius 必须大于 protection_radius，且半径不能为负")
    for alliance_code, formation_code in alliance_to_formation.items():
        if alliance_code < 1 or formation_code < 1:
            raise ValueError("类别映射中的编码必须为正整数")

    class_ids_array = np.asarray(class_ids, dtype=np.int64)
    predicted_channels = np.argmax(probabilities, axis=0)
    confidence = np.max(probabilities, axis=0)
    all_positive = positive_labels != 0
    candidates: list[NegativeCandidate] = []
    for seed in seeds:
        _check_seed(seed, shape)
        if alliance_to_formation.get(seed.alliance_code) != seed.formation_code:
            raise ValueError(f"样点的小类与大类映射不一致: {seed}")
        for row_offset, column_offset in _disk_offsets(search_radius):
            distance_squared = int(row_offset) ** 2 + int(column_offset) ** 2
            if distance_squared <= protection_radius**2:
                continue
            row = seed.row + int(row_offset)
            column = seed.column + int(column_offset)
            if not (0 <= row < height and 0 <= column < width):
                continue
            if all_positive[row, column] or (
                valid_mask is not None and not bool(valid_mask[row, column])
            ):
                continue
            predicted_alliance = int(class_ids_array[predicted_channels[row, column]])
            predicted_formation = alliance_to_formation.get(predicted_alliance)
            if predicted_formation is None:
                continue
            if predicted_alliance == seed.alliance_code:
                continue
            if exclude_same_formation and predicted_formation == seed.formation_code:
                continue
            candidates.append(
                NegativeCandidate(
                    row=row,
                    column=column,
                    anchor_alliance_code=seed.alliance_code,
                    predicted_formation_code=predicted_formation,
                    predicted_alliance_code=predicted_alliance,
                    confidence=float(confidence[row, column]),
                )
            )
    return filter_negative_candidates(
        candidates,
        min_confidence=min_confidence,
        max_candidates_per_anchor=max_candidates_per_anchor,
    )
