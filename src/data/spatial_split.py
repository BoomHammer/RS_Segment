"""Leakage-safe spatial splits and long-tail sampling metadata."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .sample_index import WindowedSampleDataset


@dataclass(frozen=True, slots=True)
class SpatialSplitManifest:
    """Reproducible split assignment and class-balancing metadata."""

    schema_version: int
    seed: int
    block_size: tuple[int, int]
    ratios: tuple[float, float, float]
    splits: dict[str, list[int]]
    blocks: dict[str, str]
    class_counts: dict[str, dict[str, int]]
    class_weights: dict[str, float]
    sampling_weights: dict[str, float]
    class_coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpatialSplitManifest:
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("不支持的空间划分 schema_version")
        return cls(
            schema_version=1,
            seed=int(payload["seed"]),
            block_size=tuple(payload["block_size"]),
            ratios=tuple(payload["ratios"]),
            splits={key: list(value) for key, value in payload["splits"].items()},
            blocks=dict(payload["blocks"]),
            class_counts={
                key: dict(value) for key, value in payload["class_counts"].items()
            },
            class_weights={
                key: float(value) for key, value in payload["class_weights"].items()
            },
            class_coverage=dict(payload.get("class_coverage", {})),
            sampling_weights={
                key: float(value) for key, value in payload["sampling_weights"].items()
            },
        )


def load_spatial_split(path: str | Path | SpatialSplitManifest) -> SpatialSplitManifest:
    """Load a spatial split manifest from memory or JSON."""

    if isinstance(path, SpatialSplitManifest):
        return path
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SpatialSplitManifest.from_dict(payload)


def _split_sizes(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    sizes = [int(value) for value in raw]
    for index in sorted(
        range(3), key=lambda item: raw[item] - sizes[item], reverse=True
    ):
        if sum(sizes) < total:
            sizes[index] += 1
    return sizes


def _stratified_block_assignment(
    grouped: dict[tuple[int, int], list[int]],
    block_classes: dict[tuple[int, int], Counter[str]],
    ratios: tuple[float, float, float],
    seed: int,
    min_class_points_per_split: int = 2,
) -> dict[tuple[int, int], str]:
    """Optimize labelled-point ratios while keeping every block indivisible."""

    names = ("train", "validation", "test")
    total_sizes = _split_sizes(len(grouped), ratios)
    labelled = [block for block in grouped if block_classes[block]]
    unlabelled = [block for block in grouped if not block_classes[block]]
    labelled_sizes = _split_sizes(len(labelled), ratios)
    labelled_sizes = [
        min(size, capacity)
        for size, capacity in zip(labelled_sizes, total_sizes, strict=True)
    ]
    for index in range(3):
        remaining = len(labelled) - sum(labelled_sizes)
        labelled_sizes[index] += min(
            remaining, total_sizes[index] - labelled_sizes[index]
        )
    if not labelled:
        shuffled = list(unlabelled)
        random.Random(seed).shuffle(shuffled)
        assigned = {}
        start = 0
        for name, size in zip(names, total_sizes, strict=True):
            for block in shuffled[start : start + size]:
                assigned[block] = name
            start += size
        return assigned

    codes = sorted(sum(block_classes.values(), Counter()))
    code_index = {code: index for index, code in enumerate(codes)}
    matrix = np.zeros((len(labelled), len(codes)), dtype=np.int64)
    for row, block in enumerate(labelled):
        for code, count in block_classes[block].items():
            matrix[row, code_index[code]] = count
    ratio_array = np.asarray(ratios, dtype=np.float64)
    class_totals = matrix.sum(axis=0)
    class_targets = ratio_array[:, None] * class_totals
    point_targets = ratio_array * class_totals.sum()
    class_presence = (matrix > 0).sum(axis=0)
    active_splits = int(np.sum(ratio_array > 0))

    def cost(counts: np.ndarray) -> float:
        class_error = np.sum((counts - class_targets) ** 2 / (class_targets + 1.0))
        point_error = 10.0 * np.sum(
            (counts.sum(axis=1) - point_targets) ** 2 / (point_targets + 1.0)
        )
        train_missing = 1_000_000.0 * np.sum(counts[0] == 0) if ratios[0] > 0 else 0.0
        feasible_everywhere = class_presence >= active_splits
        split_missing = 10_000.0 * sum(
            np.sum((counts[index] == 0) & feasible_everywhere)
            for index, ratio in enumerate(ratios)
            if ratio > 0
        )
        return float(class_error + point_error + train_missing + split_missing)

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    iterations = max(2_000, len(labelled) * 50)
    for restart in range(8):
        generator = np.random.default_rng(seed + restart)
        order = generator.permutation(len(labelled))
        assignment = np.empty(len(labelled), dtype=np.int8)
        start = 0
        for split, size in enumerate(labelled_sizes):
            assignment[order[start : start + size]] = split
            start += size
        counts = np.stack(
            [matrix[assignment == split].sum(axis=0) for split in range(3)]
        )
        current_cost = cost(counts)
        temperature = 5.0
        for _ in range(iterations):
            left, right = generator.integers(0, len(labelled), size=2)
            left_split = int(assignment[left])
            right_split = int(assignment[right])
            if left_split == right_split:
                continue
            candidate = counts.copy()
            candidate[left_split] += matrix[right] - matrix[left]
            candidate[right_split] += matrix[left] - matrix[right]
            candidate_cost = cost(candidate)
            difference = candidate_cost - current_cost
            if difference < 0 or generator.random() < np.exp(-difference / temperature):
                assignment[left], assignment[right] = right_split, left_split
                counts = candidate
                current_cost = candidate_cost
            temperature = max(0.02, temperature * 0.9997)
        if best is None or current_cost < best[0]:
            best = (current_cost, assignment.copy(), counts.copy())

    if best is None:
        raise RuntimeError("空间块优化没有产生有效划分")
    _, assignment, counts_array = best
    assigned = {
        block: names[int(assignment[index])] for index, block in enumerate(labelled)
    }
    random.Random(seed).shuffle(unlabelled)
    start = 0
    for index, name in enumerate(names):
        remaining = total_sizes[index] - labelled_sizes[index]
        for block in unlabelled[start : start + remaining]:
            assigned[block] = name
        start += remaining
    # Ratios are targets, not a reason to strand a class outside training.
    # Reserve whole blocks: no point duplication across held-out splits.
    if ratios[0] > 0:
        rare = (class_presence < active_splits) | (
            class_totals < active_splits * min_class_points_per_split
        )
        for index, block in enumerate(labelled):
            if np.any((matrix[index] > 0) & rare):
                assigned[block] = "train"
        covered = sum(
            (block_classes[block] for block in grouped if assigned[block] == "train"),
            Counter(),
        )
        missing = set(codes) - set(covered)
        while missing:
            block = max(
                labelled, key=lambda item: len(missing & set(block_classes[item]))
            )
            assigned[block] = "train"
            missing -= set(block_classes[block])
    return assigned


def build_spatial_split(
    dataset: WindowedSampleDataset,
    *,
    block_size: tuple[int, int] = (2048, 2048),
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    output: str | Path | None = None,
    min_class_points_per_split: int = 2,
) -> SpatialSplitManifest:
    """Assign whole spatial blocks to train/validation/test splits.

    Blocks, rather than individual windows, are shuffled and assigned. This
    prevents overlapping or neighboring windows from crossing split boundaries.
    Training sampling weights use inverse square-root class frequency, which
    is less aggressive than inverse frequency for highly long-tailed labels.
    """

    if len(block_size) != 2 or min(block_size) < 1:
        raise ValueError("block_size 必须是两个正整数")
    if len(ratios) != 3 or min(ratios) < 0 or sum(ratios) <= 0:
        raise ValueError("ratios 必须是三个非负数且总和大于 0")
    if min_class_points_per_split < 1:
        raise ValueError("min_class_points_per_split must be positive")
    normalized = tuple(value / sum(ratios) for value in ratios)
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for window_id, row in enumerate(dataset.index.itertuples(index=False)):
        block = (int(row.row) // block_size[1], int(row.column) // block_size[0])
        grouped[block].append(int(window_id))

    names = ("train", "validation", "test")
    block_classes: dict[tuple[int, int], Counter[str]] = {
        block: Counter() for block in grouped
    }
    for (row, column), code in getattr(
        dataset, "supervision_pixels", dataset.ground_truth_pixels
    ).items():
        block = (int(row) // block_size[1], int(column) // block_size[0])
        block_classes[block][str(code)] += 1
    split_blocks = _stratified_block_assignment(
        grouped, block_classes, normalized, seed, min_class_points_per_split
    )
    splits = {
        name: sorted(
            window_id
            for block, values in grouped.items()
            if split_blocks[block] == name
            for window_id in values
        )
        for name in names
    }

    counts_by_split: dict[str, Counter[str]] = {name: Counter() for name in names}
    window_classes: dict[int, Counter[str]] = defaultdict(Counter)
    for (row, column), code in getattr(
        dataset, "supervision_pixels", dataset.ground_truth_pixels
    ).items():
        owner = split_blocks[(int(row) // block_size[1], int(column) // block_size[0])]
        counts_by_split[owner][str(code)] += 1
        window_ids = dataset.query_windows_for_pixel(row, column)
        for window_id in window_ids:
            if window_id in splits[owner]:
                window_classes[window_id][str(code)] += 1
    total_counts = counts_by_split["train"]
    class_weights = {code: 1.0 / (count**0.5) for code, count in total_counts.items()}
    if class_weights:
        mean_weight = sum(class_weights.values()) / len(class_weights)
        class_weights = {
            code: weight / mean_weight for code, weight in class_weights.items()
        }
    sampling_weights = {}
    for window_id in splits["train"]:
        labels = window_classes[window_id]
        sampling_weights[str(window_id)] = max(
            (class_weights[code] for code in labels), default=1.0
        )
    all_codes = sorted({code for counts in block_classes.values() for code in counts})
    coverage = {}
    active = [name for name, ratio in zip(names, normalized, strict=True) if ratio > 0]
    for code in all_codes:
        per_split = {name: counts_by_split[name][code] for name in names}
        support_blocks = sum(counts[code] > 0 for counts in block_classes.values())
        coverage[code] = {
            "points": sum(per_split.values()),
            "spatial_blocks": support_blocks,
            "split_counts": per_split,
            "rare_train_only": normalized[0] > 0
            and (
                support_blocks < len(active)
                or sum(per_split.values()) < len(active) * min_class_points_per_split
            ),
            "insufficient_splits": [
                name for name in active if per_split[name] < min_class_points_per_split
            ],
            "min_points_per_split": min_class_points_per_split,
        }
    manifest = SpatialSplitManifest(
        schema_version=1,
        seed=seed,
        block_size=block_size,
        ratios=normalized,
        splits=splits,
        blocks={
            f"{row}:{column}": name for (row, column), name in split_blocks.items()
        },
        class_counts={
            name: dict(sorted(counts.items()))
            for name, counts in counts_by_split.items()
        },
        class_weights=dict(sorted(class_weights.items())),
        sampling_weights=sampling_weights,
        class_coverage=coverage,
    )
    if output is not None:
        manifest.write(output)
    return manifest
