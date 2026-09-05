"""Leakage-safe spatial splits and long-tail sampling metadata."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
) -> dict[tuple[int, int], str]:
    """Assign blocks while keeping measured classes in every requested split.

    A block is the atomic unit, so exact per-class ratios are not always
    possible. The greedy objective minimizes deviation from the requested
    class pixel counts and strongly prefers filling a class that is currently
    absent from a split. This retains the leakage-safe block boundary.
    """

    names = ("train", "validation", "test")
    sizes = _split_sizes(len(grouped), ratios)
    capacities = dict(zip(names, sizes, strict=True))
    rng = random.Random(seed)
    blocks = list(grouped)
    rng.shuffle(blocks)
    class_totals = sum(block_classes.values(), Counter())
    class_targets = {
        code: [class_totals[code] * ratio for ratio in ratios] for code in class_totals
    }
    class_presence = Counter(
        code for classes in block_classes.values() for code in classes
    )
    blocks.sort(
        key=lambda block: (
            -len(block_classes[block]),
            min(class_presence[code] for code in block_classes[block])
            if block_classes[block]
            else len(blocks),
        )
    )
    assigned: dict[tuple[int, int], str] = {}
    counts = {name: Counter() for name in names}
    for block in blocks:
        candidates = [name for name in names if capacities[name] > 0]
        if not candidates:
            raise RuntimeError("空间块分配失败：剩余空间块没有可用划分")

        def score(
            name: str, current_block: tuple[int, int] = block
        ) -> tuple[float, int]:
            missing = sum(
                1
                for code in block_classes[current_block]
                if counts[name][code] == 0 and class_presence[code] > 1
            )
            deviation = 0.0
            for code, value in block_classes[current_block].items():
                target = class_targets[code][names.index(name)]
                deviation += (counts[name][code] + value - target) ** 2 / max(
                    target, 1.0
                )
            return (-missing * 1000.0 + deviation, capacities[name])

        selected = min(candidates, key=score)
        assigned[block] = selected
        capacities[selected] -= 1
        counts[selected].update(block_classes[block])

    missing = {
        name: {
            code
            for code in class_totals
            if class_presence[code] >= 3 and counts[name][code] == 0
        }
        for name in names
        if ratios[names.index(name)] > 0
    }
    if any(missing.values()):
        details = ", ".join(
            f"{name}: {sorted(values)}" for name, values in missing.items() if values
        )
        raise ValueError(
            "按空间块进行类别分层后仍有类别缺失；请减小 block_size "
            "或提供更多跨空间块的实测样点。"
            f" 缺失={details}"
        )
    return assigned


def build_spatial_split(
    dataset: WindowedSampleDataset,
    *,
    block_size: tuple[int, int] = (2048, 2048),
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    output: str | Path | None = None,
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
    normalized = tuple(value / sum(ratios) for value in ratios)
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for window_id, row in enumerate(dataset.index.itertuples(index=False)):
        block = (int(row.row) // block_size[1], int(row.column) // block_size[0])
        grouped[block].append(int(window_id))

    names = ("train", "validation", "test")
    block_classes: dict[tuple[int, int], Counter[str]] = {
        block: Counter() for block in grouped
    }
    for (row, column), code in dataset.ground_truth_pixels.items():
        block = (int(row) // block_size[1], int(column) // block_size[0])
        block_classes[block][str(code)] += 1
    split_blocks = _stratified_block_assignment(
        grouped, block_classes, normalized, seed
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

    window_to_split = {window_id: name for name in names for window_id in splits[name]}
    counts_by_split: dict[str, Counter[str]] = {name: Counter() for name in names}
    window_classes: dict[int, Counter[str]] = defaultdict(Counter)
    for (row, column), code in dataset.ground_truth_pixels.items():
        window_ids = dataset.query_windows_for_pixel(row, column)
        for window_id in window_ids:
            split = window_to_split[window_id]
            counts_by_split[split][str(code)] += 1
            window_classes[window_id][str(code)] += 1
    total_counts = sum(counts_by_split.values(), Counter())
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
    )
    if output is not None:
        manifest.write(output)
    return manifest
