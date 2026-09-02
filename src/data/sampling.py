"""PyTorch samplers and DataLoader construction for spatial samples."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler, get_worker_info

from .sample_index import WindowedSampleDataset, sample_collate_fn
from .spatial_split import SpatialSplitManifest, load_spatial_split


class SpatialWeightedSampler(Sampler[int]):
    """Weighted sampler combining class and spatial-region priorities."""

    def __init__(
        self,
        dataset: WindowedSampleDataset,
        *,
        indices: list[int] | None = None,
        manifest: SpatialSplitManifest | str | Path | None = None,
        split: str = "train",
        region_weights: dict[str, float] | None = None,
        num_samples: int | None = None,
        replacement: bool = True,
        seed: int = 42,
    ) -> None:
        if manifest is not None:
            loaded = load_spatial_split(manifest)
            indices = loaded.splits[split]
            class_weights = loaded.class_weights
            block_size = loaded.block_size
        else:
            class_weights = {}
            block_size = (2048, 2048)
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else list(indices)
        if not self.indices:
            raise ValueError("采样索引不能为空")
        self.replacement = replacement
        self.num_samples = num_samples or len(self.indices)
        if not replacement and self.num_samples > len(self.indices):
            raise ValueError("replacement=False 时 num_samples 不能超过索引数量")
        self.seed = seed
        self.epoch = 0
        self.weights = self._build_weights(
            self.indices,
            class_weights,
            region_weights or {},
            block_size,
        )

    def _build_weights(
        self,
        indices: list[int],
        class_weights: dict[str, float],
        region_weights: dict[str, float],
        block_size: tuple[int, int],
    ) -> torch.Tensor:
        window_classes: dict[int, Counter[str]] = {}
        for (row, column), code in self.dataset.ground_truth_pixels.items():
            for window_id in self.dataset.query_windows_for_pixel(row, column):
                if window_id in indices:
                    window_classes.setdefault(window_id, Counter())[str(code)] += 1
        weights = []
        for window_id in indices:
            classes = window_classes.get(window_id, {})
            class_weight = max(
                (class_weights.get(code, 1.0) for code in classes), default=1.0
            )
            row = self.dataset.index.iloc[window_id].row
            column = self.dataset.index.iloc[window_id].column
            region = f"{int(row) // block_size[1]}:{int(column) // block_size[0]}"
            weights.append(class_weight * region_weights.get(region, 1.0))
        result = torch.tensor(weights, dtype=torch.double)
        if not torch.isfinite(result).all() or (result <= 0).any():
            raise ValueError("类别/区域采样权重必须是有限正数")
        return result

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=generator,
        )
        self.epoch += 1
        return iter([self.indices[int(position)] for position in sampled])

    def __len__(self) -> int:
        return self.num_samples


def _worker_init_fn(_: int) -> None:
    """Ensure a forked worker starts without inherited raster handles."""

    worker = get_worker_info()
    if worker is not None:
        dataset = worker.dataset
        close = getattr(dataset, "close", None)
        if close is not None:
            close()


def build_dataloader(
    dataset: WindowedSampleDataset,
    *,
    indices: list[int] | None = None,
    sampler: Sampler[int] | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader[dict[str, Any]]:
    """Build a safe DataLoader using the project's temporal collate function."""

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size 必须为正数，num_workers 不能为负数")
    if indices is not None and sampler is not None:
        raise ValueError("indices 和 sampler 不能同时提供")
    if indices is not None:
        sampler = torch.utils.data.SubsetRandomSampler(indices)
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers 需要 num_workers > 0")
    generator = torch.Generator().manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": sampler is None,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": drop_last,
        "collate_fn": sample_collate_fn,
        "worker_init_fn": _worker_init_fn,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)
