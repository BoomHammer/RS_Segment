"""Class-aware measured-window sampling with bounded repetition."""

import math

import torch
from torch.utils.data import Sampler

from data.training_policy import point_owner, point_windows


class ClassBalancedPointSampler(Sampler[tuple[int, tuple[int, int]]]):
    """Visit every independent training point once in class-balanced order.

    One covering training window is selected for each point. Interleaving class
    queues prevents majority classes from occupying the beginning of an epoch,
    while exhaustive queues guarantee that no training point is omitted.
    """

    def __init__(
        self,
        dataset,
        manifest,
        indices,
        *,
        seed=42,
    ):
        if not indices:
            raise ValueError("采样索引不能为空")
        self.seed = seed
        self.epoch = 0
        index_set = set(indices)
        self.points = {}
        for pixel, code in dataset.ground_truth_pixels.items():
            if point_owner(*pixel, manifest) != "train":
                continue
            candidates = sorted(
                index_set & set(dataset.query_windows_for_pixel(*pixel))
            )
            if candidates:
                self.points.setdefault(int(code), []).append((pixel, candidates))
        if len(self.points) != len(manifest.class_counts.get("train", self.points)):
            missing = sorted(
                set(map(int, manifest.class_counts.get("train", {})))
                - self.points.keys()
            )
            if missing:
                raise ValueError(f"类别没有可采样的独立训练点: {missing}")
        self._length = sum(len(values) for values in self.points.values())
        if self._length < 1:
            raise ValueError("训练划分中没有可采样的独立真实点")

    def __len__(self):
        return self._length

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        queues = {}
        for code, points in self.points.items():
            queues[code] = torch.randperm(len(points), generator=generator).tolist()
        classes = sorted(queues)
        class_order = torch.randperm(len(classes), generator=generator).tolist()
        classes = [classes[position] for position in class_order]
        selected = []
        while len(selected) < self._length:
            progressed = False
            for code in classes:
                if not queues[code] or len(selected) >= self._length:
                    continue
                point_index = queues[code].pop()
                pixel, candidates = self.points[code][point_index]
                choice = int(
                    torch.randint(len(candidates), (1,), generator=generator).item()
                )
                selected.append((candidates[choice], pixel))
                progressed = True
            if not progressed:
                raise RuntimeError("类别平衡采样预算无法完成")
        order = torch.randperm(len(selected), generator=generator).tolist()
        return iter(selected[position] for position in order)


class CappedClassSampler(Sampler[int]):
    """Visit every measured window, then add a weighted subset at most once.

    Unique training point counts determine priorities. No validation/test label
    frequencies are used. Each window appears at most twice per epoch.
    """

    def __init__(
        self,
        dataset,
        indices,
        class_counts,
        *,
        manifest=None,
        extra_fraction=0.25,
        seed=42,
    ):
        if not 0 <= extra_fraction <= 1 or not indices:
            raise ValueError("需要非空索引，extra_fraction 必须位于 [0, 1]")
        self.indices = list(indices)
        self.extra = math.floor(len(indices) * extra_fraction)
        self.seed = seed
        self.epoch = 0
        windows = (
            point_windows(dataset, manifest, "train")
            if manifest is not None
            else point_windows(dataset)
        )
        self.weights = torch.tensor(
            [
                sum(
                    count / math.sqrt(max(class_counts.get(code, 0), 1))
                    for code, count in windows[index].items()
                )
                / sum(windows[index].values())
                for index in indices
            ],
            dtype=torch.double,
        )

    def __len__(self):
        return len(self.indices) + self.extra

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        extra = (
            torch.multinomial(
                self.weights, self.extra, replacement=False, generator=generator
            ).tolist()
            if self.extra
            else []
        )
        positions = list(range(len(self.indices))) + extra
        order = torch.randperm(len(positions), generator=generator).tolist()
        return iter(self.indices[positions[position]] for position in order)
