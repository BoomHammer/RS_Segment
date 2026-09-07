"""Class-aware measured-window sampling with bounded repetition."""

import math

import torch
from torch.utils.data import Sampler

from data.training_policy import point_windows


class CappedClassSampler(Sampler[int]):
    """Visit every measured window, then add a weighted subset at most once.

    Unique training point counts determine priorities. No validation/test label
    frequencies are used. Each window appears at most twice per epoch.
    """

    def __init__(self, dataset, indices, class_counts, extra_fraction=0.25, seed=42):
        if not 0 <= extra_fraction <= 1 or not indices:
            raise ValueError("需要非空索引，extra_fraction 必须位于 [0, 1]")
        self.indices = list(indices)
        self.extra = math.floor(len(indices) * extra_fraction)
        self.seed = seed
        self.epoch = 0
        windows = point_windows(dataset)
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
