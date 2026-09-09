"""Synchronized spatial and continuous-feature augmentations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

SPATIAL_KEYS = (
    "core_mask",
    "supervision_split_mask",
    "dynamic",
    "static",
    "ground_truth",
    "ground_truth_mask",
    "weak_label",
    "weak_label_mask",
    "dynamic_valid_mask",
    "static_valid_mask",
    "valid_mask",
)
CONTINUOUS_KEYS = ("dynamic", "static")


@dataclass(slots=True)
class SynchronizedAugmentation:
    """Apply identical spatial transforms to all aligned sample tensors.

    ``dynamic`` and ``static`` are the only tensors receiving spectral noise
    and gain. Label tensors and masks are never spectrally modified.
    """

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate_probability: float = 0.5
    spectral_noise_std: float = 0.0
    spectral_gain_std: float = 0.0
    seed: int | None = None
    _generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        probabilities = (
            self.horizontal_flip_probability,
            self.vertical_flip_probability,
            self.rotate_probability,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("空间增强概率必须在 [0, 1] 范围内")
        if self.spectral_noise_std < 0 or self.spectral_gain_std < 0:
            raise ValueError("光谱增强标准差不能为负数")
        self._generator = torch.Generator()
        if self.seed is not None:
            self._generator.manual_seed(self.seed)

    def _random(self) -> float:
        return float(torch.rand((), generator=self._generator))

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(sample)
        horizontal_flip = self._random() < self.horizontal_flip_probability
        vertical_flip = self._random() < self.vertical_flip_probability
        rotate = self._random() < self.rotate_probability
        turns = int(torch.randint(1, 4, (), generator=self._generator)) if rotate else 0
        if horizontal_flip:
            self._flip(result, dim=-1)
        if vertical_flip:
            self._flip(result, dim=-2)
        if rotate:
            self._rotate(result, turns)
        self._spectral(result)
        result["augmentation"] = {
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "rotation_quarter_turns": turns,
            "spectral_noise_std": self.spectral_noise_std,
            "spectral_gain_std": self.spectral_gain_std,
        }
        return result

    @staticmethod
    def _flip(sample: dict[str, Any], *, dim: int) -> None:
        for key in SPATIAL_KEYS:
            value = sample.get(key)
            if isinstance(value, Tensor):
                sample[key] = torch.flip(value, dims=(dim,))

    @staticmethod
    def _rotate(sample: dict[str, Any], turns: int) -> None:
        for key in SPATIAL_KEYS:
            value = sample.get(key)
            if isinstance(value, Tensor):
                sample[key] = torch.rot90(value, turns, dims=(-2, -1))

    def _spectral(self, sample: dict[str, Any]) -> None:
        for key in CONTINUOUS_KEYS:
            value = sample.get(key)
            if not isinstance(value, Tensor) or not value.is_floating_point():
                continue
            if self.spectral_gain_std:
                gain_shape = (*value.shape[:-2], 1, 1)
                gain = 1 + self.spectral_gain_std * torch.randn(
                    gain_shape, generator=self._generator, dtype=value.dtype
                )
                value = value * gain
            if self.spectral_noise_std:
                noise = self.spectral_noise_std * torch.randn(
                    value.shape, generator=self._generator, dtype=value.dtype
                )
                value = torch.where(torch.isfinite(value), value + noise, value)
            sample[key] = value
