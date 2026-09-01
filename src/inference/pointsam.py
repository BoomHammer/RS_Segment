"""Replaceable PointSAM inference boundary and NPC prompt adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from data.npc import NegativeCandidate, PointSeed


@dataclass(frozen=True, slots=True)
class PointSAMRequest:
    """Model-agnostic image and point prompts passed to a PointSAM backend."""

    image: Any
    positive_points: tuple[PointSeed, ...]
    negative_points: tuple[NegativeCandidate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PointSAMPrediction:
    """Standardized output returned by a PointSAM backend."""

    mask: np.ndarray
    confidence: np.ndarray | None = None
    class_probabilities: np.ndarray | None = None
    mask_logits: np.ndarray | None = None


class PointSAMInferencer(Protocol):
    """Backend protocol implemented by a real or test PointSAM model."""

    def predict(self, request: PointSAMRequest) -> PointSAMPrediction:
        """Run inference for one image/window and its point prompts."""


def build_pointsam_request(
    image: Any,
    seeds: Sequence[PointSeed],
    negative_candidates: Sequence[NegativeCandidate],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PointSAMRequest:
    """Convert NPC outputs into an immutable PointSAM request."""

    return PointSAMRequest(
        image=image,
        positive_points=tuple(seeds),
        negative_points=tuple(negative_candidates),
        metadata=dict(metadata or {}),
    )


def validate_pointsam_prediction(
    prediction: PointSAMPrediction,
    *,
    spatial_shape: tuple[int, int],
) -> None:
    """Validate backend output before it enters downstream label generation."""

    if prediction.mask.shape[-2:] != spatial_shape:
        raise ValueError("PointSAM mask 的空间形状不匹配")
    if not np.isfinite(prediction.mask).all():
        raise ValueError("PointSAM mask 不能包含 NaN 或 Inf")
    if prediction.confidence is not None:
        if prediction.confidence.shape != spatial_shape:
            raise ValueError("PointSAM confidence 的形状不匹配")
        if not np.isfinite(prediction.confidence).all():
            raise ValueError("PointSAM confidence 不能包含 NaN 或 Inf")
        if ((prediction.confidence < 0) | (prediction.confidence > 1)).any():
            raise ValueError("PointSAM confidence 必须位于 [0, 1]")
    if prediction.class_probabilities is not None:
        probabilities = prediction.class_probabilities
        if probabilities.ndim != 3 or probabilities.shape[-2:] != spatial_shape:
            raise ValueError("PointSAM class_probabilities 必须是 [类别, 行, 列]")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("PointSAM class_probabilities 包含非法值")
    if prediction.mask_logits is not None:
        if prediction.mask_logits.shape != spatial_shape:
            raise ValueError("PointSAM mask_logits 的形状不匹配")
        if not np.isfinite(prediction.mask_logits).all():
            raise ValueError("PointSAM mask_logits 不能包含 NaN 或 Inf")


def run_pointsam_with_npc(
    inferencer: PointSAMInferencer,
    image: Any,
    seeds: Sequence[PointSeed],
    negative_candidates: Sequence[NegativeCandidate],
    *,
    spatial_shape: tuple[int, int],
    metadata: Mapping[str, Any] | None = None,
) -> PointSAMPrediction:
    """Run a replaceable backend with NPC-generated positive/negative prompts."""

    request = build_pointsam_request(
        image, seeds, negative_candidates, metadata=metadata
    )
    prediction = inferencer.predict(request)
    validate_pointsam_prediction(prediction, spatial_shape=spatial_shape)
    return prediction
