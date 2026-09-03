"""Backward-compatible imports for the moved weak-label implementation."""

from weak_label.generation import (  # noqa: F401
    WeakLabelGenerationConfig,
    evaluate_label_quality_from_raster,
    generate_weak_labels,
)

__all__ = [
    "WeakLabelGenerationConfig",
    "evaluate_label_quality_from_raster",
    "generate_weak_labels",
]
