"""PointSAM weak-label generation and diagnostics."""

from weak_label.generation import (
    WeakLabelGenerationConfig,
    evaluate_label_quality_from_raster,
    generate_weak_labels,
)
from weak_label.quality import (
    evaluate_label_quality,
    write_label_quality_report,
    write_label_quality_visualization,
)
from weak_label.sam_input import discover_sam_videos

__all__ = [
    "WeakLabelGenerationConfig",
    "discover_sam_videos",
    "evaluate_label_quality",
    "evaluate_label_quality_from_raster",
    "generate_weak_labels",
    "write_label_quality_report",
    "write_label_quality_visualization",
]
