"""Backward-compatible imports for moved weak-label diagnostics."""

from weak_label.quality import (  # noqa: F401
    evaluate_label_quality,
    write_label_quality_report,
    write_label_quality_visualization,
)

__all__ = [
    "evaluate_label_quality",
    "write_label_quality_report",
    "write_label_quality_visualization",
]
