"""Backward-compatible imports for the moved SAM input discovery."""

from weak_label.sam_input import (  # noqa: F401
    discover_sam_composites,
    discover_sam_images,
    discover_sam_videos,
    validate_sam_bands,
)

__all__ = [
    "discover_sam_composites",
    "discover_sam_images",
    "discover_sam_videos",
    "validate_sam_bands",
]
