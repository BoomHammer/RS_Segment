"""Metadata-only supervision selection and spatial leakage diagnostics."""

from collections import Counter
from dataclasses import replace

from data.sample_index import WindowedSampleDataset
from data.spatial_split import SpatialSplitManifest


def point_windows(dataset: WindowedSampleDataset) -> dict[int, Counter]:
    """Index measured labels in window cores without reading raster imagery."""
    result: dict[int, Counter] = {}
    for pixel, code in dataset.ground_truth_pixels.items():
        for index in dataset.query_windows_for_pixel(*pixel):
            result.setdefault(index, Counter())[str(code)] += 1
    return result


def isolate_splits(
    dataset: WindowedSampleDataset, manifest: SpatialSplitManifest
) -> SpatialSplitManifest:
    """Drop windows whose input footprint touches blocks of another split.

    Include halo context and keep overlap between windows of the same split.
    Block assignments are retained; no held-out labels enter the training set.
    """
    block_width, block_height = manifest.block_size
    halo_x, halo_y = dataset.halo
    splits = {}
    for name, indices in manifest.splits.items():
        kept = []
        for index in indices:
            record = dataset.index.iloc[index]
            row, column = int(record.row), int(record.column)
            left, top = max(0, column - halo_x), max(0, row - halo_y)
            right = min(dataset.grid.width, column + dataset.window_size[0] + halo_x)
            bottom = min(dataset.grid.height, row + dataset.window_size[1] + halo_y)
            owners = {
                manifest.blocks.get(f"{y}:{x}")
                for y in range(top // block_height, (bottom - 1) // block_height + 1)
                for x in range(left // block_width, (right - 1) // block_width + 1)
            }
            if owners == {name}:
                kept.append(index)
        splits[name] = kept
    return replace(manifest, splits=splits)


def supervision_summary(dataset, manifest) -> dict:
    """Count independent points separately from overlapping-window occurrences."""
    windows = point_windows(dataset)
    split_sets = {name: set(ids) for name, ids in manifest.splits.items()}
    counts = {name: Counter() for name in split_sets}
    shared = Counter()
    for pixel, code in dataset.ground_truth_pixels.items():
        covering = set(dataset.query_windows_for_pixel(*pixel))
        names = [name for name, ids in split_sets.items() if covering & ids]
        for name in names:
            counts[name][str(code)] += 1
        if len(names) > 1:
            shared["/".join(names)] += 1
    return {
        "windows": {name: len(ids) for name, ids in split_sets.items()},
        "point_windows": {
            name: len(ids & windows.keys()) for name, ids in split_sets.items()
        },
        "unique_points": {name: sum(c.values()) for name, c in counts.items()},
        "unique_class_counts": {name: dict(c) for name, c in counts.items()},
        "shared_points": dict(shared),
    }
