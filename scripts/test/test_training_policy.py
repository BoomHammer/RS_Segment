"""Protect split isolation including halo context and retained window overlap."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from rasterio.windows import Window

from data.balanced_sampling import ClassBalancedPointSampler
from data.spatial_split import SpatialSplitManifest, build_spatial_split
from data.training_policy import assert_supervision_isolated, isolate_splits


def test_isolation_removes_cross_split_inputs_and_keeps_same_split_overlap():
    dataset = SimpleNamespace(
        index=pd.DataFrame({"row": [0, 0, 0, 0], "column": [0, 4, 6, 8]}),
        grid=SimpleNamespace(width=16, height=8),
        window_size=(4, 4),
        halo=(0, 0),
    )
    manifest = SpatialSplitManifest(
        schema_version=1,
        seed=42,
        block_size=(8, 8),
        ratios=(0.5, 0.5, 0),
        splits={"train": [0, 1, 2], "validation": [3], "test": []},
        blocks={"0:0": "train", "0:1": "validation"},
        class_counts={},
        class_weights={},
        sampling_weights={},
    )
    isolated = isolate_splits(dataset, manifest)
    assert isolated.splits == {"train": [0, 1], "validation": [3], "test": []}
    assert manifest.splits["train"] == [0, 1, 2]
    dataset.halo = (1, 1)
    isolated = isolate_splits(dataset, manifest)
    assert isolated.splits == {"train": [0], "validation": [], "test": []}
    manifest = replace(
        manifest,
        blocks={"0:0": "train", "0:1": "train"},
        splits={"train": [0, 1, 2, 3], "validation": [], "test": []},
    )
    assert isolate_splits(dataset, manifest).splits["train"] == [0, 1, 2, 3]


def test_validation_aggregates_overlapping_windows_by_position():
    evaluate = run_path(str(Path(__file__).parents[1] / "train.py"))["_evaluate"]

    class Model(torch.nn.Module):
        def forward(self, batch):
            logits = torch.zeros(2, 2, 1, 2)
            logits[:, 0] = 2
            return {"fine_logits": logits}

    batch = {
        "ground_truth": torch.tensor([[[1, 1]], [[1, 2]]]),
        "ground_truth_mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "valid_mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "input_window": [Window(0, 0, 2, 1), Window(1, 0, 2, 1)],
    }
    result = evaluate(Model(), [batch], torch.device("cpu"))
    assert result["labeled_pixels"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["unique_point_count"] == 3
    assert result["unique_point_accuracy"] == pytest.approx(2 / 3)
    assert result["window_label_occurrences"] == 4
    assert result["confusion_matrix"] == [[2, 0], [1, 0]]


def test_class_balanced_point_sampler_uses_every_point_once():
    dataset = SimpleNamespace(
        ground_truth_pixels={(0, 0): 1, (0, 1): 1, (1, 0): 2},
        query_windows_for_pixel=lambda row, column: [row * 2 + column],
    )
    manifest = SpatialSplitManifest(
        schema_version=1,
        seed=1,
        block_size=(4, 4),
        ratios=(1, 0, 0),
        splits={"train": [0, 1, 2], "validation": [], "test": []},
        blocks={"0:0": "train"},
        class_counts={"train": {"1": 2, "2": 1}},
        class_weights={},
        sampling_weights={},
    )
    assert_supervision_isolated(dataset, manifest)
    sampler = ClassBalancedPointSampler(
        dataset,
        manifest,
        [0, 1, 2],
        seed=7,
    )
    sampled = list(sampler)
    assert len(sampled) == 3
    anchors = [pixel for _, pixel in sampled]
    assert Counter(anchors) == Counter(dataset.ground_truth_pixels.keys())


def test_point_sampler_exhausts_all_class_points():
    pixels = {(0, column): 1 for column in range(4)}
    dataset = SimpleNamespace(
        ground_truth_pixels=pixels,
        query_windows_for_pixel=lambda row, column: [column],
    )
    manifest = SpatialSplitManifest(
        schema_version=1,
        seed=1,
        block_size=(8, 8),
        ratios=(1, 0, 0),
        splits={"train": list(range(4)), "validation": [], "test": []},
        blocks={"0:0": "train"},
        class_counts={"train": {"1": 4}},
        class_weights={},
        sampling_weights={},
    )
    sampler = ClassBalancedPointSampler(
        dataset,
        manifest,
        list(range(4)),
        seed=7,
    )
    anchors = [pixel for _, pixel in sampler]
    assert len(anchors) == 4
    assert set(anchors) == set(pixels)


def test_spatial_split_balances_labelled_blocks_and_unique_points():
    rows = []
    columns = []
    ground_truth = {}
    point_window = {}
    for block in range(30):
        rows.append(0)
        columns.append(block * 10)
        if block >= 12:
            continue
        for offset in range(10):
            pixel = (offset, block * 10)
            ground_truth[pixel] = 1
            point_window[pixel] = block
        pixel = (0, block * 10 + 1)
        ground_truth[pixel] = 2
        point_window[pixel] = block
    dataset = SimpleNamespace(
        index=pd.DataFrame({"row": rows, "column": columns}),
        ground_truth_pixels=ground_truth,
        query_windows_for_pixel=lambda row, column: [point_window[(row, column)]],
    )
    manifest = build_spatial_split(
        dataset,
        block_size=(10, 10),
        ratios=(0.6, 0.2, 0.2),
        seed=42,
    )
    assert {name: len(ids) for name, ids in manifest.splits.items()} == {
        "train": 18,
        "validation": 6,
        "test": 6,
    }
    assert {
        name: sum(counts.values())
        for name, counts in manifest.class_counts.items()
    } == {"train": 77, "validation": 33, "test": 22}
    assert all(len(counts) == 2 for counts in manifest.class_counts.values())
