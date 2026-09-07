"""Protect split isolation including halo context and retained window overlap."""

from dataclasses import replace
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from rasterio.windows import Window

from data.spatial_split import SpatialSplitManifest
from data.training_policy import isolate_splits


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


def test_validation_reports_unique_points_separately_from_overlapping_windows():
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
    assert result["labeled_pixels"] == 4
    assert result["accuracy"] == 0.75
    assert result["unique_point_count"] == 3
    assert result["unique_point_accuracy"] == pytest.approx(2 / 3)
    assert result["confusion_matrix"] == [[3, 0], [1, 0]]
