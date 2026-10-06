"""Partial observations, singleton branches and leakage-safe rare-class splits."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rasterio.windows import Window

from data.augmentations import SynchronizedAugmentation
from data.balanced_sampling import ClassBalancedPointSampler
from data.labels import build_label_mapping, iter_encoded_labels, validate_labels
from data.raster_alignment import target_grid_from_raster
from data.sample_index import (
    WindowedSampleDataset,
    build_sample_index,
    sample_collate_fn,
)
from data.spatial_split import (
    SpatialSplitManifest,
    _stratified_block_assignment,
    build_spatial_split,
)
from data.training_policy import point_windows
from evaluation import collect_point_predictions, evaluate_points
from losses.supervision import combined_supervision_loss
from models.architecture import HierarchicalHeads
from test_sample_index import _write_raster
from test_weak_labels import FakeInferencer, _write_band
from weak_label.generation import WeakLabelGenerationConfig, generate_weak_labels


def label_fixture(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "X,Y,L0,L1,L2\n"
        "0.5,3.5,A,a,x\n"
        "1.5,3.5,A,,\n"
        "0.5,2.5,A,a,\n"
        "2.5,3.5,B,b,\n"
        "2.5,2.5,B,b,y\n",
        encoding="utf-8",
    )
    columns = {
        "levels": [{"name": f"L{i}", "column": f"L{i}"} for i in range(3)],
        "missing_policy": "partial",
    }
    mapping = build_label_mapping(path, label_columns=columns)
    return path, columns, mapping


def test_partial_encoding_and_validation(tmp_path):
    path, columns, mapping = label_fixture(tmp_path)
    records = list(iter_encoded_labels(path, label_columns=columns, mapping=mapping))[0]
    assert records[1].level_codes == (1, -1, -1)
    assert records[2].level_codes == (1, 1, -1)
    assert records[1].alliance_code == records[2].alliance_code == -1
    assert len(mapping["classes"]) == 2
    report = validate_labels(path, label_columns=columns, mapping=mapping)
    assert (report.valid_rows, report.invalid_rows, report.partial_rows) == (5, 0, 3)


@pytest.mark.parametrize("row", ["A,,x", ",a,x", ",,", "C,c,"])
def test_gaps_empty_paths_and_unknown_taxonomy_are_rejected(tmp_path, row):
    path, columns, _ = label_fixture(tmp_path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"0.5,0.5,{row}\n")
    with pytest.raises(ValueError):
        build_label_mapping(path, label_columns=columns)


def test_singleton_edges_are_identity_probabilities():
    parents = [[0, 1, 1], [0, 0, 1, 2]]
    head = HierarchicalHeads.from_derived(
        4, {"level_counts": [2, 3, 4], "level_parents": parents}
    )
    output = head(torch.randn(1, 4, 2, 2))
    torch.testing.assert_close(
        output["level_0_logits"][:, 0], output["level_1_logits"][:, 0]
    )
    torch.testing.assert_close(
        output["level_1_logits"][:, 1], output["level_2_logits"][:, 2]
    )
    torch.testing.assert_close(
        output["level_1_logits"][:, 2], output["level_2_logits"][:, 3]
    )
    torch.testing.assert_close(output["fine_probability"].sum(1), torch.ones(1, 2, 2))


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("known", [1, 2])
def test_partial_loss_supervises_only_observed_levels(depth, known):
    parents = [[0, 0, 1]] if depth == 2 else [[0, 0, 1], [0, 0, 1, 2]]
    derived = {
        "level_counts": [2, 3] if depth == 2 else [2, 3, 4],
        "level_parents": parents,
        "fine_to_coarse": [0, 0, 1],
    }
    head = HierarchicalHeads.from_derived(4, derived)
    output = head(torch.randn(1, 4, 2, 2))
    labels = torch.full((1, depth, 2, 2), -1, dtype=torch.long)
    labels[:, :known] = 1
    leaf = labels[:, -1]
    batch = {
        "ground_truth_levels": labels,
        "ground_truth": leaf,
        "ground_truth_mask": leaf.gt(0),
        "valid_mask": torch.ones_like(leaf, dtype=torch.bool),
        "weak_label": torch.full_like(leaf, -1),
        "weak_label_mask": torch.zeros_like(leaf, dtype=torch.bool),
    }
    losses = combined_supervision_loss(
        output,
        batch,
        fine_to_coarse=[0, 0, 1],
        level_parents=parents,
        hierarchy_weight=0,
    )
    for level in range(known, depth):
        assert losses[f"ground_truth_level_{level}_loss"].item() == 0
    expected = sum(losses[f"ground_truth_level_{level}_loss"] for level in range(known))
    torch.testing.assert_close(losses["loss"], expected)
    losses["loss"].backward()
    root = head.root if depth == 3 else head.coarse
    assert root.weight.grad.abs().sum() > 0
    if known == 1:
        experts = head.transitions if depth == 3 else head.experts
        assert all(
            p.grad is None
            or torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-7)
            for p in experts.parameters()
        )


def test_partial_points_survive_dataset_sampler_augmentation_and_split_mask(tmp_path):
    path, columns, mapping = label_fixture(tmp_path)
    dynamic, static = tmp_path / "dynamic", tmp_path / "static"
    dynamic.mkdir()
    static.mkdir()
    image = dynamic / "NDVI230101.tif"
    _write_raster(image, np.ones((4, 4), dtype=np.float32))
    _write_raster(static / "DEM.tif", np.ones((4, 4), dtype=np.float32))
    index = build_sample_index(
        dynamic_dir=dynamic,
        static_dir=static,
        label_file=path,
        target_grid=target_grid_from_raster(
            image, target_crs="EPSG:4326", resolution=1
        ),
    )
    dataset = WindowedSampleDataset(
        index, window_size=(4, 4), label_columns=columns, label_mapping=mapping
    )
    manifest = SpatialSplitManifest(
        1,
        1,
        (2, 2),
        (0.5, 0.5, 0),
        {"train": [0], "validation": [0], "test": []},
        {"0:0": "train", "0:1": "validation", "1:0": "train", "1:1": "validation"},
        {},
        {},
        {},
    )
    assert len(dataset.ground_truth_pixels) == 2
    assert len(dataset.supervision_pixels) == 5
    assert 0 in point_windows(dataset, manifest, "train")
    sampler = ClassBalancedPointSampler(dataset, manifest, [0])
    assert len(sampler) == 3
    dataset.configure_supervision_split(manifest, "train", mask_weak_labels=True)
    sample = dataset[0]
    assert sample["ground_truth_mask"].sum() == 1
    assert sample["ground_truth_levels"][0].gt(0).sum() == 3
    assert sample["ground_truth_levels"][..., 2:].eq(-1).all()
    batch = sample_collate_fn([sample])
    assert batch["ground_truth_levels"].shape == (1, 3, 4, 4)
    augmentation = SynchronizedAugmentation(
        horizontal_flip_probability=1, vertical_flip_probability=0, rotate_probability=0
    )
    augmented = augmentation(sample)
    torch.testing.assert_close(
        augmented["ground_truth_levels"], sample["ground_truth_levels"].flip(-1)
    )
    for window_id, pixel in sampler:
        selected = dataset[(window_id, pixel)]
        assert selected["ground_truth_levels"][0].gt(0).sum() == 1


def test_partial_weak_labels_are_not_promoted_to_leaf(tmp_path):
    path, columns, mapping = label_fixture(tmp_path)
    records = list(iter_encoded_labels(path, label_columns=columns, mapping=mapping))[0]
    bands = []
    for index in range(3):
        band = tmp_path / f"band{index}.tif"
        _write_band(band, np.ones((4, 4), dtype=np.float32))
        bands.append(band)
    report = generate_weak_labels(
        records[1:3],
        grid=target_grid_from_raster(bands[0], target_crs="EPSG:4326", resolution=1),
        image_paths=bands,
        inferencer=FakeInferencer(),
        output_path=tmp_path / "weak.tif",
        config=WeakLabelGenerationConfig(window_size=(4, 4)),
    )
    assert report["labeled_pixels"] == 0
    assert report["sample_quality"]["partial_label_samples"] == 2
    assert report["sample_quality"]["inference_attempted"] == 0
    assert report["sample_quality"]["invalid_samples"] == 0
    assert report["missing_classes"] == []


def test_per_level_evaluation_uses_partial_points_once():
    class Model(torch.nn.Module):
        def forward(self, batch):
            logits = torch.tensor([[[[2.0, 2.0, 2.0]], [[0.0, 0.0, 0.0]]]])
            return {
                "fine_logits": logits,
                "level_0_logits": logits,
                "level_1_logits": logits,
                "level_2_logits": logits,
            }

    levels = torch.tensor([[[[1, 1, 1]], [[1, 1, -1]], [[1, -1, -1]]]])
    batch = {
        "ground_truth_levels": levels,
        "ground_truth": levels[:, -1],
        "ground_truth_mask": levels[:, -1].gt(0),
        "valid_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        "input_window": [Window(0, 0, 3, 1)],
    }
    collected = {}
    leaf = collect_point_predictions(
        Model(),
        [batch, batch],
        torch.device("cpu"),
        "none",
        level_predictions=collected,
    )
    assert len(leaf[0]) == 1
    assert [len(collected[i][0]) for i in range(3)] == [3, 2, 1]
    assert [collected[i][3] for i in range(3)] == [6, 4, 2]
    levels[:, -1] = -1
    batch["ground_truth_mask"] = levels[:, -1].gt(0)
    report = evaluate_points(Model(), [batch], torch.device("cpu"), "none")
    assert report["selection_level"] == 1


def test_rare_classes_reserved_in_train_without_duplicating_blocks(tmp_path):
    import pandas as pd

    points = {(i * 10, j): 1 for i in range(9) for j in range(3)}
    points[(0, 4)] = 2  # One point, one spatial block.
    points.update(
        {(i * 10, 5): 3 for i in range(3)}
    )  # Three blocks, only three points.
    dataset = SimpleNamespace(
        index=pd.DataFrame({"row": [i * 10 for i in range(9)], "column": [0] * 9}),
        ground_truth_pixels=points,
        query_windows_for_pixel=lambda row, column: [row // 10],
    )
    manifest = build_spatial_split(
        dataset,
        block_size=(10, 10),
        ratios=(0.6, 0.2, 0.2),
        min_class_points_per_split=2,
    )
    for code in ("2", "3"):
        assert manifest.class_coverage[code]["rare_train_only"]
        assert manifest.class_counts["train"][code] > 0
        assert code not in manifest.class_counts["validation"]
        assert code not in manifest.class_counts["test"]
    assert manifest.class_coverage["2"]["insufficient_splits"]
    assert sorted(i for ids in manifest.splits.values() for i in ids) == list(range(9))
    path = tmp_path / "split.json"
    manifest.write(path)
    import json

    assert (
        SpatialSplitManifest.from_dict(json.loads(path.read_text())).class_coverage
        == manifest.class_coverage
    )


def test_train_coverage_survives_tiny_train_quota():
    grouped = {(i, 0): [i] for i in range(4)}
    counts = {block: Counter({str(i): 1}) for i, block in enumerate(grouped)}
    assignment = _stratified_block_assignment(grouped, counts, (0.01, 0.49, 0.5), 42)
    assert set(assignment.values()) == {"train"}
