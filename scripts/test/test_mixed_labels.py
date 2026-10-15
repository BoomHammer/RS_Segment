"""Preserve mixed pixels across loading, augmentation, loss, splits and evaluation."""

import copy
import csv

import numpy as np
import pytest
import torch

from data.augmentations import SynchronizedAugmentation
from data.balanced_sampling import ClassBalancedPointSampler
from data.labels import (
    build_label_mapping,
    iter_encoded_labels,
    validate_label_mapping,
    validate_labels,
)
from data.raster_alignment import target_grid_from_raster
from data.sample_index import (
    WindowedSampleDataset,
    build_sample_index,
    sample_collate_fn,
)
from data.spatial_split import build_spatial_split
from data.training_policy import supervision_summary
from evaluation import (
    collect_point_predictions,
    evaluate_points,
    merge_point_prediction_shards,
)
from losses.supervision import combined_supervision_loss
from models.architecture import HierarchicalHeads
from test_sample_index import _write_raster


def test_rare_classes_use_unique_coordinates_and_recursive_parent_counts(tmp_path):
    path = tmp_path / "labels.csv"
    rows = []
    for category, count in [
        (("A", "a", "tree", "oak"), 10),
        (("A", "a", "tree", "elm"), 1),
        (("A", "a", "shrub", ""), 9),
        (("A", "b", "grass", ""), 9),
        (("B", "b", "forest", ""), 10),
        (("C", "c", "grass", ""), 1),
        (("D", "d", "needle", ""), 1),
        (("D", "d", "broad", ""), 9),
    ]:
        for _ in range(count):
            rows.append([len(rows) * 0.001, 2, *category])
    rows.extend([rows[10]] * 15)  # Sixteen CSV rows still supply one coordinate.
    for i in range(6):
        for child in ("x", "y"):
            rows.append([i * 0.001, 3, "E", "e", child, ""])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["X", "Y", "L0", "L1", "L2", "L3"])
        writer.writerows(rows)
    columns = {
        "levels": [{"name": f"L{i}", "column": f"L{i}"} for i in range(4)],
        "missing_policy": "partial",
        "optional_leaf_level": True,
        "min_samples_per_fine_class": 10,
    }
    mapping = build_label_mapping(path, label_columns=columns)
    assert validate_label_mapping(mapping)["valid"]
    classes = mapping["classes"]
    assert any(c["alliance"] == "oak" for c in classes)
    assert not any(c["alliance"] == "elm" for c in classes)
    assert {
        c["formation"]: c["terminal_level"] for c in classes if c["formation"] != "A"
    } == {"B": 3, "C": 1, "D": 2, "E": 1}
    records = [
        r
        for batch in iter_encoded_labels(path, label_columns=columns, mapping=mapping)
        for r in batch
    ]
    by_code = {c["alliance_code"]: c for c in classes}
    assert by_code[records[10].alliance_code]["alliance"] == "Other_tree"
    assert records[11].level_codes[2:] == (
        -1,
        -1,
    )  # Third -> second, no guessed sibling.
    assert records[20].level_codes[1:] == (-1, -1, -1)  # Second -> first.
    assert (
        next(
            x
            for x in mapping["coarsened_classes"]
            if x["source_level_names"][-1] == "elm"
        )["ground_truth_samples"]
        == 1
    )


def mixed_dataset(tmp_path, *, reverse=False, cross_root=False):
    path = tmp_path / ("reverse.csv" if reverse else "labels.csv")
    rows = ["0.1,3.9,A,a", "0.1,3.9,A,b", "0.2,3.8,A,a", "2.5,1.5,A,a"]
    rows.extend([rows[0]] * 3)
    if cross_root:
        rows[1] = "0.1,3.9,B,b"
    path.write_text(
        "X,Y,L0,L1\n" + "\n".join(reversed(rows) if reverse else rows) + "\n",
        encoding="utf-8",
    )
    columns = {
        "levels": [{"name": f"L{i}", "column": f"L{i}"} for i in range(2)],
        "missing_policy": "partial",
    }
    mapping = build_label_mapping(path, label_columns=columns)
    dynamic, static = tmp_path / "dynamic", tmp_path / "static"
    dynamic.mkdir(exist_ok=True)
    static.mkdir(exist_ok=True)
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
    return (
        WindowedSampleDataset(
            index, window_size=(4, 4), label_columns=columns, label_mapping=mapping
        ),
        mapping,
        path,
        columns,
    )


def test_conflicting_pixels_survive_order_sampling_and_augmentation(tmp_path):
    dataset, mapping, path, columns = mixed_dataset(tmp_path)
    reversed_dataset, _, _, _ = mixed_dataset(tmp_path, reverse=True)
    first, reversed_sample = dataset[0], reversed_dataset[0]
    torch.testing.assert_close(
        first["ground_truth_observations"], reversed_sample["ground_truth_observations"]
    )
    assert first["ground_truth_observations"].shape == (3, 4)
    assert first["ground_truth_levels"][0, 0, 0] == 1
    assert first["ground_truth"][0, 0] == -1
    assert (0, 0) in dataset.supervision_pixels
    report = validate_labels(path, label_columns=columns, mapping=mapping)
    assert report.valid_rows == 7 and report.mixed_coordinate_count == 1
    augmented = copy.deepcopy(first)
    SynchronizedAugmentation._flip(augmented, dim=-1)
    SynchronizedAugmentation._rotate(augmented, 1)
    assert torch.all(augmented["ground_truth_observations"][:, :2] == 0)
    assert augmented["ground_truth_levels"][0, 0, 0] == 1
    manifest = build_spatial_split(dataset, block_size=(4, 4), ratios=(1, 0, 0))
    assert manifest.class_counts["train"] == {"1": 2, "2": 1}
    assert supervision_summary(dataset, manifest)["unique_points"]["train"] == 2
    sampler = ClassBalancedPointSampler(dataset, manifest, [0])
    assert len(sampler) == 2
    selected = next(dataset[item] for item in sampler if item[1] == (0, 0))
    assert len(selected["ground_truth_observations"]) == 3
    # Ownership filters sparse and hard labels together, including fully mixed roots.
    manifest.blocks["0:0"] = "validation"
    dataset.configure_supervision_split(manifest, "train", mask_weak_labels=True)
    assert len(dataset[0]["ground_truth_observations"]) == 0
    assert not dataset[0]["ground_truth_levels"].gt(0).any()


@pytest.mark.parametrize("cross_root", [False, True])
def test_mixed_loss_has_equal_pixel_weight_and_all_class_gradients(
    tmp_path, cross_root
):
    dataset, mapping, _, _ = mixed_dataset(tmp_path, cross_root=cross_root)
    batch = sample_collate_fn([dataset[0]])
    # Focus the loss on the mixed pixel; duplicates must not turn 2:1 into 5:1.
    batch["valid_mask"].zero_()
    batch["valid_mask"][0, 0, 0] = True
    fine = (
        torch.tensor([0.8, 0.2])
        .log()
        .reshape(1, 2, 1, 1)
        .expand(1, 2, 4, 4)
        .clone()
        .requires_grad_()
    )
    root = fine if cross_root else torch.zeros(1, 1, 4, 4, requires_grad=True)
    output = {"fine_logits": fine, "level_0_logits": root, "level_1_logits": fine}
    losses = combined_supervision_loss(
        output, batch, level_parents=mapping["level_parents"], hierarchy_weight=0
    )
    expected = -(
        2 / 3 * torch.log(torch.tensor(0.8)) + 1 / 3 * torch.log(torch.tensor(0.2))
    )
    torch.testing.assert_close(losses["ground_truth_level_1_loss"], expected)
    losses["loss"].backward()
    assert fine.grad[0, 0, 0, 0] > 0 and fine.grad[0, 1, 0, 0] < 0
    batch["valid_mask"].zero_()
    assert (
        combined_supervision_loss(
            output, batch, level_parents=mapping["level_parents"], hierarchy_weight=0
        )["loss"].item()
        == 0
    )


def test_mixed_evaluation_reports_distributions_and_merges_windows(tmp_path):
    dataset, mapping, _, _ = mixed_dataset(tmp_path)
    batch = sample_collate_fn([dataset[0]])
    head = HierarchicalHeads.from_derived(2, {**mapping, "fine_to_coarse": [0, 0]})

    class Model(torch.nn.Module):
        def forward(self, _):
            return head(torch.zeros(1, 2, 4, 4))

    report = evaluate_points(
        Model(), [batch, batch], torch.device("cpu"), "none", selection_level=0
    )
    assert report["levels"]["0"]["labeled_pixels"] == 2
    assert report["levels"]["1"]["labeled_pixels"] == 1
    assert report["mixed_pixels"]["1"]["pixel_count"] == 1
    raw = {}
    collect_point_predictions(
        Model(), [batch], torch.device("cpu"), "none", raw_predictions=raw
    )
    mixed = {}
    merge_point_prediction_shards([raw, raw], mixed_predictions=mixed)
    torch.testing.assert_close(mixed["leaf"][0], torch.tensor([[2 / 3, 1 / 3]]))
    assert mixed["leaf"][3] == 2


def test_all_mixed_root_evaluation_and_partial_reporting(tmp_path):
    dataset, mapping, path, columns = mixed_dataset(tmp_path, cross_root=True)
    batch = sample_collate_fn([dataset[0]])
    batch["valid_mask"].zero_()
    batch["valid_mask"][0, 0, 0] = True

    class Model(torch.nn.Module):
        def forward(self, _):
            logits = (
                torch.tensor([0.8, 0.2]).log().reshape(1, 2, 1, 1).expand(1, 2, 4, 4)
            )
            return {
                "fine_logits": logits,
                "level_0_logits": logits,
                "level_1_logits": logits,
            }

    report = evaluate_points(
        Model(), [batch], torch.device("cpu"), "none", selection_level=0
    )
    assert report["selection_target_type"] == "mixed_observation_distribution"
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["mixed_pixels"]["0"]["pixel_count"] == 1
    with path.open("a", encoding="utf-8") as stream:
        stream.write("3.5,0.5,A,\n")
    validation = validate_labels(path, label_columns=columns, mapping=mapping)
    assert validation.partial_rows == 1
    assert validation.missing_hierarchy_rows == []
    assert validation.partial_label_rows == [9]
