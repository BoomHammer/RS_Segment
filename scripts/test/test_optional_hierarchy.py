"""Mixed third/fourth-level labels, priority supervision and validation selection."""

import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from rasterio.windows import Window

from data.labels import (
    build_label_mapping,
    iter_encoded_labels,
    validate_label_mapping,
    validate_labels,
)
from evaluation import (
    collect_point_predictions,
    evaluate_points,
    merge_point_prediction_shards,
    summarize_point_predictions,
)
from losses.supervision import combined_supervision_loss
from models.architecture import HierarchicalHeads
from models.config import derive_model_contract, validate_checkpoint_mapping


def mixed_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "X,Y,L0,L1,L2,L3,Z0,Z1,Z2,Z3\n"
        "0.5,3.5,A,a,irrigated,maize,农田,耕地,水浇地,玉米\n"
        "1.5,3.5,A,a,irrigated,,农田,耕地,水浇地,\n"
        "2.5,3.5,A,b,shrub,apple,农田,园地,灌木园地,苹果\n"
        "0.5,2.5,B,b,forest,,森林,林地,阔叶林,\n"
        "1.5,2.5,A,a,,,农田,耕地,,\n"
        "2.5,2.5,B,,,,森林,,,\n",
        encoding="utf-8",
    )
    columns = {
        "levels": [
            {"name": f"L{i}", "column": f"L{i}", "zh_column": f"Z{i}"} for i in range(4)
        ],
        "missing_policy": "partial",
        "optional_leaf_level": True,
    }
    mapping = build_label_mapping(path, label_columns=columns)
    records = [
        r
        for batch in iter_encoded_labels(path, label_columns=columns, mapping=mapping)
        for r in batch
    ]
    return path, columns, mapping, records


def test_mixed_leaves_fallbacks_and_legend(tmp_path):
    path, columns, mapping, records = mixed_labels(tmp_path)
    assert validate_label_mapping(mapping)["valid"]
    assert mapping["level_counts"] == [2, 3, 3, 5]
    classes = {item["alliance_code"]: item for item in mapping["classes"]}
    assert classes[records[1].alliance_code]["alliance_zh"] == "其它水浇地"
    assert classes[records[3].alliance_code]["terminal_level"] == 3
    assert any(item["alliance_zh"] == "其它灌木园地" for item in classes.values())
    assert records[4].level_codes[-2:] == (-1, -1)
    assert records[5].level_codes[1:] == (-1, -1, -1)
    report = validate_labels(path, label_columns=columns, mapping=mapping)
    assert (report.valid_rows, report.invalid_rows, report.partial_rows) == (6, 0, 2)
    spec = importlib.util.spec_from_file_location(
        "mixed_predict", Path(__file__).parents[1] / "predict.py"
    )
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    legend = tmp_path / "legend.csv"
    predict._write_mapping(mapping, legend)
    with legend.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    forest = next(row for row in rows if row["L2"] == "forest")
    assert forest["L3"] == forest["L3中文"] == ""
    assert any(row["L3中文"] == "其它灌木园地" for row in rows)
    broken = copy.deepcopy(mapping)
    broken["classes"][-1]["terminal_level"] = 4
    with pytest.raises(ValueError, match="terminal_level"):
        validate_label_mapping(broken)
    coarsened = build_label_mapping(
        path, label_columns=columns, schema={"min_samples_per_fine_class": 10}
    )
    assert all(item["terminal_level"] == 1 for item in coarsened["classes"])


def test_checkpoint_binds_mixed_leaf_semantics(tmp_path):
    _, _, mapping, _ = mixed_labels(tmp_path)
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "assets": [
                    {"role": "dynamic", "name": "NDVI"},
                    {"role": "static", "name": "DEM"},
                ]
            }
        )
    )
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps(mapping))
    derived = derive_model_contract(index, mapping_file)
    validate_checkpoint_mapping(derived, mapping)
    assert derived["leaf_depths"] == [4, 4, 4, 4, 3]
    changed = copy.deepcopy(mapping)
    changed["optional_leaf_level"] = False
    with pytest.raises(ValueError, match="optional_leaf_level"):
        validate_checkpoint_mapping(derived, changed)


def batch_and_head(mapping, records):
    head = HierarchicalHeads.from_derived(4, mapping)
    observed = torch.tensor([r.level_codes for r in records]).T[None, :, None, :]
    leaf = observed[:, -1]
    batch = {
        "ground_truth_levels": observed,
        "ground_truth": leaf,
        "ground_truth_mask": leaf.gt(0),
        "weak_label": torch.ones_like(leaf),
        "weak_label_mask": torch.ones_like(leaf, dtype=torch.bool),
        "valid_mask": torch.ones_like(leaf, dtype=torch.bool),
        "input_window": [Window(0, 0, len(records), 1)],
    }
    return batch, head


def test_priorities_masks_and_fourth_level_gradients(tmp_path):
    _, _, mapping, records = mixed_labels(tmp_path)
    batch, head = batch_and_head(mapping, records)
    depths = [item["terminal_level"] for item in mapping["classes"]]
    weights = [1.0, 0.5, 0.25, 0.1]
    inputs = torch.randn(1, 4, 1, len(records))
    batch["valid_mask"][..., -1] = False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = head(inputs)
        losses = combined_supervision_loss(
            output,
            batch,
            level_parents=mapping["level_parents"],
            level_weights=weights,
            leaf_depths=depths,
            weak_label_weight=0.2,
            hierarchy_weight=0,
        )
    expected = sum(
        w
        * (
            losses[f"ground_truth_level_{i}_loss"]
            + 0.2 * losses[f"weak_label_level_{i}_loss"]
        )
        for i, w in enumerate(weights)
    )
    torch.testing.assert_close(losses["loss"], expected)
    losses["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )
    forest, _ = batch_and_head(mapping, records[3:4])
    forest["weak_label_mask"].zero_()
    losses = combined_supervision_loss(
        head(inputs[..., :1]),
        forest,
        level_parents=mapping["level_parents"],
        level_weights=weights,
        leaf_depths=depths,
    )
    assert losses["ground_truth_level_3_loss"].item() == 0
    assert losses["ground_truth_level_2_loss"].item() > 0
    # The same exclusion applies to SAM leaf labels for terminal third-level classes.
    forest["ground_truth_levels"].fill_(-1)
    forest["ground_truth_mask"].zero_()
    forest["weak_label"].fill_(records[3].alliance_code)
    forest["weak_label_mask"].fill_(True)
    losses = combined_supervision_loss(
        head(inputs[..., :1]),
        forest,
        level_parents=mapping["level_parents"],
        leaf_depths=depths,
    )
    assert losses["weak_label_level_3_loss"].item() == 0


def test_validation_prioritizes_root_and_excludes_identity_fourth(tmp_path):
    _, _, mapping, records = mixed_labels(tmp_path)
    batch, head = batch_and_head(mapping, records)
    depths = [item["terminal_level"] for item in mapping["classes"]]

    class Model(torch.nn.Module):
        def forward(self, _):
            return head(torch.zeros(1, 4, 1, len(records)))

    report = evaluate_points(
        Model(),
        [batch, batch],
        torch.device("cpu"),
        "none",
        selection_level=0,
        leaf_depths=depths,
    )
    assert report["selection_level"] == 0
    assert report["labeled_pixels"] == 6
    assert report["leaf"]["labeled_pixels"] == 4
    assert [report["levels"][str(i)]["labeled_pixels"] for i in range(4)] == [
        6,
        5,
        4,
        3,
    ]
    assert report["macro_f1"] == report["levels"]["0"]["macro_f1"]
    raw = {}
    collect_point_predictions(
        Model(),
        [batch],
        torch.device("cpu"),
        "none",
        raw_predictions=raw,
        leaf_depths=depths,
    )
    leaf, levels = merge_point_prediction_shards([raw, raw])
    merged = summarize_point_predictions(leaf, levels, selection_level=0)
    assert merged == report


@pytest.mark.parametrize(
    "weights", [[1, 0.5], [0, 0, 0, 0], [1, -1, 1, 1], [1, float("nan"), 1, 1]]
)
def test_bad_priority_weights_rejected(tmp_path, weights):
    _, _, mapping, records = mixed_labels(tmp_path)
    batch, head = batch_and_head(mapping, records)
    with pytest.raises(ValueError, match="level_weights"):
        combined_supervision_loss(
            head(torch.zeros(1, 4, 1, len(records))),
            batch,
            level_parents=mapping["level_parents"],
            level_weights=weights,
        )
