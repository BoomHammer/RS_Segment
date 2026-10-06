"""Exercise CSV-to-model contracts, conditional training and exported legends."""

import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from config import DataConfig
from data.labels import (
    build_label_mapping,
    iter_encoded_labels,
    validate_label_mapping,
    validate_labels,
    write_label_artifacts,
)
from losses.supervision import combined_supervision_loss
from models.architecture import HierarchicalHeads, SegFormerUtae
from models.config import derive_model_contract, validate_checkpoint_mapping
from test_anysat import batch as anysat_batch
from test_anysat import contract as anysat_contract
from test_maestro import batch as maestro_batch
from test_maestro import contract as maestro_contract


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _csv(tmp_path, depth):
    levels = [{"name": f"L{i}", "column": f"L{i}"} for i in range(depth)]
    columns = {"x": "lon", "y": "lat", "levels": levels}
    path = tmp_path / "labels.csv"
    rows = [("A", "a", "same"), ("A", "b", "same"), ("B", "a", "other")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lon", "lat", *[level["column"] for level in levels]])
        for index, row in enumerate(rows):
            writer.writerow([10 + index, 20, *row[:depth]])
    return path, columns


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_csv_contract_legend_and_metrics(tmp_path, depth):
    path, columns = _csv(tmp_path, depth)
    mapping_path, _ = write_label_artifacts(
        path, output_dir=tmp_path, label_columns=columns
    )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert validate_label_mapping(mapping)["valid"]
    assert mapping["level_counts"] == [2, 3, 3][:depth]
    records = [
        r
        for batch in iter_encoded_labels(path, label_columns=columns, mapping=mapping)
        for r in batch
    ]
    assert all(len(record.level_codes) == depth for record in records)
    assert validate_labels(path, label_columns=columns, mapping=mapping).valid_rows == 3
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
    derived = derive_model_contract(index, mapping_path)
    assert derived["level_counts"] == mapping["level_counts"]
    assert derived["level_parents"] == mapping["level_parents"]
    validate_checkpoint_mapping(derived, mapping)
    mismatched = copy.deepcopy(derived)
    mismatched["class_paths"][0][0] = "wrong"
    with pytest.raises(ValueError, match="class_paths"):
        validate_checkpoint_mapping(mismatched, mapping)
    spec = importlib.util.spec_from_file_location(
        "hierarchy_predict", Path(__file__).parents[1] / "predict.py"
    )
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    legend = tmp_path / "legend.csv"
    predict._write_mapping(mapping, legend)
    with legend.open(encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    assert len(rows[0]) == 1 + 2 * depth
    assert len(rows) == derived["num_classes"] + 1
    spec = importlib.util.spec_from_file_location(
        "hierarchy_metrics", Path(__file__).parents[1] / "test.py"
    )
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    count = derived["num_classes"]
    report = metrics._metrics(
        torch.arange(count),
        torch.eye(count),
        fine_to_coarse=derived["fine_to_coarse"],
        loss_sum=0,
        level_parents=derived["level_parents"],
        level_names=derived["level_names"],
    )
    assert len(report["levels"]) == depth
    assert ("coarse" in report) == (depth > 1)


def test_mapping_rejects_wrong_middle_parent(tmp_path):
    path, columns = _csv(tmp_path, 3)
    mapping = build_label_mapping(path, label_columns=columns)
    mapping["level_parents"][0][0] = 1
    with pytest.raises(ValueError, match="level_parents"):
        validate_label_mapping(mapping)


def test_missing_levels_require_explicit_skip_and_are_reported(tmp_path):
    path, columns = _csv(tmp_path, 3)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("13,20,A,a,\n")
    with pytest.raises(ValueError, match="5"):
        build_label_mapping(path, label_columns=columns)
    columns["missing_policy"] = "skip"
    mapping = build_label_mapping(path, label_columns=columns)
    report = validate_labels(path, label_columns=columns, mapping=mapping)
    assert (report.total_rows, report.valid_rows, report.invalid_rows) == (4, 3, 1)
    assert report.missing_hierarchy_rows == [5]
    assert (
        sum(
            len(batch)
            for batch in iter_encoded_labels(
                path, label_columns=columns, mapping=mapping
            )
        )
        == 3
    )


def test_non_utf8_schema_propagates_to_artifact_validation(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("lon,lat,类型\n10,20,森林\n", encoding="gb18030")
    mapping_path, report_path = write_label_artifacts(
        path,
        output_dir=tmp_path,
        label_columns={"x": "lon", "y": "lat"},
        schema={"encoding": "gb18030", "levels": [{"name": "类型", "column": "类型"}]},
    )
    assert json.loads(mapping_path.read_text(encoding="utf-8"))["level_counts"] == [1]
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid_rows"] == 1


@pytest.mark.parametrize("levels", [[], [{}], [{"name": "x", "column": "x"}] * 4])
def test_schema_rejects_invalid_levels(levels):
    with pytest.raises(ValueError):
        _ = DataConfig(label_schema={"levels": levels}).label_columns


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize(
    "architecture",
    [
        "anysat",
        "maestro_s",
        "lightweight_dual_branch",
        "segformer_utae_pretrained",
        "segformer_utae_dynamic_ablation",
        "segformer_utae_static_ablation",
    ],
)
def test_all_architectures_train_restore_and_normalize(tmp_path, depth, architecture):
    contract = anysat_contract() if architecture == "anysat" else maestro_contract()
    batch = anysat_batch() if architecture == "anysat" else maestro_batch()
    contract["architecture"] = architecture
    contract["temporal"] = {}
    derived = contract["derived"]
    derived.update(
        {
            "level_counts": [3] if depth == 1 else [2, 3, 3],
            "level_parents": [] if depth == 1 else [[0, 0, 1], [0, 1, 2]],
            "fine_to_coarse": [0, 1, 2] if depth == 1 else [0, 0, 1],
            "num_coarse_classes": 3 if depth == 1 else 2,
            "dynamic_features_count": 4,
            "static_features_count": 2,
        }
    )
    if architecture not in {"anysat", "maestro_s"}:
        # MiT's spatial reduction requires at least a 32x32 input.
        for key, value in batch.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 3
                and value.shape[-2:] == (7, 9)
            ):
                batch[key] = (
                    value[..., :1, :1].expand(*value.shape[:-2], 32, 32).clone()
                )
    model = SegFormerUtae.from_contract(contract)
    output = model(batch)
    probability = output["fine_probability"]
    torch.testing.assert_close(probability.sum(1), torch.ones_like(probability[:, 0]))
    for level, edge in enumerate(derived["level_parents"]):
        parent = output[f"level_{level}_logits"].exp()
        child = output[f"level_{level + 1}_logits"].exp()
        marginal = torch.zeros_like(parent).index_add(1, torch.tensor(edge), child)
        torch.testing.assert_close(parent, marginal)
    loss = combined_supervision_loss(
        output,
        batch,
        fine_to_coarse=derived["fine_to_coarse"],
        level_parents=derived["level_parents"],
    )
    loss["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.heads.parameters()
    )
    if depth == 1:
        expected = torch.nn.functional.nll_loss(
            output["fine_logits"], batch["ground_truth"] - 1
        )
        torch.testing.assert_close(loss["loss"], expected)
    path = tmp_path / "model.pt"
    torch.save({"contract": contract, "model": model.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    restored = SegFormerUtae.from_contract(payload["contract"]).eval()
    restored.load_state_dict(payload["model"])
    with torch.no_grad():
        torch.testing.assert_close(
            model.eval()(batch)["fine_logits"], restored(batch)["fine_logits"]
        )


def test_three_levels_bf16_masks_and_weak_weight():
    parents = [[0, 0, 1], [0, 0, 1, 2]]
    head = HierarchicalHeads.from_derived(
        4, {"level_counts": [2, 3, 4], "level_parents": parents}
    )
    inputs = torch.randn(1, 4, 3, 3)
    batch = {
        "ground_truth": torch.ones(1, 3, 3, dtype=torch.long),
        "weak_label": torch.full((1, 3, 3), 4, dtype=torch.long),
        "valid_mask": torch.ones(1, 3, 3, dtype=torch.bool),
        "ground_truth_mask": torch.ones(1, 3, 3, dtype=torch.bool),
        "weak_label_mask": torch.ones(1, 3, 3, dtype=torch.bool),
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = head(inputs)
        combined = combined_supervision_loss(
            output, batch, level_parents=parents, weak_label_weight=0.25
        )
    expected = sum(
        combined[f"ground_truth_level_{i}_loss"]
        + 0.25 * combined[f"weak_label_level_{i}_loss"]
        for i in range(3)
    )
    torch.testing.assert_close(combined["loss"], expected, atol=1e-6, rtol=1e-5)
    combined["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
    empty = copy.deepcopy(batch)
    empty["valid_mask"].zero_()
    loss = combined_supervision_loss(head(inputs), empty, level_parents=parents)["loss"]
    assert loss.item() == 0
    loss.backward()
