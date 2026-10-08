"""Prepared-data experiments must never launch preparation or resplit stages."""

import json

import pytest

import pipeline
from data.spatial_split import SpatialSplitManifest
from data.training_policy import merge_test_into_train, point_owner
from experiment_options import MODEL_CHOICES
from models.config import load_model_contract


@pytest.mark.parametrize(
    "model", [None, "segformer-utae", "utae", "segformer", "maestro", "anysat"]
)
@pytest.mark.parametrize("merge", [False, True])
def test_retrain_routes_only_training_test_prediction(
    tmp_path, monkeypatch, model, merge
):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "prepared"
    run.mkdir()
    calls = []

    def stage(root, module, arguments):
        calls.append((module, arguments))
        if module == "train":
            output = root / "experiments" / "new"
            output.mkdir(parents=True)
            (output / "model.pt").touch()
            (output / "train_log.json").write_text(
                json.dumps(
                    {
                        "checkpoint": "model.pt",
                        "supervision_policy": {"train_on_test": merge},
                    }
                )
            )

    monkeypatch.setattr(pipeline, "_run", stage)
    args = ["--retrain", str(run), "--no-pseudo-labels"]
    if model:
        args += ["--model", model]
    if merge:
        args += ["--train-on-test"]
    assert pipeline.main(args) == 0
    assert [name for name, _ in calls] == (
        ["train", "predict"] if merge else ["train", "test", "predict"]
    )
    assert "--no-pseudo-labels" in calls[0][1]
    if model:
        assert calls[0][1][calls[0][1].index("--model") + 1] == model
    assert list(run.iterdir()) == []


def test_merge_changes_point_ownership_without_mutating_source():
    source = SpatialSplitManifest(
        schema_version=1,
        seed=42,
        block_size=(10, 10),
        ratios=(0.6, 0.2, 0.2),
        splits={"train": [0], "validation": [1], "test": [2]},
        blocks={"0:0": "train", "0:1": "validation", "0:2": "test"},
        class_counts={"train": {"1": 2}, "test": {"2": 1}},
        class_weights={"1": 1.0},
        sampling_weights={},
    )
    merged = merge_test_into_train(source)
    assert merged.splits == {"train": [0, 2], "validation": [1], "test": []}
    assert point_owner(0, 20, merged) == "train"
    assert point_owner(0, 10, merged) == "validation"
    assert point_owner(0, 20, source) == "test"
    assert source.splits["train"] == [0]
    assert merged.class_counts["train"] == {"1": 2, "2": 1}
    assert "2" in merged.class_weights


@pytest.mark.parametrize("model", MODEL_CHOICES)
def test_model_override_is_applied_before_deriving_contract(tmp_path, model):
    (tmp_path / "sample_index.json").write_text(
        json.dumps(
            {
                "assets": [
                    {"role": "dynamic", "name": "NDVI"},
                    {"role": "static", "name": "DEM"},
                ],
            }
        )
    )
    (tmp_path / "label_mapping.json").write_text('{"minor_count": 2}')
    config = tmp_path / "model.yaml"
    config.write_text("model:\n  anysat:\n    resolution_m: 250.0\n")
    contract = load_model_contract(config, tmp_path, model_name=model)
    assert contract["architecture"] == MODEL_CHOICES[model]
    if model == "segformer-utae":
        assert contract["pretrained"]["freeze_stages"] == 0
    if model == "anysat":
        assert contract["anysat"]["resolution_m"] == 250.0
        assert "target_grid" in contract["derived"]
