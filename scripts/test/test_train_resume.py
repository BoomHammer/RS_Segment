"""Exercise epoch checkpoint recovery without raster data or a GPU."""

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from data.spatial_split import SpatialSplitManifest

SPEC = importlib.util.spec_from_file_location(
    "resume_train", Path(__file__).parents[1] / "train.py"
)
train = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train
SPEC.loader.exec_module(train)


def _split_manifest() -> SpatialSplitManifest:
    return SpatialSplitManifest(
        schema_version=1,
        seed=42,
        block_size=(1024, 1024),
        ratios=(0.6, 0.2, 0.2),
        splits={"train": [0], "validation": [1], "test": [2]},
        blocks={"0:0": "train", "1:0": "validation", "2:0": "test"},
        class_counts={"train": {"1": 1}},
        class_weights={"1": 1.0},
        sampling_weights={"0": 1.0},
    )


def test_resume_split_check_normalizes_json_lists(tmp_path):
    manifest = _split_manifest()
    split_path = tmp_path / "spatial_split.json"
    manifest.write(split_path)

    train._assert_resume_split_matches(split_path, manifest)


def test_resume_split_check_rejects_real_change(tmp_path):
    manifest = _split_manifest()
    split_path = tmp_path / "spatial_split.json"
    manifest.write(split_path)
    changed = SpatialSplitManifest.from_dict(
        {**manifest.to_dict(), "splits": {**manifest.splits, "train": [9]}}
    )

    with pytest.raises(ValueError, match="空间划分已改变"):
        train._assert_resume_split_matches(split_path, changed)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 2, 1, 1))

    def forward(self, batch):
        return {"fine_logits": self.logits}


class TinyLoader(list):
    def __init__(self):
        super().__init__(
            [
                {
                    "ground_truth": torch.ones(1, 1, 1, dtype=torch.long),
                    "ground_truth_mask": torch.ones(1, 1, 1, dtype=torch.bool),
                    "valid_mask": torch.ones(1, 1, 1, dtype=torch.bool),
                    "input_window": [SimpleNamespace(row_off=0, col_off=0)],
                }
            ]
        )
        self.generator = torch.Generator().manual_seed(42)


@pytest.mark.parametrize("halo", [0, 1])
def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch, halo):
    run = tmp_path / "data"
    run.mkdir()
    (run / "label_mapping.json").write_text("{}")
    (run / "spatial_split.json").write_text("{}")
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "stage2": {
                        "sampling": {"strategy": "random"},
                        "window": {"halo": [halo, halo]},
                    }
                }
            }
        )
    )
    config_path = tmp_path / "model.yaml"
    config_path.write_text("{}")
    train_path = tmp_path / "train.yaml"
    train_path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "epochs": 3,
                    "gradient_accumulation_steps": 2,
                    "early_stopping": {"enabled": False},
                },
                "supervision_policy": {"fixed_spatial_supervision": True},
            }
        )
    )
    monkeypatch.setattr(
        train,
        "WindowedSampleDataset",
        lambda **kw: SimpleNamespace(
            configure_supervision_split=lambda *args, **kwargs: None
        ),
    )
    monkeypatch.setattr(train, "point_windows", lambda *args: {0: {"1": 1}})
    monkeypatch.setattr(train, "assert_supervision_isolated", lambda *args: None)
    monkeypatch.setattr(
        train,
        "supervision_summary",
        lambda *args: {
            "unique_class_counts": {"train": {"1": 1}, "validation": {"1": 1}}
        },
    )
    monkeypatch.setattr(
        train,
        "load_spatial_split",
        lambda path: SimpleNamespace(
            splits={"train": [0], "validation": [1]},
            class_weights={},
            write=lambda path: None,
        ),
    )
    monkeypatch.setattr(train, "build_dataloader", lambda *a, **kw: TinyLoader())
    monkeypatch.setattr(
        train,
        "load_model_contract",
        lambda *a: {"derived": {"num_classes": 2, "fine_to_coarse": [0, 0]}},
    )
    monkeypatch.setattr(
        train,
        "SegFormerUtae",
        SimpleNamespace(from_contract=lambda contract: TinyModel()),
    )
    modes = []

    def loss(prediction, batch, **kwargs):
        return {"loss": -prediction["fine_logits"].log_softmax(1)[:, 0].mean()}

    monkeypatch.setattr(train, "combined_supervision_loss", loss)
    evaluate = train._evaluate

    def interrupted(model, loader, device):
        modes.append(model.training)
        if len(modes) == 2:
            raise KeyboardInterrupt
        return evaluate(model, loader, device)

    base = [
        str(run),
        "--data-config",
        str(data_path),
        "--config",
        str(config_path),
        "--train-config",
        str(train_path),
        "--device",
        "cpu",
        "--window-size",
        "2",
        "2",
        "--stride",
        "1",
        "1",
    ]
    output = tmp_path / "interrupted"
    monkeypatch.setattr(train, "_evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train.main([*base, "--output-dir", str(output)])
    assert all(modes)
    saved = torch.load(output / "last.pt", weights_only=True)
    assert saved["next_epoch"] == 1
    monkeypatch.setattr(train, "_evaluate", evaluate)
    train.main([str(run), "--resume", str(output / "last.pt"), "--device", "cpu"])
    full = tmp_path / "full"
    train.main([*base, "--output-dir", str(full)])
    recovered = torch.load(output / "last.pt", weights_only=True)
    expected = torch.load(full / "last.pt", weights_only=True)
    for key in ("model", "ema", "best_state"):
        for name in expected[key]:
            torch.testing.assert_close(recovered[key][name], expected[key][name])
    assert recovered["scheduler"] == expected["scheduler"]
    assert recovered["optimizer_steps"] == expected["optimizer_steps"] == 3
    assert recovered["train_log"]["epochs"] == expected["train_log"]["epochs"]
    log = json.loads((output / "train_log.json").read_text())
    assert log["status"] == "completed"
    assert len(log["epochs"]) == 3


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    train._atomic_save({"epoch": 1}, path)

    def fail(payload, stream):
        stream.write(b"incomplete")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        train._atomic_save({"epoch": 2}, path)
    assert torch.load(path, weights_only=True) == {"epoch": 1}


def test_evaluate_reports_macro_f1():
    logits = torch.tensor(
        [
            [
                [[5.0, 5.0, 0.0, 0.0]],
                [[0.0, 0.0, 5.0, 0.0]],
                [[0.0, 0.0, 0.0, 5.0]],
            ]
        ]
    )
    loader = [
        {
            "logits": logits,
            "ground_truth": torch.tensor([[[1, 2, 2, 3]]]),
            "ground_truth_mask": torch.ones(1, 1, 4, dtype=torch.bool),
            "valid_mask": torch.ones(1, 1, 4, dtype=torch.bool),
            "input_window": [SimpleNamespace(row_off=0, col_off=0)],
        }
    ]

    class FixedModel(torch.nn.Module):
        def forward(self, batch):
            return {"fine_logits": batch["logits"]}

    result = train._evaluate(FixedModel(), loader, torch.device("cpu"))

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(7 / 9)


def test_metrics_csv_appends_one_row_per_epoch(tmp_path):
    path = tmp_path / "epoch_metrics.csv"
    row = {field: 0.5 for field in train.METRICS_CSV_FIELDS}
    row["epoch"] = 1

    train._append_metrics_csv(path, row)
    train._append_metrics_csv(path, row)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert tuple(rows[0]) == train.METRICS_CSV_FIELDS
