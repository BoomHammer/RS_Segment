"""Tests for the model-facing stage-3 data contracts."""

import torch

from losses.supervision import combined_supervision_loss
from models.config import derive_model_contract, load_model_contract
from models.input_adapter import SegFormerUtaeInputAdapter, sanitize_batch


def _batch() -> dict[str, torch.Tensor]:
    return {
        "dynamic": torch.tensor(
            [[[[[1.0, float("nan")], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]]]
        ),
        "dynamic_mask": torch.tensor([[[True], [True]]]),
        "dynamic_time_mask": torch.tensor([[True, True]]),
        "time_encoding": torch.zeros((1, 2, 3)),
        "static": torch.tensor([[[[2.0, float("inf")], [2.0, 2.0]]]]),
        "dynamic_valid_mask": torch.tensor([[[True, True], [True, True]]]),
        "static_valid_mask": torch.tensor([[[True, True], [True, True]]]),
        "valid_mask": torch.tensor([[[True, True], [True, True]]]),
        "ground_truth": torch.tensor([[[1, -1], [-1, -1]]]),
        "ground_truth_mask": torch.tensor([[[True, False], [False, False]]]),
        "weak_label": torch.tensor([[[1, 2], [-1, -1]]]),
        "weak_label_mask": torch.tensor([[[True, True], [False, False]]]),
    }


def test_sanitize_batch_replaces_nonfinite_values() -> None:
    clean = sanitize_batch(_batch())
    assert torch.isfinite(clean["dynamic"]).all()
    assert torch.isfinite(clean["static"]).all()
    assert clean["dynamic"][0, 0, 0, 0, 1] == 0
    assert clean["static"][0, 0, 0, 1] == 0


def test_segformer_utae_adapter_returns_spatial_features() -> None:
    adapter = SegFormerUtaeInputAdapter(
        dynamic_features=1,
        static_features=1,
        dynamic_channels=4,
        static_channels=2,
        fused_channels=6,
    )
    output = adapter(_batch())
    assert output["features"].shape == (1, 6, 2, 2)
    assert output["valid_mask"].shape == (1, 1, 2, 2)
    assert torch.isfinite(output["features"]).all()


def test_combined_supervision_uses_both_label_sources() -> None:
    logits = torch.zeros((1, 3, 2, 2), requires_grad=True)
    losses = combined_supervision_loss(logits, _batch(), weak_label_weight=1.0)
    assert losses["ground_truth_loss"] > 0
    assert losses["weak_label_loss"] > 0
    losses["loss"].backward()
    assert logits.grad is not None


def test_model_contract_derives_dimensions_from_artifacts(tmp_path) -> None:
    sample_index = tmp_path / "sample_index.json"
    mapping = tmp_path / "label_mapping.json"
    sample_index.write_text(
        '{"assets": [{"role": "dynamic", "name": "NDVI"}, '
        '{"role": "dynamic", "name": "SR_B1"}, '
        '{"role": "static", "name": "DEM"}]}',
        encoding="utf-8",
    )
    mapping.write_text('{"minor_count": 72, "classes": []}', encoding="utf-8")
    contract = derive_model_contract(sample_index, mapping)
    assert contract["dynamic_features_count"] == 2
    assert contract["static_features_count"] == 1
    assert contract["num_classes"] == 72


def test_model_config_discovers_run_artifacts() -> None:
    contract = load_model_contract(
        "configs/model.yaml", "data/processed/20260902_105656"
    )
    assert contract["derived"]["dynamic_features_count"] == 15
    assert contract["derived"]["static_features_count"] == 7
    assert contract["derived"]["num_classes"] == 32
