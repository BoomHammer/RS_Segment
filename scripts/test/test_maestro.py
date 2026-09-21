"""MAESTRO temporal/missing-data, hierarchy and checkpoint integration tests."""

import copy
import json

import pytest
import torch

from data.maestro import prepare_modality, temporal_features
from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.config import load_model_contract
from models.maestro import MaestroS


def contract():
    return {
        "architecture": "maestro_s",
        "derived": {
            "dynamic_features": ["NDVI", "PR", "SR_B1", "SR_B2"],
            "static_features": ["DEM", "rain"],
            "num_classes": 3,
            "fine_to_coarse": [0, 1, 1],
        },
        "maestro": {
            "embed_dim": 24,
            "depth": 3,
            "heads": 3,
            "inter_depth": 1,
            "patch_size": 4,
            "temporal_bins": 2,
            "max_tokens": 100,
        },
        "regularization": {"dropout": 0, "drop_path": 0},
    }


def batch():
    torch.manual_seed(7)
    return {
        "dynamic": torch.randn(2, 3, 4, 7, 9),
        "static": torch.randn(2, 2, 7, 9),
        "dynamic_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "dynamic_time_mask": torch.ones(2, 3, dtype=torch.bool),
        "time_encoding": torch.tensor(
            [[[0.0, 0.0, 1.0], [0.5, 0.0, -1.0], [0.9, -0.6, 0.8]]] * 2
        ),
        "dynamic_valid_mask": torch.ones(2, 7, 9, dtype=torch.bool),
        "static_valid_mask": torch.ones(2, 7, 9, dtype=torch.bool),
        "valid_mask": torch.ones(2, 7, 9, dtype=torch.bool),
        "ground_truth": torch.randint(1, 4, (2, 7, 9)),
        "ground_truth_mask": torch.ones(2, 7, 9, dtype=torch.bool),
        "weak_label": torch.full((2, 7, 9), -1, dtype=torch.long),
        "weak_label_mask": torch.zeros(2, 7, 9, dtype=torch.bool),
    }


def test_factory_backward_hierarchy_and_checkpoint(tmp_path):
    model = SegFormerUtae.from_contract(contract())
    assert isinstance(model, MaestroS)
    inputs = batch()
    inputs["weak_label"][:, 0, 0] = 2
    inputs["weak_label_mask"][:, 0, 0] = True
    output = model(inputs)
    assert output["fine_logits"].shape == (2, 3, 7, 9)
    assert output["valid_mask"].shape == (2, 1, 7, 9)
    assert torch.allclose(
        output["fine_probability"].sum(1), torch.ones(2, 7, 9), atol=1e-6
    )
    assert torch.allclose(
        output["fine_probability"][:, 1:].sum(1),
        output["coarse_probability"][:, 1],
        atol=1e-6,
    )
    loss = combined_supervision_loss(
        output, inputs, fine_to_coarse=contract()["derived"]["fine_to_coarse"]
    )["loss"]
    assert loss > 0
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert model.tokenizers["SR"].weight.grad.abs().sum() > 0
    model.eval()
    path = tmp_path / "model.pt"
    torch.save({"contract": contract(), "model": model.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    restored = SegFormerUtae.from_contract(payload["contract"]).eval()
    restored.load_state_dict(payload["model"])
    with torch.no_grad():
        assert torch.equal(
            model(inputs)["fine_logits"], restored(inputs)["fine_logits"]
        )


def test_missing_values_and_padding_cannot_change_predictions():
    model = MaestroS(contract()).eval()
    inputs = batch()
    inputs["dynamic_time_mask"][:, 2] = False
    inputs["dynamic_mask"][:, 1, 1] = False
    inputs["static_valid_mask"][:, 0, 0] = False
    altered = copy.deepcopy(inputs)
    altered["dynamic"][:, 2] = 1e7
    altered["dynamic"][:, 1, 1] = float("nan")
    altered["static"][:, :, 0, 0] = float("inf")
    with torch.no_grad():
        assert torch.equal(model(inputs)["fine_logits"], model(altered)["fine_logits"])
    inputs["dynamic"][:] = float("nan")
    inputs["static"][:] = float("nan")
    inputs["valid_mask"][:] = False
    with torch.no_grad():
        output = model(inputs)
    assert torch.isfinite(output["fine_logits"]).all()
    assert not output["valid_mask"].any()


def test_discretization_uses_each_products_actual_dates():
    inputs = batch()
    inputs["dynamic_mask"][:, :, 1] = False
    inputs["dynamic_mask"][:, 2, 1] = True
    values, valid, dates = prepare_modality(inputs, "dynamic", [1], 4, training=False)
    assert torch.equal(values[:, 0, 0], inputs["dynamic"][:, 2, 1])
    assert valid[:, 0].all() and not valid[:, 1:].any()
    assert torch.equal(dates[:, 0, 0], inputs["time_encoding"][:, 2, 1])


def test_feature_order_is_resolved_by_name():
    inputs = batch()
    altered = copy.deepcopy(inputs)
    order = [2, 0, 3, 1]
    altered["dynamic"] = inputs["dynamic"][:, :, order]
    altered["dynamic_mask"] = inputs["dynamic_mask"][:, :, order]
    altered["dynamic_features"] = [
        [contract()["derived"]["dynamic_features"][i] for i in order]
    ] * 2
    altered["static"] = inputs["static"][:, [1, 0]]
    altered["static_features"] = [["rain", "DEM"]] * 2
    model = MaestroS(contract()).eval()
    with torch.no_grad():
        assert torch.equal(model(inputs)["fine_logits"], model(altered)["fine_logits"])


def test_token_budget_and_configuration_fail_early():
    config = contract()
    config["maestro"]["max_tokens"] = 1
    with pytest.raises(ValueError, match="token"):
        MaestroS(config)(batch())
    config["maestro"]["modalities"] = []
    with pytest.raises(ValueError, match="modalities"):
        MaestroS(config)


def test_contract_feature_subset_and_order_match_dataset(tmp_path):
    (tmp_path / "sample_index.json").write_text(
        json.dumps(
            {
                "assets": [
                    {"role": "dynamic", "name": "SR", "band": 2},
                    {"role": "dynamic", "name": "SR", "band": 1},
                    {"role": "dynamic", "name": "PR"},
                    {"role": "static", "name": "terrain"},
                    {"role": "static", "name": "climate"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "label_mapping.json").write_text(
        '{"minor_count": 1, "classes": []}', encoding="utf-8"
    )
    configuration = tmp_path / "model.yaml"
    configuration.write_text("model:\n  architecture: maestro_s\n", encoding="utf-8")
    settings = {
        "features": {
            "dynamic": ["SR_B1", "SR_B2"],
            "dynamic_order": ["SR_B2"],
            "static": ["terrain"],
        }
    }
    derived = load_model_contract(configuration, tmp_path, settings)["derived"]
    assert derived["dynamic_features"] == ["SR_B2", "SR_B1"]
    assert derived["dynamic_features_count"] == 2
    assert derived["static_features"] == ["terrain"]
    assert derived["static_features_count"] == 1
    settings["features"]["dynamic_order"] = ["PR"]
    with pytest.raises(ValueError, match="dynamic_order"):
        load_model_contract(configuration, tmp_path, settings)
    # Unfiltered modern contracts match source ordering; old contracts stay
    # byte-for-byte compatible with their previous derived ordering.
    assert load_model_contract(configuration, tmp_path)["derived"][
        "static_features"
    ] == ["terrain", "climate"]
    configuration.write_text(
        "model:\n  architecture: segformer_utae\n", encoding="utf-8"
    )
    assert load_model_contract(configuration, tmp_path)["derived"][
        "static_features"
    ] == ["climate", "terrain"]


def test_temporal_features_retain_year_and_month_semantics():
    inputs = batch()
    inputs["dynamic_times"] = [["2023-01-01", "2024-01-01", "2024-02"]] * 2
    first = temporal_features(inputs, 0, 0)
    next_year = temporal_features(inputs, 0, 1)
    month = temporal_features(inputs, 0, 2)
    assert first[4] == 0
    assert next_year[4].item() == pytest.approx(365 / 365.25)
    assert month[4].item() == pytest.approx(396 / 365.25)
    assert torch.equal(month[4:], month[4].expand(4))


def test_entirely_missing_inputs_have_finite_backward():
    model = MaestroS(contract()).train()
    inputs = batch()
    inputs["dynamic"][:] = float("nan")
    inputs["static"][:] = float("nan")
    inputs["valid_mask"][:] = False
    output = model(inputs)
    loss = combined_supervision_loss(
        output, inputs, fine_to_coarse=contract()["derived"]["fine_to_coarse"]
    )["loss"]
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
