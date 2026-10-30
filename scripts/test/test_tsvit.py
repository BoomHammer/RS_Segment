"""TSViT adaptation, masking, hierarchy and budget tests."""

import copy

import pytest
import torch

from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.tsvit import TSViTAdapted


def contract():
    return {
        "architecture": "tsvit_adapted",
        "derived": {
            "dynamic_features": ["NDVI", "SR_B1", "SR_B2"],
            "static_features": ["DEM", "rain"],
            "dynamic_features_count": 3,
            "static_features_count": 2,
            "num_classes": 3,
            "fine_to_coarse": [0, 1, 1],
        },
        "tsvit": {
            "patch_size": 4,
            "dim": 16,
            "heads": 2,
            "temporal_depth": 1,
            "spatial_depth": 1,
            "mlp_ratio": 2,
            "max_temporal_tokens": 1000,
            "max_spatial_tokens": 1000,
            "gradient_checkpointing": False,
        },
        "regularization": {"dropout": 0.0, "drop_path": 0.0},
    }


def batch():
    torch.manual_seed(17)
    return {
        "dynamic": torch.randn(2, 3, 3, 7, 9),
        "static": torch.randn(2, 2, 7, 9),
        "dynamic_mask": torch.ones(2, 3, 3, dtype=torch.bool),
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
        "dynamic_features": [["NDVI", "SR_B1", "SR_B2"]] * 2,
        "static_features": [["DEM", "rain"]] * 2,
    }


def test_factory_hierarchy_backward_odd_window_and_checkpoint(tmp_path):
    model = SegFormerUtae.from_contract(contract())
    assert isinstance(model, TSViTAdapted)
    inputs = batch()
    output = model(inputs)
    assert output["fine_logits"].shape == (2, 3, 7, 9)
    assert output["valid_mask"].shape == (2, 1, 7, 9)
    assert torch.allclose(output["fine_probability"].sum(1), torch.ones(2, 7, 9))
    assert torch.allclose(
        output["fine_probability"][:, 1:].sum(1),
        output["coarse_probability"][:, 1],
    )
    loss = combined_supervision_loss(
        output, inputs, fine_to_coarse=contract()["derived"]["fine_to_coarse"]
    )["loss"]
    loss.backward()
    assert model.dynamic_tokenizer.weight.grad is not None
    assert model.static_tokenizer.weight.grad is not None
    assert model.date_embedding.weight.grad is not None
    path = tmp_path / "tsvit.pt"
    torch.save({"contract": contract(), "model": model.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    restored = SegFormerUtae.from_contract(payload["contract"])
    restored.load_state_dict(payload["model"])


def test_missing_values_padding_and_feature_order_cannot_change_predictions():
    model = TSViTAdapted(contract()).eval()
    inputs = batch()
    inputs["dynamic_time_mask"][:, 2] = False
    inputs["dynamic_mask"][:, 1, 1] = False
    inputs["static_valid_mask"][:, 0, 0] = False
    inputs["static"][:, 0, 1, 1] = float("nan")
    altered = copy.deepcopy(inputs)
    altered["dynamic"][:, 2] = 1e7
    altered["dynamic"][:, 1, 1] = float("nan")
    altered["static"][:, :, 0, 0] = float("inf")
    with torch.no_grad():
        assert torch.equal(model(inputs)["fine_logits"], model(altered)["fine_logits"])

    order = [2, 0, 1]
    reordered = copy.deepcopy(inputs)
    reordered["dynamic"] = inputs["dynamic"][:, :, order]
    reordered["dynamic_mask"] = inputs["dynamic_mask"][:, :, order]
    reordered["dynamic_features"] = [
        [contract()["derived"]["dynamic_features"][index] for index in order]
    ] * 2
    reordered["static"] = inputs["static"][:, [1, 0]]
    reordered["static_features"] = [["rain", "DEM"]] * 2
    with torch.no_grad():
        assert torch.equal(
            model(inputs)["fine_logits"], model(reordered)["fine_logits"]
        )


def test_dates_affect_predictions_and_all_missing_is_finite():
    model = TSViTAdapted(contract()).eval()
    inputs = batch()
    shifted = copy.deepcopy(inputs)
    shifted["time_encoding"][..., 0] = (
        shifted["time_encoding"][..., 0] + 0.1
    ).clamp_max(1)
    with torch.no_grad():
        assert not torch.equal(
            model(inputs)["fine_logits"], model(shifted)["fine_logits"]
        )
    inputs["dynamic"][:] = float("nan")
    inputs["static"][:] = float("nan")
    inputs["valid_mask"][:] = False
    with torch.no_grad():
        output = model(inputs)
    assert torch.isfinite(output["fine_logits"]).all()
    assert not output["valid_mask"].any()


def test_token_budgets_fail_before_attention_allocation():
    config = contract()
    config["tsvit"]["max_temporal_tokens"] = 1
    with pytest.raises(ValueError, match="temporal token"):
        TSViTAdapted(config)(batch())
    config = contract()
    config["tsvit"]["max_spatial_tokens"] = 1
    with pytest.raises(ValueError, match="spatial token"):
        TSViTAdapted(config)(batch())
