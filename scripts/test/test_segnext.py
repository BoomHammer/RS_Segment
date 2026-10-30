"""SegNeXt-T architecture, temporal aggregation and integration tests."""

import copy

import torch

from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.segnext import MSCANTiny, SegNeXtTiny


def contract():
    return {
        "architecture": "segnext_tiny",
        "derived": {
            "dynamic_features": ["NDVI", "SR_B1", "SR_B2"],
            "static_features": ["DEM", "rain"],
            "dynamic_features_count": 3,
            "static_features_count": 2,
            "num_classes": 3,
            "fine_to_coarse": [0, 1, 1],
        },
        "segnext": {
            "dropout": 0.0,
            "drop_path": 0.0,
            "decoder_channels": 32,
            "decoder_dropout": 0.0,
            "nmf_rank": 4,
            "nmf_train_steps": 1,
            "nmf_eval_steps": 1,
        },
    }


def batch():
    torch.manual_seed(23)
    return {
        "dynamic": torch.randn(2, 3, 3, 35, 37),
        "static": torch.randn(2, 2, 35, 37),
        "dynamic_mask": torch.ones(2, 3, 3, dtype=torch.bool),
        "dynamic_time_mask": torch.ones(2, 3, dtype=torch.bool),
        "time_encoding": torch.rand(2, 3, 3),
        "dynamic_valid_mask": torch.ones(2, 35, 37, dtype=torch.bool),
        "static_valid_mask": torch.ones(2, 35, 37, dtype=torch.bool),
        "valid_mask": torch.ones(2, 35, 37, dtype=torch.bool),
        "ground_truth": torch.randint(1, 4, (2, 35, 37)),
        "ground_truth_mask": torch.ones(2, 35, 37, dtype=torch.bool),
        "weak_label": torch.full((2, 35, 37), -1, dtype=torch.long),
        "weak_label_mask": torch.zeros(2, 35, 37, dtype=torch.bool),
        "dynamic_features": [["NDVI", "SR_B1", "SR_B2"]] * 2,
        "static_features": [["DEM", "rain"]] * 2,
    }


def test_factory_tiny_configuration_hierarchy_backward_and_checkpoint(tmp_path):
    model = SegFormerUtae.from_contract(contract())
    assert isinstance(model, SegNeXtTiny)
    assert MSCANTiny.channels == (32, 64, 160, 256)
    assert MSCANTiny.depths == (3, 3, 5, 2)
    assert MSCANTiny.ratios == (8, 8, 4, 4)
    inputs = batch()
    output = model(inputs)
    assert output["fine_logits"].shape == (2, 3, 35, 37)
    assert output["valid_mask"].shape == (2, 1, 35, 37)
    assert torch.allclose(output["fine_probability"].sum(1), torch.ones(2, 35, 37))
    assert torch.allclose(
        output["fine_probability"][:, 1:].sum(1),
        output["coarse_probability"][:, 1],
    )
    loss = combined_supervision_loss(
        output, inputs, fine_to_coarse=contract()["derived"]["fine_to_coarse"]
    )["loss"]
    loss.backward()
    assert model.encoder.embeddings[0][0].weight.grad is not None
    assert model.decoder.hamburger.input.weight.grad is not None
    path = tmp_path / "segnext.pt"
    torch.save({"contract": contract(), "model": model.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    restored = SegFormerUtae.from_contract(payload["contract"]).eval()
    restored.load_state_dict(payload["model"])
    model.eval()
    with torch.no_grad():
        assert torch.equal(
            model(inputs)["fine_logits"], restored(inputs)["fine_logits"]
        )


def test_masked_frames_missing_values_and_feature_order_do_not_change_output():
    model = SegNeXtTiny(contract()).eval()
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


def test_spatial_baseline_is_temporal_permutation_invariant_and_finite_when_empty():
    model = SegNeXtTiny(contract()).eval()
    inputs = batch()
    permuted = copy.deepcopy(inputs)
    order = [2, 0, 1]
    for name in ("dynamic", "dynamic_mask", "dynamic_time_mask", "time_encoding"):
        permuted[name] = inputs[name][:, order]
    with torch.no_grad():
        assert torch.allclose(
            model(inputs)["fine_logits"],
            model(permuted)["fine_logits"],
            rtol=1e-6,
            atol=1e-5,
        )
    inputs["dynamic"][:] = float("nan")
    inputs["static"][:] = float("nan")
    inputs["valid_mask"][:] = False
    with torch.no_grad():
        output = model(inputs)
    assert torch.isfinite(output["fine_logits"]).all()
    assert not output["valid_mask"].any()
