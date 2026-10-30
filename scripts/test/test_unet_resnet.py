"""U-Net–ResNet34 spatial baseline integration tests."""

import copy

import torch

from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.unet_resnet import UNetResNet34


def contract():
    return {
        "architecture": "unet_resnet34",
        "derived": {
            "dynamic_features": ["NDVI", "SR_B1"],
            "static_features": ["DEM"],
            "dynamic_features_count": 2,
            "static_features_count": 1,
            "num_classes": 3,
            "fine_to_coarse": [0, 1, 1],
        },
        "unet": {"decoder_channels": 16, "dropout": 0.0},
    }


def batch():
    torch.manual_seed(31)
    return {
        "dynamic": torch.randn(2, 3, 2, 35, 37),
        "static": torch.randn(2, 1, 35, 37),
        "dynamic_mask": torch.ones(2, 3, 2, dtype=torch.bool),
        "dynamic_time_mask": torch.ones(2, 3, dtype=torch.bool),
        "time_encoding": torch.rand(2, 3, 3),
        "dynamic_valid_mask": torch.ones(2, 35, 37, dtype=torch.bool),
        "static_valid_mask": torch.ones(2, 35, 37, dtype=torch.bool),
        "valid_mask": torch.ones(2, 35, 37, dtype=torch.bool),
        "ground_truth": torch.randint(1, 4, (2, 35, 37)),
        "ground_truth_mask": torch.ones(2, 35, 37, dtype=torch.bool),
        "weak_label": torch.full((2, 35, 37), -1, dtype=torch.long),
        "weak_label_mask": torch.zeros(2, 35, 37, dtype=torch.bool),
        "dynamic_features": [["NDVI", "SR_B1"]] * 2,
        "static_features": [["DEM"]] * 2,
    }


def test_factory_shape_hierarchy_backward_and_checkpoint(tmp_path):
    model = SegFormerUtae.from_contract(contract())
    assert isinstance(model, UNetResNet34)
    inputs = batch()
    output = model(inputs)
    assert output["fine_logits"].shape == (2, 3, 35, 37)
    assert torch.allclose(output["fine_probability"].sum(1), torch.ones(2, 35, 37))
    loss = combined_supervision_loss(
        output, inputs, fine_to_coarse=contract()["derived"]["fine_to_coarse"]
    )["loss"]
    loss.backward()
    assert model.encoder.stem[0].weight.grad is not None
    assert model.decoder4.convolutions[0].weight.grad is not None
    path = tmp_path / "unet.pt"
    torch.save({"contract": contract(), "model": model.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    restored = SegFormerUtae.from_contract(payload["contract"])
    restored.load_state_dict(payload["model"])


def test_masked_inputs_and_feature_order_are_authoritative():
    model = UNetResNet34(contract()).eval()
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

    reordered = copy.deepcopy(inputs)
    reordered["dynamic"] = inputs["dynamic"][:, :, [1, 0]]
    reordered["dynamic_mask"] = inputs["dynamic_mask"][:, :, [1, 0]]
    reordered["dynamic_features"] = [["SR_B1", "NDVI"]] * 2
    with torch.no_grad():
        assert torch.equal(
            model(inputs)["fine_logits"], model(reordered)["fine_logits"]
        )
