"""AnySat sensor, official-core equivalence and training integration checks."""

import copy

import pytest
import torch

from data.anysat import prepare_sensor, resolve_resolution
from losses.supervision import combined_supervision_loss
from models.anysat import AnySatSegmentation
from models.anysat_core import AnySatCore, position, self_block
from models.architecture import SegFormerUtae


def contract():
    return {
        "architecture": "anysat",
        "derived": {
            "dynamic_features": ["NDVI", "PR", "SR_B1", "SR_B2"],
            "static_features": ["DEM", "rain"],
            "num_classes": 3,
            "fine_to_coarse": [0, 1, 1],
        },
        "anysat": {
            "embed_dim": 24,
            "heads": 3,
            "depth": 1,
            "patch_size": 4,
            "subpatch_size": 2,
            "projector_chunk_size": 11,
            "spatial_chunk_size": 5,
            "dense_modality": "SR",
            "decoder_channels": 8,
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
        "dynamic_times": [["2023-01-01", "2023-04", "2024-12-31"]] * 2,
        "dynamic_valid_mask": torch.ones(2, 7, 9, dtype=torch.bool),
        "static_valid_mask": torch.ones(2, 7, 9, dtype=torch.bool),
        "valid_mask": torch.ones(2, 7, 9, dtype=torch.bool),
        "ground_truth": torch.randint(1, 4, (2, 7, 9)),
        "ground_truth_mask": torch.ones(2, 7, 9, dtype=torch.bool),
        "weak_label": torch.full((2, 7, 9), -1, dtype=torch.long),
        "weak_label_mask": torch.zeros(2, 7, 9, dtype=torch.bool),
    }


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_factory_backward_hierarchy_checkpoint(tmp_path):
    model = SegFormerUtae.from_contract(contract())
    assert isinstance(model, AnySatSegmentation)
    inputs = batch()
    output = model(inputs)
    assert output["fine_logits"].shape == (2, 3, 7, 9)
    assert output["valid_mask"].shape == (2, 1, 7, 9)
    torch.testing.assert_close(output["fine_probability"].sum(1), torch.ones(2, 7, 9))
    torch.testing.assert_close(
        output["fine_probability"][:, 1:].sum(1), output["coarse_probability"][:, 1]
    )
    loss = combined_supervision_loss(output, inputs, fine_to_coarse=[0, 1, 1])["loss"]
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.projectors["SR"].mlp[0].weight.grad.abs().sum() > 0
    path = tmp_path / "anysat.pt"
    torch.save({"contract": contract(), "model": model.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    restored = SegFormerUtae.from_contract(payload["contract"]).eval()
    restored.load_state_dict(payload["model"])
    with torch.no_grad():
        torch.testing.assert_close(
            model.eval()(inputs)["fine_logits"],
            restored(inputs)["fine_logits"],
            rtol=0,
            atol=0,
        )


def test_masks_feature_order_and_all_missing_backward():
    inputs = batch()
    inputs["dynamic_time_mask"][:, 2] = False
    inputs["dynamic_mask"][:, 1, 1] = False
    inputs["static_valid_mask"][:, 0, 0] = False
    altered = copy.deepcopy(inputs)
    altered["dynamic"][:, 2] = 1e7
    altered["dynamic"][:, 1, 1] = float("nan")
    altered["static"][:, :, 0, 0] = float("inf")
    model = AnySatSegmentation(contract()).eval()
    with torch.no_grad():
        expected = model(inputs)["fine_logits"]
        torch.testing.assert_close(
            expected, model(altered)["fine_logits"], rtol=0, atol=0
        )
        order = [2, 0, 3, 1]
        altered["dynamic"] = altered["dynamic"][:, :, order]
        altered["dynamic_mask"] = altered["dynamic_mask"][:, :, order]
        altered["dynamic_features"] = [
            [contract()["derived"]["dynamic_features"][i] for i in order]
        ] * 2
        altered["static"] = altered["static"][:, [1, 0]]
        altered["static_features"] = [["rain", "DEM"]] * 2
        torch.testing.assert_close(
            expected, model(altered)["fine_logits"], rtol=0, atol=0
        )
    inputs["dynamic"][:] = float("nan")
    inputs["static"][:] = float("nan")
    inputs["valid_mask"][:] = False
    output = model.train()(inputs)
    assert torch.isfinite(output["fine_logits"]).all()
    loss = combined_supervision_loss(output, inputs, fine_to_coarse=[0, 1, 1])["loss"]
    assert loss == 0
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_actual_dates_masks_and_no_temporal_subsampling():
    inputs = batch()
    spec = {"role": "dynamic", "features": ["PR"]}
    names = contract()["derived"]["dynamic_features"]
    values, valid, dates = prepare_sensor(inputs, spec, names)
    assert values.shape[1] == 3
    assert dates[0].tolist() == [0, 90, 365]
    inputs["dynamic_mask"][:, :2, 1] = False
    inputs["dynamic"][0, 2, 1, 0, 0] = float("nan")
    values, valid, dates = prepare_sensor(inputs, spec, names)
    assert values.shape[1] == 1 and dates[0].item() == 365
    assert not valid[0, 0, 0, 0, 0] and values[0, 0, 0, 0, 0] == 0


def test_resolution_uses_target_grid_units():
    grid = {"crs": "EPSG:5070", "transform": [30, 0, 0, 0, -30, 0]}
    assert resolve_resolution({}, grid) == 30
    with pytest.raises(ValueError, match="GSD"):
        resolve_resolution({"resolution_m": 250}, grid)
    grid["crs"] = "EPSG:4326"
    with pytest.raises(ValueError, match="nominal"):
        resolve_resolution({}, grid)
    assert resolve_resolution({"resolution_m": 250}, grid) == 250


def test_sdpa_matches_official_layers_with_all_valid_inputs():
    core = AnySatCore(24, 3, 1, 0, False).eval()
    x = torch.randn(2, 5, 24)
    valid = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        for block in (core.blocks[0], core.spatial_encoder.predictor_blocks[0]):
            torch.testing.assert_close(
                self_block(block, x, valid, 2), block(x), rtol=1e-5, atol=2e-6
            )
        # Match the complete official combiner including CLS, position and iRPE.
        tokens = torch.randn(2, 3, 4, 24)
        pos = position(24, 2, 100, tokens)
        native = (tokens + pos[:, None, 1:]).reshape(2, 12, 24)
        native = torch.cat((core.cls_token.expand(2, -1, -1), native), 1)
        native = core.blocks[0](native)
        native = core.blocks[-1].forward_release(native, n_modalities=3, scale=100)
        actual = core(tokens, torch.ones(2, 3, 4, dtype=torch.bool), 2, 100)
        torch.testing.assert_close(actual, native[:, 1:], rtol=1e-5, atol=2e-6)


def test_budget_and_pretrained_core_are_strict(tmp_path):
    config = contract()
    config["anysat"]["max_tokens"] = 1
    with pytest.raises(ValueError, match="token"):
        AnySatSegmentation(config)(batch())
    config = contract()
    config["anysat"]["max_subpatch_tokens"] = 1
    with pytest.raises(ValueError, match="subpatch token"):
        AnySatSegmentation(config)
    config = contract()
    path = tmp_path / "core.pth"
    model = AnySatSegmentation(config)
    torch.save({"state_dict": model.core.state_dict()}, path)
    config["anysat"]["pretrained_path"] = str(path)
    restored = AnySatSegmentation(config)
    assert restored.initialize_pretrained()["core_tensors"] > 0
    for key, value in model.core.state_dict().items():
        assert torch.equal(value, restored.core.state_dict()[key])
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="complete core"):
        restored.initialize_pretrained()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_hierarchical_backward():
    model = AnySatSegmentation(contract()).cuda().train()
    inputs = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in batch().items()
    }
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(inputs)
        loss = combined_supervision_loss(output, inputs, fine_to_coarse=[0, 1, 1])[
            "loss"
        ]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
