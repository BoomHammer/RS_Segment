"""Check controlled initialization, encoder interfaces and masked gradients."""

import copy

import pytest
import torch

from models.architecture import SegFormerUtae


def contract(architecture="segformer_utae"):
    return {
        "architecture": architecture,
        "temporal": {"projection_channels": 64, "frame_chunk_size": 2},
        "static": {"projection_channels": 32},
        "fusion": {"output_channels": 32},
        "regularization": {"dropout": 0.1, "drop_path": 0.05},
        "pretrained": {"freeze_stages": 2},
        "derived": {
            "dynamic_features_count": 2,
            "static_features_count": 2,
            "num_classes": 3,
            "num_coarse_classes": 2,
            "fine_to_coarse": [0, 0, 1],
        },
    }


@pytest.mark.parametrize("branch", ["dynamic", "static"])
def test_common_modules_and_rng_are_identical(branch):
    torch.manual_seed(42)
    control = SegFormerUtae.from_contract(contract())
    after_control = torch.get_rng_state()
    torch.manual_seed(42)
    model = SegFormerUtae.from_contract(contract(f"segformer_utae_{branch}_ablation"))
    assert torch.equal(after_control, torch.get_rng_state())
    prefixes = ["fusions.", "decoder.", "heads.", "adapter."]
    prefixes.append("static_encoder." if branch == "dynamic" else "dynamic_encoder.")
    common = control.state_dict()
    for name, value in model.state_dict().items():
        if any(name.startswith(prefix) for prefix in prefixes):
            torch.testing.assert_close(value, common[name], rtol=0, atol=0)


@pytest.mark.parametrize("branch", ["dynamic", "static"])
def test_branch_forward_backward_and_missing_date(branch):
    torch.set_num_threads(2)
    model = SegFormerUtae.from_contract(contract(f"segformer_utae_{branch}_ablation"))
    batch = {
        "dynamic": torch.randn(1, 3, 2, 64, 64),
        "static": torch.randn(1, 2, 64, 64),
        "dynamic_mask": torch.ones(1, 3, 2, dtype=torch.bool),
        "dynamic_time_mask": torch.tensor([[True, True, False]]),
        "time_encoding": torch.zeros(1, 3, 3),
        "dynamic_valid_mask": torch.ones(1, 64, 64, dtype=torch.bool),
        "static_valid_mask": torch.ones(1, 64, 64, dtype=torch.bool),
        "valid_mask": torch.ones(1, 64, 64, dtype=torch.bool),
    }
    batch["dynamic_mask"][:, 2] = False
    output = model(batch)
    assert output["fine_logits"].shape == (1, 3, 64, 64)
    loss = -output["fine_logits"][:, 0, 30, 30].mean()
    loss.backward()
    encoder = model.dynamic_encoder if branch == "dynamic" else model.static_encoder
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    model.eval()
    altered = copy.deepcopy(batch)
    altered["dynamic"][:, 2] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(
            model(batch)["fine_logits"], model(altered)["fine_logits"]
        )
