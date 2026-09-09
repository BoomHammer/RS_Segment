"""Native AMP selection, accumulated updates and FP16 mask stability."""

from contextlib import nullcontext

import pytest
import torch

from models.pretrained_utae import MaskedUTAE, PretrainedSegFormerUTAE
from models.utae.ltae import LTAE2d, ScaledDotProductAttention
from precision import resolve_amp_dtype, scaled_optimizer_step


@pytest.mark.parametrize("requested", ["auto", "bfloat16", "float16", "none"])
def test_cpu_keeps_full_precision(requested):
    assert resolve_amp_dtype(torch.device("cpu"), requested) is None


def test_amp_validates_configuration():
    with pytest.raises(ValueError, match="amp_dtype"):
        resolve_amp_dtype(torch.device("cpu"), "typo")


@pytest.mark.parametrize("native_bf16", [False, True])
def test_amp_uses_native_hardware_support(monkeypatch, native_bf16):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    def supported(*, including_emulation):
        assert including_emulation is False
        return native_bf16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    device = torch.device("cuda:0")
    expected = torch.bfloat16 if native_bf16 else torch.float16
    assert resolve_amp_dtype(device) == expected
    assert resolve_amp_dtype(device, "float16") == torch.float16
    assert resolve_amp_dtype(device, "none") is None
    if native_bf16:
        assert resolve_amp_dtype(device, "bfloat16") == expected
    else:
        with pytest.warns(UserWarning, match="FP16"):
            assert resolve_amp_dtype(device, "bfloat16") == expected


def test_scaler_accumulation_clipping_overflow_and_restore():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    for _ in range(2):
        scaler.scale(parameter.square() / 2).backward()
    # True accumulated gradient is 2; clipping it to 1 gives an update of 0.1.
    assert scaled_optimizer_step(optimizer, scaler, max_norm=1.0)
    torch.testing.assert_close(parameter, torch.tensor(0.9))
    state = scaler.state_dict()
    restored = torch.amp.GradScaler("cpu")
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    optimizer.zero_grad(set_to_none=True)
    restored.scale(parameter * float("inf")).backward()
    before = parameter.detach().clone()
    assert not scaled_optimizer_step(optimizer, restored, max_norm=1.0)
    torch.testing.assert_close(parameter, before)
    assert restored.get_scale() == 64.0


def test_temporal_position_broadcast_preserves_sample_boundaries():
    torch.manual_seed(42)
    encoder = LTAE2d(
        in_channels=8, n_head=2, d_k=2, d_model=8, mlp=(8, 8), return_att=True
    ).eval()
    values = torch.randn(2, 3, 8, 2, 2)
    positions = torch.tensor([[1.0, 8.0, 16.0], [4.0, 12.0, 31.0]])
    with torch.no_grad():
        together, attention = encoder(values, positions)
        singles = [encoder(values[i : i + 1], positions[i : i + 1]) for i in range(2)]
    torch.testing.assert_close(together, torch.cat([item[0] for item in singles]))
    torch.testing.assert_close(attention, torch.cat([item[1] for item in singles], 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP16 required")
def test_fp16_fully_masked_attention_has_finite_backward():
    attention = ScaledDotProductAttention(2.0, attn_dropout=0.0).cuda()
    query = torch.randn(2, 4, device="cuda", requires_grad=True)
    keys = torch.randn(2, 3, 4, device="cuda", requires_grad=True)
    values = torch.randn(2, 3, 4, device="cuda", requires_grad=True)
    mask = torch.tensor([[True, True, True], [False, True, False]], device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        output, weights = attention(query, keys, values, pad_mask=mask)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(weights[0]) == 0
    torch.testing.assert_close(weights[1].sum(), weights.new_tensor(1.0))
    output.float().square().sum().backward()
    for value in (query, keys, values):
        assert torch.isfinite(value.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP16 required")
def test_fp16_grouped_skip_handles_absent_pixels():
    features = torch.randn(
        1, 3, 4, 8, 8, device="cuda", dtype=torch.float16, requires_grad=True
    )
    attention = torch.rand(
        2, 1, 3, 2, 2, device="cuda", dtype=torch.float16, requires_grad=True
    )
    present = torch.ones(1, 3, 1, 8, 8, device="cuda", dtype=torch.bool)
    present[..., :4, :4] = False
    with torch.autocast("cuda", dtype=torch.float16):
        output = MaskedUTAE._aggregate(features, attention, present)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[..., :4, :4]) == 0
    output.float().square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(attention.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP16 required")
@pytest.mark.parametrize("padded", [False, True])
def test_full_model_fp16_preserves_masks_hierarchy_and_gradients(padded):
    torch.manual_seed(42)
    contract = {
        "derived": {
            "static_features_count": 1,
            "dynamic_features_count": 2,
            "num_coarse_classes": 2,
            "fine_to_coarse": [0, 0, 1],
        }
    }
    model = PretrainedSegFormerUTAE(contract).cuda().train()
    batch = {
        "dynamic": torch.randn(1, 3, 2, 32, 32, device="cuda"),
        "static": torch.randn(1, 1, 32, 32, device="cuda"),
        "dynamic_mask": torch.ones(1, 3, 2, device="cuda", dtype=torch.bool),
        "dynamic_time_mask": torch.tensor([[True, True, not padded]], device="cuda"),
        "time_encoding": torch.rand(1, 3, 3, device="cuda"),
        **{
            key: torch.ones(1, 32, 32, device="cuda", dtype=torch.bool)
            for key in ("dynamic_valid_mask", "static_valid_mask", "valid_mask")
        },
    }
    batch["dynamic_valid_mask"][..., :16, :16] = False
    with torch.autocast("cuda", dtype=torch.float16):
        prediction = model(batch)
        loss = -prediction["fine_logits"][:, 0].mean()
    probability = prediction["fine_probability"]
    assert probability.shape == (1, 3, 32, 32)
    torch.testing.assert_close(probability.sum(1), torch.ones_like(probability[:, 0]))
    torch.testing.assert_close(
        probability[:, :2].sum(1), prediction["coarse_probability"][:, 0]
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    scaler.scale(loss).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.static_stem.weight.grad is not None
    assert model.dynamic_encoder.in_conv.conv.conv[0].weight.grad is not None
    assert scaled_optimizer_step(optimizer, scaler, max_norm=1.0)
