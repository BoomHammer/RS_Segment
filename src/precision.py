"""Select native CUDA mixed precision and safely update scaled gradients."""

from __future__ import annotations

import warnings

import torch


def resolve_amp_dtype(
    device: torch.device, requested: str = "auto"
) -> torch.dtype | None:
    """Prefer BF16 on supported GPUs, FP16 on older CUDA GPUs, FP32 on CPU."""

    if requested not in {"auto", "bfloat16", "float16", "none"}:
        raise ValueError("amp_dtype 必须是 auto、bfloat16、float16 或 none")
    if device.type != "cuda" or requested == "none":
        return None
    if requested == "float16":
        return torch.float16
    with torch.cuda.device(device):
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if native_bf16:
        return torch.bfloat16
    if requested == "bfloat16":
        warnings.warn(
            "当前 GPU 不支持原生 BF16，自动使用 FP16；训练时启用 GradScaler。",
            stacklevel=2,
        )
    return torch.float16


def scaled_optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    max_norm: float | None = None,
) -> bool:
    """Unscale before clipping; report whether overflow skipped the update."""

    if max_norm is not None:
        scaler.unscale_(optimizer)
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return not scaler.is_enabled() or scaler.get_scale() >= previous_scale
