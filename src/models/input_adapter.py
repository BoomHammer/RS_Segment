"""Model-facing input adapters for the SegFormer + U-TAE pipeline."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def sanitize_batch(batch: dict[str, Any], *, fill_value: float = 0.0) -> dict[str, Any]:
    """Create finite tensors plus authoritative masks for model computation."""

    dynamic = batch["dynamic"].float()
    static = batch["static"].float()
    dynamic_mask = batch["dynamic_mask"].bool()
    dynamic_mask = dynamic_mask[:, :, :, None, None]
    dynamic_valid = batch["dynamic_valid_mask"].bool()[:, None, None, :, :]
    static_valid = batch["static_valid_mask"].bool()[:, None, :, :]
    dynamic_ok = torch.isfinite(dynamic) & dynamic_mask & dynamic_valid
    static_ok = torch.isfinite(static) & static_valid
    return {
        **batch,
        # The fill value is only a computational sentinel. The masks preserve
        # the distinction between an invalid pixel and a valid physical zero.
        "dynamic": torch.where(
            dynamic_ok, dynamic, torch.full_like(dynamic, fill_value)
        ),
        "static": torch.where(static_ok, static, torch.full_like(static, fill_value)),
        "dynamic_value_mask": dynamic_ok,
        "static_value_mask": static_ok,
        "dynamic_mask": batch["dynamic_mask"].bool(),
        "dynamic_time_mask": batch["dynamic_time_mask"].bool(),
        "dynamic_valid_mask": batch["dynamic_valid_mask"].bool(),
        "static_valid_mask": batch["static_valid_mask"].bool(),
        "valid_mask": batch["valid_mask"].bool(),
    }


class TemporalAttentionEncoder(nn.Module):
    """Lightweight U-TAE-style per-pixel temporal attention encoder."""

    def __init__(
        self,
        in_features: int,
        channels: int = 64,
        time_encoding_dim: int = 3,
    ) -> None:
        super().__init__()
        if in_features < 1 or channels < 1:
            raise ValueError("in_features 和 channels 必须为正数")
        self.projection = nn.Conv2d(in_features, channels, kernel_size=1, bias=False)
        self.time_projection = nn.Linear(time_encoding_dim, channels)
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(
        self,
        dynamic: Tensor,
        dynamic_mask: Tensor,
        dynamic_time_mask: Tensor,
        time_encoding: Tensor,
        dynamic_value_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return fused spatial features and a pixel validity mask."""

        if dynamic.ndim != 5:
            raise ValueError("dynamic 必须是 [B, T, F, H, W]")
        batch_size, time_steps, _, height, width = dynamic.shape
        projected = self.projection(dynamic.flatten(0, 1)).unflatten(
            0, (batch_size, time_steps)
        )
        projected = projected + self.time_projection(time_encoding).view(
            batch_size, time_steps, -1, 1, 1
        )
        scores = self.attention(projected.flatten(0, 1)).view(
            batch_size, time_steps, height, width
        )
        feature_present = dynamic_mask.any(dim=2)
        pixel_present = dynamic_value_mask.any(dim=2)
        valid_time = dynamic_time_mask & feature_present
        valid_time = valid_time[:, :, None, None] & pixel_present
        scores = scores.masked_fill(~valid_time, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(valid_time, weights, 0.0)
        fused = (projected * weights[:, :, None]).sum(dim=1)
        valid_pixels = valid_time.any(dim=1)[:, None]
        return fused, valid_pixels


class SegFormerUtaeInputAdapter(nn.Module):
    """Prepare sanitized multi-source inputs for a SegFormer spatial encoder."""

    def __init__(
        self,
        dynamic_features: int,
        static_features: int,
        *,
        dynamic_channels: int = 64,
        static_channels: int = 32,
        fused_channels: int = 96,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        if static_features < 1:
            raise ValueError("static_features 必须为正数")
        self.fill_value = fill_value
        self.temporal = TemporalAttentionEncoder(
            dynamic_features, channels=dynamic_channels
        )
        self.static = nn.Sequential(
            nn.Conv2d(static_features, static_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dynamic_channels + static_channels, fused_channels, 1),
            nn.GELU(),
        )

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> SegFormerUtaeInputAdapter:
        """Build the adapter from dimensions derived from JSON artifacts."""

        derived = dict(contract.get("derived", contract))
        temporal = dict(contract.get("temporal", {}))
        static = dict(contract.get("static", {}))
        fusion = dict(contract.get("fusion", {}))
        input_config = dict(contract.get("input", {}))
        return cls(
            dynamic_features=int(derived["dynamic_features_count"]),
            static_features=int(derived["static_features_count"]),
            dynamic_channels=int(temporal.get("projection_channels", 64)),
            static_channels=int(static.get("projection_channels", 32)),
            fused_channels=int(fusion.get("output_channels", 96)),
            fill_value=float(input_config.get("invalid_fill_value", 0.0)),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Return model-safe fused features and validity masks."""

        clean = sanitize_batch(batch, fill_value=self.fill_value)
        temporal, temporal_valid = self.temporal(
            clean["dynamic"],
            clean["dynamic_mask"],
            clean["dynamic_time_mask"],
            clean["time_encoding"],
            clean["dynamic_value_mask"],
        )
        static = self.static(clean["static"])
        fused = self.fusion(torch.cat((temporal, static), dim=1))
        valid = clean["valid_mask"][:, None] & temporal_valid
        return {
            "features": fused,
            "dynamic_features": temporal,
            "static_features": static,
            "valid_mask": valid,
        }
