"""Memory-bounded TSViT adaptation for heterogeneous raster time series.

The temporal-then-spatial factorization and multiple learnable class tokens
follow the CVPR 2023 TSViT design. Project-specific adaptations are explicit:
static variables are fused after temporal encoding, missing values have presence
channels and attention masks, and the shared feature map uses the project's
hierarchical dense head.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.architecture import DropPath, HierarchicalHeads
from models.input_adapter import sanitize_batch


class Attention(nn.Module):
    """Masked multi-head attention backed by memory-efficient SDPA."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        batch, length, dim = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        # The sentinel is enabled only when a sequence is entirely missing.
        k = F.pad(k, (0, 0, 0, 1))
        v = F.pad(v, (0, 0, 0, 1))
        allowed = torch.cat((valid, ~valid.any(1, keepdim=True)), dim=1)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None, None]
        )
        output = output.transpose(1, 2).reshape(batch, length, dim)
        return self.projection(output).masked_fill(~valid[..., None], 0.0)


class Block(nn.Module):
    def __init__(
        self, dim: int, heads: int, mlp_ratio: int, dropout: float, drop_path: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), valid))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.masked_fill(~valid[..., None], 0.0)


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        mlp_ratio: int,
        dropout: float,
        drop_path: float,
        checkpointing: bool,
    ) -> None:
        super().__init__()
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            Block(dim, heads, mlp_ratio, dropout, rate) for rate in rates
        )
        self.norm = nn.LayerNorm(dim)
        self.checkpointing = checkpointing

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        for block in self.blocks:
            if self.training and self.checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, valid, use_reentrant=False)
            else:
                x = block(x, valid)
        return self.norm(x).masked_fill(~valid[..., None], 0.0)


def _spatial_encoding(
    height: int, width: int, dim: int, device: torch.device
) -> Tensor:
    """Resolution-independent 2-D encoding for clipped boundary windows."""

    quarter = dim // 4
    frequency = torch.exp(
        -math.log(10000)
        * torch.arange(quarter, device=device, dtype=torch.float32)
        / max(quarter, 1)
    )
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = x.flatten()[:, None] * frequency
    y = y.flatten()[:, None] * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)


class TSViTAdapted(nn.Module):
    """TSViT adapted to streamed dynamic/static windows and hierarchy heads."""

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__()
        self.contract = contract
        self.derived = dict(contract["derived"])
        settings = dict(contract.get("tsvit", {}))
        self.patch_size = int(settings.get("patch_size", 16))
        self.dim = int(settings.get("dim", 64))
        heads = int(settings.get("heads", 4))
        temporal_depth = int(settings.get("temporal_depth", 4))
        spatial_depth = int(settings.get("spatial_depth", 4))
        mlp_ratio = int(settings.get("mlp_ratio", 4))
        self.max_temporal_tokens = int(settings.get("max_temporal_tokens", 65536))
        self.max_spatial_tokens = int(settings.get("max_spatial_tokens", 32768))
        checkpointing = bool(settings.get("gradient_checkpointing", True))
        if (
            self.patch_size < 1
            or self.dim < 4
            or self.dim % 4
            or heads < 1
            or self.dim % heads
            or min(temporal_depth, spatial_depth, mlp_ratio) < 1
            or min(self.max_temporal_tokens, self.max_spatial_tokens) < 1
        ):
            raise ValueError("TSViT dimensions/depth/patch_size/token budget 无效")

        dynamic_features = int(self.derived["dynamic_features_count"])
        static_features = int(self.derived["static_features_count"])
        classes = int(self.derived["num_classes"])
        self.class_count = classes
        self.dynamic_tokenizer = nn.Conv2d(
            dynamic_features * 2,
            self.dim,
            self.patch_size,
            stride=self.patch_size,
        )
        self.static_tokenizer = nn.Conv2d(
            static_features * 2,
            self.dim,
            self.patch_size,
            stride=self.patch_size,
        )
        # TSViT's acquisition-time-specific lookup table, including leap day.
        self.date_embedding = nn.Embedding(366, self.dim)
        self.class_tokens = nn.Parameter(torch.randn(1, classes, self.dim) * 0.02)
        dropout = float(contract.get("regularization", {}).get("dropout", 0.0))
        drop_path = float(contract.get("regularization", {}).get("drop_path", 0.0))
        self.temporal_encoder = Encoder(
            self.dim,
            heads,
            temporal_depth,
            mlp_ratio,
            dropout,
            drop_path,
            checkpointing,
        )
        self.static_fusion = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.spatial_encoder = Encoder(
            self.dim,
            heads,
            spatial_depth,
            mlp_ratio,
            dropout,
            drop_path,
            checkpointing,
        )
        self.dropout = nn.Dropout(dropout)

        def pixel_head(count: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(self.dim, count * self.patch_size**2, 1),
                nn.PixelShuffle(self.patch_size),
            )

        self.heads = HierarchicalHeads.from_derived(self.dim, self.derived, pixel_head)

    def _ordered_inputs(
        self, clean: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dynamic = clean["dynamic"]
        dynamic_presence = clean["dynamic_value_mask"]
        static = clean["static"]
        static_presence = clean["static_value_mask"]
        for role in ("dynamic", "static"):
            expected = list(self.derived[f"{role}_features"])
            provided = clean.get(f"{role}_features")
            if provided is None:
                if clean[role].shape[-3] != len(expected):
                    raise ValueError(f"{role} 通道数与 TSViT contract 不一致")
                continue
            names = list(provided[0])
            if any(list(item) != names for item in provided):
                raise ValueError("TSViT 批内特征顺序必须一致")
            if sorted(names) != sorted(expected):
                raise ValueError(f"TSViT {role} 特征与 contract 不一致")
            indices = [names.index(name) for name in expected]
            if role == "dynamic":
                dynamic = dynamic[:, :, indices]
                dynamic_presence = dynamic_presence[:, :, indices]
            else:
                static = static[:, indices]
                static_presence = static_presence[:, indices]
        return dynamic, dynamic_presence, static, static_presence

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        clean = sanitize_batch(
            batch,
            fill_value=float(
                self.contract.get("input", {}).get("invalid_fill_value", 0)
            ),
        )
        dynamic, dynamic_presence, static, static_presence = self._ordered_inputs(clean)
        batch_size, frames, _, height, width = dynamic.shape
        patch = self.patch_size
        grid_h, grid_w = math.ceil(height / patch), math.ceil(width / patch)
        locations = grid_h * grid_w
        temporal_tokens = batch_size * locations * (self.class_count + frames)
        spatial_tokens = batch_size * self.class_count * locations
        if temporal_tokens > self.max_temporal_tokens:
            raise ValueError(
                f"TSViT temporal token 数 {temporal_tokens} 超过上限 "
                f"{self.max_temporal_tokens}；请缩小窗口/批量或增大 patch_size"
            )
        if spatial_tokens > self.max_spatial_tokens:
            raise ValueError(
                f"TSViT spatial token 数 {spatial_tokens} 超过上限 "
                f"{self.max_spatial_tokens}；请缩小窗口/批量或增大 patch_size"
            )
        pad = (0, grid_w * patch - width, 0, grid_h * patch - height)

        dynamic_input = torch.cat(
            (dynamic, dynamic_presence.to(dynamic.dtype)), dim=2
        ).flatten(0, 1)
        dynamic_tokens = (
            self.dynamic_tokenizer(F.pad(dynamic_input, pad))
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size, frames, locations, self.dim)
            .permute(0, 2, 1, 3)
        )
        frame_pixel_valid = (
            dynamic_presence.any(2) & clean["dynamic_time_mask"][:, :, None, None]
        )
        frame_valid = (
            F.max_pool2d(
                F.pad(frame_pixel_valid.float().flatten(0, 1)[:, None], pad),
                patch,
                stride=patch,
            )
            .flatten(1)
            .bool()
            .reshape(batch_size, frames, locations)
            .permute(0, 2, 1)
        )
        day = (clean["time_encoding"][..., 0] * 365.25).round().long().clamp(0, 365)
        dynamic_tokens = dynamic_tokens + self.date_embedding(day)[:, None]
        dynamic_tokens = dynamic_tokens.masked_fill(~frame_valid[..., None], 0.0)

        class_tokens = self.class_tokens.expand(batch_size * locations, -1, -1)
        temporal = dynamic_tokens.reshape(batch_size * locations, frames, self.dim)
        temporal_valid = frame_valid.reshape(batch_size * locations, frames)
        valid = torch.cat(
            (
                torch.ones(
                    batch_size * locations,
                    self.class_count,
                    dtype=torch.bool,
                    device=dynamic.device,
                ),
                temporal_valid,
            ),
            dim=1,
        )
        temporal = self.temporal_encoder(
            torch.cat((class_tokens, temporal), dim=1), valid
        )[:, : self.class_count]

        static_input = torch.cat((static, static_presence.to(static.dtype)), dim=1)
        static_tokens = (
            self.static_tokenizer(F.pad(static_input, pad)).flatten(2).transpose(1, 2)
        )
        static_valid = (
            F.max_pool2d(
                F.pad(static_presence.any(1, keepdim=True).float(), pad),
                patch,
                stride=patch,
            )
            .flatten(1)
            .bool()
        )
        static_for_classes = (
            static_tokens[:, :, None]
            .expand(-1, -1, self.class_count, -1)
            .reshape(batch_size * locations, self.class_count, self.dim)
        )
        temporal = self.static_fusion(torch.cat((temporal, static_for_classes), -1))
        temporal = temporal.reshape(
            batch_size, locations, self.class_count, self.dim
        ).permute(0, 2, 1, 3)

        location_valid = frame_valid.any(2) | static_valid
        spatial_valid = (
            location_valid[:, None]
            .expand(-1, self.class_count, -1)
            .reshape(batch_size * self.class_count, locations)
        )
        spatial = temporal.reshape(batch_size * self.class_count, locations, self.dim)
        position = _spatial_encoding(grid_h, grid_w, self.dim, dynamic.device).to(
            spatial.dtype
        )
        spatial = self.spatial_encoder(
            self.dropout(spatial + position[None]), spatial_valid
        )
        # Multiple TSViT class streams remain distinct through both encoders.
        # Their mean is the explicit adapter into the project's shared hierarchy head.
        features = spatial.reshape(
            batch_size, self.class_count, grid_h, grid_w, self.dim
        ).mean(1)
        features = features.permute(0, 3, 1, 2)
        output = self.heads(features)
        for key in tuple(output):
            output[key] = output[key][..., :height, :width]
        output["valid_mask"] = clean["valid_mask"][:, None]
        return output
