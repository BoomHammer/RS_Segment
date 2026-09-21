"""MAESTRO-S downstream encoder adapted to streamed MODIS/terrain windows.

Reference: https://arxiv.org/html/2508.10894v2 (3.1, 3.3, 4.4).
Small dimensions follow IGNF/MAESTRO maestro/ssl/mae.py: 384/12/6, MLP ratio 2.
This is a local implementation, not an official checkpoint-compatible wrapper.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from data.maestro import prepare_modality
from models.architecture import DropPath, HierarchicalHeads


class Attention(nn.Module):
    """SDPA avoids explicitly retaining quadratic attention matrices."""

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
        # A zero sentinel handles all-missing samples on every SDPA backend.
        k = F.pad(k, (0, 0, 0, 1))
        v = F.pad(v, (0, 0, 0, 1))
        allowed = torch.cat((valid, ~valid.any(1, keepdim=True)), dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None, None])
        out = self.projection(out.transpose(1, 2).reshape(batch, length, dim))
        return out.masked_fill(~valid[..., None], 0.0)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: int, drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, dim)
        )
        self.drop = DropPath(drop)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        x = x + self.drop(self.attention(self.norm1(x), valid))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x.masked_fill(~valid[..., None], 0.0)


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        ratio: int,
        drop: float,
        checkpointing: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(dim, heads, ratio, drop) for _ in range(depth)
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


class AttentivePool(nn.Module):
    """Learned multi-head query pools modality/date tokens at each location."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        batch, dates, locations, dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * locations, dates, dim)
        valid = valid.permute(0, 2, 1).reshape(batch * locations, dates)
        k, v = (
            self.kv(self.norm(x))
            .reshape(-1, dates, 2, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q = self.query.reshape(1, self.heads, 1, dim // self.heads)
        allowed = torch.cat((valid, ~valid.any(1, keepdim=True)), dim=1)
        pooled = F.scaled_dot_product_attention(
            q,
            F.pad(k, (0, 0, 0, 1)),
            F.pad(v, (0, 0, 0, 1)),
            attn_mask=allowed[:, None, None],
        ).reshape(batch * locations, dim)
        pooled = self.output_norm(pooled).masked_fill(~valid.any(1, keepdim=True), 0.0)
        return pooled.reshape(batch, locations, dim)


def modality_specs(derived: dict[str, Any], settings: dict[str, Any]) -> list[dict]:
    """Bind product-specific tokenizers to the persisted input feature names."""
    specs = settings.get("modalities")
    if specs is None:
        products: dict[str, list[str]] = {}
        for feature in derived["dynamic_features"]:
            products.setdefault(re.sub(r"_B\d+$", "", feature), []).append(feature)
        specs = [
            {
                "name": name,
                "role": "dynamic",
                "features": features,
                "group": "optical"
                if name.upper() in {"SR", "NDVI", "EVI", "FPAR", "GPP", "LAI"}
                else "environment",
            }
            for name, features in products.items()
        ]
        specs.append(
            {
                "name": "static",
                "role": "static",
                "group": "static",
                "features": derived["static_features"],
            }
        )
    specs = [dict(spec) for spec in specs]
    for role in ("dynamic", "static"):
        names = [
            name for spec in specs if spec["role"] == role for name in spec["features"]
        ]
        if sorted(names) != sorted(derived[f"{role}_features"]):
            raise ValueError(f"MAESTRO modalities 必须恰好覆盖所有 {role} 特征一次")
    identifiers = [spec["name"] for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("MAESTRO modality 名称不可重复")
    for spec in specs:
        if spec["role"] not in {"dynamic", "static"} or not spec["features"]:
            raise ValueError("MAESTRO modality role/features 无效")
        for key in ("name", "group"):
            if not spec[key] or "." in spec[key]:
                raise ValueError("MAESTRO modality/group 名称非空且不能包含点")
        spec["indices"] = [
            derived[f"{spec['role']}_features"].index(name) for name in spec["features"]
        ]
        spec["temporal_bins"] = (
            1
            if spec["role"] == "static"
            else int(spec.get("temporal_bins", settings.get("temporal_bins", 4)))
        )
        if spec["temporal_bins"] < 1:
            raise ValueError("temporal_bins 必须为正数")
    return specs


class MaestroS(nn.Module):
    """Nine group-specific blocks + three joint blocks, with hierarchical head."""

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__()
        self.contract = contract
        self.derived = contract["derived"]
        settings = contract.get("maestro", {})
        dim = int(settings.get("embed_dim", 384))
        depth = int(settings.get("depth", 12))
        heads = int(settings.get("heads", 6))
        inter_depth = int(settings.get("inter_depth", 3))
        ratio = int(settings.get("mlp_ratio", 2))
        self.patch_size = int(settings.get("patch_size", 32))
        self.max_tokens = int(settings.get("max_tokens", 4096))
        if (
            heads < 1
            or dim < 12
            or dim % heads
            or (dim - 8) % 4
            or not 0 < inter_depth < depth
            or ratio < 1
            or self.patch_size < 1
            or self.max_tokens < 1
        ):
            raise ValueError("MAESTRO dimensions/depth/patch_size/token budget 无效")
        self.dim = dim
        self.specs = modality_specs(self.derived, settings)
        self.groups = list(dict.fromkeys(spec["group"] for spec in self.specs))
        self.tokenizers = nn.ModuleDict(
            {
                spec["name"]: nn.Conv2d(
                    len(spec["features"]) * 2,
                    dim,
                    self.patch_size,
                    stride=self.patch_size,
                )
                for spec in self.specs
            }
        )
        drop = float(contract.get("regularization", {}).get("drop_path", 0.05))
        checkpointing = bool(settings.get("gradient_checkpointing", True))
        self.encoders = nn.ModuleDict(
            {
                group: Encoder(
                    dim, heads, depth - inter_depth, ratio, drop, checkpointing
                )
                for group in self.groups
            }
        )
        self.fusion = Encoder(dim, heads, inter_depth, ratio, drop, checkpointing)
        self.pool = AttentivePool(dim, heads)
        self.dropout = nn.Dropout(
            float(contract.get("regularization", {}).get("dropout", 0.2))
        )
        mapping = self.derived["fine_to_coarse"]
        if len(mapping) != int(self.derived["num_classes"]):
            raise ValueError("fine_to_coarse 长度与类别数不一致")
        self.heads = HierarchicalHeads(dim, max(mapping) + 1, mapping)

        # Dense patch unprojection, as in the paper's PixelifyHead. Compute
        # hierarchical probabilities AFTER unpatchifying coarse/fine logits.
        def pixel_head(classes: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(dim, classes * self.patch_size**2, 1),
                nn.PixelShuffle(self.patch_size),
            )

        self.heads.coarse = pixel_head(max(mapping) + 1)
        self.heads.experts = nn.ModuleList(
            pixel_head(mapping.count(parent)) for parent in range(max(mapping) + 1)
        )

    def spatial_encoding(self, height: int, width: int, device: torch.device) -> Tensor:
        # Existing rasters already share a target grid/GSD; there is no native
        # 90 m DEM grid at this interface. All token grids are therefore equal.
        quarter = (self.dim - 8) // 4
        frequency = torch.exp(
            -math.log(10000) * torch.arange(quarter, device=device) / quarter
        )
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        x, y = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        height, width = batch["static"].shape[-2:]
        patch = self.patch_size
        grid_h, grid_w = math.ceil(height / patch), math.ceil(width / patch)
        locations = grid_h * grid_w
        count = locations * sum(spec["temporal_bins"] for spec in self.specs)
        if count > self.max_tokens:
            raise ValueError(
                f"MAESTRO token 数 {count} 超过上限 {self.max_tokens}；"
                "请缩小窗口/temporal_bins 或增大 patch_size"
            )
        for role in ("dynamic", "static"):
            if batch[role].shape[-3] != len(self.derived[f"{role}_features"]):
                raise ValueError(f"{role} 通道数与 MAESTRO contract 不一致")
        spatial = self.spatial_encoding(grid_h, grid_w, batch["static"].device)
        grouped: dict[str, list[Tensor]] = {group: [] for group in self.groups}
        masks: dict[str, list[Tensor]] = {group: [] for group in self.groups}
        for spec in self.specs:
            indices = spec["indices"]
            feature_lists = batch.get(f"{spec['role']}_features")
            if feature_lists is not None:
                names = feature_lists[0]
                if any(names != other for other in feature_lists):
                    raise ValueError("批内特征顺序必须一致")
                indices = [names.index(name) for name in spec["features"]]
            values, valid, dates = prepare_modality(
                batch,
                spec["role"],
                indices,
                spec["temporal_bins"],
                training=self.training,
            )
            b, t, _, _, _ = values.shape
            pad = (0, grid_w * patch - width, 0, grid_h * patch - height)
            # Presence channels distinguish physical zero from missing values.
            inputs = F.pad(
                torch.cat((values, valid.to(values.dtype)), dim=2).flatten(0, 1), pad
            )
            tokens = self.tokenizers[spec["name"]](inputs).flatten(2).transpose(1, 2)
            token_valid = (
                F.max_pool2d(
                    F.pad(valid.any(2).float().flatten(0, 1)[:, None], pad),
                    patch,
                    stride=patch,
                )
                .flatten(1)
                .bool()
            )
            encoding = torch.cat(
                (
                    spatial[None].expand(b * t, -1, -1),
                    dates.reshape(b * t, 1, 8).expand(-1, locations, -1),
                ),
                -1,
            )
            tokens = (tokens + encoding.to(tokens.dtype)).masked_fill(
                ~token_valid[..., None], 0
            )
            grouped[spec["group"]].append(tokens.reshape(b, t * locations, self.dim))
            masks[spec["group"]].append(token_valid.reshape(b, t * locations))
        encoded, valid_groups = [], []
        for group in self.groups:
            valid = torch.cat(masks[group], dim=1)
            encoded.append(
                self.encoders[group](torch.cat(grouped[group], dim=1), valid)
            )
            valid_groups.append(valid)
        valid = torch.cat(valid_groups, dim=1)
        encoded = self.fusion(torch.cat(encoded, dim=1), valid)
        b = encoded.shape[0]
        pooled = self.pool(
            encoded.reshape(b, -1, locations, self.dim), valid.reshape(b, -1, locations)
        )
        features = pooled.transpose(1, 2).reshape(b, self.dim, grid_h, grid_w)
        output = self.heads(self.dropout(features))
        output = {key: value[..., :height, :width] for key, value in output.items()}
        output["valid_mask"] = batch["valid_mask"][:, None].bool()
        return output
