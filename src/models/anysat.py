"""AnySat dense hierarchical segmentation for the existing target-grid dataset."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from data.anysat import prepare_sensor
from models.anysat_core import AnySatCore
from models.architecture import HierarchicalHeads


def sensor_specs(derived: dict, settings: dict) -> list[dict]:
    specs = settings.get("modalities")
    if specs is None:
        products: dict[str, list[str]] = {}
        for name in derived["dynamic_features"]:
            products.setdefault(re.sub(r"_B\d+$", "", name), []).append(name)
        specs = [
            dict(name=name, role="dynamic", features=names)
            for name, names in products.items()
        ]
        specs.append(
            dict(name="static", role="static", features=derived["static_features"])
        )
    specs = [dict(spec) for spec in specs]
    identifiers = [spec["name"] for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("AnySat modality names must be unique")
    for spec in specs:
        if (
            not spec["name"]
            or "." in spec["name"]
            or not spec["features"]
            or spec["role"] not in {"dynamic", "static"}
        ):
            raise ValueError("Invalid AnySat modality name/role/features")
    for role in ("dynamic", "static"):
        names = [
            name for spec in specs if spec["role"] == role for name in spec["features"]
        ]
        if sorted(names) != sorted(derived[f"{role}_features"]):
            raise ValueError(f"AnySat modalities must cover every {role} feature once")
    return specs


class SensorProjector(nn.Module):
    """Sensor-specific sub-patch MLP and lightweight temporal attention.

    New projectors are trained on the project's normalization and channel schema;
    climate/terrain/indices are never passed off as pretrained optical bands.
    """

    def __init__(
        self, channels: int, dim: int, heads: int, subpatch: int, temporal: bool
    ) -> None:
        super().__init__()
        self.temporal = temporal
        self.heads = heads
        self.mlp = nn.Sequential(
            nn.Linear(channels * subpatch**2 * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        if temporal:
            self.key = nn.Linear(dim, heads * 8)
            self.query = nn.Parameter(torch.randn(heads, 8) / math.sqrt(8))
            self.norm = nn.LayerNorm(dim)
            self.output = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim)
            )
            self.register_buffer(
                "denominator",
                367 ** (2 * (torch.arange(dim) // 2) / dim),
                persistent=False,
            )

    def forward(self, values: Tensor, valid: Tensor, dates: Tensor) -> Tensor:
        # [locations, time, flattened subpatch channels]; validity stays explicit.
        x = self.mlp(torch.cat((values, valid.to(values.dtype)), -1))
        usable = valid.any(-1)
        if not self.temporal:
            return x[:, 0].masked_fill(~usable[:, :1], 0)
        phase = dates[..., None] / self.denominator
        encoding = torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), -1)
        x = self.norm(x) + encoding.flatten(-2).to(x.dtype)
        count, times, dim = x.shape
        k = self.key(x).reshape(count, times, self.heads, 8).transpose(1, 2)
        v = x.reshape(count, times, self.heads, dim // self.heads).transpose(1, 2)
        q = self.query[None, :, None].to(k.dtype)
        allowed = torch.cat((usable, ~usable.any(-1, keepdim=True)), -1)
        value = F.scaled_dot_product_attention(
            q,
            F.pad(k, (0, 0, 0, 1)),
            F.pad(v, (0, 0, 0, 1)),
            attn_mask=allowed[:, None, None],
        ).reshape(count, dim)
        return self.output(value).masked_fill(~usable.any(-1, keepdim=True), 0)


class AnySatSegmentation(nn.Module):
    """Pinned official AnySat core with new sensor projectors and dense heads."""

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__()
        self.contract = contract
        self.derived = contract["derived"]
        settings = contract.get("anysat", {})
        sizes = {"tiny": (256, 4, 2), "small": (512, 8, 4), "base": (768, 12, 6)}
        size = settings.get("size", "tiny")
        if size not in sizes:
            raise ValueError("AnySat size must be tiny, small or base")
        dim, heads, depth = sizes[size]
        dim = int(settings.get("embed_dim", dim))
        heads = int(settings.get("heads", heads))
        depth = int(settings.get("depth", depth))
        self.patch = int(settings.get("patch_size", 32))
        self.subpatch = int(settings.get("subpatch_size", 4))
        self.max_tokens = int(settings.get("max_tokens", 2048))
        self.max_subpatch_tokens = int(settings.get("max_subpatch_tokens", 257))
        self.projector_chunk = int(settings.get("projector_chunk_size", 256))
        self.spatial_chunk = int(settings.get("spatial_chunk_size", 16))
        self.resolution = float(settings.get("resolution_m", 250.0))
        self.checkpointing = bool(settings.get("gradient_checkpointing", True))
        if (
            min(
                dim,
                heads,
                depth,
                self.patch,
                self.subpatch,
                self.max_tokens,
                self.max_subpatch_tokens,
                self.projector_chunk,
                self.spatial_chunk,
            )
            < 1
            or dim % heads
            or dim % 4
            or self.patch % self.subpatch
            or not math.isfinite(self.resolution)
            or self.resolution <= 0
        ):
            raise ValueError("Invalid AnySat dimensions, patch sizes or memory budget")
        if (self.patch // self.subpatch) ** 2 + 1 > self.max_subpatch_tokens:
            raise ValueError(
                "AnySat subpatch token budget exceeded; increase subpatch_size "
                "or reduce patch_size"
            )
        self.specs = sensor_specs(self.derived, settings)
        self.dense_modality = settings.get("dense_modality", self.specs[0]["name"])
        if self.dense_modality not in {spec["name"] for spec in self.specs}:
            raise ValueError("AnySat dense_modality is not in modalities")
        self.projectors = nn.ModuleDict(
            {
                spec["name"]: SensorProjector(
                    len(spec["features"]),
                    dim,
                    heads,
                    self.subpatch,
                    spec["role"] == "dynamic",
                )
                for spec in self.specs
            }
        )
        self.core = AnySatCore(
            dim,
            heads,
            depth,
            float(contract.get("regularization", {}).get("drop_path", 0)),
            self.checkpointing,
        )
        channels = int(settings.get("decoder_channels", 64))
        if channels < 1:
            raise ValueError("decoder_channels must be positive")
        self.decoder = nn.Sequential(
            nn.Conv2d(dim * 2, channels * self.subpatch**2, 1),
            nn.PixelShuffle(self.subpatch),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(float(contract.get("regularization", {}).get("dropout", 0.2))),
        )
        mapping = self.derived["fine_to_coarse"]
        if len(mapping) != self.derived["num_classes"]:
            raise ValueError("AnySat fine_to_coarse differs from num_classes")
        self.heads = HierarchicalHeads.from_derived(channels, self.derived)

    def initialize_pretrained(self) -> dict[str, Any] | None:
        """Explicit initialization only; checkpoint restoration never downloads."""
        path = self.contract.get("anysat", {}).get("pretrained_path")
        if not path:
            return None
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        source = payload.get("state_dict", payload)
        expected = self.core.state_dict()
        missing = [
            key
            for key, value in expected.items()
            if key not in source or source[key].shape != value.shape
        ]
        if missing:
            raise ValueError(
                "Official AnySat checkpoint does not match the complete core: "
                + ", ".join(missing[:8])
            )
        self.core.load_state_dict({key: source[key] for key in expected}, strict=True)
        return {
            "path": str(path),
            "core_tensors": len(expected),
            "projectors": "new sensor projectors, randomly initialized",
        }

    def run_chunk(self, function, *args):
        if self.training and self.checkpointing and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def encode_sensor(
        self, values: Tensor, valid: Tensor, dates: Tensor, name: str, padded: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, times, channels, height, width = values.shape
        pad = (0, padded - width, 0, padded - height)
        sub = self.subpatch
        grid = padded // sub
        # Only one sensor at a time, then bounded location chunks through L-TAE.
        values = F.pixel_unshuffle(F.pad(values.flatten(0, 1), pad), sub)
        valid = F.pixel_unshuffle(F.pad(valid.flatten(0, 1).float(), pad), sub).bool()
        values = values.reshape(batch, times, channels * sub**2, -1).permute(0, 3, 1, 2)
        valid = valid.reshape(batch, times, channels * sub**2, -1).permute(0, 3, 1, 2)
        projected = []
        for sample in range(batch):
            chunks = []
            for start in range(0, grid * grid, self.projector_chunk):
                end = min(start + self.projector_chunk, grid * grid)
                chunks.append(
                    self.run_chunk(
                        self.projectors[name],
                        values[sample, start:end],
                        valid[sample, start:end],
                        dates[sample : sample + 1].expand(end - start, -1),
                    )
                )
            projected.append(torch.cat(chunks))
        x = torch.stack(projected)
        ok = valid.any(dim=(-1, -2))
        side = self.patch // sub
        patches = padded // self.patch
        x = x.reshape(batch, patches, side, patches, side, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch * patches**2, side**2, -1)
        ok = ok.reshape(batch, patches, side, patches, side)
        ok = ok.permute(0, 1, 3, 2, 4).reshape(batch * patches**2, side**2)
        tokens, dense = [], []
        for start in range(0, len(x), self.spatial_chunk):
            token, subs = self.run_chunk(
                self.core.spatial_encoder,
                x[start : start + self.spatial_chunk],
                ok[start : start + self.spatial_chunk],
                side,
                self.resolution * sub,
            )
            tokens.append(token)
            if name == self.dense_modality:
                dense.append(subs)
        patch_valid = ok.any(-1).reshape(batch, patches**2)
        tokens = torch.cat(tokens).reshape(batch, patches**2, -1)
        tokens = tokens.masked_fill(~patch_valid[..., None], 0)
        return tokens, patch_valid, torch.cat(dense) if dense else x.new_empty(0)

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        height, width = batch["static"].shape[-2:]
        if batch["dynamic"].shape[-2:] != (height, width):
            raise ValueError(
                "AnySat expects modalities aligned on the dataset target grid"
            )
        side = math.ceil(max(height, width) / self.patch)
        count = 1 + side**2 * len(self.specs)
        if count > self.max_tokens:
            raise ValueError(
                f"AnySat token budget exceeded: {count} > {self.max_tokens}; "
                "increase patch_size or reduce window size"
            )
        tokens, masks = [], []
        for spec in self.specs:
            values, valid, dates = prepare_sensor(
                batch, spec, self.derived[f"{spec['role']}_features"]
            )
            token, mask, subs = self.encode_sensor(
                values, valid, dates, spec["name"], side * self.patch
            )
            tokens.append(token)
            masks.append(mask)
            if spec["name"] == self.dense_modality:
                dense = subs
        fused = self.core(
            torch.stack(tokens, 1),
            torch.stack(masks, 1),
            side,
            self.resolution * self.patch / 10,
        )
        batch_size, _, dim = fused.shape
        subside = self.patch // self.subpatch
        fused = fused.reshape(-1, 1, dim).expand(-1, subside**2, -1)
        dense = torch.cat((fused, dense), -1)
        dense = dense.reshape(batch_size, side, side, subside, subside, dim * 2)
        dense = dense.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size, dim * 2, side * subside, side * subside
        )
        features = self.decoder(dense)[..., :height, :width]
        output = self.heads(features)
        output["valid_mask"] = batch["valid_mask"][:, None].bool()
        return output
