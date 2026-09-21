"""Memory-conscious dual-branch SegFormer + U-TAE segmentation model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.input_adapter import SegFormerUtaeInputAdapter


class DropPath(nn.Module):
    """Stochastic depth for residual feature blocks."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop_path 必须位于 [0, 1)")
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_probability
        return x * mask / keep_probability


class ConvNormAct(nn.Sequential):
    """Overlap-patch projection used by the lightweight MiT stages."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )


class SpatialStage(nn.Module):
    """A compact spatial transformer substitute suitable for 24 GB GPUs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.projection = ConvNormAct(in_channels, out_channels, 2)
        self.drop_path = DropPath(drop_path)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        out_channels, out_channels, 3, padding=1, groups=out_channels
                    ),
                    nn.GroupNorm(min(8, out_channels), out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 1),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        for block in self.blocks:
            x = x + self.drop_path(block(x))
        return x


class MiTB1Encoder(nn.Module):
    """Four-scale MiT-B1-style encoder with convolutional efficient mixing."""

    channels = (64, 128, 320, 512)

    def __init__(self, in_channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, 32, 2)
        self.stages = nn.ModuleList(
            [
                SpatialStage(32, 64, drop_path=drop_path),
                SpatialStage(64, 128, drop_path=drop_path),
                SpatialStage(128, 320, drop_path=drop_path),
                SpatialStage(320, 512, drop_path=drop_path),
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class DynamicMultiScaleEncoder(nn.Module):
    """U-TAE-style temporal encoder followed by spatial multi-scale stages."""

    def __init__(self, channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.stem = ConvNormAct(channels, 32, 2)
        self.stages = nn.ModuleList(
            [
                SpatialStage(32, 64, drop_path=drop_path),
                SpatialStage(64, 128, drop_path=drop_path),
                SpatialStage(128, 320, drop_path=drop_path),
                SpatialStage(320, 512, drop_path=drop_path),
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class GatedScaleFusion(nn.Module):
    """Fuse matching static/dynamic scales with a learned per-pixel gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.static_projection = nn.Conv2d(channels, channels, 1)
        self.dynamic_projection = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())

    def forward(self, static: Tensor, dynamic: Tensor) -> Tensor:
        pair = torch.cat((static, dynamic), dim=1)
        gate = self.gate(pair)
        return gate * self.static_projection(static) + (
            1.0 - gate
        ) * self.dynamic_projection(dynamic)


class SharedDecoder(nn.Module):
    """Fuse four scales at the finest encoder resolution."""

    def __init__(
        self, channels: Sequence[int], output_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(c, output_channels, 1) for c in channels
        )
        self.refine = nn.Sequential(
            nn.Conv2d(output_channels * len(channels), output_channels, 3, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        target_size = features[0].shape[-2:]
        upsampled = [self.projections[0](features[0])]
        upsampled.extend(
            F.interpolate(
                self.projections[i](feature),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for i, feature in enumerate(features[1:], 1)
        )
        return self.refine(torch.cat(upsampled, dim=1))


class HierarchicalHeads(nn.Module):
    """Coarse head and conditional fine experts, combined without argmax routing."""

    def __init__(
        self, channels: int, num_coarse: int, fine_to_coarse: Sequence[int]
    ) -> None:
        super().__init__()
        self.num_coarse = num_coarse
        self.fine_to_coarse = tuple(int(value) for value in fine_to_coarse)
        if len(self.fine_to_coarse) < 1 or min(self.fine_to_coarse) < 0:
            raise ValueError("fine_to_coarse 必须是非空的 0-based 映射")
        self.coarse = nn.Conv2d(channels, num_coarse, 1)
        expert_sizes = [self.fine_to_coarse.count(index) for index in range(num_coarse)]
        if not all(expert_sizes):
            raise ValueError("每个大类必须至少包含一个小类")
        self.experts = nn.ModuleList(
            nn.Conv2d(channels, size, 1) for size in expert_sizes
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        coarse_logits = self.coarse(features)
        coarse_log_probability = coarse_logits.log_softmax(dim=1)
        expert_log_probability = torch.full(
            (features.shape[0], len(self.fine_to_coarse), *coarse_logits.shape[-2:]),
            torch.finfo(coarse_log_probability.dtype).min,
            dtype=coarse_log_probability.dtype,
            device=features.device,
        )
        for coarse_index, expert in enumerate(self.experts):
            fine_indices = [
                index
                for index, parent in enumerate(self.fine_to_coarse)
                if parent == coarse_index
            ]
            expert_log_probability[:, fine_indices] = (
                expert(features).log_softmax(dim=1)
                + coarse_log_probability[:, coarse_index : coarse_index + 1]
            )
        return {
            "coarse_logits": coarse_logits,
            "fine_logits": expert_log_probability,
            "coarse_probability": coarse_logits.softmax(dim=1),
            "fine_probability": expert_log_probability.exp(),
        }


class SegFormerUtae(nn.Module):
    """End-to-end dual-branch model consuming the stage-2 batch contract."""

    def __init__(self, contract: dict[str, Any], fine_to_coarse: Sequence[int]) -> None:
        super().__init__()
        derived = dict(contract.get("derived", contract))
        temporal = dict(contract.get("temporal", {}))
        static = dict(contract.get("static", {}))
        fusion = dict(contract.get("fusion", {}))
        regularization = dict(contract.get("regularization", {}))
        dropout = float(regularization.get("dropout", 0.0))
        drop_path = float(regularization.get("drop_path", 0.0))
        self.adapter = SegFormerUtaeInputAdapter.from_contract(contract)
        self.static_encoder = MiTB1Encoder(
            int(static.get("projection_channels", 32)), drop_path=drop_path
        )
        self.dynamic_encoder = DynamicMultiScaleEncoder(
            int(temporal.get("projection_channels", 64)), drop_path=drop_path
        )
        self.fusions = nn.ModuleList(GatedScaleFusion(c) for c in MiTB1Encoder.channels)
        decoder_channels = int(fusion.get("output_channels", 96))
        self.decoder = SharedDecoder(
            MiTB1Encoder.channels, decoder_channels, dropout=dropout
        )
        self.heads = HierarchicalHeads(
            decoder_channels, max(fine_to_coarse) + 1, fine_to_coarse
        )
        if len(fine_to_coarse) != int(derived["num_classes"]):
            raise ValueError("fine_to_coarse 长度必须等于 contract 的类别数")

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> SegFormerUtae:
        if contract.get("architecture") == "anysat":
            from models.anysat import AnySatSegmentation

            return AnySatSegmentation(contract)
        if contract.get("architecture") == "maestro_s":
            from models.maestro import MaestroS

            return MaestroS(contract)
        if contract.get("architecture") in {
            "segformer_utae_dynamic_ablation",
            "segformer_utae_static_ablation",
        }:
            from models.branch_ablation import BranchAblation

            return BranchAblation(contract)
        if contract.get("architecture") == "segformer_utae_pretrained":
            from models.pretrained_utae import PretrainedSegFormerUTAE

            return PretrainedSegFormerUTAE(contract)
        mapping = contract.get("fine_to_coarse")
        if mapping is None:
            mapping = dict(contract.get("derived", {})).get("fine_to_coarse")
        if mapping is None:
            raise ValueError("contract 缺少 fine_to_coarse 层级映射")
        return cls(contract, mapping)

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        adapted = self.adapter(batch)
        static_features = self.static_encoder(adapted["static_features"])
        dynamic_features = self.dynamic_encoder(adapted["dynamic_features"])
        fused = [
            fusion(static, dynamic)
            for fusion, static, dynamic in zip(
                self.fusions, static_features, dynamic_features, strict=True
            )
        ]
        output = self.heads(self.decoder(fused))
        target_size = batch["static"].shape[-2:]
        for key in ("coarse_logits", "fine_logits"):
            output[key] = F.interpolate(
                output[key], size=target_size, mode="bilinear", align_corners=False
            )
        output["coarse_probability"] = output["coarse_logits"].softmax(dim=1)
        output["fine_probability"] = output["fine_logits"].exp()
        output["valid_mask"] = F.interpolate(
            adapted["valid_mask"].float(), size=target_size, mode="nearest"
        ).bool()
        return output
