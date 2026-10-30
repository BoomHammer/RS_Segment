"""U-Net with a ResNet-34 encoder for the E05 spatial baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.architecture import HierarchicalHeads
from models.input_adapter import sanitize_batch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


def _residual_stage(
    in_channels: int, out_channels: int, blocks: int, stride: int
) -> nn.Sequential:
    layers = [BasicBlock(in_channels, out_channels, stride)]
    layers.extend(BasicBlock(out_channels, out_channels) for _ in range(blocks - 1))
    return nn.Sequential(*layers)


class ResNet34Encoder(nn.Module):
    channels = (64, 64, 128, 256, 512)

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = _residual_stage(64, 64, 3, 1)
        self.layer2 = _residual_stage(64, 128, 4, 2)
        self.layer3 = _residual_stage(128, 256, 6, 2)
        self.layer4 = _residual_stage(256, 512, 3, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        stem = self.stem(x)
        stage1 = self.layer1(self.pool(stem))
        stage2 = self.layer2(stage1)
        stage3 = self.layer3(stage2)
        stage4 = self.layer4(stage3)
        return stem, stage1, stage2, stage3, stage4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convolutions(torch.cat((x, skip), dim=1))


class UNetResNet34(nn.Module):
    """ResNet-34 encoder with a skip-connected U-Net decoder."""

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__()
        self.contract = contract
        self.derived = dict(contract["derived"])
        input_channels = 2 * (
            int(self.derived["dynamic_features_count"])
            + int(self.derived["static_features_count"])
        )
        self.encoder = ResNet34Encoder(input_channels)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        decoder_channels = int(contract.get("unet", {}).get("decoder_channels", 64))
        if decoder_channels < 1:
            raise ValueError("U-Net decoder_channels 必须为正数")
        self.refine = nn.Sequential(
            nn.Conv2d(64, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(contract.get("unet", {}).get("dropout", 0.1))),
        )
        self.heads = HierarchicalHeads.from_derived(decoder_channels, self.derived)

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
                    raise ValueError(f"{role} 通道数与 U-Net contract 不一致")
                continue
            names = list(provided[0])
            if any(list(item) != names for item in provided):
                raise ValueError("U-Net 批内特征顺序必须一致")
            if sorted(names) != sorted(expected):
                raise ValueError(f"U-Net {role} 特征与 contract 不一致")
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
        dynamic, presence, static, static_presence = self._ordered_inputs(clean)
        presence = presence & clean["dynamic_time_mask"][:, :, None, None, None]
        temporal_mean = (dynamic * presence).sum(1) / presence.sum(1).clamp_min(1)
        inputs = torch.cat(
            (
                temporal_mean,
                presence.any(1).to(dynamic.dtype),
                static,
                static_presence.to(static.dtype),
            ),
            dim=1,
        )
        height, width = inputs.shape[-2:]
        padding = (0, (-width) % 32, 0, (-height) % 32)
        stem, stage1, stage2, stage3, stage4 = self.encoder(F.pad(inputs, padding))
        decoded = self.decoder4(stage4, stage3)
        decoded = self.decoder3(decoded, stage2)
        decoded = self.decoder2(decoded, stage1)
        decoded = self.decoder1(decoded, stem)
        decoded = F.interpolate(
            self.refine(decoded),
            size=(height + padding[3], width + padding[1]),
            mode="bilinear",
            align_corners=False,
        )[..., :height, :width]
        with torch.autocast(device_type=decoded.device.type, enabled=False):
            output = self.heads(decoded.float())
        output["valid_mask"] = clean["valid_mask"][:, None]
        return output
