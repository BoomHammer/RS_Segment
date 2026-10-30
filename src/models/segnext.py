"""SegNeXt-T adapted to the project's spatial baseline input contract.

MSCAN-T and LightHamHead follow the official SegNeXt configuration. The NMF
basis initialization is persisted instead of redrawn on every forward pass so
overlap-consistency training and sliding-window inference stay deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.architecture import DropPath, HierarchicalHeads
from models.input_adapter import sanitize_batch


class ConvMLP(nn.Module):
    def __init__(self, channels: int, ratio: int, dropout: float) -> None:
        super().__init__()
        hidden = channels * ratio
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden, bias=True
        )
        self.activation = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.depthwise(self.fc1(x)))
        return self.dropout(self.fc2(self.dropout(x)))


class MultiScaleConvolutionalAttention(nn.Module):
    """Depthwise 5x5 plus separable 7/11/21 strip-convolution branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    (1, kernel),
                    padding=(0, kernel // 2),
                    groups=channels,
                ),
                nn.Conv2d(
                    channels,
                    channels,
                    (kernel, 1),
                    padding=(kernel // 2, 0),
                    groups=channels,
                ),
            )
            for kernel in (7, 11, 21)
        )
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        attention = self.local(x)
        attention = attention + sum(branch(attention) for branch in self.branches)
        return self.mix(attention) * residual


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection_in = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.gate = MultiScaleConvolutionalAttention(channels)
        self.projection_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.gate(self.activation(self.projection_in(x)))
        return self.projection_out(x) + residual


class MSCABlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attention = SpatialAttention(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = ConvMLP(channels, ratio, dropout)
        self.drop_path = DropPath(drop_path)
        self.layer_scale1 = nn.Parameter(torch.full((channels, 1, 1), 1e-2))
        self.layer_scale2 = nn.Parameter(torch.full((channels, 1, 1), 1e-2))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.layer_scale1 * self.attention(self.norm1(x)))
        return x + self.drop_path(self.layer_scale2 * self.mlp(self.norm2(x)))


class Stem(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )


class OverlapPatchEmbed(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )


class MSCANTiny(nn.Module):
    """Official MSCAN-T dimensions, ratios and stage depths."""

    channels = (32, 64, 160, 256)
    depths = (3, 3, 5, 2)
    ratios = (8, 8, 4, 4)

    def __init__(
        self,
        in_channels: int,
        *,
        dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                Stem(in_channels, self.channels[0]),
                *(
                    OverlapPatchEmbed(source, target)
                    for source, target in zip(
                        self.channels[:-1], self.channels[1:], strict=True
                    )
                ),
            ]
        )
        rates = torch.linspace(0, drop_path, sum(self.depths)).tolist()
        offset = 0
        self.stages = nn.ModuleList()
        for channels, depth, ratio in zip(
            self.channels, self.depths, self.ratios, strict=True
        ):
            self.stages.append(
                nn.Sequential(
                    *(
                        MSCABlock(channels, ratio, dropout, rates[offset + block_index])
                        for block_index in range(depth)
                    )
                )
            )
            offset += depth
        self.norms = nn.ModuleList(nn.LayerNorm(value) for value in self.channels)

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = []
        for embedding, blocks, norm in zip(
            self.embeddings, self.stages, self.norms, strict=True
        ):
            x = embedding(x)
            height, width = x.shape[-2:]
            x = blocks(x)
            x = (
                norm(x.flatten(2).transpose(1, 2))
                .transpose(1, 2)
                .reshape(x.shape[0], x.shape[1], height, width)
            )
            outputs.append(x)
        return outputs


class NMF2D(nn.Module):
    """Deterministic non-negative matrix factorization used by Hamburger."""

    def __init__(
        self,
        channels: int,
        rank: int = 16,
        train_steps: int = 6,
        eval_steps: int = 7,
    ) -> None:
        super().__init__()
        if channels < 1 or rank < 1 or min(train_steps, eval_steps) < 1:
            raise ValueError("SegNeXt NMF channels/rank/steps 必须为正数")
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        generator = torch.Generator().manual_seed(0)
        bases = torch.rand(1, channels, rank, generator=generator)
        self.register_buffer("initial_bases", F.normalize(bases, dim=1))

    @staticmethod
    def _step(x: Tensor, bases: Tensor, coefficients: Tensor) -> tuple[Tensor, Tensor]:
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coefficients.bmm(bases.transpose(1, 2).bmm(bases))
        coefficients = coefficients * numerator / denominator.clamp_min(1e-6)
        numerator = torch.bmm(x, coefficients)
        denominator = bases.bmm(coefficients.transpose(1, 2).bmm(coefficients))
        bases = bases * numerator / denominator.clamp_min(1e-6)
        return bases, coefficients

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        batch, channels, height, width = x.shape
        values = x.float().reshape(batch, channels, height * width)
        bases = self.initial_bases.expand(batch, -1, -1).to(values)
        coefficients = torch.bmm(values.transpose(1, 2), bases).softmax(dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coefficients = self._step(values, bases, coefficients)
        numerator = torch.bmm(values.transpose(1, 2), bases)
        denominator = coefficients.bmm(bases.transpose(1, 2).bmm(bases))
        coefficients = coefficients * numerator / denominator.clamp_min(1e-6)
        result = torch.bmm(bases, coefficients.transpose(1, 2))
        return result.reshape(batch, channels, height, width).to(dtype)


class Hamburger(nn.Module):
    def __init__(
        self, channels: int, rank: int, train_steps: int, eval_steps: int
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(channels, channels, 1)
        self.nmf = NMF2D(channels, rank, train_steps, eval_steps)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.GroupNorm(32, channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        filling = self.output(self.nmf(F.relu(self.input(x))))
        return F.relu(x + filling)


class LightHamHead(nn.Module):
    """Fuse MSCAN stages 2-4 and apply the SegNeXt Hamburger decoder."""

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 256,
        rank: int = 16,
        train_steps: int = 6,
        eval_steps: int = 7,
    ) -> None:
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(sum(in_channels), channels, 1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )
        self.hamburger = Hamburger(channels, rank, train_steps, eval_steps)
        self.align = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        target_size = features[0].shape[-2:]
        resized = [features[0]]
        resized.extend(
            F.interpolate(
                feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for feature in features[1:]
        )
        return self.align(self.hamburger(self.squeeze(torch.cat(resized, dim=1))))


class SegNeXtTiny(nn.Module):
    """MSCAN-T + LightHamHead with E03-compatible temporal mean inputs."""

    def __init__(self, contract: dict[str, Any]) -> None:
        super().__init__()
        self.contract = contract
        self.derived = dict(contract["derived"])
        settings = dict(contract.get("segnext", {}))
        input_channels = 2 * (
            int(self.derived["dynamic_features_count"])
            + int(self.derived["static_features_count"])
        )
        self.encoder = MSCANTiny(
            input_channels,
            dropout=float(settings.get("dropout", 0.0)),
            drop_path=float(settings.get("drop_path", 0.1)),
        )
        decoder_channels = int(settings.get("decoder_channels", 256))
        if decoder_channels < 32 or decoder_channels % 32:
            raise ValueError("SegNeXt decoder_channels 必须是至少 32 的 32 倍数")
        self.decoder = LightHamHead(
            MSCANTiny.channels[1:],
            channels=decoder_channels,
            rank=int(settings.get("nmf_rank", 16)),
            train_steps=int(settings.get("nmf_train_steps", 6)),
            eval_steps=int(settings.get("nmf_eval_steps", 7)),
        )
        self.dropout = nn.Dropout2d(float(settings.get("decoder_dropout", 0.1)))
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
                    raise ValueError(f"{role} 通道数与 SegNeXt contract 不一致")
                continue
            names = list(provided[0])
            if any(list(item) != names for item in provided):
                raise ValueError("SegNeXt 批内特征顺序必须一致")
            if sorted(names) != sorted(expected):
                raise ValueError(f"SegNeXt {role} 特征与 contract 不一致")
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
        features = self.encoder(F.pad(inputs, padding))
        decoded = self.dropout(self.decoder(features[1:]))
        decoded = F.interpolate(
            decoded,
            size=(height + padding[3], width + padding[1]),
            mode="bilinear",
            align_corners=False,
        )[..., :height, :width]
        with torch.autocast(device_type=decoded.device.type, enabled=False):
            output = self.heads(decoded.float())
        output["valid_mask"] = clean["valid_mask"][:, None]
        return output
