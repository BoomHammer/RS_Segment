"""Pretrained MiT-B1 and the official U-TAE encoder/decoder in a hybrid model."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import SegformerConfig, SegformerModel
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

from models.architecture import GatedScaleFusion, HierarchicalHeads
from models.input_adapter import sanitize_batch
from models.utae.utae import UTAE


class MaskedUTAE(UTAE):
    """Official U-TAE blocks with chunked encoding and authoritative masks.

    Keep the full resolution encoder, bottleneck L-TAE, grouped attention
    skips and every upsampling block. Checkpoint frame chunks to bound memory.
    Whole padded dates are removed per sample before temporal normalization.
    """

    def __init__(self, input_dim: int, chunk_size: int = 2):
        super().__init__(input_dim=input_dim, encoder=True, pad_value=None)
        if chunk_size < 1:
            raise ValueError("frame_chunk_size 必须为正整数")
        self.chunk_size = chunk_size
        # The hybrid uses feature maps, so the original final classifier is unused.
        self.out_conv = nn.Identity()

    def _encode(self, block, sequence):
        batch, time = sequence.shape[:2]
        frames = sequence.flatten(0, 1)
        outputs = []
        for chunk in frames.split(self.chunk_size):
            if self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(block, chunk, use_reentrant=False))
            else:
                outputs.append(block(chunk))
        return torch.cat(outputs).unflatten(0, (batch, time))

    @staticmethod
    def _aggregate(features, attention, present):
        heads, batch, time = attention.shape[:3]
        height, width = features.shape[-2:]
        weights = F.interpolate(
            attention.flatten(0, 1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(heads, batch, time, height, width)
        valid = (
            F.adaptive_max_pool2d(present.flatten(0, 1).float(), (height, width))
            .unflatten(0, (batch, time))
            .squeeze(2)
            .bool()
        )
        weights = weights * valid[None]
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        # Accumulate channel groups separately instead of materializing H*B*T*C.
        return torch.cat(
            [
                (part * weights[i, :, :, None].to(part.dtype)).sum(dim=1)
                for i, part in enumerate(features.chunk(heads, dim=2))
            ],
            dim=1,
        )

    def forward(self, sequence, positions, present):
        per_sample = []
        for sample in range(sequence.shape[0]):
            active = present[sample].flatten(1).any(dim=1)
            if not active.any():
                height, width = sequence.shape[-2:]
                per_sample.append(
                    [
                        sequence.new_zeros(1, channels, height // 2**i, width // 2**i)
                        for i, channels in enumerate(self.decoder_widths)
                    ]
                )
                continue
            values = sequence[sample : sample + 1, active]
            valid = present[sample : sample + 1, active]
            dates = positions[sample : sample + 1, active]
            features = [self._encode(self.in_conv, values)]
            for block in self.down_blocks:
                features.append(self._encode(block, features[-1]))
            coarse_valid = (
                F.adaptive_max_pool2d(
                    valid.flatten(0, 1).float(), features[-1].shape[-2:]
                )
                .unflatten(0, (1, int(active.sum())))
                .squeeze(2)
                .bool()
            )
            output, attention = self.temporal_encoder(
                features[-1], batch_positions=dates, pad_mask=~coarse_valid
            )
            output = output * coarse_valid.any(dim=1)[:, None]
            maps = [output]
            for i, block in enumerate(self.up_blocks):
                skip = self._aggregate(features[-i - 2], attention, valid)
                output = block(output, skip)
                maps.append(output)
            per_sample.append(list(reversed(maps)))
        return [torch.cat(items) for items in zip(*per_sample, strict=True)]


class PretrainedSegFormerUTAE(nn.Module):
    """MiT attention stages + full U-TAE + SegFormer MLP decoder + hierarchy.

    Construction never accesses the network. Initial training explicitly loads
    pretrained MiT weights; inference restores the self-contained checkpoint.
    """

    def __init__(self, contract):
        super().__init__()
        self.contract = contract
        derived = contract["derived"]
        settings = contract.get("pretrained", {})
        self.freeze_stages = int(settings.get("freeze_stages", 2))
        if not 0 <= self.freeze_stages <= 4:
            raise ValueError("freeze_stages 必须位于 [0, 4]")
        self.static_stem = nn.Conv2d(2 * derived["static_features_count"], 3, 1)
        self.static_encoder = SegformerModel(
            SegformerConfig(
                depths=[2, 2, 2, 2],
                hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                drop_path_rate=float(
                    contract.get("regularization", {}).get("drop_path", 0.1)
                ),
            )
        )
        for group in (
            self.static_encoder.encoder.patch_embeddings,
            self.static_encoder.encoder.block,
            self.static_encoder.encoder.layer_norm,
        ):
            for stage in list(group)[: self.freeze_stages]:
                stage.requires_grad_(False)
        self.dynamic_encoder = MaskedUTAE(
            2 * derived["dynamic_features_count"],
            chunk_size=int(contract.get("temporal", {}).get("frame_chunk_size", 2)),
        )
        widths = self.dynamic_encoder.decoder_widths
        self.static_projections = nn.ModuleList(
            nn.Conv2d(source, target, 1)
            for source, target in zip([64, 128, 320, 512], widths, strict=True)
        )
        self.fusions = nn.ModuleList(GatedScaleFusion(width) for width in widths)
        decoder_width = int(contract.get("fusion", {}).get("output_channels", 64))
        self.decoder = SegformerDecodeHead(
            SegformerConfig(
                hidden_sizes=list(widths),
                decoder_hidden_size=decoder_width,
                classifier_dropout_prob=float(
                    contract.get("regularization", {}).get("dropout", 0.2)
                ),
            )
        )
        self.decoder.classifier = nn.Identity()
        self.heads = HierarchicalHeads(
            decoder_width, derived["num_coarse_classes"], derived["fine_to_coarse"]
        )

    def initialize_pretrained(self):
        folder = Path(self.contract["pretrained"]["path"])
        if not folder.is_dir():
            raise FileNotFoundError(
                f"MiT 预训练权重不存在: {folder}；先运行 scripts/download_mit.py"
            )
        provenance_path = folder / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else {}
        )
        expected_revision = self.contract["pretrained"].get("revision")
        if expected_revision and provenance.get("revision") != expected_revision:
            raise ValueError("本地 MiT 权重 revision 与配置不一致，请使用指定版本下载")
        encoder, information = SegformerModel.from_pretrained(
            folder, local_files_only=True, output_loading_info=True
        )
        if information["missing_keys"] or information.get("mismatched_keys"):
            raise ValueError(f"MiT 预训练编码器权重不完整: {information}")
        self.static_encoder.load_state_dict(encoder.state_dict(), strict=True)
        return {
            "path": str(folder.resolve()),
            "provenance": provenance,
            "missing_keys": [],
            "unexpected_keys": information["unexpected_keys"],
        }

    def train(self, mode=True):
        super().train(mode)
        # Keep stochastic depth disabled in frozen pretrained stages.
        for stage in list(self.static_encoder.encoder.block)[: self.freeze_stages]:
            stage.eval()
        return self

    def forward(self, batch):
        clean = sanitize_batch(batch)
        dynamic = torch.cat(
            [clean["dynamic"], clean["dynamic_value_mask"].float()], dim=2
        )
        present = clean["dynamic_value_mask"].any(dim=2, keepdim=True)
        present = present & clean["dynamic_time_mask"][:, :, None, None, None]
        static = torch.cat([clean["static"], clean["static_value_mask"].float()], dim=1)
        height, width = static.shape[-2:]
        # U-TAE transposed convolutions require matching sizes at all four scales.
        bottom, right = (-height) % 32, (-width) % 32
        static = F.pad(static, (0, right, 0, bottom))
        dynamic = F.pad(dynamic, (0, right, 0, bottom))
        present = F.pad(present, (0, right, 0, bottom))
        stem = self.static_stem(static)

        def spatial(value):
            return tuple(
                self.static_encoder(value, output_hidden_states=True).hidden_states
            )

        static_maps = (
            checkpoint(spatial, stem, use_reentrant=False)
            if self.training and torch.is_grad_enabled()
            else spatial(stem)
        )
        dynamic_maps = self.dynamic_encoder(
            dynamic, clean["time_encoding"][..., 0] * 365.25, present
        )
        fused = [
            fusion(
                F.interpolate(
                    projection(spatial_map),
                    size=temporal_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                temporal_map,
            )
            for projection, fusion, spatial_map, temporal_map in zip(
                self.static_projections,
                self.fusions,
                static_maps,
                dynamic_maps,
                strict=True,
            )
        ]
        # Classify at final resolution so the conditional hierarchy stays normalized.
        features = self.decoder(fused)[..., :height, :width]
        with torch.autocast(device_type=features.device.type, enabled=False):
            output = self.heads(features.float())
        output["valid_mask"] = clean["valid_mask"][:, None]
        return output
