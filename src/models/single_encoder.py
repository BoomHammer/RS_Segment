"""Single-encoder baselines retaining both dynamic and static input sources."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import SegformerConfig, SegformerModel

from models.architecture import HierarchicalHeads, SharedDecoder
from models.input_adapter import sanitize_batch
from models.pretrained_utae import MaskedUTAE


class SingleEncoder(nn.Module):
    """U-TAE uses dates; SegFormer uses masked temporal means, never flattened dates."""

    def __init__(self, contract):
        super().__init__()
        derived = contract["derived"]
        self.is_utae = contract["architecture"] == "utae"
        channels = 2 * (
            derived["dynamic_features_count"] + derived["static_features_count"]
        )
        if self.is_utae:
            self.encoder = MaskedUTAE(
                channels,
                chunk_size=int(contract.get("temporal", {}).get("frame_chunk_size", 2)),
                normalization="group",
            )
            widths = self.encoder.decoder_widths
        else:
            widths = [64, 128, 320, 512]
            self.encoder = SegformerModel(
                SegformerConfig(
                    num_channels=channels,
                    depths=[2, 2, 2, 2],
                    hidden_sizes=widths,
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                )
            )
        output_channels = int(contract.get("fusion", {}).get("output_channels", 96))
        self.decoder = SharedDecoder(widths, output_channels, dropout=0.2)
        self.heads = HierarchicalHeads.from_derived(output_channels, derived)

    def forward(self, batch):
        clean = sanitize_batch(batch)
        dynamic = clean["dynamic"]
        mask = (
            clean["dynamic_value_mask"]
            & clean["dynamic_time_mask"][:, :, None, None, None]
        )
        static = torch.cat((clean["static"], clean["static_value_mask"].float()), 1)
        height, width = static.shape[-2:]
        padding = (0, (-width) % 32, 0, (-height) % 32)
        if self.is_utae:
            sequence = torch.cat(
                (
                    dynamic * mask,
                    mask.to(dynamic.dtype),
                    static[:, None].expand(-1, dynamic.shape[1], -1, -1, -1),
                ),
                2,
            )
            maps = self.encoder(
                F.pad(sequence, padding),
                clean["time_encoding"][..., 0] * 365.25,
                F.pad(mask.any(2, keepdim=True), padding),
            )
        else:
            mean = (dynamic * mask).sum(1) / mask.sum(1).clamp_min(1)
            inputs = F.pad(torch.cat((mean, mask.any(1).float(), static), 1), padding)

            def encode(value):
                return tuple(
                    self.encoder(value, output_hidden_states=True).hidden_states
                )

            maps = (
                checkpoint(encode, inputs, use_reentrant=False)
                if self.training
                else encode(inputs)
            )
        features = F.interpolate(
            self.decoder(maps),
            size=(height + padding[3], width + padding[1]),
            mode="bilinear",
            align_corners=False,
        )[..., :height, :width]
        with torch.autocast(device_type=features.device.type, enabled=False):
            output = self.heads(features.float())
        output["valid_mask"] = clean["valid_mask"][:, None]
        return output
