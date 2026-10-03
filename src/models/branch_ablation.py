"""Replace one encoder while retaining the lightweight fusion and decoder."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import SegformerConfig, SegformerModel

from models.architecture import MiTB1Encoder, SegFormerUtae
from models.input_adapter import sanitize_batch
from models.pretrained_utae import MaskedUTAE


class BranchAblation(SegFormerUtae):
    """Public branch interfaces are four maps at strides 4, 8, 16 and 32."""

    def __init__(self, contract):
        # Construct common modules in exactly the same RNG order as the control.
        super().__init__(contract, contract["derived"]["fine_to_coarse"])
        self.contract = contract
        self.replace_dynamic = (
            contract["architecture"] == "segformer_utae_dynamic_ablation"
        )
        self.freeze_stages = int(contract.get("pretrained", {}).get("freeze_stages", 2))
        if not 0 <= self.freeze_stages <= 4:
            raise ValueError("freeze_stages must be between zero and four")
        # Replacement construction must not advance the common training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1042)
            if self.replace_dynamic:
                self.dynamic_encoder = MaskedUTAE(
                    2 * contract["derived"]["dynamic_features_count"],
                    chunk_size=int(contract["temporal"].get("frame_chunk_size", 2)),
                    normalization="group",
                )
                self.dynamic_projections = nn.ModuleList(
                    nn.Conv2d(source, target, 1)
                    for source, target in zip(
                        self.dynamic_encoder.decoder_widths,
                        MiTB1Encoder.channels,
                        strict=True,
                    )
                )
            else:
                self.static_stem = nn.Conv2d(
                    2 * contract["derived"]["static_features_count"], 3, 1
                )
                self.static_encoder = SegformerModel(
                    SegformerConfig(
                        depths=[2, 2, 2, 2],
                        hidden_sizes=list(MiTB1Encoder.channels),
                        num_attention_heads=[1, 2, 5, 8],
                        sr_ratios=[8, 4, 2, 1],
                        drop_path_rate=float(contract["regularization"]["drop_path"]),
                    )
                )
                for group in (
                    self.static_encoder.encoder.patch_embeddings,
                    self.static_encoder.encoder.block,
                    self.static_encoder.encoder.layer_norm,
                ):
                    for stage in list(group)[: self.freeze_stages]:
                        stage.requires_grad_(False)

    def initialize_pretrained(self):
        if self.replace_dynamic:
            return None
        # Reuse the existing local-only loader and revision checks.
        from models.pretrained_utae import PretrainedSegFormerUTAE

        with torch.random.fork_rng(devices=[]):
            return PretrainedSegFormerUTAE.initialize_pretrained(self)

    def train(self, mode=True):
        super().train(mode)
        if not self.replace_dynamic:
            for stage in list(self.static_encoder.encoder.block)[: self.freeze_stages]:
                stage.eval()
        return self

    def forward(self, batch):
        clean = sanitize_batch(batch)
        height, width = clean["static"].shape[-2:]
        if self.replace_dynamic:
            static_maps = self.static_encoder(self.adapter.static(clean["static"]))
            device_type = clean["dynamic"].device.type
            dtype = (
                torch.get_autocast_dtype(device_type)
                if torch.is_autocast_enabled(device_type)
                else clean["dynamic"].dtype
            )
            dynamic = torch.cat(
                (clean["dynamic"].to(dtype), clean["dynamic_value_mask"].to(dtype)),
                dim=2,
            )
            present = clean["dynamic_value_mask"].any(dim=2, keepdim=True)
            present = present & clean["dynamic_time_mask"][:, :, None, None, None]
            padding = (0, (-width) % 32, 0, (-height) % 32)
            maps = self.dynamic_encoder(
                F.pad(dynamic, padding),
                clean["time_encoding"][..., 0] * 365.25,
                F.pad(present, padding),
            )
            dynamic_maps = []
            for i, (value, projection, reference) in enumerate(
                zip(maps, self.dynamic_projections, static_maps, strict=True)
            ):
                # Crop padding before adapting U-TAE strides 1/2/4/8 to 4/8/16/32.
                value = value[
                    ..., : (height + 2**i - 1) // 2**i, : (width + 2**i - 1) // 2**i
                ]
                value = F.adaptive_avg_pool2d(value, reference.shape[-2:])
                dynamic_maps.append(projection(value))
        else:
            temporal, _ = self.adapter.temporal(
                clean["dynamic"],
                clean["dynamic_mask"],
                clean["dynamic_time_mask"],
                clean["time_encoding"],
                clean["dynamic_value_mask"],
            )
            dynamic_maps = self.dynamic_encoder(temporal)
            stem = self.static_stem(
                torch.cat((clean["static"], clean["static_value_mask"].float()), dim=1)
            )

            def encode(value):
                return tuple(
                    self.static_encoder(value, output_hidden_states=True).hidden_states
                )

            static_maps = (
                checkpoint(encode, stem, use_reentrant=False)
                if self.training and torch.is_grad_enabled()
                else encode(stem)
            )
        for static, dynamic in zip(static_maps, dynamic_maps, strict=True):
            if static.shape != dynamic.shape:
                raise ValueError("Branch maps do not match the shared fusion interface")
        fused = [
            fusion(static, dynamic)
            for fusion, static, dynamic in zip(
                self.fusions, static_maps, dynamic_maps, strict=True
            )
        ]
        # Identical head placement/interpolation to the lightweight control.
        output = self.heads(self.decoder(fused))
        for key in ("coarse_logits", "fine_logits"):
            output[key] = F.interpolate(
                output[key], size=(height, width), mode="bilinear", align_corners=False
            )
        output["coarse_probability"] = output["coarse_logits"].softmax(dim=1)
        output["fine_probability"] = output["fine_logits"].exp()
        output["valid_mask"] = clean["valid_mask"][:, None]
        return output
