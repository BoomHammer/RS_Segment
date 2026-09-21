"""Mask-aware execution of the pinned official AnySat transformer layers.

Parameter names/shapes match AnyModule and TransformerMulti. SDPA replaces
explicit attention matrices; iRPE and the official learned cross-query remain.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.anysat_vendor.pos_embed import get_2d_sincos_pos_embed_with_scale
from models.anysat_vendor.utils_ViT import Block, BlockTransformer, CrossBlockMulti


def position(dim: int, side: int, spacing: float, reference: Tensor) -> Tensor:
    return get_2d_sincos_pos_embed_with_scale(dim, side, spacing, cls_token=True).to(
        device=reference.device, dtype=reference.dtype
    )


def self_block(block: nn.Module, x: Tensor, valid: Tensor, side: int) -> Tensor:
    attention = block.attn
    batch, count, dim = x.shape
    q, k, v = (
        attention.qkv(block.norm1(x))
        .reshape(batch, count, 3, attention.num_heads, dim // attention.num_heads)
        .permute(2, 0, 3, 1, 4)
        .unbind(0)
    )
    if hasattr(attention, "rpe_k"):
        bias = attention.rpe_k(q * attention.scale, height=side, width=side)
        mask = bias.masked_fill(~valid[:, None, None], float("-inf"))
    else:
        q, k = attention.q_norm(q), attention.k_norm(k)
        mask = valid[:, None, None]
    value = (
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=attention.attn_drop.p if block.training else 0.0,
        )
        .transpose(1, 2)
        .reshape(batch, count, dim)
    )
    value = attention.proj_drop(attention.proj(value))
    if isinstance(block, Block):
        x = x + block.drop_path1(block.ls1(value))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    else:
        x = x + block.drop_path(value)
        x = x + block.drop_path(block.mlp(block.norm2(x)))
    return x.masked_fill(~valid[..., None], 0)


class SpatialEncoder(nn.Module):
    """Shared official iRPE sub-patch transformer with bounded patch batches."""

    def __init__(self, dim: int, heads: int, depth: int, drop: float) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.predictor_blocks = nn.ModuleList(
            BlockTransformer(dim, heads, qkv_bias=True, drop_path=drop)
            for _ in range(depth)
        )
        self.predictor_norm = nn.LayerNorm(dim)

    def forward(
        self, x: Tensor, valid: Tensor, side: int, spacing: float
    ) -> tuple[Tensor, Tensor]:
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), 1)
        valid = F.pad(valid, (1, 0), value=True)
        x = x + position(x.shape[-1], side, spacing, x)
        for block in self.predictor_blocks:
            x = self_block(block, x, valid, side)
        x = self.predictor_norm(x)
        return x[:, 0], x[:, 1:].masked_fill(~valid[:, 1:, None], 0)


class AnySatCore(nn.Module):
    """Official shared spatial encoder, combiner blocks and cross-attention."""

    def __init__(
        self, dim: int, heads: int, depth: int, drop: float, checkpointing: bool
    ) -> None:
        super().__init__()
        self.spatial_encoder = SpatialEncoder(dim, heads, depth, drop)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    heads,
                    qkv_bias=True,
                    norm_layer=norm,
                    drop_path=drop,
                    flash_attn=False,
                )
                for _ in range(depth)
            ]
            + [
                CrossBlockMulti(
                    dim,
                    heads,
                    qkv_bias=True,
                    norm_layer=norm,
                    drop_path=drop,
                    release=True,
                )
            ]
        )
        self.checkpointing = checkpointing

    def cross(
        self, x: Tensor, valid: Tensor, side: int, spacing: float, modalities: int
    ) -> Tensor:
        block = self.blocks[-1]
        attention = block.attn
        batch, count, dim = x.shape
        query = attention.q_learned + position(dim, side, spacing, x)
        q = (
            query.expand(batch, -1, -1)
            .reshape(
                batch, side * side + 1, attention.num_heads, dim // attention.num_heads
            )
            .transpose(1, 2)
        )
        normalized = block.norm1(x)
        k = (
            attention.wk(normalized)
            .reshape(batch, count, attention.num_heads, dim // attention.num_heads)
            .transpose(1, 2)
        )
        v = (
            attention.wv(normalized)
            .reshape(batch, count, attention.num_heads, dim // attention.num_heads)
            .transpose(1, 2)
        )
        # Upstream cross attention applies iRPE to the unscaled query.
        bias = attention.rpe_k(q, height=side, width=side)
        bias = torch.cat((bias[..., :1], bias[..., 1:].repeat(1, 1, 1, modalities)), -1)
        bias = bias.masked_fill(~valid[:, None, None], float("-inf"))
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        value = value.transpose(1, 2).reshape(batch, side * side + 1, dim)
        value = block.drop_path(attention.proj_drop(attention.proj(value)))
        return value + block.drop_path(block.mlp(block.norm2(value)))

    def forward(
        self, tokens: Tensor, valid: Tensor, side: int, spacing: float
    ) -> Tensor:
        batch, modalities, locations, dim = tokens.shape
        pos = position(dim, side, spacing, tokens)
        x = (tokens + pos[:, None, 1:]).reshape(batch, -1, dim)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), 1)
        valid = F.pad(valid.reshape(batch, -1), (1, 0), value=True)
        for block in self.blocks[:-1]:
            if self.training and self.checkpointing and torch.is_grad_enabled():
                x = checkpoint(self_block, block, x, valid, side, use_reentrant=False)
            else:
                x = self_block(block, x, valid, side)
        if self.training and self.checkpointing and torch.is_grad_enabled():
            x = checkpoint(
                self.cross, x, valid, side, spacing, modalities, use_reentrant=False
            )
        else:
            x = self.cross(x, valid, side, spacing, modalities)
        return x[:, 1 : locations + 1]
