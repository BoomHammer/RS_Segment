"""Bounded, mask-aware temporal discretization of existing window batches."""

from __future__ import annotations

from datetime import date
from typing import Any

import torch
from torch import Tensor


def temporal_features(batch: dict[str, Any], sample: int, index: int) -> Tensor:
    """Eight MAESTRO date features; date-only observations have hour zero."""
    encoding = batch["time_encoding"][sample, index]
    elapsed = encoding[0]
    if "dynamic_times" in batch:
        stamp = batch["dynamic_times"][sample][index]
        acquired = date.fromisoformat(stamp + "-01" if len(stamp) == 7 else stamp)
        # A fixed reference shared by every window and modality, also across years.
        elapsed = encoding.new_tensor((acquired - date(2023, 1, 1)).days / 365.25)
    return torch.stack(
        (
            encoding[1],
            encoding[2],
            encoding.new_zeros(()),
            encoding.new_ones(()),
            elapsed,
            elapsed,
            elapsed,
            elapsed,
        )
    )


def prepare_modality(
    batch: dict[str, Any],
    role: str,
    indices: list[int],
    bins: int,
    *,
    training: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select real observations per product, never fabricate absent dates.

    Discretize only present observations, randomly in training and using the
    closest observation to the pixel-wise median in evaluation. Short sequences
    are padded with invalid slots. Return values, per-value validity and dates.
    """
    source = batch[role]
    if role == "static":
        values = source[:, indices].unsqueeze(1)
        valid = torch.isfinite(values) & batch["static_valid_mask"][:, None, None]
        return (
            torch.where(valid, values, 0.0),
            valid,
            values.new_zeros((values.shape[0], 1, 8)),
        )
    batch_size, _, _, height, width = source.shape
    values = source.new_zeros((batch_size, bins, len(indices), height, width))
    valid = torch.zeros_like(values, dtype=torch.bool)
    dates = source.new_zeros((batch_size, bins, 8))
    for sample in range(batch_size):
        present = batch["dynamic_mask"][sample, :, indices].any(-1)
        present = present & batch["dynamic_time_mask"][sample]
        candidates = present.nonzero().flatten()
        if candidates.numel() == 0:
            continue
        raw = source[sample, :, indices][candidates]
        ok = torch.isfinite(raw)
        ok &= batch["dynamic_mask"][sample, :, indices][candidates, :, None, None]
        ok &= batch["dynamic_valid_mask"][sample, None, None]
        usable = ok.flatten(1).any(1)
        candidates, raw, ok = candidates[usable], raw[usable], ok[usable]
        if candidates.numel() == 0:
            continue
        for slot, positions in enumerate(
            torch.tensor_split(
                torch.arange(len(candidates), device=source.device),
                min(bins, len(candidates)),
            )
        ):
            if training:
                selected = positions[
                    torch.randint(len(positions), (), device=source.device)
                ]
            else:
                observations = raw[positions].masked_fill(~ok[positions], float("nan"))
                median = observations.nanmedian(dim=0).values
                distance = (observations - median).abs().nan_to_num().flatten(1).sum(1)
                distance /= ok[positions].flatten(1).sum(1).clamp_min(1)
                selected = positions[distance.argmin()]
            valid[sample, slot] = ok[selected]
            values[sample, slot] = torch.where(ok[selected], raw[selected], 0.0)
            dates[sample, slot] = temporal_features(
                batch, sample, int(candidates[selected])
            )
    return values, valid, dates
