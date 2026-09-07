"""Construct a second overlapping view entirely inside a training window."""

import torch

from data.augmentations import SPATIAL_KEYS


def overlapping_view(batch, crop_margin=32):
    height, width = batch["static"].shape[-2:]
    if crop_margin < 1 or min(height, width) <= 2 * crop_margin:
        raise ValueError("重叠裁剪边距必须为正数且小于窗口边长的一半")
    top, left = (torch.randint(0, 2, (2,)) * crop_margin).tolist()
    result = dict(batch)
    for key in SPATIAL_KEYS:
        if key in batch:
            result[key] = batch[key][
                ..., top : top + height - crop_margin, left : left + width - crop_margin
            ]
    return result, top, left
