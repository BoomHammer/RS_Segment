"""Map streamed, normalized windows to named AnySat sensor observations."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import torch
from torch import Tensor


def resolve_resolution(settings: dict, grid: dict) -> float:
    """Use the aligned grid's GSD, never the native sensor resolution."""
    configured = settings.get("resolution_m")
    if grid:
        from pyproj import CRS

        crs = CRS.from_user_input(grid["crs"])
        transform = grid["transform"]
        if crs.is_projected:
            units = crs.axis_info[0].unit_conversion_factor
            x = math.hypot(transform[0], transform[3]) * units
            y = math.hypot(transform[1], transform[4]) * units
            if not math.isclose(x, y, rel_tol=1e-4):
                raise ValueError("AnySat requires square target-grid pixels")
            if configured is not None and not math.isclose(
                float(configured), x, rel_tol=1e-4
            ):
                raise ValueError("AnySat resolution_m differs from the target-grid GSD")
            configured = x
        elif configured is None:
            raise ValueError(
                "Angular target grids require explicit nominal resolution_m"
            )
    resolution = float(250 if configured is None else configured)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("AnySat resolution_m must be finite and positive")
    return resolution


def prepare_sensor(
    batch: dict[str, Any], spec: dict[str, Any], expected: list[str]
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep actual dates and per-channel NoData; never synthesize observations.

    Monthly rasters use their existing first-of-month convention. Dates are
    zero-based day-of-year as required by AnySat, including leap-year day 365.
    Inputs have already been scaled and standardized by WindowedSampleDataset.
    """
    role = spec["role"]
    source = batch[role]
    feature_lists = batch.get(f"{role}_features")
    names = expected if feature_lists is None else feature_lists[0]
    if feature_lists is not None and any(names != row for row in feature_lists):
        raise ValueError("AnySat requires consistent feature order within a batch")
    if len(names) != source.shape[-3] or sorted(names) != sorted(expected):
        raise ValueError(f"AnySat {role} features differ from the saved contract")
    indices = [names.index(name) for name in spec["features"]]
    if role == "static":
        values = source[:, indices].unsqueeze(1)
        valid = torch.isfinite(values) & batch["static_valid_mask"][:, None, None]
        return torch.where(valid, values, 0), valid, values.new_zeros(values.shape[:2])

    present = batch["dynamic_mask"][:, :, indices].bool()
    present = present & batch["dynamic_time_mask"][:, :, None].bool()
    # Remove only globally absent dates, without thinning any sensor's time series.
    slots = present.any(dim=(0, 2)).nonzero().flatten()
    if slots.numel() == 0:
        slots = torch.zeros(1, dtype=torch.long, device=source.device)
    values = source[:, slots][:, :, indices]
    valid = torch.isfinite(values) & present[:, slots, :, None, None]
    valid = valid & batch["dynamic_valid_mask"][:, None, None].bool()
    dates = values.new_zeros(values.shape[:2])
    if "dynamic_times" in batch:
        for sample, stamps in enumerate(batch["dynamic_times"]):
            for position, slot in enumerate(slots.tolist()):
                if slot < len(stamps):
                    stamp = stamps[slot]
                    acquired = date.fromisoformat(
                        stamp + "-01" if len(stamp) == 7 else stamp
                    )
                    dates[sample, position] = acquired.timetuple().tm_yday - 1
    else:
        # The dataset's first time_encoding component is DOY / 365.25.
        dates = (batch["time_encoding"][:, slots, 0] * 365.25).round()
    dates = torch.where(present[:, slots].any(-1), dates, 0)
    if not torch.isfinite(dates).all():
        raise ValueError("AnySat observation dates must be finite")
    return torch.where(valid, values, 0), valid, dates
