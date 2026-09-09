"""Product-specific valid ranges and scale factors for raster inputs."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ValueRange:
    """Valid raw-DN interval and conversion from DN to physical units."""

    minimum: float
    maximum: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite((self.minimum, self.maximum, self.scale)).all():
            raise ValueError("值域和 Scale 必须是有限数值")
        if self.minimum > self.maximum:
            raise ValueError("值域 Min 不得大于 Max")
        if self.scale == 0:
            raise ValueError("Scale 不得为 0")


def load_value_ranges(path: str | Path | None) -> dict[str, ValueRange]:
    """Load case-insensitive product rules from ``Value_range.csv``."""

    if path is None:
        return {}
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Data", "Min", "Max", "Scale"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"值域文件缺少列: {sorted(required)}")
        result: dict[str, ValueRange] = {}
        for line_number, row in enumerate(reader, start=2):
            key = str(row.get("Data", "")).strip().casefold()
            if not key:
                raise ValueError(f"值域文件第 {line_number} 行 Data 为空")
            if key in result:
                raise ValueError(f"值域文件 Data 重复: {row['Data']}")
            try:
                minimum = float(str(row.get("Min", "")).strip())
                maximum = float(str(row.get("Max", "")).strip())
                scale_text = str(row.get("Scale", "")).strip()
                scale = float(scale_text) if scale_text else 1.0
            except ValueError as exc:
                raise ValueError(f"值域文件第 {line_number} 行包含非法数值") from exc
            result[key] = ValueRange(minimum, maximum, scale)
    return result


def value_range_for(feature: str, ranges: dict[str, ValueRange]) -> ValueRange | None:
    """Resolve an indexed feature name to its product-level value-range rule."""

    name = feature.casefold()
    candidates = [name]
    if name.endswith("aspect"):
        candidates.append("aspect")
    if name.endswith("slope"):
        candidates.append("slope")
    candidates.append(re.sub(r"_b\d+$", "", name))
    candidates.append(re.sub(r"\d{4}$", "", name))
    if name.startswith("dsm"):
        candidates.append("dsm")
    for candidate in candidates:
        if candidate in ranges:
            return ranges[candidate]
    return None


def valid_and_scaled(values: np.ndarray, rule: ValueRange) -> np.ndarray:
    """Mask raw values outside a product rule, then apply its scale factor."""

    result = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(result) & (result >= rule.minimum) & (result <= rule.maximum)
    result[~valid] = np.nan
    result[valid] *= rule.scale
    return result
