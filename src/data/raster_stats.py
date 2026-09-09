"""Streaming statistics for large raster datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass(slots=True)
class RasterStatistics:
    """Population statistics accumulated without retaining raster pixels."""

    count: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    @property
    def variance(self) -> float:
        """Return population variance (the normalization-statistics convention)."""

        return self._m2 / self.count if self.count else float("nan")

    @property
    def standard_deviation(self) -> float:
        """Return population standard deviation."""

        return float(np.sqrt(self.variance))

    def update(self, values: np.ndarray) -> None:
        """Merge a finite one-dimensional batch using Welford's algorithm."""

        batch = np.asarray(values, dtype=np.float64).ravel()
        batch = batch[np.isfinite(batch)]
        batch_count = int(batch.size)
        if not batch_count:
            return

        batch_mean = float(np.mean(batch, dtype=np.float64))
        differences = batch - batch_mean
        batch_m2 = float(np.dot(differences, differences))
        if not self.count:
            self.count = batch_count
            self.mean = batch_mean
            self._m2 = batch_m2
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self._m2 += batch_m2 + delta * delta * self.count * batch_count / total_count
        self.mean += delta * batch_count / total_count
        self.count = total_count

    def merge(self, other: RasterStatistics) -> None:
        """Merge another accumulator, preserving streaming precision."""

        if not other.count:
            return
        if not self.count:
            self.count, self.mean, self._m2 = other.count, other.mean, other._m2
            return

        total_count = self.count + other.count
        delta = other.mean - self.mean
        self._m2 += other._m2 + delta * delta * self.count * other.count / total_count
        self.mean += delta * other.count / total_count
        self.count = total_count

    def to_dict(self) -> dict[str, float | int | None]:
        """Return the public statistics representation used for JSON output."""

        if not self.count:
            return {
                "count": 0,
                "mean": None,
                "variance": None,
                "standard_deviation": None,
            }
        return {
            "count": self.count,
            "mean": self.mean if self.count else float("nan"),
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
        }


def _windows(width: int, height: int, window_size: tuple[int, int]) -> Iterable[Window]:
    window_width, window_height = window_size
    if window_width < 1 or window_height < 1:
        raise ValueError("window_size 必须是正整数")
    for row in range(0, height, window_height):
        for column in range(0, width, window_width):
            yield Window(
                column,
                row,
                min(window_width, width - column),
                min(window_height, height - row),
            )


def stream_raster_statistics(
    path: str | Path,
    *,
    band: int = 1,
    window_size: tuple[int, int] = (1024, 1024),
    nodata: float | int | None = None,
    valid_minimum: float | None = None,
    valid_maximum: float | None = None,
    scale: float = 1.0,
) -> RasterStatistics:
    """Calculate statistics for one band by reading bounded raster windows.

    ``nodata=None`` uses the GeoTIFF's declared NoData value. Regardless of
    metadata, NaN and infinite values are always excluded. Product ranges are
    evaluated on raw values before the retained values are multiplied by ``scale``.
    """

    if valid_minimum is not None and valid_maximum is not None:
        if valid_minimum > valid_maximum:
            raise ValueError("valid_minimum 不得大于 valid_maximum")
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("scale 必须是非零有限数值")

    statistics = RasterStatistics()
    with rasterio.open(path) as dataset:
        if not 1 <= band <= dataset.count:
            raise ValueError(f"band 必须在 1 到 {dataset.count} 之间")
        missing_value = dataset.nodata if nodata is None else nodata
        for window in _windows(dataset.width, dataset.height, window_size):
            masked = dataset.read(band, window=window, masked=True)
            values = np.asarray(masked.data)
            valid = ~np.ma.getmaskarray(masked) & np.isfinite(values)
            if missing_value is not None:
                valid &= values != missing_value
            if valid_minimum is not None:
                valid &= values >= valid_minimum
            if valid_maximum is not None:
                valid &= values <= valid_maximum
            statistics.update(values[valid].astype(np.float64) * scale)
    return statistics


def stream_rasters_statistics(
    paths: Sequence[str | Path],
    *,
    band: int = 1,
    window_size: tuple[int, int] = (1024, 1024),
    nodata: float | int | None = None,
) -> RasterStatistics:
    """Aggregate statistics from multiple compatible raster files."""

    statistics = RasterStatistics()
    for path in paths:
        statistics.merge(
            stream_raster_statistics(
                path, band=band, window_size=window_size, nodata=nodata
            )
        )
    return statistics


def write_statistics(statistics: RasterStatistics, output: str | Path) -> None:
    """Persist global statistics as UTF-8 JSON."""

    Path(output).write_text(
        json.dumps(statistics.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
