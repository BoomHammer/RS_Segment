"""Target-grid construction and bounded, CRS-aware raster access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window


@dataclass(slots=True, frozen=True)
class TargetGrid:
    """A common output grid shared by all source rasters and labels."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def resolution(self) -> tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))

    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (self.width, self.height)
        return left, bottom, right, top


def _resampling(value: str | Resampling) -> Resampling:
    if isinstance(value, Resampling):
        return value
    try:
        return Resampling[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"不支持的重采样方法: {value}") from exc


def target_grid_from_raster(
    path: str | Path,
    *,
    target_crs: str | CRS,
    resolution: float | tuple[float, float],
) -> TargetGrid:
    """Create a north-up grid covering a source raster at an explicit resolution."""

    with rasterio.open(path) as source:
        crs = CRS.from_user_input(target_crs)
        if isinstance(resolution, (int, float)):
            resolution = (float(resolution), float(resolution))
        if len(resolution) != 2 or min(resolution) <= 0:
            raise ValueError("resolution 必须是一个正数或两个正数")
        transform, width, height = calculate_default_transform(
            source.crs,
            crs,
            source.width,
            source.height,
            *source.bounds,
            resolution=tuple(float(item) for item in resolution),
        )
    return TargetGrid(crs, transform, width, height)


def open_aligned_raster(
    path: str | Path,
    grid: TargetGrid,
    *,
    resampling: str | Resampling = "nearest",
) -> tuple[Any, WarpedVRT]:
    """Open a source and a window-readable VRT on the shared target grid.

    The returned source must remain open until the VRT is closed. Prefer the
    context manager helper below for normal use.
    """

    source = rasterio.open(path)
    vrt = WarpedVRT(
        source,
        crs=grid.crs,
        transform=grid.transform,
        width=grid.width,
        height=grid.height,
        resampling=_resampling(resampling),
        nodata=source.nodata,
    )
    return source, vrt


class aligned_raster:
    """Context manager for bounded reads on a target grid."""

    def __init__(
        self, path: str | Path, grid: TargetGrid, *, resampling: str = "nearest"
    ):
        self.path = path
        self.grid = grid
        self.resampling = resampling
        self.source: Any = None
        self.dataset: WarpedVRT | None = None

    def __enter__(self) -> WarpedVRT:
        self.source, self.dataset = open_aligned_raster(
            self.path, self.grid, resampling=self.resampling
        )
        return self.dataset

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.dataset is not None:
            self.dataset.close()
        if self.source is not None:
            self.source.close()


def read_aligned_window(
    path: str | Path,
    grid: TargetGrid,
    window: Window,
    *,
    band: int = 1,
    resampling: str = "nearest",
    masked: bool = True,
) -> np.ndarray:
    """Read only one bounded target-grid window, reprojecting on demand."""

    with aligned_raster(path, grid, resampling=resampling) as dataset:
        if not 1 <= band <= dataset.count:
            raise ValueError(f"band 必须在 1 到 {dataset.count} 之间")
        return dataset.read(band, window=window, masked=masked)


def locate_points(
    points: list[tuple[float, float]],
    *,
    point_crs: str | CRS,
    grid: TargetGrid,
) -> list[dict[str, Any]]:
    """Transform points and report target pixel indices and in-grid status."""

    transformer = Transformer.from_crs(
        CRS.from_user_input(point_crs), grid.crs, always_xy=True
    )
    left, bottom, right, top = grid.bounds()
    result = []
    for x, y in points:
        target_x, target_y = transformer.transform(x, y)
        column = int(np.floor((target_x - grid.transform.c) / grid.transform.a))
        row = int(np.floor((target_y - grid.transform.f) / grid.transform.e))
        inside = left <= target_x < right and bottom < target_y <= top
        result.append(
            {
                "x": x,
                "y": y,
                "target_x": target_x,
                "target_y": target_y,
                "row": row,
                "column": column,
                "inside": (
                    inside and 0 <= row < grid.height and 0 <= column < grid.width
                ),
            }
        )
    return result


def point_pixel_values(
    path: str | Path,
    points: list[tuple[float, float]],
    *,
    point_crs: str | CRS,
    grid: TargetGrid,
    band: int = 1,
) -> list[float | None]:
    """Small定位 smoke test: sample aligned pixels without loading the raster."""

    locations = locate_points(points, point_crs=point_crs, grid=grid)
    values: list[float | None] = []
    with aligned_raster(path, grid) as dataset:
        for location in locations:
            if not location["inside"]:
                values.append(None)
                continue
            value = dataset.read(
                band,
                window=Window(location["column"], location["row"], 1, 1),
            )[0, 0]
            values.append(
                None
                if dataset.nodata is not None and value == dataset.nodata
                else float(value)
            )
    return values


def grid_from_config(
    config: dict[str, Any], reference_raster: str | Path
) -> TargetGrid:
    """Build a target grid from the ``target_grid`` YAML section."""

    crs = config.get("crs", "EPSG:4326")
    resolution = config.get("resolution", [0.00225, 0.00225])
    return target_grid_from_raster(
        reference_raster,
        target_crs=crs,
        resolution=tuple(resolution) if isinstance(resolution, list) else resolution,
    )
