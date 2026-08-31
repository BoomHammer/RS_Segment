from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from data.raster_alignment import (
    TargetGrid,
    locate_points,
    point_pixel_values,
    read_aligned_window,
    target_grid_from_raster,
)


def _write_raster(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 4, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(np.arange(16, dtype=np.float32).reshape(4, 4), 1)


def test_target_grid_location_and_window_read(tmp_path: Path) -> None:
    path = tmp_path / "source.tif"
    _write_raster(path)
    grid = target_grid_from_raster(path, target_crs="EPSG:4326", resolution=1)

    locations = locate_points(
        [(0.5, 3.5), (99, 99)], point_crs="EPSG:4326", grid=grid
    )
    values = point_pixel_values(
        path,
        [(0.5, 3.5), (99, 99)],
        point_crs="EPSG:4326",
        grid=grid,
    )
    window = read_aligned_window(path, grid, rasterio.windows.Window(0, 0, 2, 2))

    assert isinstance(grid, TargetGrid)
    assert locations[0]["inside"] and locations[0]["row"] == 0
    assert not locations[1]["inside"]
    assert values == [0.0, None]
    np.testing.assert_array_equal(window, [[0, 1], [4, 5]])
