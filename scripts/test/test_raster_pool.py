"""Reader reuse must preserve pixels and survive Windows worker serialization."""

import pickle

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from data.raster_alignment import target_grid_from_raster
from data.raster_pool import RasterReaderPool


def test_reader_pool_matches_scoped_reads_and_closes_evicted_files(tmp_path):
    paths = [tmp_path / f"raster{i}.tif" for i in range(2)]
    for i, path in enumerate(paths):
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
        ) as out:
            out.write(np.arange(16, dtype=np.float32).reshape(4, 4) + i, 1)
    grid = target_grid_from_raster(paths[0], target_crs="EPSG:4326", resolution=1)
    scoped = RasterReaderPool(grid, 0)
    cached = RasterReaderPool(grid, 1)
    window = Window(0, 0, 2, 2)
    with scoped.borrow(paths[0], "bilinear") as reader:
        expected = reader.read(1, window=window)
    with cached.borrow(paths[0], "bilinear") as first:
        np.testing.assert_array_equal(first.read(1, window=window), expected)
    with cached.borrow(paths[0], "bilinear") as second:
        assert first is second and not first.closed
    restored = pickle.loads(pickle.dumps(cached))
    assert not restored._readers
    with restored.borrow(paths[0], "bilinear") as reader:
        np.testing.assert_array_equal(reader.read(1, window=window), expected)
    with cached.borrow(paths[1], "bilinear"):
        assert first.closed
    cached.close()
    restored.close()
    assert not cached._readers and not restored._readers
