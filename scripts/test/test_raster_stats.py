from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from data.raster_stats import (
    RasterStatistics,
    stream_raster_statistics,
    write_statistics,
)


def _write_raster(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        nodata=-9999,
        transform=from_origin(0, values.shape[0], 1, 1),
    ) as dataset:
        dataset.write(values.astype("float32"), 1)


def test_statistics_use_windows_and_ignore_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "sample.tif"
    _write_raster(
        path,
        np.array([[1, 2, -9999], [4, np.nan, np.inf]], dtype=np.float32),
    )

    result = stream_raster_statistics(path, window_size=(2, 1))

    assert result.count == 3
    assert result.mean == 7 / 3
    assert np.isclose(result.variance, 14 / 9)


def test_statistics_support_explicit_nodata_override(tmp_path: Path) -> None:
    path = tmp_path / "override.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        nodata=None,
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.array([[1, -9999], [3, 999]], dtype=np.float32), 1)

    result = stream_raster_statistics(path, nodata=999, window_size=(1, 1))

    assert result.count == 3
    assert np.isclose(result.mean, (1 - 9999 + 3) / 3)


def test_all_invalid_windows_return_zero_count_and_null_json_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.tif"
    _write_raster(
        path,
        np.array(
            [[-9999, np.nan, -9999, np.nan], [np.nan, -9999, np.nan, -9999]],
            dtype=np.float32,
        ),
    )

    result = stream_raster_statistics(path, window_size=(2, 1))

    assert result.count == 0
    assert np.isnan(result.variance)
    assert result.to_dict() == {
        "count": 0,
        "mean": None,
        "variance": None,
        "standard_deviation": None,
    }


@pytest.mark.parametrize(
    ("dtype", "nodata", "values"),
    [
        ("uint16", 65535, np.array([[1, 2, 65535]], dtype=np.uint16)),
        ("int16", -9999, np.array([[-2, 4, -9999]], dtype=np.int16)),
        ("float32", -9999, np.array([[0.5, 1.5, -9999]], dtype=np.float32)),
    ],
)
def test_statistics_support_integer_and_float_raster_types(
    tmp_path: Path,
    dtype: str,
    nodata: int,
    values: np.ndarray,
) -> None:
    path = tmp_path / f"values_{dtype}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=dtype,
        nodata=nodata,
        transform=from_origin(0, 1, 1, 1),
    ) as dataset:
        dataset.write(values, 1)

    result = stream_raster_statistics(path, window_size=(2, 1))
    expected = values[values != nodata].astype(np.float64)

    assert result.count == expected.size
    assert np.isclose(result.mean, np.mean(expected))
    assert np.isclose(result.variance, np.var(expected))


def test_online_statistics_preserve_precision_for_large_values() -> None:
    values = np.array([1e12, 1e12 + 1, 1e12 + 2, 1e12 + 3], dtype=np.float64)
    result = RasterStatistics()
    for value in values:
        result.update(np.array([value]))

    assert result.count == values.size
    assert result.mean == 1e12 + 1.5
    assert np.isclose(result.variance, np.var(values), rtol=0, atol=1e-12)


def test_online_accumulator_merge_matches_global_statistics() -> None:
    first = RasterStatistics()
    second = RasterStatistics()
    first.update(np.array([1.0, 2.0]))
    second.update(np.array([10.0, 20.0]))

    first.merge(second)

    assert first.count == 4
    assert first.mean == 8.25
    assert np.isclose(first.variance, np.var([1, 2, 10, 20]))


def test_write_statistics(tmp_path: Path) -> None:
    output = tmp_path / "stats.json"
    statistics = RasterStatistics()
    statistics.update(np.array([1.0, 3.0]))

    write_statistics(statistics, output)

    assert '"count": 2' in output.read_text(encoding="utf-8")
