"""Check hierarchy matching and physical, globally aligned DEM reads."""

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from inference.elevation import ElevationConstraint


@pytest.fixture
def dem_setup(tmp_path):
    path = tmp_path / "dem.tif"
    transform = from_origin(0, 1000, 100, 100)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=transform,
    ) as source:
        # No nodata metadata: fallback sentinel must be masked before resampling.
        source.write(np.array([[1399, 1400], [1401, -9999]], dtype=np.int16), 1)
        source.scales = (2.0,)
        source.offsets = (1.0,)
    grid = SimpleNamespace(
        width=3,
        height=2,
        crs="EPSG:3857",
        transform=transform,
    )
    mapping = {
        "classes": [
            {"alliance_code": 1, "level_names_zh": ["草地", "高寒草甸", "嵩草草甸"]},
            {"alliance_code": 2, "level_names_zh": ["草地", "高寒草甸", "杂类草草甸"]},
            {"alliance_code": 3, "level_names": ["Grassland", "Alpine_Steppe"]},
            {"alliance_code": 4, "level_names_zh": ["草地", "温带草甸"]},
        ]
    }
    config = {
        "dem_path": str(path),
        "classes": ["高寒草甸", "Alpine_Steppe"],
        "min_elevation_m": 2800,
        "resampling": "nearest",
    }
    return config, mapping, grid


def test_descendants_scaled_dem_and_uncovered_pixels(dem_setup):
    constraint = ElevationConstraint(*dem_setup)
    assert constraint.class_codes == {1, 2, 3}
    with constraint.open() as reader:
        elevation, allowed, missing = constraint.read(reader, Window(0, 0, 3, 2))
    np.testing.assert_allclose(
        elevation,
        [[2799, 2801, np.nan], [2803, np.nan, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_array_equal(allowed, [[False, True, False], [True, False, False]])
    np.testing.assert_array_equal(missing, [[False, False, True], [False, True, True]])


def test_global_resampling_is_independent_of_row_blocks(dem_setup):
    config, mapping, grid = dem_setup
    config["resampling"] = "bilinear"
    grid.width, grid.height = 4, 4
    grid.transform = from_origin(0, 1000, 50, 50)
    constraint = ElevationConstraint(config, mapping, grid)
    with constraint.open() as reader:
        whole = constraint.read(reader, Window(0, 0, 4, 4))
        parts = [constraint.read(reader, Window(0, row, 4, 1)) for row in range(4)]
    for index in range(3):
        np.testing.assert_allclose(
            whole[index],
            np.concatenate([part[index] for part in parts]),
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"classes": ["不存在的类别"]}, "未匹配"),
        ({"classes": "高寒草甸"}, "列表"),
        ({"min_elevation_m": float("nan")}, "有限"),
        ({"missing_dem": "ignore"}, "missing_dem"),
        ({"resampling": "cubic"}, "resampling"),
        ({"dem_path": None}, "dem_path"),
    ],
)
def test_invalid_configuration_fails_early(dem_setup, options, message):
    config, mapping, grid = dem_setup
    with pytest.raises(ValueError, match=message):
        ElevationConstraint({**config, **options}, mapping, grid)
