import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from data_stats import main


def test_compute_stats_cli_uses_config_and_writes_output(tmp_path: Path) -> None:
    raster_directory = tmp_path / "dynamic"
    raster_directory.mkdir()
    raster_path = raster_directory / "NDVI230101.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        nodata=-9999,
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.array([[1, 2], [3, -9999]], dtype="float32"), 1)

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "data:\n"
        f"  dynamic: {raster_directory.as_posix()}\n"
        "  raster:\n"
        "    nodata: -9999\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "results" / "stats.json"

    assert main(["--config", str(config_path), "--output", str(output_path)]) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["window_size"] == [1024, 1024]
    assert payload["groups"][0]["category"] == "NDVI"
    assert payload["groups"][0]["statistics"]["count"] == 3
    assert payload["groups"][0]["statistics"]["mean"] == 2.0
    assert "statistics" not in payload


def test_compute_stats_groups_dynamic_series_and_static_files(tmp_path: Path) -> None:
    dynamic_directory = tmp_path / "dynamic"
    static_directory = tmp_path / "static"
    dynamic_directory.mkdir()
    static_directory.mkdir()

    for path, values in (
        (dynamic_directory / "SR230101B1.tif", [[1, 2], [3, -9999]]),
        (dynamic_directory / "SR230102B1.tif", [[5, 6], [7, -9999]]),
        (static_directory / "DSM.tif", [[10, 11], [12, -9999]]),
    ):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            nodata=-9999,
            transform=from_origin(0, 2, 1, 1),
        ) as dataset:
            dataset.write(np.array(values, dtype="float32"), 1)

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "data:\n"
        f"  dynamic: {dynamic_directory.as_posix()}\n"
        f"  static: {static_directory.as_posix()}\n"
        "  raster:\n"
        "    nodata: -9999\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "stats.json"

    assert main(["--config", str(config_path), "--output", str(output_path)]) == 0

    groups = {
        group["category"]: group
        for group in json.loads(output_path.read_text(encoding="utf-8"))["groups"]
    }
    assert groups["SR_B1"]["statistics"]["count"] == 6
    assert len(groups["SR_B1"]["files"]) == 2
    assert groups["SR_B1"]["files"][0]["date"] == "2023-01-01"
    assert "date" not in groups["DSM"]["files"][0]
    assert groups["DSM"]["statistics"]["mean"] == 11.0


def test_compute_stats_applies_configured_product_range_and_scale(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("data_stats.configure_logging", lambda: None)
    dynamic_directory = tmp_path / "dynamic"
    dynamic_directory.mkdir()
    static_directory = tmp_path / "static"
    static_directory.mkdir()
    with rasterio.open(
        static_directory / "COPERNICUS_DEM_100M.tif",
        "w",
        driver="GTiff",
        width=5,
        height=1,
        count=1,
        dtype="float32",
        nodata=-9999,
        transform=from_origin(0, 1, 1, 1),
    ) as dataset:
        dataset.write(
            np.array([[100, 300, -9999, np.nan, np.inf]], dtype=np.float32), 1
        )
    with rasterio.open(
        dynamic_directory / "LST230101.tif",
        "w",
        driver="GTiff",
        width=3,
        height=1,
        count=1,
        dtype="uint16",
        transform=from_origin(0, 1, 1, 1),
    ) as dataset:
        dataset.write(np.array([[0, 7500, 10000]], dtype=np.uint16), 1)
    range_path = tmp_path / "ranges.csv"
    range_path.write_text("Data,Min,Max,Scale\nLST,7500,65535,0.02\n", encoding="utf-8")
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "data:\n"
        f"  dynamic: {dynamic_directory.as_posix()}\n"
        f"  static: {static_directory.as_posix()}\n"
        f"  processed: {(tmp_path / 'processed').as_posix()}\n"
        "  stage2:\n"
        f"    value_range_file: {range_path.as_posix()}\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "stats.json"

    assert main(["--config", str(config_path), "--output", str(output_path)]) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    groups = {group["category"]: group for group in payload["groups"]}
    group = groups["LST"]
    assert payload["schema_version"] == 4
    assert group["statistics"]["count"] == 2
    assert group["statistics"]["mean"] == 175
    assert group["value_range"] == {
        "minimum": 7500.0,
        "maximum": 65535.0,
        "scale": 0.02,
    }
    dem = groups["COPERNICUS_DEM_100M"]
    assert dem["value_range"] is None
    assert dem["statistics"]["count"] == 2
    assert dem["statistics"]["mean"] == 200
    assert payload["missing_value_ranges"] == ["COPERNICUS_DEM_100M"]
    assert "COPERNICUS_DEM_100M" in caplog.text

    cached_path = tmp_path / "processed" / "raster_stats_cached.json"
    cached_path.parent.mkdir()
    cached_path.write_text(json.dumps(payload), encoding="utf-8")
    caplog.clear()
    assert main(["--config", str(config_path), "--output", str(output_path)]) == 0
    assert "COPERNICUS_DEM_100M" in caplog.text
