from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from prepare_data import main


def test_prepare_data_writes_statistics_in_timestamped_run(tmp_path: Path) -> None:
    data = tmp_path / "data"
    dynamic = data / "raw" / "dynamic"
    static = data / "raw" / "static"
    labels = data / "labels"
    processed = data / "processed"
    for directory in (dynamic, static, labels, processed):
        directory.mkdir(parents=True)
    raster = dynamic / "NDVI230101.tif"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        nodata=-9999,
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype=np.float32), 1)
    config = tmp_path / "data.yaml"
    config.write_text(
        "data:\n"
        f"  root: {data.as_posix()}\n"
        f"  labels: {labels.as_posix()}\n"
        f"  raw: {(data / 'raw').as_posix()}\n"
        f"  dynamic: {dynamic.as_posix()}\n"
        f"  static: {static.as_posix()}\n"
        f"  processed: {processed.as_posix()}\n"
        "  required_subdirectories: "
        "[labels, raw, raw/dynamic, raw/static, processed]\n",
        encoding="utf-8",
    )

    assert main(["--config", str(config)]) == 0

    runs = [path for path in processed.iterdir() if path.is_dir()]
    assert len(runs) == 1
    statistics = list(runs[0].glob("raster_stats_*.json"))
    assert len(statistics) == 1

    # A repeated run still gets its own directory while reusing matching statistics.
    assert main(["--config", str(config)]) == 0
    runs = [path for path in processed.iterdir() if path.is_dir()]
    assert len(runs) == 2
    assert all(len(list(run.glob("raster_stats_*.json"))) == 1 for run in runs)
