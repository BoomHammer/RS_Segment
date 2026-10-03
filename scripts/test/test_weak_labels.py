import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from data.labels import LabelRecord
from data.raster_alignment import target_grid_from_raster
from data.weak_labels import (
    WeakLabelGenerationConfig,
    evaluate_label_quality_from_raster,
    generate_weak_labels,
)
from inference.pointsam import PointSAMPrediction


class FakeInferencer:
    def predict(self, request):
        height, width = request.image.shape[-2:]
        mask = np.zeros((height, width), dtype=bool)
        point = request.positive_points[0]
        mask[point.row, point.column] = True
        return PointSAMPrediction(mask=mask, confidence=np.ones((height, width)))


def _write_band(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 4, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def test_streaming_weak_labels_write_geo_tiff_and_quality(tmp_path: Path) -> None:
    bands = []
    for index in range(3):
        path = tmp_path / f"band_{index}.tif"
        _write_band(path, np.ones((4, 4), dtype=np.float32))
        bands.append(path)
    grid = target_grid_from_raster(bands[0], target_crs="EPSG:4326", resolution=1)
    records = [LabelRecord("1", 0.5, 3.5, "F", "A", "F", "A", 1, 7)]
    output = tmp_path / "weak_labels.tif"
    report = generate_weak_labels(
        records,
        grid=grid,
        image_paths=bands,
        inferencer=FakeInferencer(),
        output_path=output,
        config=WeakLabelGenerationConfig(window_size=(4, 4)),
        alliance_names={7: "Test alliance"},
        provenance_path=tmp_path / "seed_ids.tif",
        sample_outcomes_path=tmp_path / "outcomes.json",
    )
    with rasterio.open(output) as dataset:
        assert dataset.crs.to_string() == "EPSG:4326"
        assert dataset.nodata == -9999
        assert dataset.read(1)[0, 0] == 7
    assert report["labeled_pixels"] == 1
    assert report["sample_quality"]["invalid_samples"] == 0
    assert report["class_distribution"]["7"]["alliance"] == "Test alliance"
    with rasterio.open(tmp_path / "seed_ids.tif") as seeds:
        assert seeds.read(1)[0, 0] == 1
        assert (seeds.read(1) > 0).sum() == 1
    outcomes = json.loads((tmp_path / "outcomes.json").read_text())
    assert outcomes[0]["record_index"] == "1"
    assert outcomes[0]["seed_id"] == 1


def test_quality_report_detects_conflicting_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.tif"
    _write_band(path, np.array([[1, -1], [0, -9999]], dtype=np.float32))
    report = evaluate_label_quality_from_raster(path)
    assert report["invalid_pixel_count"] == 1
    assert report["valid_pixels"] == 3
