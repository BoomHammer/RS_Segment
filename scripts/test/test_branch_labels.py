"""Verify training-only prompt selection and aligned provenance clipping."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin

from data.labels import LabelRecord
from data.raster_alignment import TargetGrid


def test_prompt_deduplication_and_held_out_exclusion(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("prepare_branch_labels")
    records = [
        LabelRecord("old", 0.5, 3.5, "F", "A", "F", "A", 1, 1),
        LabelRecord("latest", 0.5, 3.5, "F", "B", "F", "B", 1, 2),
        LabelRecord("validation", 2.5, 3.5, "F", "C", "F", "C", 1, 3),
        LabelRecord("test", 0.5, 1.5, "F", "D", "F", "D", 1, 4),
    ]
    monkeypatch.setattr(module, "iter_encoded_labels", lambda *a, **k: [records])
    dataset = SimpleNamespace(
        grid=TargetGrid(CRS.from_epsg(4326), from_origin(0, 4, 1, 1), 4, 4),
        ground_truth_pixels={(0, 0): 2, (0, 2): 3, (2, 0): 4},
    )
    manifest = SimpleNamespace(
        blocks={"0:0": "train", "0:1": "validation", "1:0": "test"},
        block_size=(2, 2),
        class_counts={"train": {"2": 1}},
    )
    config = SimpleNamespace(
        data=SimpleNamespace(label_file="unused", label_columns={})
    )
    selected = module.training_records(config, {}, dataset, manifest)
    assert len(selected) == 1
    assert selected[0][1].index == "latest"


def test_training_block_clipping_preserves_source_ids(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("prepare_branch_labels")
    profile = {
        "driver": "GTiff",
        "width": 4,
        "height": 4,
        "count": 1,
        "dtype": "int32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 4, 1, 1),
        "nodata": -9999,
    }
    for name, value in (("raw.tif", 2), ("seeds.tif", 17)):
        with rasterio.open(tmp_path / name, "w", **profile) as target:
            target.write(np.full((4, 4), value, dtype=np.int32), 1)
    manifest = SimpleNamespace(block_size=(2, 2), blocks={"0:0": "train"})
    module.restrict_to_training(
        tmp_path / "raw.tif",
        tmp_path / "seeds.tif",
        tmp_path / "train.tif",
        tmp_path / "train_seeds.tif",
        manifest,
    )
    with (
        rasterio.open(tmp_path / "train.tif") as labels,
        rasterio.open(tmp_path / "train_seeds.tif") as ids,
    ):
        values, sources = labels.read(1), ids.read(1)
        assert (values > 0).sum() == 4
        np.testing.assert_array_equal(values > 0, sources > 0)
        assert (sources[values > 0] == 17).all()
