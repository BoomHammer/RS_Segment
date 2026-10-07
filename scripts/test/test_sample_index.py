"""Tests for the unified sample index and lazy TorchGeo dataset."""

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from data.raster_alignment import target_grid_from_raster
from data.sample_index import (
    WindowedSampleDataset,
    _validate_statistics_value_ranges,
    build_sample_index,
    load_sample_index,
    sample_collate_fn,
)
from data.sampling import SpatialWeightedSampler, build_dataloader
from data.spatial_split import SpatialSplitManifest, build_spatial_split
from data.value_ranges import ValueRange


def _write_raster(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 4, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)


def test_dataset_accepts_missing_product_range(tmp_path, caplog):
    dynamic = tmp_path / "dynamic"
    static = tmp_path / "static"
    dynamic.mkdir()
    static.mkdir()
    path = dynamic / "NDVI230101.tif"
    _write_raster(path, np.full((4, 4), 2, dtype=np.float32))
    _write_raster(
        static / "COPERNICUS_DEM_100M.tif", np.full((4, 4), 300, dtype=np.float32)
    )
    ranges = tmp_path / "ranges.csv"
    ranges.write_text("Data,Min,Max,Scale\nNDVI,0,10,0.5\n", encoding="utf-8")
    index = build_sample_index(
        dynamic_dir=dynamic,
        static_dir=static,
        target_grid=target_grid_from_raster(path, target_crs="EPSG:4326", resolution=1),
    )
    payload = {
        "schema_version": 4,
        "groups": [
            {
                "category": "COPERNICUS_DEM_100M",
                "value_range": None,
                "statistics": {"mean": 100, "standard_deviation": 100},
            }
        ],
    }
    dataset = WindowedSampleDataset(
        index,
        window_size=(2, 2),
        statistics=payload,
        stage2={"value_range_file": str(ranges)},
    )
    torch.testing.assert_close(dataset[0]["static"], torch.full((1, 2, 2), 2.0))
    baseline = dataset[0]
    dataset.profile_steps = 1
    profiled = dataset[0]
    for key, value in baseline.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(profiled[key], value, equal_nan=True)
    batched = sample_collate_fn([profiled])
    timings = batched["_performance"]["samples"][0]["seconds"]
    assert timings["read_window_s"] >= timings["read_asset_s"]
    assert timings["read_asset_s"] >= timings["raster_open_read_close_s"]
    assert timings["normalize_s"] >= 0
    assert timings["stack_s"] >= 0
    assert "_performance" not in dataset[0]
    assert "COPERNICUS_DEM_100M" in caplog.text
    with pytest.raises(ValueError, match="缺少有效值规则"):
        _validate_statistics_value_ranges(
            payload, {"copernicus_dem_100m": ValueRange(0, 1000)}
        )
    payload["groups"][0]["value_range"] = {"minimum": 0, "maximum": 1000, "scale": 1}
    with pytest.raises(ValueError, match="规则已变化"):
        _validate_statistics_value_ranges(payload, {"ndvi": ValueRange(0, 10, 0.5)})


def test_case_insensitive_statistics_and_legacy_snapshot_compatibility(tmp_path):
    dynamic = tmp_path / "dynamic"
    static = tmp_path / "static"
    dynamic.mkdir()
    static.mkdir()
    dynamic_path = dynamic / "NDVI230101.tif"
    _write_raster(dynamic_path, np.full((4, 4), 2, dtype=np.float32))
    _write_raster(static / "DSM100aspect.tif", np.full((4, 4), 300, dtype=np.float32))
    index = build_sample_index(
        dynamic_dir=dynamic,
        static_dir=static,
        target_grid=target_grid_from_raster(
            dynamic_path, target_crs="EPSG:4326", resolution=1
        ),
    )
    statistics = {
        "groups": [
            {"category": "NDVI", "statistics": {"mean": 1, "standard_deviation": 2}},
            {
                "category": "DSM100ASPECT",
                "statistics": {"mean": 180, "standard_deviation": 120},
            },
        ]
    }
    legacy = WindowedSampleDataset(index, window_size=(2, 2), statistics=statistics)
    corrected = WindowedSampleDataset(
        index,
        window_size=(2, 2),
        statistics=statistics,
        stage2={
            "normalization": {"case_insensitive": True, "require_statistics": True}
        },
    )
    torch.testing.assert_close(legacy[0]["static"], torch.full((1, 2, 2), 300.0))
    torch.testing.assert_close(corrected[0]["static"], torch.ones(1, 2, 2))
    with pytest.raises(ValueError, match="标准化统计"):
        WindowedSampleDataset(
            index,
            window_size=(2, 2),
            stage2={"normalization": {"require_statistics": True}},
        )


def test_sample_index_round_trip_and_dataset_reads_both_labels(tmp_path: Path) -> None:
    dynamic = tmp_path / "dynamic"
    static = tmp_path / "static"
    dynamic.mkdir()
    static.mkdir()
    dynamic_path = dynamic / "NDVI230101.tif"
    _write_raster(dynamic_path, np.arange(16, dtype=np.float32).reshape(4, 4))
    _write_raster(dynamic / "SR230101B2.tif", np.full((4, 4), 2, dtype=np.int16))
    _write_raster(static / "DEM.tif", np.ones((4, 4), dtype=np.float32))
    weak_path = tmp_path / "weak_labels.tif"
    _write_raster(weak_path, np.full((4, 4), 7, dtype=np.int16))

    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "Index",
                "X",
                "Y",
                "Eng_Formation",
                "Eng_Alliance",
                "Formation",
                "Alliance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Index": "p1",
                "X": "0.5",
                "Y": "3.5",
                "Eng_Formation": "major",
                "Eng_Alliance": "minor",
                "Formation": "大类",
                "Alliance": "小类",
            }
        )

    grid = target_grid_from_raster(dynamic_path, target_crs="EPSG:4326", resolution=1)
    index_path = tmp_path / "sample_index.json"
    build_sample_index(
        dynamic_dir=dynamic,
        static_dir=static,
        target_grid=grid,
        label_file=labels_path,
        weak_label_file=weak_path,
        output=index_path,
    )
    index = load_sample_index(index_path)
    statistics = {
        "groups": [
            {"category": "NDVI", "statistics": {"mean": 1, "standard_deviation": 2}},
            {"category": "DEM", "statistics": {"mean": 1, "standard_deviation": 2}},
        ]
    }
    dataset = WindowedSampleDataset(index, window_size=(2, 2), statistics=statistics)

    sample = dataset[0]
    assert len(dataset) == 4
    assert sample["dynamic"].shape == (1, 2, 2, 2)
    assert sample["dynamic_mask"].tolist() == [[True, True]]
    assert sample["static"].shape == (1, 2, 2)
    assert sample["ground_truth"].shape == (2, 2)
    assert sample["ground_truth_mask"].sum() == 1
    assert sample["ground_truth"][0, 0] == 1
    assert sample["weak_label_mask"].all()
    assert torch.all(sample["weak_label"] == 7)
    assert torch.allclose(sample["dynamic"][0, 0, 0, 0], torch.tensor(-0.5))
    assert torch.allclose(sample["dynamic"][0, 1, 0, 0], torch.tensor(2.0))
    assert torch.allclose(sample["static"][:, 0, 0], torch.tensor([0.0]))
    assert sample["sample_status"]["boundary_clipped"] is False
    short_sample = dict(sample)
    short_sample["dynamic"] = sample["dynamic"][:0]
    short_sample["dynamic_mask"] = sample["dynamic_mask"][:0]
    short_sample["dynamic_time_mask"] = sample["dynamic_time_mask"][:0]
    short_sample["time_encoding"] = sample["time_encoding"][:0]
    short_sample["dynamic_times"] = []
    batch = sample_collate_fn([sample, short_sample])
    assert batch["dynamic"].shape == (2, 1, 2, 2, 2)
    assert batch["dynamic_time_mask"].tolist() == [[True], [False]]
    manifest = build_spatial_split(
        dataset, block_size=(2, 2), ratios=(0.5, 0.25, 0.25), seed=7
    )
    split_ids = [window_id for ids in manifest.splits.values() for window_id in ids]
    assert sorted(split_ids) == list(range(len(dataset)))
    assert not (set(manifest.splits["train"]) & set(manifest.splits["test"]))
    sampler = SpatialWeightedSampler(dataset, manifest=manifest, seed=3)
    assert len(sampler) == len(manifest.splits["train"])
    assert set(iter(sampler)).issubset(set(manifest.splits["train"]))
    loader = build_dataloader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    assert batch["dynamic"].shape == (1, 1, 2, 2, 2)


def test_valid_mask_survives_partial_static_nodata(tmp_path: Path) -> None:
    dynamic = tmp_path / "dynamic"
    static = tmp_path / "static"
    dynamic.mkdir()
    static.mkdir()
    _write_raster(dynamic / "NDVI230101.tif", np.ones((2, 2), dtype=np.float32))
    first = np.ones((2, 2), dtype=np.float32)
    first[0, 0] = -9999
    _write_raster(static / "A.tif", first)
    _write_raster(static / "B.tif", np.ones((2, 2), dtype=np.float32))
    grid = target_grid_from_raster(
        dynamic / "NDVI230101.tif", target_crs="EPSG:4326", resolution=1
    )
    index = build_sample_index(dynamic_dir=dynamic, static_dir=static, target_grid=grid)
    sample = WindowedSampleDataset(index, window_size=(2, 2))[0]
    assert sample["valid_mask"].tolist() == [[True, True], [True, True]]


def test_fixed_split_ownership_masks_ground_truth_and_weak_labels(tmp_path: Path):
    dynamic = tmp_path / "dynamic"
    static = tmp_path / "static"
    dynamic.mkdir()
    static.mkdir()
    dynamic_path = dynamic / "NDVI230101.tif"
    _write_raster(dynamic_path, np.ones((4, 4), dtype=np.float32))
    _write_raster(static / "DEM.tif", np.ones((4, 4), dtype=np.float32))
    weak_path = tmp_path / "weak.tif"
    _write_raster(weak_path, np.ones((4, 4), dtype=np.int16))
    grid = target_grid_from_raster(dynamic_path, target_crs="EPSG:4326", resolution=1)
    index = build_sample_index(
        dynamic_dir=dynamic,
        static_dir=static,
        target_grid=grid,
        weak_label_file=weak_path,
    )
    train = WindowedSampleDataset(index, window_size=(4, 4))
    validation = WindowedSampleDataset(index, window_size=(4, 4), use_weak_labels=False)
    train._ground_truth = {(0, 0): 1, (1, 0): 1, (0, 2): 1}
    validation._ground_truth = dict(train._ground_truth)
    manifest = SpatialSplitManifest(
        schema_version=1,
        seed=1,
        block_size=(2, 2),
        ratios=(0.5, 0.5, 0),
        splits={"train": [0], "validation": [0], "test": []},
        blocks={
            "0:0": "train",
            "0:1": "validation",
            "1:0": "train",
            "1:1": "validation",
        },
        class_counts={},
        class_weights={},
        sampling_weights={},
    )
    train.configure_supervision_split(manifest, "train", mask_weak_labels=True)
    validation.configure_supervision_split(
        manifest, "validation", mask_weak_labels=False
    )
    train_sample = train[0]
    validation_sample = validation[0]
    assert train_sample["ground_truth_mask"].nonzero().tolist() == [[0, 0], [1, 0]]
    selected_sample = train[(0, (0, 0))]
    assert selected_sample["ground_truth_mask"].nonzero().tolist() == [[0, 0]]
    assert validation_sample["ground_truth_mask"].nonzero().tolist() == [[0, 2]]
    assert not (
        train_sample["ground_truth_mask"] & validation_sample["ground_truth_mask"]
    ).any()
    assert train_sample["weak_label_mask"][:, :2].all()
    assert not train_sample["weak_label_mask"][:, 2:].any()
