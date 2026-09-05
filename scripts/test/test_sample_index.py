"""Tests for the unified sample index and lazy TorchGeo dataset."""

import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from data.raster_alignment import target_grid_from_raster
from data.sample_index import (
    WindowedSampleDataset,
    build_sample_index,
    load_sample_index,
    sample_collate_fn,
)
from data.sampling import SpatialWeightedSampler, build_dataloader
from data.spatial_split import build_spatial_split


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
