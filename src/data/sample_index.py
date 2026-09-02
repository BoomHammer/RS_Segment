"""Unified raster/sample index and lazy TorchGeo dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import CRS
from rasterio.windows import Window
from shapely.geometry import Point, box
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox

from .filename_parser import RasterMetadata, scan_dynamic_directory
from .labels import LabelRecord, build_label_mapping, iter_encoded_labels
from .raster_alignment import TargetGrid, aligned_raster, locate_points

INDEX_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class RasterAsset:
    """One lazily readable raster asset in the unified index."""

    path: str
    role: str
    name: str
    category: str | None
    temporal_resolution: str | None
    date: str | None
    month: str | None
    band: int | None
    crs: str
    bounds: tuple[float, float, float, float]
    width: int
    height: int
    count: int
    dtype: str
    nodata: float | int | None
    resampling: str = "nearest"


@dataclass(frozen=True, slots=True)
class SampleIndex:
    """Serializable contract shared by indexing and dataset construction."""

    schema_version: int
    created_at: str
    target_grid: dict[str, Any]
    assets: tuple[RasterAsset, ...]
    ground_truth: dict[str, Any] | None
    weak_label: dict[str, Any] | None
    temporal_groups: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "target_grid": self.target_grid,
            "assets": [asdict(asset) for asset in self.assets],
            "ground_truth": self.ground_truth,
            "weak_label": self.weak_label,
            "temporal_groups": list(self.temporal_groups),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SampleIndex:
        version = int(payload.get("schema_version", -1))
        if version not in {1, INDEX_SCHEMA_VERSION}:
            raise ValueError("不支持的样本索引 schema_version")
        assets = []
        for item in payload["assets"]:
            asset = dict(item)
            asset.setdefault(
                "resampling",
                "bilinear" if asset.get("role") in {"dynamic", "static"} else "nearest",
            )
            assets.append(RasterAsset(**asset))
        assets_tuple = tuple(assets)
        return cls(
            schema_version=INDEX_SCHEMA_VERSION,
            created_at=str(payload["created_at"]),
            target_grid=dict(payload["target_grid"]),
            assets=assets_tuple,
            ground_truth=payload.get("ground_truth"),
            weak_label=payload.get("weak_label"),
            temporal_groups=tuple(payload.get("temporal_groups", ()))
            or _temporal_groups(list(assets_tuple)),
        )


def _asset(
    path: Path,
    role: str,
    metadata: RasterMetadata | None = None,
    *,
    resampling: str,
) -> RasterAsset:
    with rasterio.open(path) as dataset:
        bounds = tuple(float(value) for value in dataset.bounds)
        return RasterAsset(
            path=str(path),
            role=role,
            name=metadata.category if metadata else path.stem,
            category=metadata.category if metadata else None,
            temporal_resolution=metadata.temporal_resolution if metadata else None,
            date=metadata.date if metadata else None,
            month=metadata.month if metadata else None,
            band=metadata.band if metadata else None,
            crs=str(dataset.crs),
            bounds=bounds,
            width=dataset.width,
            height=dataset.height,
            count=dataset.count,
            dtype=str(dataset.dtypes[0]),
            nodata=dataset.nodata,
            resampling=resampling,
        )


def _feature_name(asset: RasterAsset) -> str:
    return (
        f"{asset.category}_B{asset.band}"
        if asset.band is not None
        else str(asset.category)
    )


def _timestamp(asset: RasterAsset) -> str:
    return asset.date or asset.month or "static"


def _time_encoding(timestamp: str) -> list[float]:
    """Encode date/month timestamps as normalized and cyclical time features."""

    if len(timestamp) == 7:
        value = datetime.strptime(timestamp, "%Y-%m").date()
    else:
        value = datetime.strptime(timestamp, "%Y-%m-%d").date()
    day = value.timetuple().tm_yday
    angle = 2 * np.pi * (day - 1) / 365.25
    return [float((day - 1) / 365.25), float(np.sin(angle)), float(np.cos(angle))]


def _temporal_groups(assets: list[RasterAsset]) -> tuple[dict[str, Any], ...]:
    groups: dict[str, list[RasterAsset]] = {}
    for asset in assets:
        if asset.role == "dynamic":
            groups.setdefault(_timestamp(asset), []).append(asset)
    return tuple(
        {
            "timestamp": timestamp,
            "features": tuple(
                {"name": _feature_name(asset), "path": asset.path}
                for asset in sorted(items, key=_feature_name)
            ),
        }
        for timestamp, items in sorted(groups.items())
    )


def _grid_dict(grid: TargetGrid) -> dict[str, Any]:
    return {
        "crs": grid.crs.to_string(),
        "transform": list(grid.transform),
        "width": grid.width,
        "height": grid.height,
    }


def build_sample_index(
    *,
    dynamic_dir: str | Path,
    static_dir: str | Path,
    target_grid: TargetGrid,
    label_file: str | Path | None = None,
    weak_label_file: str | Path | None = None,
    label_crs: str = "EPSG:4326",
    continuous_resampling: str = "bilinear",
    categorical_resampling: str = "nearest",
    output: str | Path | None = None,
) -> SampleIndex:
    """Discover assets and write a bounded, versioned sample index.

    The CSV and weak-label paths are recorded separately so consumers cannot
    accidentally replace the measured labels with the weak labels.
    """

    dynamic = scan_dynamic_directory(dynamic_dir)
    assets = [
        _asset(Path(item.path), "dynamic", item, resampling=continuous_resampling)
        for item in dynamic
    ]
    assets.extend(
        _asset(path, "static", resampling=continuous_resampling)
        for path in sorted(Path(static_dir).glob("*.tif"))
    )
    ground_truth = (
        {"path": str(Path(label_file)), "crs": label_crs, "format": "csv"}
        if label_file is not None
        else None
    )
    weak_label = (
        {
            "path": str(Path(weak_label_file)),
            "format": "geotiff",
            "resampling": categorical_resampling,
        }
        if weak_label_file is not None
        else None
    )
    index = SampleIndex(
        schema_version=INDEX_SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        target_grid=_grid_dict(target_grid),
        assets=tuple(assets),
        ground_truth=ground_truth,
        weak_label=weak_label,
        temporal_groups=_temporal_groups(assets),
    )
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return index


def load_sample_index(path: str | Path) -> SampleIndex:
    """Load and validate a previously written sample index."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SampleIndex.from_dict(payload)


def load_raster_statistics(
    source: str | Path | dict[str, Any] | None,
) -> dict[str, tuple[float, float]]:
    """Load ``category -> (mean, standard deviation)`` normalization constants."""

    if source is None:
        return {}
    payload = (
        json.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(source, (str, Path))
        else source
    )
    result = {}
    for group in payload.get("groups", []):
        statistics = group.get("statistics", {})
        mean = statistics.get("mean")
        standard_deviation = statistics.get("standard_deviation")
        if mean is not None and standard_deviation not in {None, 0}:
            result[str(group["category"])] = (float(mean), float(standard_deviation))
    return result


def _grid_from_index(index: SampleIndex) -> TargetGrid:
    from affine import Affine

    grid = index.target_grid
    return TargetGrid(
        crs=CRS.from_user_input(grid["crs"]),
        transform=Affine(*grid["transform"]),
        width=int(grid["width"]),
        height=int(grid["height"]),
    )


def _window(row: int, column: int, size: tuple[int, int], grid: TargetGrid) -> Window:
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("window_size 必须是两个正整数")
    left = min(max(column, 0), max(grid.width - width, 0))
    top = min(max(row, 0), max(grid.height - height, 0))
    return Window(left, top, min(width, grid.width), min(height, grid.height))


class WindowedSampleDataset(GeoDataset):
    """TorchGeo dataset that reads every source only for the requested window.

    Rasterio handles are deliberately scoped to one read. This makes dataset
    copies in multiprocessing workers independent and avoids inherited handles.
    """

    def __init__(
        self,
        index: SampleIndex | str | Path,
        *,
        window_size: tuple[int, int] = (256, 256),
        stride: tuple[int, int] | None = None,
        label_columns: dict[str, str] | None = None,
        label_mapping: dict[str, Any] | None = None,
        statistics: str | Path | dict[str, Any] | None = None,
        nodata: float | int = -9999,
        stage2: dict[str, Any] | None = None,
        transforms: Any = None,
    ) -> None:
        self.sample_index = (
            load_sample_index(index) if isinstance(index, (str, Path)) else index
        )
        self.grid = _grid_from_index(self.sample_index)
        self.window_size = window_size
        self.stride = stride or window_size
        self.transforms = transforms
        self.statistics = load_raster_statistics(statistics)
        self.nodata = float(nodata)
        self._label_columns = label_columns or {}
        self._ground_truth: dict[tuple[int, int], int] = {}
        self._load_ground_truth(label_mapping)
        self._configure_stage2(stage2 or {})
        rows = range(0, self.grid.height, self.stride[1])
        columns = range(0, self.grid.width, self.stride[0])
        records = []
        for row in rows:
            for column in columns:
                window = _window(row, column, window_size, self.grid)
                left, bottom, right, top = rasterio.windows.bounds(
                    window, self.grid.transform
                )
                records.append(
                    {
                        "row": row,
                        "column": column,
                        "geometry": box(left, bottom, right, top),
                    }
                )
        geometry = gpd.GeoSeries(
            [item.pop("geometry") for item in records], crs=self.grid.crs
        )
        self.index = gpd.GeoDataFrame(records, geometry=geometry)
        self.index.index = pd.IntervalIndex.from_tuples(
            [(datetime.min, datetime.max)] * len(records),
            closed="both",
            name="datetime",
        )
        self._res = self.grid.resolution

    def _configure_stage2(self, config: dict[str, Any]) -> None:
        features = dict(config.get("features", {}))
        dynamic_assets = [
            asset for asset in self.sample_index.assets if asset.role == "dynamic"
        ]
        static_assets = [
            asset for asset in self.sample_index.assets if asset.role == "static"
        ]
        selected_dynamic = set(features.get("dynamic", []))
        selected_static = set(features.get("static", []))
        if selected_dynamic:
            dynamic_assets = [
                asset
                for asset in dynamic_assets
                if _feature_name(asset) in selected_dynamic
            ]
        if selected_static:
            static_assets = [
                asset for asset in static_assets if asset.name in selected_static
            ]
        order = list(features.get("dynamic_order", []))
        feature_order = order + sorted(
            {_feature_name(asset) for asset in dynamic_assets} - set(order)
        )
        time_config = dict(config.get("time", {}))
        start = str(time_config.get("start", "0001-01-01"))
        end = str(time_config.get("end", "9999-12-31"))
        start_date = datetime.fromisoformat(start).date()
        end_date = datetime.fromisoformat(end).date()
        dynamic_assets = [
            asset
            for asset in dynamic_assets
            if start_date
            <= datetime.fromisoformat(
                _timestamp(asset) + "-01"
                if len(_timestamp(asset)) == 7
                else _timestamp(asset)
            ).date()
            <= end_date
        ]
        times = sorted({_timestamp(asset) for asset in dynamic_assets})
        lookup = {(_timestamp(asset), _feature_name(asset)) for asset in dynamic_assets}
        if time_config.get("missing", "mask_nan") == "drop":
            times = [
                timestamp
                for timestamp in times
                if all((timestamp, feature) in lookup for feature in feature_order)
            ]
            dynamic_assets = [
                asset for asset in dynamic_assets if _timestamp(asset) in times
            ]
        self._dynamic_assets = dynamic_assets
        self._static_assets = static_assets
        self._dynamic_features = feature_order or sorted(
            {_feature_name(asset) for asset in dynamic_assets}
        )
        self._dynamic_times = times

    def _load_ground_truth(self, mapping: dict[str, Any] | None) -> None:
        source = self.sample_index.ground_truth
        if source is None:
            return
        if mapping is None:
            mapping = build_label_mapping(
                source["path"], label_columns=self._label_columns
            )
        records: list[LabelRecord] = [
            record
            for batch in iter_encoded_labels(
                source["path"], label_columns=self._label_columns, mapping=mapping
            )
            for record in batch
        ]
        locations = locate_points(
            [(record.x, record.y) for record in records],
            point_crs=source.get("crs", "EPSG:4326"),
            grid=self.grid,
        )
        self._ground_truth = {
            (item["row"], item["column"]): record.alliance_code
            for item, record in zip(locations, records, strict=True)
            if item["inside"]
        }

    def __getitem__(self, index: int | BoundingBox) -> dict[str, Any]:
        if isinstance(index, int):
            row = self.index.iloc[index]
            window = _window(int(row.row), int(row.column), self.window_size, self.grid)
        elif isinstance(index, BoundingBox):
            window = self._window_from_bounds(index)
        else:
            raise TypeError("样本索引必须是整数或 TorchGeo BoundingBox")
        sample = self._read_window(window)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _window_from_bounds(self, query: BoundingBox) -> Window:
        grid_left, grid_bottom, grid_right, grid_top = self.grid.bounds()
        left = max(query.minx, grid_left)
        right = min(query.maxx, grid_right)
        bottom = max(query.miny, grid_bottom)
        top = min(query.maxy, grid_top)
        if left >= right or bottom >= top:
            raise IndexError("空间查询与目标网格没有交集")
        column = int(np.floor((left - grid_left) / self.grid.resolution[0]))
        row = int(np.floor((top - grid_top) / self.grid.transform.e))
        width = int(np.ceil((right - left) / self.grid.resolution[0]))
        height = int(np.ceil((top - bottom) / self.grid.resolution[1]))
        return Window(
            column,
            row,
            min(width, self.grid.width - column),
            min(height, self.grid.height - row),
        )

    def _read_asset(self, asset: RasterAsset, window: Window) -> np.ndarray:
        with aligned_raster(
            asset.path, self.grid, resampling=asset.resampling
        ) as dataset:
            # ``asset.band`` is the spectral identifier in filenames such as
            # SR230218B2.tif; each such file is itself a single-band GeoTIFF.
            band = asset.band if asset.count > 1 and asset.band else 1
            if band > dataset.count:
                band = 1
            values = dataset.read(band, window=window, masked=True)
            result = np.asarray(
                values.astype(np.float32).filled(np.nan), dtype=np.float32
            )
        invalid = ~np.isfinite(result)
        if asset.nodata is not None:
            invalid |= result == float(asset.nodata)
        invalid |= result == self.nodata
        result[invalid] = np.nan
        return result

    def _normalized(self, values: np.ndarray, key: str) -> np.ndarray:
        statistics = self.statistics.get(key)
        if statistics is None:
            return values
        mean, standard_deviation = statistics
        return (values - mean) / standard_deviation

    def _read_window(self, window: Window) -> dict[str, Any]:
        dynamic_assets = self._dynamic_assets
        dynamic_features = self._dynamic_features
        dynamic_times = self._dynamic_times
        asset_lookup = {
            (
                asset.date or asset.month or "static",
                _feature_name(asset),
            ): asset
            for asset in dynamic_assets
        }
        dynamic_values: list[list[np.ndarray]] = []
        dynamic_mask = np.zeros((len(dynamic_times), len(dynamic_features)), dtype=bool)
        for time_index, timestamp in enumerate(dynamic_times):
            time_values = []
            for feature_index, feature in enumerate(dynamic_features):
                asset = asset_lookup.get((timestamp, feature))
                if asset is None:
                    time_values.append(
                        np.full(
                            (int(window.height), int(window.width)),
                            np.nan,
                            dtype=np.float32,
                        )
                    )
                else:
                    time_values.append(
                        self._normalized(
                            self._read_asset(asset, window), _feature_name(asset)
                        )
                    )
                    dynamic_mask[time_index, feature_index] = True
            dynamic_values.append(time_values)
        dynamic_array = np.stack(dynamic_values)
        static = self._static_assets
        static_array = np.stack(
            [
                self._normalized(self._read_asset(asset, window), asset.name)
                for asset in static
            ]
        )
        dynamic_valid = np.isfinite(dynamic_array).any(axis=(0, 1))
        static_valid = np.isfinite(static_array).all(axis=0)
        shape = dynamic_array.shape[-2:]
        weak = np.full(shape, -1, dtype=np.int64)
        if self.sample_index.weak_label is not None:
            weak_values = self._read_asset(
                _asset(
                    Path(self.sample_index.weak_label["path"]),
                    "weak_label",
                    resampling=self.sample_index.weak_label.get(
                        "resampling", "nearest"
                    ),
                ),
                window,
            )
            weak = np.where(np.isfinite(weak_values), weak_values, -1).astype(np.int64)
        ground_truth = np.full(shape, -1, dtype=np.int64)
        row_start, col_start = int(window.row_off), int(window.col_off)
        for (row, column), code in self._ground_truth.items():
            local_row, local_column = row - row_start, column - col_start
            if 0 <= local_row < shape[0] and 0 <= local_column < shape[1]:
                ground_truth[local_row, local_column] = code
        boundary_clipped = (
            int(window.width) != self.window_size[0]
            or int(window.height) != self.window_size[1]
        )
        return {
            "dynamic": torch.from_numpy(dynamic_array),
            "dynamic_mask": torch.from_numpy(dynamic_mask),
            "dynamic_time_mask": torch.from_numpy(dynamic_mask.any(axis=1)),
            "time_encoding": torch.tensor(
                [_time_encoding(timestamp) for timestamp in dynamic_times],
                dtype=torch.float32,
            ),
            "dynamic_times": dynamic_times,
            "dynamic_features": dynamic_features,
            "static": torch.from_numpy(static_array),
            "dynamic_valid_mask": torch.from_numpy(dynamic_valid),
            "static_valid_mask": torch.from_numpy(static_valid),
            "valid_mask": torch.from_numpy(dynamic_valid & static_valid),
            "ground_truth": torch.from_numpy(ground_truth),
            "ground_truth_mask": torch.from_numpy(ground_truth >= 0),
            "weak_label": torch.from_numpy(weak),
            "weak_label_mask": torch.from_numpy(weak >= 0),
            "crs": self.grid.crs,
            "bounds": rasterio.windows.bounds(window, self.grid.transform),
            "window": window,
            "sample_status": {
                "boundary_clipped": boundary_clipped,
                "window": [
                    int(window.col_off),
                    int(window.row_off),
                    int(window.width),
                    int(window.height),
                ],
                "ground_truth_pixels": int((ground_truth >= 0).sum()),
                "weak_label_pixels": int((weak >= 0).sum()),
                "valid_pixels": int((dynamic_valid & static_valid).sum()),
            },
        }

    def query_windows(self, query: BoundingBox) -> list[int]:
        """Return indexed windows intersecting a TorchGeo spatial query."""

        geometry = box(query.minx, query.miny, query.maxx, query.maxy)
        return np.flatnonzero(self.index.intersects(geometry)).tolist()

    @property
    def ground_truth_pixels(self) -> dict[tuple[int, int], int]:
        """Return target-grid point labels used to build split statistics."""

        return dict(self._ground_truth)

    def query_windows_for_pixel(self, row: int, column: int) -> list[int]:
        """Return window positions covering one target-grid pixel."""

        point = self.grid.transform * (column + 0.5, row + 0.5)
        return np.flatnonzero(self.index.intersects(Point(point))).tolist()

    def __len__(self) -> int:
        return len(self.index)

    def close(self) -> None:
        """Close worker-local resources; reads currently use scoped handles."""


def sample_collate_fn(
    samples: list[dict[str, Any]], *, pad_value: float = float("nan")
) -> dict[str, Any]:
    """Collate variable-length temporal samples without loading extra rasters."""

    if not samples:
        raise ValueError("samples 不能为空")
    max_time = max(int(sample["dynamic"].shape[0]) for sample in samples)
    max_features = max(int(sample["dynamic"].shape[1]) for sample in samples)
    _, _, height, width = samples[0]["dynamic"].shape
    dynamic = torch.full(
        (len(samples), max_time, max_features, height, width),
        pad_value,
        dtype=torch.float32,
    )
    feature_mask = torch.zeros((len(samples), max_time, max_features), dtype=torch.bool)
    time_mask = torch.zeros((len(samples), max_time), dtype=torch.bool)
    time_encoding = torch.zeros((len(samples), max_time, 3), dtype=torch.float32)
    for batch_index, sample in enumerate(samples):
        current_time, current_features = sample["dynamic"].shape[:2]
        dynamic[batch_index, :current_time, :current_features] = sample["dynamic"]
        feature_mask[batch_index, :current_time, :current_features] = sample[
            "dynamic_mask"
        ]
        time_mask[batch_index, :current_time] = sample["dynamic_time_mask"]
        time_encoding[batch_index, :current_time] = sample["time_encoding"]
    return {
        "dynamic": dynamic,
        "dynamic_mask": feature_mask,
        "dynamic_time_mask": time_mask,
        "time_encoding": time_encoding,
        "static": torch.stack([sample["static"] for sample in samples]),
        "dynamic_valid_mask": torch.stack(
            [sample["dynamic_valid_mask"] for sample in samples]
        ),
        "static_valid_mask": torch.stack(
            [sample["static_valid_mask"] for sample in samples]
        ),
        "valid_mask": torch.stack([sample["valid_mask"] for sample in samples]),
        "ground_truth": torch.stack([sample["ground_truth"] for sample in samples]),
        "ground_truth_mask": torch.stack(
            [sample["ground_truth_mask"] for sample in samples]
        ),
        "weak_label": torch.stack([sample["weak_label"] for sample in samples]),
        "weak_label_mask": torch.stack(
            [sample["weak_label_mask"] for sample in samples]
        ),
        "dynamic_times": [sample["dynamic_times"] for sample in samples],
        "dynamic_features": [sample["dynamic_features"] for sample in samples],
        "sample_status": [sample["sample_status"] for sample in samples],
        "crs": [sample["crs"] for sample in samples],
        "bounds": [sample["bounds"] for sample in samples],
        "window": [sample["window"] for sample in samples],
    }
