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
from .raster_alignment import TargetGrid, locate_points
from .raster_pool import RasterReaderPool
from .value_ranges import (
    ValueRange,
    load_value_ranges,
    valid_and_scaled,
    value_range_for,
)

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


def _statistics_payload(source: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return {}
    return (
        json.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(source, (str, Path))
        else source
    )


def _validate_statistics_value_ranges(
    payload: dict[str, Any], ranges: dict[str, ValueRange]
) -> None:
    """Reject raw-DN statistics when scaled product rules are configured."""

    if not payload:
        return
    if int(payload.get("schema_version", 0)) < 4:
        raise ValueError("栅格统计量未应用有效值范围和 Scale，请重新生成统计量")
    for group in payload.get("groups", []):
        category = str(group["category"])
        expected = value_range_for(category, ranges)
        recorded = group.get("value_range")
        if expected is None or recorded is None:
            raise ValueError(f"栅格统计量缺少有效值规则: {category}")
        actual = ValueRange(
            float(recorded["minimum"]),
            float(recorded["maximum"]),
            float(recorded["scale"]),
        )
        if actual != expected:
            raise ValueError(f"栅格统计量有效值规则已变化: {category}")


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


def _expanded_window(window: Window, halo: tuple[int, int], grid: TargetGrid) -> Window:
    halo_x, halo_y = halo
    left = max(0, int(window.col_off) - halo_x)
    top = max(0, int(window.row_off) - halo_y)
    right = min(grid.width, int(window.col_off + window.width) + halo_x)
    bottom = min(grid.height, int(window.row_off + window.height) + halo_y)
    return Window(left, top, right - left, bottom - top)


class WindowedSampleDataset(GeoDataset):
    """TorchGeo dataset that reads every source only for the requested window.

    Rasterio handles are scoped to one read by default. Optional bounded pools
    reuse metadata within each worker; handles are never serialized to workers.
    """

    def __init__(
        self,
        index: SampleIndex | str | Path,
        *,
        window_size: tuple[int, int] = (256, 256),
        stride: tuple[int, int] | None = None,
        grid_offset: tuple[int, int] = (0, 0),
        halo: tuple[int, int] = (0, 0),
        label_columns: dict[str, str] | None = None,
        label_mapping: dict[str, Any] | None = None,
        statistics: str | Path | dict[str, Any] | None = None,
        nodata: float | int = -9999,
        stage2: dict[str, Any] | None = None,
        use_weak_labels: bool = True,
        transforms: Any = None,
    ) -> None:
        self.sample_index = (
            load_sample_index(index) if isinstance(index, (str, Path)) else index
        )
        self.grid = _grid_from_index(self.sample_index)
        self.window_size = window_size
        self.stride = stride or window_size
        if len(grid_offset) != 2 or min(grid_offset) < 0:
            raise ValueError("grid_offset 必须是两个非负整数")
        if grid_offset[0] >= self.stride[0] or grid_offset[1] >= self.stride[1]:
            raise ValueError("grid_offset 必须小于 stride")
        self.grid_offset = grid_offset
        if len(halo) != 2 or min(halo) < 0:
            raise ValueError("halo 必须是两个非负整数")
        self.halo = halo
        self.transforms = transforms
        self.use_weak_labels = use_weak_labels
        self._supervision_split: str | None = None
        self._supervision_block_size: tuple[int, int] | None = None
        self._supervision_blocks: dict[str, str] = {}
        self._mask_weak_labels_by_split = False
        stage2_config = stage2 or {}
        self._raster_pool = RasterReaderPool(
            self.grid,
            max_items=int(stage2_config.get("io", {}).get("max_open_rasters", 0)),
        )
        self.value_ranges = load_value_ranges(stage2_config.get("value_range_file"))
        statistics_payload = _statistics_payload(statistics)
        if self.value_ranges:
            _validate_statistics_value_ranges(statistics_payload, self.value_ranges)
        self.statistics = load_raster_statistics(statistics_payload)
        self.nodata = float(nodata)
        self._label_columns = label_columns or {}
        self._ground_truth: dict[tuple[int, int], int] = {}
        self._load_ground_truth(label_mapping)
        self._configure_stage2(stage2_config)
        rows = range(self.grid_offset[1], self.grid.height, self.stride[1])
        columns = range(self.grid_offset[0], self.grid.width, self.stride[0])
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
        normalization = dict(config.get("normalization", {}))
        # Preserve the input convention of old experiment snapshots. New runs
        # opt in explicitly and persist the setting alongside their weights.
        self.normalization_case_insensitive = bool(
            normalization.get("case_insensitive", False)
        )
        self._casefold_statistics = {}
        for key, value in self.statistics.items():
            folded = key.casefold()
            if (
                self.normalization_case_insensitive
                and folded in self._casefold_statistics
                and self._casefold_statistics[folded] != value
            ):
                raise ValueError(f"大小写不敏感统计键冲突: {key}")
            self._casefold_statistics[folded] = value
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
        if self.value_ranges:
            missing_ranges = sorted(
                {
                    self._value_range_key(asset)
                    for asset in (*dynamic_assets, *static_assets)
                    if self._value_range_for_asset(asset) is None
                }
            )
            if missing_ranges:
                raise ValueError(f"输入特征缺少有效值范围: {missing_ranges}")
        if normalization.get("require_statistics", False):
            keys = [_feature_name(asset) for asset in dynamic_assets]
            keys.extend(asset.name for asset in static_assets)
            missing = sorted({key for key in keys if self._statistics_for(key) is None})
            if missing:
                raise ValueError(f"输入特征缺少标准化统计: {missing}")

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

    def configure_supervision_split(
        self, manifest: Any, split: str, *, mask_weak_labels: bool
    ) -> None:
        """Expose supervision owned by one fixed spatial split only."""

        if split not in manifest.splits:
            raise ValueError(f"未知监督划分: {split}")
        self._supervision_split = split
        self._supervision_block_size = tuple(manifest.block_size)
        self._supervision_blocks = dict(manifest.blocks)
        self._mask_weak_labels_by_split = mask_weak_labels

    def _owner(self, row: int, column: int) -> str | None:
        if self._supervision_block_size is None:
            return None
        width, height = self._supervision_block_size
        return self._supervision_blocks.get(f"{row // height}:{column // width}")

    def _owned_mask(self, window: Window) -> np.ndarray:
        shape = (int(window.height), int(window.width))
        if self._supervision_split is None or self._supervision_block_size is None:
            return np.ones(shape, dtype=bool)
        rows = np.arange(int(window.row_off), int(window.row_off + window.height))
        columns = np.arange(int(window.col_off), int(window.col_off + window.width))
        width, height = self._supervision_block_size
        block_rows = rows // height
        block_columns = columns // width
        result = np.zeros(shape, dtype=bool)
        for block_row in np.unique(block_rows):
            for block_column in np.unique(block_columns):
                if (
                    self._supervision_blocks.get(f"{block_row}:{block_column}")
                    == self._supervision_split
                ):
                    result[
                        np.ix_(block_rows == block_row, block_columns == block_column)
                    ] = True
        return result

    def __getitem__(
        self, index: int | tuple[int, tuple[int, int]] | BoundingBox
    ) -> dict[str, Any]:
        selected_ground_truth = None
        if isinstance(index, tuple):
            index, selected_ground_truth = index
        if isinstance(index, int):
            row = self.index.iloc[index]
            window = _window(int(row.row), int(row.column), self.window_size, self.grid)
        elif isinstance(index, BoundingBox):
            window = self._window_from_bounds(index)
        else:
            raise TypeError("样本索引必须是整数或 TorchGeo BoundingBox")
        input_window = _expanded_window(window, self.halo, self.grid)
        sample = (
            self._read_window(input_window)
            if selected_ground_truth is None
            else self._read_window(input_window, selected_ground_truth)
        )
        sample["window"] = window
        sample["input_window"] = input_window
        core_mask = torch.zeros_like(sample["valid_mask"])
        offset_y = int(window.row_off - input_window.row_off)
        offset_x = int(window.col_off - input_window.col_off)
        core_mask[
            offset_y : offset_y + int(window.height),
            offset_x : offset_x + int(window.width),
        ] = True
        sample["core_mask"] = core_mask
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
        with self._raster_pool.borrow(asset.path, asset.resampling) as dataset:
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
        value_range = self._value_range_for_asset(asset)
        if value_range is not None:
            result = valid_and_scaled(result, value_range)
        return result

    @staticmethod
    def _value_range_key(asset: RasterAsset) -> str:
        return _feature_name(asset) if asset.role == "dynamic" else asset.name

    def _value_range_for_asset(self, asset: RasterAsset) -> ValueRange | None:
        if asset.role not in {"dynamic", "static"}:
            return None
        return value_range_for(self._value_range_key(asset), self.value_ranges)

    def _normalized(self, values: np.ndarray, key: str) -> np.ndarray:
        statistics = self._statistics_for(key)
        if statistics is None:
            return values
        mean, standard_deviation = statistics
        return (values - mean) / standard_deviation

    def _statistics_for(self, key: str) -> tuple[float, float] | None:
        if self.normalization_case_insensitive:
            return self._casefold_statistics.get(key.casefold())
        return self.statistics.get(key)

    def _read_window(
        self, window: Window, selected_ground_truth: tuple[int, int] | None = None
    ) -> dict[str, Any]:
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
        # A pixel is usable when at least one input observation is usable.
        # Clouds or swaths missing from one frame/layer must not erase other
        # sources. Only the union of "any dynamic valid" and "any static
        # valid" is output-valid; all-inputs-invalid remains NoData.
        dynamic_valid = np.isfinite(dynamic_array).any(axis=(0, 1))
        static_valid = np.isfinite(static_array).any(axis=0)
        shape = dynamic_array.shape[-2:]
        supervision_split_mask = self._owned_mask(window)
        weak = np.full(shape, -1, dtype=np.int64)
        if self.use_weak_labels and self.sample_index.weak_label is not None:
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
            if self._mask_weak_labels_by_split:
                weak = np.where(supervision_split_mask, weak, -1)
        ground_truth = np.full(shape, -1, dtype=np.int64)
        row_start, col_start = int(window.row_off), int(window.col_off)
        for (row, column), code in self._ground_truth.items():
            if (
                selected_ground_truth is not None
                and (row, column) != selected_ground_truth
            ):
                continue
            if (
                self._supervision_split is not None
                and self._owner(row, column) != self._supervision_split
            ):
                continue
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
            "static_features": [asset.name for asset in static],
            "dynamic_valid_mask": torch.from_numpy(dynamic_valid),
            "static_valid_mask": torch.from_numpy(static_valid),
            "valid_mask": torch.from_numpy(dynamic_valid | static_valid),
            "supervision_split_mask": torch.from_numpy(supervision_split_mask),
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
                "valid_pixels": int((dynamic_valid | static_valid).sum()),
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
        """Close only this dataset's worker-local raster readers."""
        self._raster_pool.close()


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
        "supervision_split_mask": torch.stack(
            [
                sample.get(
                    "supervision_split_mask", torch.ones_like(sample["valid_mask"])
                )
                for sample in samples
            ]
        ),
        "core_mask": torch.stack(
            [
                sample.get("core_mask", torch.ones_like(sample["valid_mask"]))
                for sample in samples
            ]
        ),
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
        **(
            {"static_features": [sample["static_features"] for sample in samples]}
            if all("static_features" in sample for sample in samples)
            else {}
        ),
        "sample_status": [sample["sample_status"] for sample in samples],
        "crs": [sample["crs"] for sample in samples],
        "bounds": [sample["bounds"] for sample in samples],
        "window": [sample["window"] for sample in samples],
        "input_window": [sample["input_window"] for sample in samples],
    }
