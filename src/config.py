"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    """Paths and expectations used by data preparation commands."""

    root: Path = Path("data")
    labels: Path = Path("data/labels")
    raw: Path = Path("data/raw")
    dynamic: Path = Path("data/raw/dynamic")
    static: Path = Path("data/raw/static")
    processed: Path = Path("data/processed")
    label_file: Path | None = None
    sam2_checkpoint: Path | None = None
    output_nodata: int = -9999
    required_subdirectories: list[str] = field(
        default_factory=lambda: ["labels", "raw"]
    )
    label_crs: str = "EPSG:4326"
    label_schema: dict[str, Any] = field(default_factory=dict)
    target_grid: dict[str, Any] = field(default_factory=dict)
    raster: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    stage2: dict[str, Any] = field(default_factory=dict)
    dynamic_filename: dict[str, Any] = field(default_factory=dict)
    metadata_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def label_columns(self) -> dict[str, Any]:
        """Return CSV columns from the unified label schema."""

        columns = dict(self.label_schema.get("columns", {}))
        if "levels" in self.label_schema:
            from data.label_hierarchy import label_levels

            columns["levels"] = self.label_schema["levels"]
            label_levels(columns)
        columns["encoding"] = self.label_schema.get("encoding", "utf-8-sig")
        columns["missing_policy"] = self.label_schema.get("missing_policy", "error")
        if columns["missing_policy"] not in {"error", "skip", "partial"}:
            raise ValueError(
                "label_schema.missing_policy 必须为 error、skip 或 partial"
            )
        return columns


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: dict[str, Any] = field(default_factory=dict)


def _resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML configuration and resolve paths relative to its file."""

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}

    data = raw.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("配置中的 data 必须是对象")
    stage2 = dict(data.get("stage2", {}))
    for key in ("statistics_file", "split_file", "value_range_file"):
        if stage2.get(key) is not None:
            stage2[key] = str(_resolve_path(stage2[key], config_path.parent))
    return AppConfig(
        model=dict(raw.get("model", {})),
        data=DataConfig(
            root=_resolve_path(data.get("root", "data"), config_path.parent)
            or config_path.parent,
            labels=_resolve_path(data.get("labels", "data/labels"), config_path.parent)
            or config_path.parent,
            raw=_resolve_path(data.get("raw", "data/raw"), config_path.parent)
            or config_path.parent,
            dynamic=_resolve_path(
                data.get("dynamic", "data/raw/dynamic"), config_path.parent
            )
            or config_path.parent,
            static=_resolve_path(
                data.get("static", "data/raw/static"), config_path.parent
            )
            or config_path.parent,
            processed=_resolve_path(
                data.get("processed", "data/processed"), config_path.parent
            )
            or config_path.parent,
            label_file=_resolve_path(data.get("label_file"), config_path.parent),
            sam2_checkpoint=_resolve_path(
                data.get("sam2_checkpoint"), config_path.parent
            ),
            output_nodata=int(data.get("output_nodata", -9999)),
            required_subdirectories=list(
                data.get("required_subdirectories", ["labels", "raw"])
            ),
            label_crs=str(data.get("label_crs", "EPSG:4326")),
            label_schema=dict(data.get("label_schema", {})),
            target_grid=dict(data.get("target_grid", {})),
            raster=dict(data.get("raster", {})),
            sampling=dict(data.get("sampling", {})),
            stage2=stage2,
            dynamic_filename=dict(data.get("dynamic_filename", {})),
            metadata_schema=dict(data.get("metadata_schema", {})),
        ),
    )
