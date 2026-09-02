"""Run all stage-1 preprocessing artifacts in one timestamped directory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import AppConfig
from data.labels import write_label_artifacts
from data_stats import compute_statistics


def run_preprocessing(
    config: AppConfig,
    *,
    band: int = 1,
    window_size: tuple[int, int] = (1024, 1024),
    nodata: float | int | None = None,
    timestamp: str | None = None,
    skip_statistics: bool = False,
) -> Path:
    """Write mapping, validation, and raster statistics for one preprocessing run."""

    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.data.processed / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    label_path = config.data.label_file or (config.data.labels / "labels.csv")
    write_label_artifacts(
        label_path,
        output_dir=run_dir,
        label_columns=config.data.label_columns,
        label_crs=config.data.label_crs,
        schema=config.data.label_schema,
        output_nodata=config.data.output_nodata,
        mapping_file=run_dir / f"label_mapping_{run_timestamp}.json",
        validation_report=run_dir / f"label_validation_{run_timestamp}.json",
    )
    if not skip_statistics:
        payload = compute_statistics(
            config,
            band=band,
            window_size=window_size,
            nodata=nodata,
        )
        stats_path = run_dir / f"raster_stats_{run_timestamp}.json"
        stats_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return run_dir
