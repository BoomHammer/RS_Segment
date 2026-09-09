"""Configuration-driven command for streaming raster statistics."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from config import AppConfig, load_config
from data.filename_parser import parse_filename
from data.raster_stats import RasterStatistics, stream_raster_statistics
from data.value_ranges import load_value_ranges, value_range_for
from logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def _input_files(config: AppConfig) -> list[Path]:
    candidates = (config.data.dynamic, config.data.static)
    files: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            LOGGER.warning("跳过不存在的输入目录: %s", candidate)
            continue
        files.extend(sorted(candidate.glob("*.tif")))
    return list(dict.fromkeys(files))


def _input_signature(files: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]


def _file_category_and_time(
    config: AppConfig,
    path: Path,
) -> tuple[str, dict[str, str]]:
    if path.parent == config.data.dynamic:
        metadata = parse_filename(path)
        category = metadata.category
        if metadata.band is not None:
            category = f"{category}_B{metadata.band}"
        time = (
            {"date": metadata.date}
            if metadata.date is not None
            else {"month": metadata.month}
        )
        return category, time
    return path.stem.upper(), {}


def _cache_key(
    files: list[Path],
    *,
    band: int,
    window_size: tuple[int, int],
    nodata: float | int | None,
    value_range_file: Path | None,
) -> dict[str, object]:
    return {
        "schema_version": 4,
        "files": _input_signature(files),
        "band": band,
        "window_size": list(window_size),
        "nodata": nodata,
        "value_range_file": (
            _input_signature([value_range_file])[0]
            if value_range_file is not None
            else None
        ),
    }


def _find_cached_result(
    directory: Path,
    key: dict[str, object],
) -> Path | None:
    paths = sorted(
        directory.rglob("raster_stats_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("cache_key") == key:
            return path
    return None


def _default_output_path(directory: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"raster_stats_{timestamp}.json"
    counter = 1
    while path.exists():
        path = directory / f"raster_stats_{timestamp}_{counter}.json"
        counter += 1
    return path


def compute_statistics(
    config: AppConfig,
    *,
    band: int,
    window_size: tuple[int, int],
    nodata: float | int | None,
) -> dict[str, object]:
    """Compute statistics for all configured dynamic and static rasters."""

    files = _input_files(config)
    if not files:
        raise FileNotFoundError("dynamic 和 static 目录中没有找到 .tif 影像")

    configured_nodata = config.data.raster.get("nodata")
    missing_value = configured_nodata if nodata is None else nodata
    value_range_path = config.data.stage2.get("value_range_file")
    value_range_file = Path(value_range_path) if value_range_path is not None else None
    value_ranges = load_value_ranges(value_range_file)
    group_records: dict[str, list[dict[str, object]]] = {}
    group_statistics: dict[str, RasterStatistics] = {}
    progress = tqdm(files, desc="计算栅格统计量", unit="file")
    for path in progress:
        progress.set_postfix_str(path.name)
        category, time = _file_category_and_time(config, path)
        value_range = value_range_for(category, value_ranges)
        if value_ranges and value_range is None:
            raise ValueError(f"输入特征缺少有效值范围: {category}")
        statistics = stream_raster_statistics(
            path,
            band=band,
            window_size=window_size,
            nodata=missing_value,
            valid_minimum=value_range.minimum if value_range is not None else None,
            valid_maximum=value_range.maximum if value_range is not None else None,
            scale=value_range.scale if value_range is not None else 1.0,
        )
        group_statistics.setdefault(category, RasterStatistics()).merge(statistics)
        group_records.setdefault(category, []).append(
            {
                "path": str(path),
                **time,
                "value_range": (
                    {
                        "minimum": value_range.minimum,
                        "maximum": value_range.maximum,
                        "scale": value_range.scale,
                    }
                    if value_range is not None
                    else None
                ),
                **statistics.to_dict(),
            }
        )
    groups = [
        {
            "category": category,
            "statistics": group_statistics[category].to_dict(),
            "value_range": group_records[category][0]["value_range"],
            "files": group_records[category],
        }
        for category in sorted(group_statistics)
    ]
    return {
        "schema_version": 4,
        "cache_key": _cache_key(
            files,
            band=band,
            window_size=window_size,
            nodata=missing_value,
            value_range_file=value_range_file,
        ),
        "band": band,
        "window_size": list(window_size),
        "nodata": missing_value,
        "groups": groups,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the streaming-statistics command."""

    parser = argparse.ArgumentParser(description="计算全部 GeoTIFF 流式统计量")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="YAML 配置文件，默认 configs/data.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSON 路径，默认写入 data.processed 并自动添加时间戳",
    )
    parser.add_argument("--band", type=int, default=1, help="统计的波段编号")
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="读取窗口的宽和高，默认 1024 1024",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="覆盖配置中的 NoData 值；不填写时使用 data.raster.nodata",
    )
    args = parser.parse_args(argv)
    configure_logging()
    try:
        config = load_config(args.config)
        files = _input_files(config)
        configured_nodata = config.data.raster.get("nodata")
        missing_value = configured_nodata if args.nodata is None else args.nodata
        value_range_path = config.data.stage2.get("value_range_file")
        value_range_file = (
            Path(value_range_path) if value_range_path is not None else None
        )
        key = _cache_key(
            files,
            band=args.band,
            window_size=tuple(args.window_size),
            nodata=missing_value,
            value_range_file=value_range_file,
        )
        cached_path = _find_cached_result(config.data.processed, key)
        if cached_path is not None:
            if args.output is not None:
                output = args.output.resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                if cached_path.resolve() != output:
                    shutil.copy2(cached_path, output)
                LOGGER.info("输入数据未变化，复制已有统计量到: %s", output)
            else:
                LOGGER.info("输入数据未变化，复用已有统计量: %s", cached_path)
            return 0

        output = args.output or _default_output_path(config.data.processed)
        payload = compute_statistics(
            config,
            band=args.band,
            window_size=tuple(args.window_size),
            nodata=args.nodata,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        LOGGER.error("统计失败: %s", exc)
        return 2
    LOGGER.info("统计结果已写入 %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
