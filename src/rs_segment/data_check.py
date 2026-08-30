"""Configuration-driven data layout validation command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rs_segment.config import AppConfig, load_config
from rs_segment.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def check_data(config: AppConfig) -> list[str]:
    """Return human-readable validation errors for the configured data layout."""

    data = config.data
    errors: list[str] = []
    if not data.root.is_dir():
        errors.append(f"数据根目录不存在: {data.root}")
    for name in data.required_subdirectories:
        path = data.root / name
        if not path.is_dir():
            errors.append(f"必需目录不存在: {path}")
    for path in (data.labels, data.raw):
        if not path.is_dir():
            errors.append(f"配置路径不存在: {path}")
    if data.label_file is not None and not data.label_file.is_file():
        errors.append(f"标签文件不存在: {data.label_file}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查遥感数据目录和标签配置")
    parser.add_argument("--config", required=True, type=Path, help="YAML 配置文件")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        errors = check_data(load_config(args.config))
    except (OSError, ValueError) as exc:
        LOGGER.error("无法加载配置: %s", exc)
        return 2
    if errors:
        for error in errors:
            LOGGER.error(error)
        return 1
    LOGGER.info("数据检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
