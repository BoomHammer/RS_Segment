"""Run the fixed data-layout check and raster-statistics preparation flow."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from config import load_config
from data_check import main as check_data_main
from data_stats import main as compute_stats_main


def _new_run_path(processed: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = processed / timestamp
    counter = 1
    while run.exists():
        run = processed / f"{timestamp}_{counter}"
        counter += 1
    return run, run.name


def main(argv: list[str] | None = None) -> int:
    """Validate configured data and then compute or reuse raster statistics."""

    parser = argparse.ArgumentParser(description="检查数据并生成统一预处理目录")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    check_status = check_data_main(["--config", str(config_path)])
    if check_status:
        return check_status
    config = load_config(config_path)
    run, run_name = _new_run_path(config.data.processed)
    output = run / f"raster_stats_{run_name}.json"
    status = compute_stats_main(["--config", str(config_path), "--output", str(output)])
    if status == 0:
        print(f"数据准备产物: {run}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
