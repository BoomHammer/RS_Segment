"""Run label validation and raster statistics as one preprocessing job."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config
from preprocessing import run_preprocessing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行完整数据预处理并生成统一产物目录")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dataset.yaml")
    )
    parser.add_argument("--band", type=int, default=1)
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=(1024, 1024),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--nodata", type=float, default=None)
    args = parser.parse_args(argv)
    run_dir = run_preprocessing(
        load_config(args.config),
        band=args.band,
        window_size=tuple(args.window_size),
        nodata=args.nodata,
    )
    print(f"预处理产物: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
