"""Validate sample labels and write the processed mapping/report artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from config import load_config
from data.labels import write_label_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查并编码样点标签 CSV")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data = config.data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = data.processed / timestamp
    mapping, report = write_label_artifacts(
        data.label_file or (data.labels / "labels.csv"),
        output_dir=run_dir,
        label_columns=data.label_columns,
        label_crs=data.label_crs,
        schema=data.label_schema,
        output_nodata=data.output_nodata,
        mapping_file=run_dir / f"label_mapping_{timestamp}.json",
        validation_report=run_dir / f"label_validation_{timestamp}.json",
    )
    print(f"映射: {mapping}")
    print(f"报告: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
