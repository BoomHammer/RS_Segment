"""Run the fixed data-layout check and raster-statistics preparation flow."""

from __future__ import annotations

from pathlib import Path

from data_check import main as check_data_main
from data_stats import main as compute_stats_main


def main() -> int:
    """Validate configured data and then compute or reuse raster statistics."""

    config = str(Path("configs/data.yaml"))
    check_status = check_data_main(["--config", config])
    if check_status:
        return check_status
    return compute_stats_main(["--config", config])


if __name__ == "__main__":
    raise SystemExit(main())
