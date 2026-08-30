"""Project logging setup."""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a consistent console logger for command-line entry points."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
