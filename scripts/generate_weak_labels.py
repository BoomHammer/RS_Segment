"""Compatibility launcher for the renamed weak-label entry point."""

import runpy
from pathlib import Path


def main() -> int:
    runpy.run_path(str(Path(__file__).with_name("weak_label.py")), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
