"""Run the default data preparation command without installing the package."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "src" / "prepare_data.py"),
        run_name="__main__",
    )
