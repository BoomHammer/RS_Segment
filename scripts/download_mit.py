"""Download a revision-pinned official NVIDIA MiT-B1 into the workspace."""

import argparse
import json
from pathlib import Path

from huggingface_hub import model_info, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pretrained/mit-b1"))
    parser.add_argument(
        "--revision", default="13ddceec4e8bdf401e7cd7acf5aebc526222518c"
    )
    args = parser.parse_args()
    revision = model_info("nvidia/mit-b1", revision=args.revision).sha
    snapshot_download(
        "nvidia/mit-b1",
        revision=revision,
        local_dir=args.output,
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "README.md",
        ],
    )
    (args.output / "provenance.json").write_text(
        json.dumps({"repo_id": "nvidia/mit-b1", "revision": revision}, indent=2),
        encoding="utf-8",
    )
    print(f"已下载官方 MiT-B1: {args.output.resolve()}，revision={revision}")


if __name__ == "__main__":
    main()
