"""Download the official AnySat base checkpoint and record its exact revision."""

import argparse
import json
from pathlib import Path

from huggingface_hub import model_info, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("third_party/pretrained/anysat")
    )
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    repository = "g-astruc/AnySat"
    revision = model_info(repository, revision=args.revision).sha
    snapshot_download(
        repository,
        revision=revision,
        local_dir=args.output,
        allow_patterns=["models/AnySat.pth"],
    )
    (args.output / "provenance.json").write_text(
        json.dumps({"repo_id": repository, "revision": revision}, indent=2),
        encoding="utf-8",
    )
    print(args.output.resolve() / "models/AnySat.pth")


if __name__ == "__main__":
    main()
