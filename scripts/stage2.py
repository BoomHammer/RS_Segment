"""Compatibility launcher for the unified stage-2 dataset entry point."""

from __future__ import annotations

from datasets import main as datasets_main


def main(argv: list[str] | None = None) -> int:
    return datasets_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
