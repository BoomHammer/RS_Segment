"""Parse dynamic remote-sensing raster filenames into metadata records."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import yaml

DATE_PATTERN = re.compile(
    r"^(?P<category>[A-Za-z]+)(?P<date>\d{6})(?:B(?P<band>\d+))?\.tif$",
    re.IGNORECASE,
)
MONTH_PATTERN = re.compile(
    r"^(?P<category>[A-Za-z]+)(?P<month>\d{4})\.tif$", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class RasterMetadata:
    """The stable JSON/YAML schema for one raster file."""

    path: str
    filename: str
    category: str
    temporal_resolution: str
    date: str | None
    month: str | None
    band: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-serializable record."""

        return asdict(self)


def _parse_date(value: str) -> str:
    return date(2000 + int(value[:2]), int(value[2:4]), int(value[4:])).isoformat()


def _parse_month(value: str) -> str:
    year = 2000 + int(value[:2])
    month = int(value[2:])
    if not 1 <= month <= 12:
        raise ValueError(f"月份无效: {value}")
    return f"{year:04d}-{month:02d}"


def parse_filename(
    path: str | Path,
) -> RasterMetadata:
    """Parse one dynamic ``.tif`` filename.

    Six trailing digits mean a date (``TYPEyymmdd.tif``), optionally with
    ``B<n>``. Four trailing digits mean a month (``TYPEyymm.tif``). The
    category is never used to decide the temporal resolution.
    """

    file_path = Path(path)
    match = DATE_PATTERN.fullmatch(file_path.name)
    if match:
        return RasterMetadata(
            path=str(file_path),
            filename=file_path.name,
            category=match.group("category").upper(),
            temporal_resolution="date",
            date=_parse_date(match.group("date")),
            month=None,
            band=int(match.group("band")) if match.group("band") else None,
        )

    match = MONTH_PATTERN.fullmatch(file_path.name)
    if match:
        return RasterMetadata(
            path=str(file_path),
            filename=file_path.name,
            category=match.group("category").upper(),
            temporal_resolution="month",
            date=None,
            month=_parse_month(match.group("month")),
            band=None,
        )

    raise ValueError(f"无法解析动态影像文件名: {file_path.name}")


def scan_dynamic_directory(
    directory: str | Path,
) -> list[RasterMetadata]:
    """Parse all top-level TIFFs in a dynamic-data directory."""

    return [parse_filename(path) for path in sorted(Path(directory).glob("*.tif"))]


def write_metadata(records: list[RasterMetadata], output: str | Path) -> None:
    """Write metadata using JSON or YAML, selected by the output suffix."""

    output_path = Path(output)
    payload = {"schema_version": 1, "records": [record.to_dict() for record in records]}
    if output_path.suffix.lower() == ".json":
        content = json.dumps(payload, ensure_ascii=False, indent=2)
    elif output_path.suffix.lower() in {".yaml", ".yml"}:
        content = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    else:
        raise ValueError("输出文件必须使用 .json、.yaml 或 .yml 后缀")
    output_path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="解析动态遥感影像文件名")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_metadata(scan_dynamic_directory(args.directory), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
