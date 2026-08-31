"""Streaming reading, validation, and encoding of sample-label CSV files."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyproj import CRS, Transformer


@dataclass(slots=True, frozen=True)
class LabelRecord:
    """One validated CSV row with numeric hierarchical labels."""

    index: str
    x: float
    y: float
    major: str
    minor: str
    major_english: str
    minor_english: str
    formation_code: int
    alliance_code: int


@dataclass(slots=True)
class ValidationReport:
    """Counters and examples produced without retaining the whole CSV."""

    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    duplicate_points: int = 0
    conflicting_duplicate_points: int = 0
    unknown_categories: int = 0
    errors: Counter[str] = field(default_factory=Counter)
    class_counts: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "duplicate_points": self.duplicate_points,
            "conflicting_duplicate_points": self.conflicting_duplicate_points,
            "unknown_categories": self.unknown_categories,
            "errors": dict(self.errors),
            "class_counts": dict(self.class_counts),
        }


def _required(schema: Mapping[str, Any], key: str, default: str) -> str:
    return str(schema.get(key, default))


def _field(columns: Mapping[str, str], key: str, default: str) -> str:
    return str(columns.get(key, default))


def _read_float(value: str, name: str) -> float:
    result = float(value.strip())
    if not math.isfinite(result):
        raise ValueError(f"{name} 不是有限数值")
    return result


def iter_label_rows(
    path: str | Path,
    *,
    label_columns: Mapping[str, str],
    label_crs: str = "EPSG:4326",
    batch_size: int = 4096,
) -> Iterator[list[dict[str, str]]]:
    """Yield CSV rows in bounded batches after checking the input header."""

    if batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    del label_crs  # Kept in the API so readers share the coordinate contract.
    required = {
        _field(label_columns, "x", "X"),
        _field(label_columns, "y", "Y"),
        _field(label_columns, "formation", "Eng_Formation"),
        _field(label_columns, "alliance", "Eng_Alliance"),
        _field(label_columns, "chn_formation", "Formation"),
        _field(label_columns, "chn_alliance", "Alliance"),
    }
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"标签 CSV 缺少字段: {sorted(missing)}")
        batch: list[dict[str, str]] = []
        for row in reader:
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def build_label_mapping(
    path: str | Path,
    *,
    label_columns: Mapping[str, str],
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic major/minor mapping from the CSV."""

    schema = schema or {}
    major_values: dict[str, tuple[str, str]] = {}
    pairs: dict[tuple[str, str], tuple[str, str]] = {}
    for batch in iter_label_rows(path, label_columns=label_columns):
        for row in batch:
            major = row[_field(label_columns, "formation", "Eng_Formation")].strip()
            minor = row[_field(label_columns, "alliance", "Eng_Alliance")].strip()
            pair = (major, minor)
            major_values.setdefault(
                major,
                (
                    row[_field(label_columns, "chn_formation", "Formation")].strip(),
                    major,
                ),
            )
            pairs.setdefault(
                pair,
                (
                    row[_field(label_columns, "chn_alliance", "Alliance")].strip(),
                    minor,
                ),
            )
    first_id = int(schema.get("first_id", 1))
    majors = sorted(major_values)
    major_ids = {value: first_id + i for i, value in enumerate(majors)}
    minors = sorted(pairs)
    records = []
    for offset, (major, minor) in enumerate(minors):
        minor_id = first_id + offset
        records.append(
            {
                "formation_code": major_ids[major],
                "alliance_code": minor_id,
                "formation": major,
                "alliance": minor,
                "formation_zh": major_values[major][0],
                "alliance_zh": pairs[(major, minor)][0],
            }
        )
    return {
        "version": int(schema.get("version", 1)),
        "major_count": len(majors),
        "minor_count": len(records),
        "classes": records,
    }


def validate_labels(
    path: str | Path,
    *,
    label_columns: Mapping[str, str],
    label_crs: str = "EPSG:4326",
    mapping: Mapping[str, Any] | None = None,
    batch_size: int = 4096,
) -> ValidationReport:
    """Validate coordinates, duplicates, required categories, and mapping membership."""

    report = ValidationReport()
    source_crs = CRS.from_user_input(label_crs)
    to_wgs84 = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    classes = {
        (item["formation"], item["alliance"])
        for item in (mapping or {}).get("classes", [])
    }
    seen: dict[tuple[float, float], tuple[str, str]] = {}
    for batch in iter_label_rows(
        path, label_columns=label_columns, batch_size=batch_size
    ):
        for row in batch:
            report.total_rows += 1
            try:
                x = _read_float(row[_field(label_columns, "x", "X")], "X")
                y = _read_float(row[_field(label_columns, "y", "Y")], "Y")
                lon, lat = to_wgs84.transform(x, y)
                if not -180 <= lon <= 180 or not -90 <= lat <= 90:
                    raise ValueError("坐标超出 WGS84 范围")
                major = row[_field(label_columns, "formation", "Eng_Formation")].strip()
                minor = row[_field(label_columns, "alliance", "Eng_Alliance")].strip()
                if not major or not minor:
                    raise ValueError("类别为空")
                if classes and (major, minor) not in classes:
                    report.unknown_categories += 1
                    raise ValueError("类别不在标签映射中")
                key = (round(x, 9), round(y, 9))
                previous = seen.get(key)
                if previous is not None:
                    report.duplicate_points += 1
                    if previous != (major, minor):
                        report.conflicting_duplicate_points += 1
                else:
                    seen[key] = (major, minor)
                report.class_counts[f"{major}|{minor}"] += 1
                report.valid_rows += 1
            except (KeyError, TypeError, ValueError) as exc:
                report.invalid_rows += 1
                report.errors[str(exc)] += 1
    return report


def iter_encoded_labels(
    path: str | Path,
    *,
    label_columns: Mapping[str, str],
    mapping: Mapping[str, Any],
    batch_size: int = 4096,
) -> Iterator[list[LabelRecord]]:
    """Yield bounded batches with categorical IDs suitable for training."""

    lookup = {
        (item["formation"], item["alliance"]): item
        for item in mapping.get("classes", [])
    }
    for batch in iter_label_rows(
        path, label_columns=label_columns, batch_size=batch_size
    ):
        encoded: list[LabelRecord] = []
        for row in batch:
            major = row[_field(label_columns, "formation", "Eng_Formation")].strip()
            minor = row[_field(label_columns, "alliance", "Eng_Alliance")].strip()
            item = lookup.get((major, minor))
            if item is None:
                raise ValueError(f"类别不在标签映射中: {major}|{minor}")
            encoded.append(
                LabelRecord(
                    index=row.get(_field(label_columns, "index", "Index"), ""),
                    x=_read_float(row[_field(label_columns, "x", "X")], "X"),
                    y=_read_float(row[_field(label_columns, "y", "Y")], "Y"),
                    major=major,
                    minor=minor,
                    major_english=major,
                    minor_english=minor,
                    formation_code=int(item["formation_code"]),
                    alliance_code=int(item["alliance_code"]),
                )
            )
        yield encoded


def write_label_artifacts(
    path: str | Path,
    *,
    output_dir: str | Path,
    label_columns: Mapping[str, str],
    label_crs: str = "EPSG:4326",
    schema: Mapping[str, Any] | None = None,
    output_nodata: int = -9999,
    mapping_file: str | Path | None = None,
    validation_report: str | Path | None = None,
) -> tuple[Path, Path]:
    """Create the stable mapping and validation JSON artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapping = build_label_mapping(path, label_columns=label_columns, schema=schema)
    report = validate_labels(
        path,
        label_columns=label_columns,
        label_crs=label_crs,
        mapping=mapping,
    )
    schema = schema or {}
    mapping_path = (
        Path(mapping_file)
        if mapping_file is not None
        else output / "label_mapping.json"
    )
    report_path = (
        Path(validation_report)
        if validation_report is not None
        else output / "label_validation.json"
    )
    mapping["output_nodata"] = output_nodata
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return mapping_path, report_path
