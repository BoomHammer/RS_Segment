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

from data.label_hierarchy import (
    build_hierarchy,
    category_path,
    known_prefix,
    label_levels,
    leaf_aliases,
    mapping_path,
    prefix_codes,
    resolve_category_path,
    validate_hierarchy,
)
from data.raster_alignment import TargetGrid, locate_points


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
    level_codes: tuple[int, ...] = ()


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
    invalid_row_examples: list[dict[str, Any]] = field(default_factory=list)
    missing_hierarchy_rows: list[int] = field(default_factory=list)
    partial_label_rows: list[int] = field(default_factory=list)
    partial_depth_counts: Counter[str] = field(default_factory=Counter)
    coarsened_rows: int = 0
    mixed_coordinate_count: int = 0
    mixed_coordinate_examples: list[dict[str, Any]] = field(default_factory=list)
    partial_rows: int = 0

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
            "invalid_row_examples": self.invalid_row_examples,
            "missing_hierarchy_rows": self.missing_hierarchy_rows,
            "partial_rows": self.partial_rows,
            "partial_label_rows": self.partial_label_rows,
            "partial_depth_counts": dict(self.partial_depth_counts),
            "coarsened_rows": self.coarsened_rows,
            "mixed_coordinate_count": self.mixed_coordinate_count,
            "mixed_coordinate_examples": self.mixed_coordinate_examples,
            "hierarchy_row_notes": {
                "partial_label_rows": (
                    "合法部分标签的 CSV 行号（包含表头）；只监督已知上级，不是错误"
                ),
                "missing_hierarchy_rows": (
                    "无效层级缺失的 CSV 行号示例；不包含合法部分标签"
                ),
            },
        }


def _required(schema: Mapping[str, Any], key: str, default: str) -> str:
    return str(schema.get(key, default))


def _field(columns: Mapping[str, Any], key: str, default: str) -> str:
    return str(columns.get(key, default))


def _read_float(value: str, name: str) -> float:
    result = float(value.strip())
    if not math.isfinite(result):
        raise ValueError(f"{name} 不是有限数值")
    return result


def _grid_fine_class_counts(
    path: str | Path,
    *,
    label_columns: Mapping[str, Any],
    label_crs: str,
    grid: TargetGrid,
    batch_size: int = 4096,
) -> Counter[tuple[str, ...]]:
    """Count unique in-grid ground-truth pixels for each complete leaf path."""
    levels = label_levels(label_columns)
    pixel_paths: dict[tuple[int, int], tuple[str, ...]] = {}
    for batch in iter_label_rows(
        path, label_columns=label_columns, batch_size=batch_size
    ):
        paths = []
        points = []
        for row in batch:
            names = category_path(row, levels)
            if not all(names):
                continue
            try:
                point = (
                    _read_float(row[_field(label_columns, "x", "X")], "X"),
                    _read_float(row[_field(label_columns, "y", "Y")], "Y"),
                )
            except (KeyError, TypeError, ValueError):
                continue
            paths.append(names)
            points.append(point)
        locations = locate_points(points, point_crs=label_crs, grid=grid)
        for names, location in zip(paths, locations, strict=True):
            if location["inside"]:
                pixel_paths[(location["row"], location["column"])] = names
    return Counter(pixel_paths.values())


def iter_label_rows(
    path: str | Path,
    *,
    label_columns: Mapping[str, Any],
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
    }
    for level in label_levels(label_columns):
        required.add(level["column"])
        if level.get("zh_column"):
            required.add(level["zh_column"])
    with Path(path).open(
        newline="", encoding=label_columns.get("encoding", "utf-8-sig")
    ) as stream:
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
    label_columns: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    label_crs: str = "EPSG:4326",
    sample_grid: TargetGrid | None = None,
) -> dict[str, Any]:
    """Build stable path-based IDs for the configured classification levels."""

    schema = schema or {}
    optional_leaf_level = bool(
        schema.get(
            "optional_leaf_level", label_columns.get("optional_leaf_level", False)
        )
    )
    if int(schema.get("first_id", 1)) != 1:
        raise ValueError("训练标签 first_id 必须为 1")
    if (
        "levels" in schema
        or "levels" in label_columns
        or schema.get("missing_policy", label_columns.get("missing_policy"))
        == "partial"
    ):
        label_columns = dict(label_columns)
        if "levels" in schema:
            label_columns["levels"] = schema["levels"]
        if "encoding" in schema:
            label_columns["encoding"] = schema["encoding"]
        fine_class_counts = (
            _grid_fine_class_counts(
                path,
                label_columns=label_columns,
                label_crs=label_crs,
                grid=sample_grid,
            )
            if sample_grid is not None and not optional_leaf_level
            else None
        )
        return build_hierarchy(
            iter_label_rows(path, label_columns=label_columns),
            label_levels(label_columns),
            int(schema.get("version", 2)),
            schema.get("missing_policy", label_columns.get("missing_policy", "error")),
            int(
                schema.get(
                    "min_samples_per_fine_class",
                    label_columns.get("min_samples_per_fine_class", 1),
                )
            ),
            (
                _field(label_columns, "x", "X"),
                _field(label_columns, "y", "Y"),
            ),
            fine_class_counts,
            optional_leaf_level=optional_leaf_level,
        )
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


def validate_label_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the persisted hierarchy and its legacy root/leaf aliases."""

    if "levels" in mapping:
        return validate_hierarchy(mapping)
    classes = list(mapping.get("classes", []))
    required = {"formation_code", "alliance_code", "formation", "alliance"}
    errors: list[str] = []
    if not classes:
        errors.append("classes 不能为空")
    if int(mapping.get("major_count", -1)) < 1:
        errors.append("major_count 必须为正数")
    if int(mapping.get("minor_count", -1)) != len(classes):
        errors.append("minor_count 与 classes 数量不一致")
    alliance_to_formation: dict[int, int] = {}
    formation_names: dict[int, str] = {}
    pairs: set[tuple[str, str]] = set()
    for index, item in enumerate(classes):
        missing = required - set(item)
        if missing:
            errors.append(f"classes[{index}] 缺少字段: {sorted(missing)}")
            continue
        formation_code = int(item["formation_code"])
        alliance_code = int(item["alliance_code"])
        formation = str(item["formation"])
        alliance = str(item["alliance"])
        previous = alliance_to_formation.get(alliance_code)
        if previous is not None and previous != formation_code:
            errors.append(f"小类 {alliance_code} 对应了多个大类")
        alliance_to_formation[alliance_code] = formation_code
        previous_name = formation_names.get(formation_code)
        if previous_name is not None and previous_name != formation:
            errors.append(f"大类编码 {formation_code} 对应了多个名称")
        formation_names[formation_code] = formation
        pair = (formation, alliance)
        if pair in pairs:
            errors.append(f"重复的大类-小类组合: {formation}|{alliance}")
        pairs.add(pair)
    if len(alliance_to_formation) != len(classes):
        errors.append("alliance_code 不唯一")
    if len(formation_names) != int(mapping.get("major_count", -1)):
        errors.append("major_count 与实际大类编码数量不一致")
    if errors:
        raise ValueError("标签映射校验失败: " + "; ".join(errors))
    return {
        "valid": True,
        "major_count": len(formation_names),
        "minor_count": len(alliance_to_formation),
        "alliance_to_formation": {
            str(alliance): formation
            for alliance, formation in sorted(alliance_to_formation.items())
        },
    }


def validate_labels(
    path: str | Path,
    *,
    label_columns: Mapping[str, Any],
    label_crs: str = "EPSG:4326",
    mapping: Mapping[str, Any] | None = None,
    batch_size: int = 4096,
) -> ValidationReport:
    """Validate coordinates, duplicates, required categories, and mapping membership."""

    report = ValidationReport()
    source_crs = CRS.from_user_input(label_crs)
    to_wgs84 = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    label_columns = dict(label_columns)
    if mapping and "levels" in mapping:
        label_columns["levels"] = mapping["levels"]
    levels = label_levels(label_columns)
    classes = {mapping_path(item) for item in (mapping or {}).get("classes", [])}
    aliases = leaf_aliases(mapping or {})
    partial = (mapping or {}).get(
        "missing_policy", label_columns.get("missing_policy")
    ) == "partial"
    prefixes = {path[:depth] for path in classes for depth in range(1, len(path) + 1)}
    seen: dict[tuple[float, float], set[tuple[str, ...]]] = {}
    mixed_coordinates = set()
    coarsened = {
        tuple(item["source_level_names"])
        for item in (mapping or {}).get("coarsened_classes", [])
    }
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
                raw_names = category_path(row, levels)
                try:
                    raw_prefix = known_prefix(raw_names)
                except ValueError:
                    if len(report.missing_hierarchy_rows) < 100:
                        report.missing_hierarchy_rows.append(report.total_rows + 1)
                    raise
                names = resolve_category_path(raw_names, mapping or {})
                if not all(names):
                    if not partial:
                        if len(report.missing_hierarchy_rows) < 100:
                            report.missing_hierarchy_rows.append(report.total_rows + 1)
                        raise ValueError("类别为空")
                    names = known_prefix(names)
                if classes and names not in prefixes and names not in aliases:
                    report.unknown_categories += 1
                    raise ValueError("类别不在标签映射中")
                if len(raw_prefix) < len(levels) and not (
                    (mapping or {}).get("optional_leaf_level") and len(raw_prefix) >= 3
                ):
                    report.partial_rows += 1
                    report.partial_label_rows.append(report.total_rows + 1)
                    report.partial_depth_counts[str(len(raw_prefix))] += 1
                if raw_prefix in coarsened:
                    report.coarsened_rows += 1
                key = (round(x, 9), round(y, 9))
                previous = seen.get(key)
                if previous is not None:
                    report.duplicate_points += 1
                    if any(
                        p[: min(len(p), len(names))] != names[: min(len(p), len(names))]
                        for p in previous
                    ):
                        report.conflicting_duplicate_points += 1
                        mixed_coordinates.add(key)
                seen.setdefault(key, set()).add(names)
                report.class_counts["|".join(names)] += 1
                report.valid_rows += 1
            except (KeyError, TypeError, ValueError) as exc:
                report.invalid_rows += 1
                report.errors[str(exc)] += 1
                if len(report.invalid_row_examples) < 100:
                    report.invalid_row_examples.append(
                        {
                            "csv_line": report.total_rows + 1,
                            "index": row.get(
                                _field(label_columns, "index", "Index"), ""
                            ),
                            "x": row.get(_field(label_columns, "x", "X")),
                            "y": row.get(_field(label_columns, "y", "Y")),
                            "error": str(exc),
                        }
                    )
    report.mixed_coordinate_count = len(mixed_coordinates)
    report.mixed_coordinate_examples = [
        {
            "x": x,
            "y": y,
            "observed_paths": [list(path) for path in sorted(seen[(x, y)])],
        }
        for x, y in sorted(mixed_coordinates)[:100]
    ]
    return report


def iter_encoded_labels(
    path: str | Path,
    *,
    label_columns: Mapping[str, Any],
    mapping: Mapping[str, Any],
    batch_size: int = 4096,
) -> Iterator[list[LabelRecord]]:
    """Yield bounded batches with categorical IDs suitable for training."""

    label_columns = dict(label_columns)
    if "levels" in mapping:
        label_columns["levels"] = mapping["levels"]
    levels = label_levels(label_columns)
    lookup = {mapping_path(item): item for item in mapping.get("classes", [])}
    aliases = leaf_aliases(mapping)
    prefixes = prefix_codes(mapping)
    for batch in iter_label_rows(
        path, label_columns=label_columns, batch_size=batch_size
    ):
        encoded: list[LabelRecord] = []
        for row in batch:
            try:
                names = resolve_category_path(category_path(row, levels), mapping)
                if not all(names) and mapping.get("missing_policy", "error") == "skip":
                    continue
                major, minor = names[0], names[-1]
                item = lookup.get(aliases.get(names, names))
                if not all(names) and mapping.get("missing_policy") == "partial":
                    prefix = known_prefix(names)
                    if prefix not in prefixes:
                        raise ValueError(f"部分标签不在类别体系中: {prefix}")
                    codes = prefixes[prefix] + (-1,) * (len(levels) - len(prefix))
                    item = {
                        "formation_code": codes[0],
                        "alliance_code": -1,
                        "level_codes": codes,
                    }
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
                        level_codes=tuple(
                            item.get(
                                "level_codes",
                                (item["formation_code"], item["alliance_code"]),
                            )
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        yield encoded


def write_label_artifacts(
    path: str | Path,
    *,
    output_dir: str | Path,
    label_columns: Mapping[str, Any],
    label_crs: str = "EPSG:4326",
    schema: Mapping[str, Any] | None = None,
    output_nodata: int = -9999,
    mapping_file: str | Path | None = None,
    validation_report: str | Path | None = None,
    sample_grid: TargetGrid | None = None,
) -> tuple[Path, Path]:
    """Create the stable mapping and validation JSON artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_columns = dict(label_columns)
    for key in ("levels", "encoding", "missing_policy"):
        if key in (schema or {}):
            label_columns[key] = schema[key]
    mapping = build_label_mapping(
        path,
        label_columns=label_columns,
        schema=schema,
        label_crs=label_crs,
        sample_grid=sample_grid,
    )
    mapping_validation = validate_label_mapping(mapping)
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
    report_payload = report.to_dict()
    report_payload["mapping_validation"] = mapping_validation
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return mapping_path, report_path
