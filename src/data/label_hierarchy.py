"""CSV hierarchy definitions and stable path-based categorical identifiers."""

from collections.abc import Mapping
from typing import Any


def label_levels(columns: Mapping[str, Any]) -> list[dict[str, str]]:
    """Resolve explicit root-to-leaf levels, or the legacy two-level schema."""
    levels = columns.get("levels")
    if levels is None:
        return [
            {
                "name": "formation",
                "column": columns.get("formation", "Eng_Formation"),
                "zh_column": columns.get("chn_formation", "Formation"),
            },
            {
                "name": "alliance",
                "column": columns.get("alliance", "Eng_Alliance"),
                "zh_column": columns.get("chn_alliance", "Alliance"),
            },
        ]
    if not isinstance(levels, list) or not 1 <= len(levels) <= 3:
        raise ValueError("label_schema.levels 必须包含 1、2 或 3 层（从粗到细）")
    names = []
    for level in levels:
        if not isinstance(level, dict) or any(
            not isinstance(level.get(key), str) or not level[key].strip()
            for key in ("name", "column")
        ):
            raise ValueError("每层必须指定非空 name 和 column")
        names.append(level["name"])
    if len(set(names)) != len(names):
        raise ValueError("层级 name 不得重复")
    if len({level["column"] for level in levels}) != len(levels):
        raise ValueError("不同层级不能使用同一个 CSV column")
    return [dict(level) for level in levels]


def category_path(row, levels):
    return tuple((row.get(level["column"]) or "").strip() for level in levels)


def mapping_path(item):
    return tuple(item.get("level_names", (item["formation"], item["alliance"])))


def known_prefix(path):
    """Only trailing missing levels are meaningful partial supervision."""
    depth = next((index for index, value in enumerate(path) if not value), len(path))
    if depth == 0 or any(path[depth:]):
        raise ValueError("层级标签必须从一级开始连续填写，只允许末尾层级为空")
    return path[:depth]


def prefix_codes(mapping):
    """Look up observed ancestor paths without inventing a missing child label."""
    result = {}
    for item in mapping["classes"]:
        names = mapping_path(item)
        codes = item.get("level_codes", [item["formation_code"], item["alliance_code"]])
        for depth in range(1, len(names) + 1):
            result[names[:depth]] = tuple(codes[:depth])
    return result


def build_hierarchy(rows, levels, version=2, missing_policy="error"):
    if missing_policy not in {"error", "skip", "partial"}:
        raise ValueError("missing_policy 必须为 error、skip 或 partial")
    paths = {}
    partial_paths = set()
    row_number = 1
    for batch in rows:
        for row in batch:
            row_number += 1
            path = category_path(row, levels)
            if not all(path):
                if missing_policy == "partial":
                    partial_paths.add(known_prefix(path))
                    continue
                if missing_policy == "skip":
                    continue
                raise ValueError(f"CSV 第 {row_number} 行类别为空，无法生成层级映射")
            localized = tuple(
                (row.get(level.get("zh_column", "")) or "").strip() for level in levels
            )
            paths.setdefault(path, localized)
    if not paths:
        raise ValueError("标签 CSV 没有完整类别路径，无法确定最终类别体系")
    for prefix in partial_paths:
        if not any(path[: len(prefix)] == prefix for path in paths):
            raise ValueError(
                f"部分标签 {prefix} 没有已知完整类别路径；请先补充类别体系样本"
            )
    identifiers = [
        {
            path: index + 1
            for index, path in enumerate(sorted({p[:depth] for p in paths}))
        }
        for depth in range(1, len(levels) + 1)
    ]
    parents = [
        [identifiers[depth - 1][path[:-1]] - 1 for path in identifiers[depth]]
        for depth in range(1, len(levels))
    ]
    classes = []
    for path in sorted(paths):
        codes = [ids[path[:depth]] for depth, ids in enumerate(identifiers, 1)]
        classes.append(
            {
                "formation_code": codes[0],
                "alliance_code": codes[-1],
                "formation": path[0],
                "alliance": path[-1],
                "formation_zh": paths[path][0],
                "alliance_zh": paths[path][-1],
                "level_codes": codes,
                "level_names": list(path),
                "level_names_zh": list(paths[path]),
            }
        )
    return {
        "version": version,
        "missing_policy": missing_policy,
        "levels": levels,
        "level_counts": [len(ids) for ids in identifiers],
        "level_parents": parents,
        "major_count": len(identifiers[0]),
        "minor_count": len(classes),
        "classes": classes,
    }


def validate_hierarchy(mapping):
    levels = label_levels({"levels": mapping["levels"]})
    classes = mapping.get("classes", [])
    # Reconstruct canonical IDs and every adjacent parent edge from the paths.
    rows = []
    for item in classes:
        names = item.get("level_names", [])
        if len(names) != len(levels):
            raise ValueError("level_names 与层级数量不一致")
        rows.append(
            {level["column"]: name for level, name in zip(levels, names, strict=True)}
        )
    expected = build_hierarchy([rows], levels)
    for key in ("level_counts", "level_parents", "major_count", "minor_count"):
        if mapping.get(key) != expected[key]:
            raise ValueError(f"标签映射 {key} 与层级路径不一致")
    ordered = sorted(classes, key=lambda item: item["alliance_code"])
    if len(ordered) != len(expected["classes"]):
        raise ValueError("标签映射包含重复路径")
    for item, canonical in zip(ordered, expected["classes"], strict=True):
        for key in (
            "level_codes",
            "level_names",
            "formation_code",
            "alliance_code",
            "formation",
            "alliance",
        ):
            if item.get(key) != canonical[key]:
                raise ValueError(f"标签映射 {key} 与层级路径不一致")
    return {
        "valid": True,
        "major_count": mapping["major_count"],
        "minor_count": mapping["minor_count"],
        "alliance_to_formation": {
            str(c["alliance_code"]): c["formation_code"] for c in classes
        },
    }
