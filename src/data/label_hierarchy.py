"""CSV hierarchy definitions and stable path-based categorical identifiers."""

from collections import Counter
from collections.abc import Mapping
from math import isfinite
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
    if not isinstance(levels, list) or not 1 <= len(levels) <= 4:
        raise ValueError("label_schema.levels 必须包含 1～4 层（从粗到细）")
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


def leaf_aliases(mapping):
    """Return original rare leaf paths mapped to their promoted parent leaves."""
    return {
        tuple(item["source_level_names"]): tuple(item["target_level_names"])
        for item in mapping.get("promoted_fine_classes", [])
    }


def resolve_category_path(path, mapping):
    """Resolve an optional fourth level before applying partial-label rules."""
    if mapping.get("path_aliases") is not None:
        prefix = known_prefix(path)
        aliases = {
            tuple(item["source_level_names"]): tuple(item["target_level_names"])
            for item in mapping["path_aliases"]
        }
        prefix = aliases.get(prefix, prefix)
        terminals = {
            tuple(item["level_names"][: item["terminal_level"]]): tuple(
                item["level_names"]
            )
            for item in mapping["classes"]
        }
        return terminals.get(prefix, prefix + ("",) * (len(path) - len(prefix)))
    if mapping.get("optional_leaf_level") and len(known_prefix(path)) == 3:
        defaults = {
            tuple(item["source_level_names"]): tuple(item["target_level_names"])
            for item in mapping["default_leaf_paths"]
        }
        return defaults.get(path[:3], path)
    return path


def active_level_classes(counts, parents, leaf_depths):
    """Identify semantic classes while excluding padded coarse terminal nodes."""
    active = [[False] * count for count in counts]
    if len(leaf_depths) != counts[-1] or any(
        type(depth) is not int or not 1 <= depth <= len(counts) for depth in leaf_depths
    ):
        raise ValueError("leaf_depths 与末层类别不一致")
    for leaf, terminal in enumerate(leaf_depths):
        code = leaf
        for depth in reversed(range(len(counts))):
            if terminal > depth:
                active[depth][code] = True
            if depth:
                code = parents[depth - 1][code]
    return active


def _optional_leaf_hierarchy(
    rows, levels, version, missing_policy, threshold, sample_columns
):
    """Retain third-level leaves and add a fallback only to subdivided branches."""
    paths = {}
    samples = {}
    row_index = 0
    for batch in rows:
        for row in batch:
            row_index += 1
            path = category_path(row, levels)
            prefix = known_prefix(path)
            if len(prefix) < 3 and missing_policy != "partial":
                if missing_policy == "skip":
                    continue
                raise ValueError("类别为空，无法生成层级映射")
            localized = tuple(
                (row.get(level.get("zh_column", "")) or "").strip()
                for level in levels[: len(prefix)]
            )
            paths.setdefault(prefix, localized)
            try:
                sample = (
                    tuple(float(row[column]) for column in sample_columns)
                    if sample_columns
                    else (row_index,)
                )
            except (TypeError, ValueError, KeyError):
                continue
            if all(isfinite(value) for value in sample):
                for depth in range(1, len(prefix) + 1):
                    samples.setdefault(prefix[:depth], set()).add(sample)
    counts = {prefix: len(values) for prefix, values in samples.items()}
    subdivided = {path[:3] for path in paths if len(path) == 4}
    defaults = {}
    normalized = {}
    aliases = {}
    promotions = []
    for path, localized in paths.items():
        target, target_zh = path, localized
        reason = None
        # Parent counts include all descendants, deduplicated by coordinate.
        for depth in (2, 3):
            if len(path) >= depth and counts.get(path[:depth], 0) < threshold:
                target, target_zh = path[: depth - 1], localized[: depth - 1]
                reason = "rare_parent"
                break
        if reason is None and len(path) >= 3 and path[:3] in subdivided:
            fallback = (*path[:3], f"Other_{path[2]}")
            fallback_zh = (*localized[:3], f"其它{localized[2] or path[2]}")
            normalized.setdefault(fallback, fallback_zh)
            defaults[path[:3]] = fallback
            if len(path) == 3 or counts.get(path, 0) < threshold:
                target, target_zh = fallback, fallback_zh
                reason = "rare_fourth" if len(path) == 4 else "unspecified_fourth"
        normalized.setdefault(target, target_zh)
        if target != path:
            aliases[path] = target
        if reason in {"rare_parent", "rare_fourth"}:
            promotions.append(
                {
                    "source_level_names": list(path),
                    "target_level_names": list(target),
                    "ground_truth_samples": counts.get(path, 0),
                    "reason": reason,
                }
            )
    # Coarse-only observations remain partial when retained descendants exist.
    # A branch with no surviving descendants becomes a coarse terminal class.
    terminals = {
        path: localized
        for path, localized in normalized.items()
        if not any(
            len(other) > len(path) and other[: len(path)] == path
            for other in normalized
        )
    }
    canonical_rows = []
    terminal_depths = {}
    for path, localized in terminals.items():
        padded = path + (path[-1],) * (4 - len(path))
        padded_zh = localized + (localized[-1],) * (4 - len(path))
        terminal_depths[padded] = len(path)
        row = {}
        for level, name, zh in zip(levels, padded, padded_zh, strict=True):
            row[level["column"]] = name
            if level.get("zh_column"):
                row[level["zh_column"]] = zh or name
        canonical_rows.append(row)
    mapping = build_hierarchy([canonical_rows], levels, version, missing_policy)
    mapping["optional_leaf_level"] = True
    mapping["min_samples_per_fine_class"] = threshold
    mapping["sample_count_unit"] = "unique_coordinates"
    mapping["coarsened_classes"] = promotions
    mapping["path_aliases"] = [
        {"source_level_names": list(source), "target_level_names": list(target)}
        for source, target in sorted(aliases.items())
    ]
    mapping["default_leaf_paths"] = [
        {"source_level_names": list(source), "target_level_names": list(target)}
        for source, target in sorted(defaults.items())
    ]
    for item in mapping["classes"]:
        item["terminal_level"] = terminal_depths[tuple(item["level_names"])]
    return mapping


def build_hierarchy(
    rows,
    levels,
    version=2,
    missing_policy="error",
    min_samples_per_fine_class=1,
    sample_columns=None,
    fine_class_counts=None,
    optional_leaf_level=False,
):
    if missing_policy not in {"error", "skip", "partial"}:
        raise ValueError("missing_policy 必须为 error、skip 或 partial")
    if type(min_samples_per_fine_class) is not int or min_samples_per_fine_class < 1:
        raise ValueError("min_samples_per_fine_class 必须是正整数")
    if optional_leaf_level:
        if len(levels) != 4:
            raise ValueError("optional_leaf_level 需要四层")
        return _optional_leaf_hierarchy(
            rows,
            levels,
            version,
            missing_policy,
            min_samples_per_fine_class,
            sample_columns,
        )
    if min_samples_per_fine_class > 1 and len(levels) < 2:
        raise ValueError("单层分类无法将低频细类并入更高一级")
    paths = {}
    path_counts = Counter()
    path_samples = {}
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
            if sample_columns is None:
                path_counts[path] += 1
            else:
                sample = tuple(
                    str(row.get(column, "")).strip() for column in sample_columns
                )
                if all(sample):
                    path_samples.setdefault(path, set()).add(sample)
    if not paths:
        raise ValueError("标签 CSV 没有完整类别路径，无法确定最终类别体系")
    if sample_columns is not None:
        path_counts.update({path: len(path_samples.get(path, set())) for path in paths})
    if fine_class_counts is not None:
        for path in paths:
            path_counts[path] = int(fine_class_counts.get(path, 0))
    for prefix in partial_paths:
        if not any(path[: len(prefix)] == prefix for path in paths):
            raise ValueError(
                f"部分标签 {prefix} 没有已知完整类别路径；请先补充类别体系样本"
            )
    promoted = {}
    effective_paths = {}
    for path, localized in paths.items():
        target = path
        target_localized = localized
        if path_counts[path] < min_samples_per_fine_class:
            target = (*path[:-1], path[-2])
            parent_localized = localized[-2] or path[-2]
            target_localized = (*localized[:-1], parent_localized)
            promoted[path] = target
        effective_paths.setdefault(target, target_localized)
    identifiers = [
        {
            path: index + 1
            for index, path in enumerate(sorted({p[:depth] for p in effective_paths}))
        }
        for depth in range(1, len(levels) + 1)
    ]
    parents = [
        [identifiers[depth - 1][path[:-1]] - 1 for path in identifiers[depth]]
        for depth in range(1, len(levels))
    ]
    classes = []
    for path in sorted(effective_paths):
        codes = [ids[path[:depth]] for depth, ids in enumerate(identifiers, 1)]
        classes.append(
            {
                "formation_code": codes[0],
                "alliance_code": codes[-1],
                "formation": path[0],
                "alliance": path[-1],
                "formation_zh": effective_paths[path][0],
                "alliance_zh": effective_paths[path][-1],
                "level_codes": codes,
                "level_names": list(path),
                "level_names_zh": list(effective_paths[path]),
            }
        )
    return {
        "version": version,
        "missing_policy": missing_policy,
        "min_samples_per_fine_class": min_samples_per_fine_class,
        "promoted_fine_classes": [
            {
                "source_level_names": list(source),
                "target_level_names": list(target),
                "ground_truth_samples": path_counts[source],
            }
            for source, target in sorted(promoted.items())
        ],
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
    canonical_paths = {mapping_path(item) for item in classes}
    if mapping.get("optional_leaf_level"):
        if len(levels) != 4:
            raise ValueError("可选四级类的层数无效")
        semantic_paths = set()
        for item in classes:
            terminal = item.get("terminal_level", 0)
            path = mapping_path(item)
            if terminal not in {1, 2, 3, 4} or path[terminal:] != (
                path[terminal - 1],
            ) * (4 - terminal):
                raise ValueError("terminal_level 与类别路径不一致")
            semantic_paths.add(path[:terminal])
        if any(
            a != b and b[: len(a)] == a for a in semantic_paths for b in semantic_paths
        ):
            raise ValueError("终止类别不能同时拥有细分类别")
        for entry in mapping.get("path_aliases", []):
            target = tuple(entry["target_level_names"])
            if not any(path[: len(target)] == target for path in semantic_paths):
                raise ValueError("path_aliases 指向不存在的上级或其它类别")
        children = Counter(path[:3] for path in canonical_paths)
        expected_defaults = {}
        for item in classes:
            path = mapping_path(item)
            if item["terminal_level"] < 3:
                continue
            split = children[path[:3]] > 1 or path[-1] != path[2]
            target = (*path[:3], f"Other_{path[2]}" if split else path[2])
            if target not in canonical_paths or item.get("terminal_level") != (
                4 if split else 3
            ):
                raise ValueError("可选四级类缺少其它类或 terminal_level 不正确")
            if split or "path_aliases" not in mapping:
                expected_defaults[path[:3]] = target
        actual_defaults = {
            tuple(item["source_level_names"]): tuple(item["target_level_names"])
            for item in mapping.get("default_leaf_paths", [])
        }
        if actual_defaults != expected_defaults:
            raise ValueError("default_leaf_paths 与类别路径不一致")
    for source, target in leaf_aliases(mapping).items():
        if (
            len(source) != len(levels)
            or len(target) != len(levels)
            or target not in canonical_paths
            or target[:-1] != source[:-1]
            or target[-1] != source[-2]
        ):
            raise ValueError("promoted_fine_classes 包含无效的细类上移映射")
    return {
        "valid": True,
        "major_count": mapping["major_count"],
        "minor_count": mapping["minor_count"],
        "min_samples_per_fine_class": int(mapping.get("min_samples_per_fine_class", 1)),
        "promoted_fine_class_count": len(mapping.get("promoted_fine_classes", [])),
        "alliance_to_formation": {
            str(c["alliance_code"]): c["formation_code"] for c in classes
        },
    }
