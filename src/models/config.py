"""Derive model input dimensions from stage-1 and stage-2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 产物必须是对象: {path}")
    return value


def _discover(run: str | Path, pattern: str) -> Path:
    candidates = sorted(Path(run).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"run 目录中找不到 {pattern}: {run}")
    return candidates[-1]


def _feature_name(asset: dict[str, Any]) -> str:
    name = str(asset["name"])
    band = asset.get("band")
    return f"{name}_B{band}" if band is not None else name


def validate_checkpoint_mapping(
    derived: dict[str, Any], mapping: dict[str, Any]
) -> None:
    """Prevent a replaced hierarchy from silently changing checkpoint legends."""
    if "levels" not in mapping and "level_counts" not in derived:
        return  # Preserve legacy checkpoints and their original artifact contract.
    from data.labels import validate_label_mapping

    validate_label_mapping(mapping)
    classes = sorted(mapping["classes"], key=lambda item: item["alliance_code"])
    expected = {
        "level_counts": mapping.get("level_counts"),
        "level_parents": mapping.get("level_parents"),
        "class_paths": [item.get("level_names") for item in classes],
        "num_classes": len(classes),
        "fine_to_coarse": [item["formation_code"] - 1 for item in classes],
    }
    for key, value in expected.items():
        if derived.get(key) != value:
            raise ValueError(
                f"checkpoint 与标签映射的 {key} 不一致，请使用训练时的映射"
            )


def derive_model_contract(
    sample_index: str | Path, label_mapping: str | Path
) -> dict[str, Any]:
    """Derive model dimensions without duplicating data-discovery logic."""

    index = _read_json(sample_index)
    mapping = _read_json(label_mapping)
    hierarchy = {}
    if "levels" in mapping:
        from data.labels import validate_label_mapping

        validate_label_mapping(mapping)
        hierarchy = {key: mapping[key] for key in ("level_counts", "level_parents")}
        hierarchy["level_names"] = [level["name"] for level in mapping["levels"]]
        hierarchy["class_paths"] = [
            item["level_names"]
            for item in sorted(
                mapping["classes"], key=lambda item: item["alliance_code"]
            )
        ]
    assets = index.get("assets", [])
    dynamic_features = sorted(
        {_feature_name(asset) for asset in assets if asset.get("role") == "dynamic"}
    )
    static_features = sorted(
        {asset["name"] for asset in assets if asset.get("role") == "static"}
    )
    classes = mapping.get("classes", [])
    num_classes = int(mapping.get("minor_count") or len(classes))
    if not dynamic_features or not static_features or num_classes < 1:
        raise ValueError("阶段产物无法推导动态特征、静态特征或类别数")
    ordered_classes = sorted(classes, key=lambda item: int(item["alliance_code"]))
    # Some lightweight interface fixtures contain only minor_count.  They can
    # still describe tensor dimensions, but cannot describe a real hierarchy.
    fine_to_coarse = (
        [int(item["formation_code"]) - 1 for item in ordered_classes]
        if ordered_classes
        else [0] * num_classes
    )
    major_count = int(mapping.get("major_count") or (max(fine_to_coarse) + 1))
    if len(fine_to_coarse) != num_classes:
        raise ValueError("标签映射的小类数量与 minor_count 不一致")
    return {
        **hierarchy,
        "dynamic_features": dynamic_features,
        "static_features": static_features,
        "dynamic_features_count": len(dynamic_features),
        "static_features_count": len(static_features),
        "num_classes": num_classes,
        "num_coarse_classes": major_count,
        "fine_to_coarse": fine_to_coarse,
        "sample_index": str(Path(sample_index).resolve()),
        "label_mapping": str(Path(label_mapping).resolve()),
    }


def load_model_contract(
    config_path: str | Path,
    run: str | Path,
    stage2: dict[str, Any] | None = None,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Load architecture settings and append dimensions derived from artifacts."""

    with Path(config_path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    model = dict(raw.get("model", {}))
    if model_name is not None:
        from experiment_options import MODEL_CHOICES

        model["architecture"] = MODEL_CHOICES[model_name]
    pretrained = dict(model.get("pretrained", {}))
    if model_name == "segformer-utae" and not pretrained.get("path"):
        pretrained["freeze_stages"] = 0
    if model_name in {"segformer-utae", "utae"}:
        model.setdefault("temporal", {}).setdefault("normalization", "group")
    if pretrained.get("path") is not None:
        path = Path(pretrained["path"])
        pretrained["path"] = str(
            path if path.is_absolute() else (Path(config_path).parent / path).resolve()
        )
    model["pretrained"] = pretrained
    if model.get("architecture") == "anysat":
        settings = dict(model.get("anysat", {}))
        if settings.get("pretrained_path"):
            path = Path(settings["pretrained_path"])
            settings["pretrained_path"] = str(
                path
                if path.is_absolute()
                else (Path(config_path).parent / path).resolve()
            )
        model["anysat"] = settings
    artifacts = dict(model.get("artifacts", {}))
    sample_index = artifacts.get("sample_index", "auto")
    label_mapping = artifacts.get("label_mapping", "auto")
    sample_index_path = (
        Path(sample_index)
        if sample_index != "auto"
        else _discover(run, "sample_index.json")
    )
    label_mapping_path = (
        Path(label_mapping)
        if label_mapping != "auto"
        else _discover(run, "label_mapping*.json")
    )
    derived = derive_model_contract(sample_index_path, label_mapping_path)
    if model.get("architecture") == "anysat":
        from data.anysat import resolve_resolution

        grid = _read_json(sample_index_path).get("target_grid", {})
        derived["target_grid"] = grid
        model["anysat"]["resolution_m"] = resolve_resolution(model["anysat"], grid)
    if model.get("architecture") in {"maestro_s", "anysat"}:
        # Keep legacy contract ordering unchanged for old checkpoint resumes.
        # MAESTRO binds static tokenizers to actual dataset/index channel order.
        derived["static_features"] = list(
            dict.fromkeys(
                asset["name"]
                for asset in _read_json(sample_index_path)["assets"]
                if asset.get("role") == "static"
            )
        )
        features = (stage2 or {}).get("features", {})
        for role in ("dynamic", "static"):
            available = derived[f"{role}_features"]
            selected = list(features.get(role) or available)
            if len(set(selected)) != len(selected) or set(selected) - set(available):
                raise ValueError(f"{role} 特征选择重复或不在样本索引中")
            ordered = [name for name in available if name in selected]
            if role == "dynamic":
                first = list(features.get("dynamic_order", []))
                if len(set(first)) != len(first) or set(first) - set(ordered):
                    raise ValueError("dynamic_order 必须是已选择动态特征的无重复子集")
                ordered = first + [name for name in ordered if name not in first]
            derived[f"{role}_features"] = ordered
            derived[f"{role}_features_count"] = len(ordered)
    model["derived"] = derived
    return model
