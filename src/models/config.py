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


def derive_model_contract(
    sample_index: str | Path, label_mapping: str | Path
) -> dict[str, Any]:
    """Derive model dimensions without duplicating data-discovery logic."""

    index = _read_json(sample_index)
    mapping = _read_json(label_mapping)
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
    return {
        "dynamic_features": dynamic_features,
        "static_features": static_features,
        "dynamic_features_count": len(dynamic_features),
        "static_features_count": len(static_features),
        "num_classes": num_classes,
        "sample_index": str(Path(sample_index).resolve()),
        "label_mapping": str(Path(label_mapping).resolve()),
    }


def load_model_contract(config_path: str | Path, run: str | Path) -> dict[str, Any]:
    """Load architecture settings and append dimensions derived from artifacts."""

    with Path(config_path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    model = dict(raw.get("model", {}))
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
        else _discover(run, "label_mapping_*.json")
    )
    derived = derive_model_contract(sample_index_path, label_mapping_path)
    model["derived"] = derived
    return model
