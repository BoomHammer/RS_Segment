"""Reproducible E08-E15 mutations, kept separate from frozen E01-E04 paths."""

from __future__ import annotations

from copy import deepcopy


def apply_training_profile(config: dict, experiment: str | None) -> dict:
    """Return an isolated training config for one registered experiment."""

    result = deepcopy(config)
    if experiment is None:
        return result
    policy = result.setdefault("supervision_policy", {})
    policy["experiment"] = experiment
    # Ignore the stored PointSAM raster; E08/E09 generate their own supervision,
    # while E10-E15 compare against E02's measured-label protocol.
    if experiment in {"E08", "E09", "E10", "E11", "E12", "E13", "E14", "E15"}:
        policy["disable_weak_labels"] = True
    if experiment == "E08":
        policy["generated_weak_labels"] = True
        policy["persist_generated_weak_labels"] = True
        result["region_growing"] = {
            "mode": "feature",
            "distance_threshold": 0.75,
            "max_radius": 8,
        }
    elif experiment == "E09":
        policy["persist_generated_weak_labels"] = True
        result["region_growing"] = {
            "mode": "prediction",
            "confidence_threshold": 0.95,
            "max_iterations": 64,
            "consistency_weight": 1.0,
            "self_training_start_epoch": 20,
            "self_training_threshold": 0.95,
            "self_training_weight": 0.5,
        }
    elif experiment == "E14":
        result.setdefault("overlap_consistency", {})["weight"] = 0.0
    return result


def apply_model_profile(contract: dict, experiment: str | None) -> dict:
    """Select only the module under test while retaining the E02 backbone."""

    result = deepcopy(contract)
    if experiment is None:
        return result
    result["architecture"] = "segformer_utae_pretrained"
    result.setdefault("pretrained", {})["freeze_stages"] = 0
    result.setdefault("temporal", {}).setdefault("normalization", "group")
    settings = result.setdefault("experiment", {})
    settings["id"] = experiment
    if experiment == "E09":
        settings["classification_head"] = "crgnet"
    elif experiment == "E10":
        settings["classification_head"] = "flat"
    elif experiment == "E11":
        settings["classification_head"] = "independent"
        result.setdefault("supervision", {})["hierarchy_weight"] = 0.0
    elif experiment == "E12":
        settings["fusion"] = "concat"
    elif experiment == "E13":
        settings["static_encoder"] = "convolutional"
    elif experiment == "E15":
        settings["classification_head"] = "hssn"
        settings["hssn_projection_channels"] = 256
        result.setdefault("supervision", {})["tree_triplet_weight"] = 0.5
    return result
