"""Conservative conditional-head training and validation-only model selection."""

import math

import torch

from models.architecture import HierarchicalHeads
from models.hierarchy import MultiLevelHeads


def configure_refinement(config, requested=False):
    """Resolve and persist the separate second-stage training policy."""
    settings = config.setdefault("head_refinement", {})
    if not (requested or settings.get("enabled", False)):
        return None
    defaults = {
        "epochs": 5,
        "learning_rate": 3e-5,
        "max_accuracy_drop": 0.005,
        "extra_fraction": 0.25,
        "max_point_repeats": 3,
    }
    for key, value in defaults.items():
        settings.setdefault(key, value)
    settings["enabled"] = True
    if not 0 <= settings["max_accuracy_drop"] < 1:
        raise ValueError("head_refinement.max_accuracy_drop must be in [0, 1)")
    if not 0 <= settings["extra_fraction"] <= 1:
        raise ValueError("head_refinement.extra_fraction must be in [0, 1]")
    if type(settings["max_point_repeats"]) is not int or not (
        1 <= settings["max_point_repeats"] <= 3
    ):
        raise ValueError("head_refinement.max_point_repeats must be 1, 2 or 3")
    if type(settings["epochs"]) is not int or settings["epochs"] < 1:
        raise ValueError("head_refinement.epochs must be a positive integer")
    rate = settings["learning_rate"]
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("head_refinement.learning_rate must be finite and positive")
    policy = config.setdefault("supervision_policy", {})
    if policy.get("train_on_test", False):
        raise ValueError("Head refinement requires an independent test split")
    policy.update(
        fixed_spatial_supervision=True,
        measured_windows_only=True,
        skip_unlabeled_validation=True,
        class_weighting="none",
        disable_weak_labels=True,
    )
    config.setdefault("training", {}).update(
        epochs=settings["epochs"], selection_level=None
    )
    config.setdefault("optimizer", {})["learning_rate"] = rate
    # Avoid averaging or rounding frozen parameters/buffers in the EMA model.
    config.setdefault("ema", {})["enabled"] = False
    config.setdefault("dataloader", {})["drop_last"] = False
    return settings


def freeze_for_refinement(model):
    """Freeze all features, buffers and the root; return the conditional heads."""
    heads = getattr(model, "heads", None)
    if isinstance(heads, MultiLevelHeads):
        if len(heads.counts) < 2:
            raise ValueError("Head refinement requires at least two hierarchy levels")
        conditional = heads.transitions
    elif isinstance(heads, HierarchicalHeads):
        conditional = heads.experts
    else:
        raise ValueError("Head refinement requires supported hierarchical heads")
    model.requires_grad_(False)
    conditional.requires_grad_(True)
    set_refinement_mode(model, conditional)
    return conditional


def set_refinement_mode(model, conditional):
    """Keep frozen dropout/BatchNorm in evaluation mode even during training."""
    model.eval()
    conditional.train()


def class_diagnostics(report):
    """Add hard-label precision/recall diagnostics without changing predictions."""
    result = dict(report)
    if "confusion_matrix" not in report:
        return result
    matrix = torch.as_tensor(report["confusion_matrix"], dtype=torch.int64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    true_positive = matrix.diag()
    precision = true_positive.double() / predicted.clamp_min(1)
    recall = true_positive.double() / support.clamp_min(1)
    present = support > 0
    result.update(
        predicted_support=predicted.tolist(),
        per_class_precision=precision.tolist(),
        per_class_recall=recall.tolist(),
        supported_zero_recall_classes=(present & true_positive.eq(0))
        .nonzero()
        .flatten()
        .add(1)
        .tolist(),
        supported_macro_precision=float(precision[present].mean()),
    )
    return result


def diagnostic_report(report):
    result = class_diagnostics(report)
    result["leaf"] = (
        class_diagnostics(report["leaf"]) if report.get("leaf") is not None else None
    )
    result["levels"] = {
        key: class_diagnostics(value) for key, value in report.get("levels", {}).items()
    }
    return result


def selection_summary(report):
    """Only complete, unambiguous leaf labels define the accuracy constraint."""
    leaf = report.get("leaf")
    if leaf is None or not leaf.get("labeled_pixels", 0):
        raise ValueError(
            "Head refinement requires unambiguous leaf validation labels; "
            "partial/mixed-label agreement cannot substitute for leaf accuracy"
        )
    leaf = class_diagnostics(leaf)
    result = {
        "accuracy": float(leaf["accuracy"]),
        "zero_recall_count": len(leaf["supported_zero_recall_classes"]),
        "macro_f1": float(leaf["macro_f1"]),
        "macro_precision": float(leaf["supported_macro_precision"]),
        "class_support": leaf["class_support"],
    }
    if not all(
        math.isfinite(result[key])
        for key in ("accuracy", "macro_f1", "macro_precision")
    ):
        raise ValueError("Non-finite head-refinement validation metric")
    return result


def select_candidate(candidate, baseline, best, max_accuracy_drop):
    """Apply the fixed accuracy floor, then rank coverage, F1 and precision."""
    if candidate["class_support"] != baseline["class_support"]:
        raise ValueError("Validation class support changed during head refinement")
    eligible = candidate["accuracy"] + 1e-8 >= (
        baseline["accuracy"] - max_accuracy_drop
    )

    def rank(value):
        return (
            -value["zero_recall_count"],
            value["macro_f1"],
            value["macro_precision"],
            value["accuracy"],
        )

    return eligible, eligible and rank(candidate) > rank(best)
