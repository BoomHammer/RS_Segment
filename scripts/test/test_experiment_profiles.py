"""Contract tests for the isolated E08-E15 experiment implementations."""

import numpy as np
import pytest
import torch
from rasterio.windows import Window

from experiment_profiles import apply_model_profile, apply_training_profile
from generated_weak_labels import GeneratedWeakLabelStore
from losses.experiment import (
    crgnet_loss,
    feature_region_labels,
    hssn_loss,
    prediction_region_labels,
)
from losses.supervision import combined_supervision_loss
from models.experiment_heads import build_experiment_heads
from models.pretrained_utae import (
    ConcatenationFusion,
    ConvolutionalStaticEncoder,
    PretrainedSegFormerUTAE,
)

DERIVED = {
    "num_classes": 3,
    "fine_to_coarse": [0, 0, 1],
    "level_counts": [2, 3],
    "level_parents": [[0, 0, 1]],
}


def batch():
    labels = torch.zeros(1, 7, 7, dtype=torch.long)
    labels[0, 3, 3] = 1
    mask = labels.gt(0)
    return {
        "dynamic": torch.zeros(1, 2, 2, 7, 7),
        "dynamic_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "dynamic_valid_mask": torch.ones(1, 7, 7, dtype=torch.bool),
        "dynamic_time_mask": torch.ones(1, 2, dtype=torch.bool),
        "static": torch.zeros(1, 2, 7, 7),
        "static_valid_mask": torch.ones(1, 7, 7, dtype=torch.bool),
        "ground_truth": labels,
        "ground_truth_mask": mask,
        "ground_truth_levels": torch.stack((labels, labels), dim=1),
        "weak_label": torch.zeros_like(labels),
        "weak_label_mask": torch.zeros_like(mask),
        "valid_mask": torch.ones_like(mask),
        "core_mask": torch.ones_like(mask),
        "supervision_split_mask": torch.ones_like(mask),
    }


@pytest.mark.parametrize("experiment", [f"E{number:02d}" for number in range(8, 16)])
def test_profiles_are_isolated_and_disable_dataset_weak_labels(experiment):
    training = {"overlap_consistency": {"weight": 0.1}}
    model = {"pretrained": {"freeze_stages": 2}, "derived": DERIVED}
    profiled_training = apply_training_profile(training, experiment)
    profiled_model = apply_model_profile(model, experiment)
    assert profiled_training["supervision_policy"]["experiment"] == experiment
    assert profiled_training["supervision_policy"]["disable_weak_labels"]
    assert profiled_model["experiment"]["id"] == experiment
    assert training == {"overlap_consistency": {"weight": 0.1}}
    assert "experiment" not in model


@pytest.mark.parametrize(
    "mode", ["flat", "independent", "crgnet", "sam_crgnet", "hssn"]
)
def test_experiment_heads_have_normalized_leaf_output_and_gradients(mode):
    head = build_experiment_heads(8, DERIVED, {"classification_head": mode})
    features = torch.randn(2, 8, 7, 7, requires_grad=True)
    output = head(features)
    assert output["fine_probability"].shape == (2, 3, 7, 7)
    torch.testing.assert_close(output["fine_probability"].sum(1), torch.ones(2, 7, 7))
    output["fine_logits"].mean().backward()
    assert features.grad is not None


def test_flat_head_loss_uses_only_leaf_classifier():
    head = build_experiment_heads(8, DERIVED, {"classification_head": "flat"})
    output = head(torch.randn(1, 8, 7, 7))
    loss = combined_supervision_loss(output, batch(), fine_to_coarse=[0, 0, 1])
    assert torch.isfinite(loss["loss"])


def test_feature_region_growing_is_local_and_excludes_seed():
    labels, mask = feature_region_labels(batch(), distance_threshold=0.1, max_radius=2)
    assert not mask[0, 3, 3]
    assert mask.sum() == 24
    assert labels[mask].eq(1).all()


def test_prediction_region_growing_and_crgnet_loss():
    data = batch()
    probability = torch.zeros(1, 3, 7, 7)
    probability[:, 0] = 0.99
    probability[:, 1:] = 0.005
    labels, mask = prediction_region_labels(probability, data, threshold=0.95)
    assert mask.all()
    assert labels.eq(1).all()
    head = build_experiment_heads(8, DERIVED, {"classification_head": "crgnet"})
    output = head(torch.randn(1, 8, 7, 7))
    loss = crgnet_loss(
        output,
        data,
        {"confidence_threshold": 0.0, "self_training_threshold": 0.0},
        epoch=20,
    )
    assert torch.isfinite(loss["loss"])
    assert loss["self_training_loss"] > 0
    assert data["self_training_label_mask"].all()
    assert data["self_training_label"].min() >= 1


def test_generated_weak_labels_are_persisted(tmp_path):
    data = batch()
    data["weak_label"][:, 2:5, 2:5] = 2
    data["weak_label_mask"] = data["weak_label"].gt(0)
    data["window"] = [Window(10, 20, 7, 7)]
    data["input_window"] = [Window(9, 19, 9, 9)]
    store = GeneratedWeakLabelStore(tmp_path, rank=0, experiment="E08")
    store.write_batch(data, epoch=3, batch_index=1)
    store.finish_epoch(epoch=3, batches=1)

    path = tmp_path / "generated_weak_labels/latest/rank_000/batch_000001.npz"
    with np.load(path) as saved:
        assert saved["labels"].shape == (1, 7, 7)
        assert saved["labels"][0, 3, 3] == 2
        assert saved["window"].tolist() == [[10, 20, 7, 7]]
    manifest = tmp_path / "generated_weak_labels/latest/rank_000/manifest.json"
    assert '"epoch": 3' in manifest.read_text(encoding="utf-8")


def test_hssn_tree_min_and_projection_receive_gradients():
    data = batch()
    # Add same-class and different-class samples so the triplet path is exercised.
    data["ground_truth"][0, 2, 2] = 1
    data["ground_truth"][0, 4, 4] = 3
    data["ground_truth_mask"] = data["ground_truth"].gt(0)
    data["ground_truth_levels"] = torch.stack(
        (
            torch.where(data["ground_truth"].eq(3), 2, data["ground_truth"]),
            data["ground_truth"],
        ),
        dim=1,
    )
    head = build_experiment_heads(8, DERIVED, {"classification_head": "hssn"})
    output = head(torch.randn(1, 8, 7, 7, requires_grad=True))
    losses = hssn_loss(output, data, DERIVED)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert head.projection[0].weight.grad is not None


@pytest.mark.parametrize(
    "experiment", ["E09", "E09.1", "E10", "E11", "E12", "E13", "E15"]
)
def test_profiles_build_the_real_hybrid_variant(experiment):
    contract = apply_model_profile(
        {
            "derived": {
                **DERIVED,
                "dynamic_features_count": 2,
                "static_features_count": 1,
            },
            "pretrained": {"freeze_stages": 0},
            "temporal": {"normalization": "group"},
            "fusion": {"output_channels": 16},
        },
        experiment,
    )
    model = PretrainedSegFormerUTAE(contract)
    assert model.contract["experiment"]["id"] == experiment
    if experiment == "E12":
        assert all(isinstance(item, ConcatenationFusion) for item in model.fusions)
    if experiment == "E13":
        assert isinstance(model.static_encoder, ConvolutionalStaticEncoder)
