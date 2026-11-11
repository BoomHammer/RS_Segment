"""E09.1 keeps SAM labels and safe partial/mixed four-level supervision."""

import json
from copy import deepcopy

import pytest
import torch

import pipeline
from experiment_profiles import apply_model_profile, apply_training_profile
from losses.sam_dual_head import sam_dual_head_loss
from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.experiment_heads import build_experiment_heads

DERIVED = {
    "num_classes": 5,
    "fine_to_coarse": [0, 0, 0, 0, 1],
    "level_counts": [2, 3, 4, 5],
    "level_parents": [[0, 0, 1], [0, 0, 1, 2], [0, 0, 1, 2, 3]],
    "leaf_depths": [4, 4, 3, 3, 3],
}
SUPERVISION = {
    "level_weights": [1.0, 0.5, 0.25, 0.1],
    "focal_gamma": 2.0,
    "weak_label_weight": 0.5,
    "hierarchy_weight": 0.0,
}


def make_batch():
    shape = (1, 3, 3)
    labels = torch.full(shape, -1, dtype=torch.long)
    levels = torch.full((1, 4, 3, 3), -1, dtype=torch.long)
    return {
        "ground_truth": labels,
        "ground_truth_mask": labels.gt(0),
        "ground_truth_levels": levels,
        "ground_truth_observations": torch.empty(0, 7, dtype=torch.long),
        "weak_label": torch.ones(shape, dtype=torch.long),
        "weak_label_mask": torch.ones(shape, dtype=torch.bool),
        "valid_mask": torch.ones(shape, dtype=torch.bool),
        "core_mask": torch.ones(shape, dtype=torch.bool),
        "supervision_split_mask": torch.ones(shape, dtype=torch.bool),
    }


def forward():
    torch.manual_seed(7)
    head = build_experiment_heads(4, DERIVED, {"classification_head": "sam_crgnet"})
    return head, head(torch.randn(1, 4, 3, 3))


def test_profile_preserves_sam_and_existing_supervision():
    original = {
        "supervision_policy": {"disable_weak_labels": True},
        "overlap_consistency": {"weight": 0.1},
        "training": {"selection_level": 2},
        "region_growing": {"mode": "prediction"},
    }
    before = deepcopy(original)
    result = apply_training_profile(original, "E09.1")
    assert original == before
    assert not result["supervision_policy"]["disable_weak_labels"]
    assert not result["supervision_policy"]["generated_weak_labels"]
    assert not result["supervision_policy"]["persist_generated_weak_labels"]
    assert "region_growing" not in result
    assert result["training"] == original["training"]
    assert result["overlap_consistency"] == original["overlap_consistency"]
    model = apply_model_profile({"supervision": SUPERVISION}, "E09.1")
    assert model["supervision"] == SUPERVISION
    assert model["experiment"]["classification_head"] == "sam_crgnet"


def test_pipeline_forwards_e091_without_preprocessing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    calls = []

    def stage(root, module, arguments):
        calls.append((module, arguments))
        if module == "train":
            output = root / "experiments" / "e091"
            output.mkdir(parents=True)
            (output / "model.pt").touch()
            (output / "train_log.json").write_text(
                json.dumps({"checkpoint": "model.pt", "supervision_policy": {}})
            )

    monkeypatch.setattr(pipeline, "_run", stage)
    assert pipeline.main(["--retrain", str(prepared), "--experiment", "E09.1"]) == 0
    assert [name for name, _ in calls] == ["train", "predict"]
    args = calls[0][1]
    assert args[args.index("--experiment") + 1] == "E09.1"
    assert "--no-pseudo-labels" not in args
    assert list(prepared.iterdir()) == []


def test_four_level_average_survives_checkpoint_roundtrip():
    head, output = forward()
    restored = build_experiment_heads(4, DERIVED, {"classification_head": "sam_crgnet"})
    restored.load_state_dict(head.state_dict(), strict=True)
    assert output["head_mode"] == "sam_crgnet"
    for depth in range(4):
        key = f"level_{depth}_logits"
        expected = (
            output["base_output"][key].exp() + output["expanded_output"][key].exp()
        ) / 2
        torch.testing.assert_close(output[key].exp(), expected)
        torch.testing.assert_close(expected.sum(1), torch.ones(1, 3, 3))
    for depth, parents in enumerate(DERIVED["level_parents"]):
        child = output[f"level_{depth + 1}_logits"].exp()
        aggregate = torch.zeros_like(output[f"level_{depth}_logits"]).index_add(
            1, torch.tensor(parents), child
        )
        torch.testing.assert_close(aggregate, output[f"level_{depth}_logits"].exp())


@pytest.mark.parametrize("kind", ["empty", "invalid", "partial", "mixed", "terminal"])
def test_degenerate_and_partial_batches_have_finite_loss_and_gradients(kind):
    data = make_batch()
    data["weak_label_mask"].zero_()
    if kind == "invalid":
        data["valid_mask"].zero_()
    elif kind == "partial":
        data["ground_truth_levels"][0, 0, 1, 1] = 1
    elif kind == "mixed":
        data["ground_truth_levels"][0, :3, 1, 1] = 1
        data["ground_truth_observations"] = torch.tensor(
            [[0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 2]]
        )
    elif kind == "terminal":
        data["ground_truth"][0, 1, 1] = 3
        data["ground_truth_mask"][0, 1, 1] = True
        data["ground_truth_levels"][0, :, 1, 1] = torch.tensor([1, 1, 2, 3])
    head, output = forward()
    losses = sam_dual_head_loss(
        output, data, DERIVED, SUPERVISION, {"consistency_weight": 0.0}
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    for parameter in head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    if kind in {"partial", "mixed", "terminal"}:
        assert losses["ground_truth_loss"] > 0
        assert head.base.root.weight.grad.abs().sum() > 0
    if kind == "partial":
        assert head.base.transitions[-1][0].weight.grad.abs().sum() == 0
    if kind == "mixed":
        assert head.base.transitions[-1][0].weight.grad.abs().sum() > 0
    if kind == "terminal":
        # A third-level terminal leaf must not acquire fabricated fourth labels.
        assert head.base.transitions[-1][0].weight.grad.abs().sum() == 0


def test_sam_domain_excludes_measured_mixed_halo_and_heldout_pixels():
    data = make_batch()
    labels_before = data["weak_label"].clone()
    valid_before = data["valid_mask"].clone()
    data["ground_truth_levels"][0, 0, 0, 0] = 1
    data["ground_truth_observations"] = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 2]]
    )
    data["core_mask"][0, 0, 2] = False
    data["supervision_split_mask"][0, 1, 0] = False
    _, output = forward()
    losses = sam_dual_head_loss(output, data, DERIVED, SUPERVISION, {})
    assert data["weak_label_mask"].sum() == 5
    assert not data["weak_label_mask"][0, 0].any()
    assert not data["weak_label_mask"][0, 1, 0]
    torch.testing.assert_close(data["weak_label"], labels_before)
    torch.testing.assert_close(data["valid_mask"], valid_before)
    assert losses["sam_lovasz_loss"] > 0


def test_sam_gradients_reach_only_expanded_head_without_consistency():
    data = make_batch()
    head, output = forward()
    losses = sam_dual_head_loss(
        output, data, DERIVED, SUPERVISION, {"consistency_weight": 0.0}
    )
    losses["loss"].backward()
    assert head.base.root.weight.grad.abs().sum() == 0
    assert head.expanded.root.weight.grad.abs().sum() > 0


def test_consistency_trains_both_heads_on_unlabeled_train_domain():
    data = make_batch()
    data["weak_label_mask"].zero_()
    head, output = forward()
    losses = sam_dual_head_loss(output, data, DERIVED, SUPERVISION, {})
    assert losses["ground_truth_loss"] == losses["weak_label_loss"] == 0
    assert losses["consistency_loss"] > 0
    losses["loss"].backward()
    assert head.base.root.weight.grad.abs().sum() > 0
    assert head.expanded.root.weight.grad.abs().sum() > 0


def test_original_hierarchy_and_class_weights_are_used():
    data = make_batch()
    data["ground_truth"][0, 1, 1] = 1
    data["ground_truth_mask"][0, 1, 1] = True
    data["ground_truth_levels"][0, :, 1, 1] = 1
    _, output = forward()
    weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0])
    losses = sam_dual_head_loss(
        output, data, DERIVED, SUPERVISION, {}, class_weights=weights
    )
    expected = combined_supervision_loss(
        output["expanded_output"],
        data,
        fine_to_coarse=DERIVED["fine_to_coarse"],
        level_parents=DERIVED["level_parents"],
        leaf_depths=DERIVED["leaf_depths"],
        class_weights=weights,
        **SUPERVISION,
    )
    torch.testing.assert_close(
        losses["sam_hierarchical_loss"],
        expected["weak_label_coarse_loss"] + expected["weak_label_fine_loss"],
    )
    torch.testing.assert_close(
        losses["loss"],
        losses["ground_truth_loss"]
        + 0.5 * losses["weak_label_loss"]
        + losses["consistency_loss"],
    )


def test_real_hybrid_backward_with_overlap_and_accumulation():
    from data.overlap import overlapping_view
    from losses.overlap import overlap_consistency_loss

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        derived = {
            **DERIVED,
            "dynamic_features_count": 2,
            "static_features_count": 1,
        }
        contract = apply_model_profile(
            {
                "derived": derived,
                "temporal": {"normalization": "group"},
                "fusion": {"output_channels": 16},
            },
            "E09.1",
        )
        model = SegFormerUtae.from_contract(contract).train()
        shape = (1, 32, 32)
        labels = torch.full(shape, -1, dtype=torch.long)
        labels[:, 16, 16] = 1
        data = {
            "dynamic": torch.randn(1, 2, 2, 32, 32),
            "dynamic_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "dynamic_time_mask": torch.ones(1, 2, dtype=torch.bool),
            "time_encoding": torch.zeros(1, 2, 3),
            "static": torch.randn(1, 1, 32, 32),
            "ground_truth": labels,
            "ground_truth_mask": labels.gt(0),
            "ground_truth_levels": labels[:, None].repeat(1, 4, 1, 1),
            "weak_label": torch.ones(shape, dtype=torch.long),
        }
        for key in (
            "valid_mask",
            "core_mask",
            "supervision_split_mask",
            "weak_label_mask",
            "static_valid_mask",
            "dynamic_valid_mask",
        ):
            data[key] = torch.ones(shape, dtype=torch.bool)
        for _ in range(2):
            output = model(data)
            losses = sam_dual_head_loss(output, data, derived, SUPERVISION, {})
            (losses["loss"] / 2).backward()
            view, top, left = overlapping_view(data, 8)
            cropped = model(view)
            overlap = overlap_consistency_loss(
                output["fine_logits"],
                cropped["fine_logits"],
                view["valid_mask"] & view["core_mask"],
                top,
                left,
                border=2,
            )
            (0.1 * overlap / 2).backward()
        for name in (
            "static_encoder.encoder.patch_embeddings.0.proj.weight",
            "heads.base.root.weight",
            "heads.expanded.root.weight",
        ):
            parameter = dict(model.named_parameters())[name]
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    finally:
        torch.set_num_threads(previous_threads)
