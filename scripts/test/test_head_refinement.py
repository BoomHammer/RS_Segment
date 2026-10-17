"""Check frozen features, bounded supervision and accuracy-constrained fallback."""

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from data.balanced_sampling import CappedPointSampler
from data.spatial_split import SpatialSplitManifest
from evaluation import point_metrics
from head_refinement import (
    configure_refinement,
    diagnostic_report,
    freeze_for_refinement,
    select_candidate,
    selection_summary,
    set_refinement_mode,
)
from models.architecture import HierarchicalHeads
from models.hierarchy import MultiLevelHeads


class RefinementModel(torch.nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 1), torch.nn.BatchNorm2d(4), torch.nn.Dropout(0.5)
        )
        self.heads = (
            HierarchicalHeads(4, 2, [0, 0, 1])
            if depth == 2
            else MultiLevelHeads(
                4,
                [2] * (depth - 1) + [3],
                [[0, 1]] * (depth - 2) + [[0, 0, 1]],
            )
        )

    def forward(self, batch):
        return self.heads(self.encoder(batch["features"]))


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_only_conditional_heads_change_including_batchnorm_buffers(depth):
    model = RefinementModel(depth)
    heads = freeze_for_refinement(model)
    batch = {"features": torch.randn(1, 2, 2, 2)}
    original = {k: v.clone() for k, v in model.state_dict().items()}
    root = model(batch)["coarse_probability"].detach().clone()
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.05
    )
    for _ in range(3):
        model.train()
        set_refinement_mode(model, heads)
        optimizer.zero_grad()
        loss = -model(batch)["fine_logits"].log_softmax(1)[:, 0].mean()
        loss.backward()
        optimizer.step()
    assert not model.encoder.training
    assert any(not torch.equal(original[k], model.state_dict()[k]) for k in trainable)
    for name, value in model.state_dict().items():
        if name not in trainable:
            assert torch.equal(value, original[name]), name
    assert torch.equal(root, model(batch)["coarse_probability"])


def test_single_level_rejected():
    model = RefinementModel()
    model.heads = MultiLevelHeads(4, [3], [])
    with pytest.raises(ValueError, match="at least two"):
        freeze_for_refinement(model)


@pytest.mark.parametrize("repeat_cap", [1, 2, 3])
def test_point_cap_keeps_all_points_and_mixed_pixels_once(repeat_cap):
    hard = {(0, x): 1 if x < 20 else 2 for x in range(22)}
    pixels = {**hard, (0, 22): -1, (0, 23): -2}
    memberships = {p: (code,) for p, code in pixels.items()}
    memberships[(0, 23)] = (1, 2)
    dataset = SimpleNamespace(
        ground_truth_pixels=hard,
        supervision_pixels=pixels,
        supervision_memberships=memberships,
        query_windows_for_pixel=lambda *_: [0, 1],
    )
    manifest = SimpleNamespace(
        blocks={"0:0": "train"}, block_size=(100, 100), class_counts={}
    )
    sampler = CappedPointSampler(
        dataset,
        manifest,
        [0, 1],
        extra_fraction=1,
        max_point_repeats=repeat_cap,
    )
    for epoch in range(3):
        visits = list(sampler)
        counts = Counter(pixel for _, pixel in visits)
        assert len(visits) == len(sampler)
        assert set(counts) == set(pixels)
        assert max(counts.values()) <= repeat_cap
        assert counts[(0, 22)] == counts[(0, 23)] == 1
        assert all(counts[(0, x)] == 1 for x in range(20))
        sampler.epoch = epoch
        assert list(sampler) == visits  # Includes restart/reproducibility behavior.


def report_from_confusion(matrix):
    targets, predictions = [], []
    for target, row in enumerate(matrix):
        for predicted, count in enumerate(row):
            targets.extend([target] * count)
            predictions.extend([predicted] * count)
    probabilities = torch.nn.functional.one_hot(
        torch.tensor(predictions), len(matrix)
    ).float()
    leaf = point_metrics(torch.tensor(targets), probabilities, len(targets))
    return {**leaf, "leaf": leaf, "levels": {}}


def test_selection_preserves_accuracy_floor_and_excludes_absent_classes():
    baseline = selection_summary(
        report_from_confusion(
            [[800, 0, 0, 0], [100, 0, 0, 0], [100, 0, 0, 0], [0, 0, 0, 0]]
        )
    )
    assert baseline["zero_recall_count"] == 2
    rejected = selection_summary(
        report_from_confusion(
            [[780, 20, 0, 0], [99, 1, 0, 0], [99, 0, 1, 0], [0, 0, 0, 0]]
        )
    )
    assert select_candidate(rejected, baseline, baseline, 0.005) == (False, False)
    accepted = selection_summary(
        report_from_confusion(
            [[797, 3, 0, 0], [99, 1, 0, 0], [100, 0, 0, 0], [0, 0, 0, 0]]
        )
    )
    assert select_candidate(accepted, baseline, baseline, 0.005) == (True, True)
    assert select_candidate(baseline, baseline, baseline, 0.005) == (True, False)
    # The floor always refers to the initial model, never the last accepted epoch.
    worse = {**accepted, "accuracy": 0.793, "zero_recall_count": 0}
    assert select_candidate(worse, baseline, accepted, 0.005) == (False, False)
    changed = {**accepted, "class_support": [800, 100, 99, 0]}
    with pytest.raises(ValueError, match="support changed"):
        select_candidate(changed, baseline, baseline, 0.005)
    with pytest.raises(ValueError, match="unambiguous leaf"):
        selection_summary({"leaf": None, "accuracy": 1.0})


def test_diagnostics_keep_existing_metrics_and_report_false_positives():
    report = report_from_confusion([[5, 1, 0], [0, 0, 2], [0, 0, 0]])
    diagnostic = diagnostic_report(report)
    assert diagnostic["accuracy"] == report["accuracy"]
    assert diagnostic["macro_f1"] == report["macro_f1"]
    assert diagnostic["leaf"]["per_class_precision"] == [1.0, 0.0, 0.0]
    assert diagnostic["leaf"]["supported_zero_recall_classes"] == [2]


def test_refinement_is_opt_in_and_disables_stacked_class_correction():
    config = {"training": {"epochs": 30, "selection_level": 0}}
    assert configure_refinement(config) is None
    assert config["training"]["epochs"] == 30
    settings = configure_refinement(config, True)
    assert settings["max_accuracy_drop"] == 0.005
    assert config["training"]["selection_level"] is None
    assert config["training"]["epochs"] == 5
    assert config["ema"]["enabled"] is False
    assert config["supervision_policy"]["class_weighting"] == "none"
    assert config["supervision_policy"]["disable_weak_labels"] is True
    with pytest.raises(ValueError, match="independent test"):
        configure_refinement({"supervision_policy": {"train_on_test": True}}, True)


@pytest.mark.parametrize("accepted", [False, True])
def test_refinement_training_fallback_and_resume(tmp_path, monkeypatch, accepted):
    spec = importlib.util.spec_from_file_location(
        "refinement_train", Path(__file__).parents[1] / "train.py"
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    run, source = tmp_path / "data", tmp_path / "source"
    run.mkdir()
    source.mkdir()
    (run / "label_mapping.json").write_text("{}")
    manifest = SpatialSplitManifest(
        schema_version=1,
        seed=42,
        block_size=(10, 10),
        ratios=(0.6, 0.2, 0.2),
        splits={"train": [0], "validation": [1], "test": [2]},
        blocks={"0:0": "train", "1:0": "validation", "2:0": "test"},
        class_counts={"train": {"1": 1}},
        class_weights={},
        sampling_weights={},
    )
    manifest.write(run / "spatial_split.json")
    manifest.write(source / "spatial_split.json")
    (source / "train_log.json").write_text("{}")
    contract = {"derived": {"num_classes": 3, "fine_to_coarse": [0, 0, 1]}}
    torch.manual_seed(42)
    initial = RefinementModel().state_dict()
    checkpoint = source / "model.pt"
    torch.save({"model": initial, "contract": contract}, checkpoint)
    data_path, train_path = tmp_path / "data.yaml", tmp_path / "train.yaml"
    data_path.write_text("data: {}")
    train_path.write_text(
        yaml.safe_dump(
            {
                "training": {"amp_dtype": "none", "early_stopping": {"enabled": False}},
                "dataloader": {"num_workers": 0},
                "head_refinement": {"epochs": 3, "learning_rate": 0.01},
            }
        )
    )
    monkeypatch.setattr(train, "load_model_contract", lambda *a, **kw: contract)
    monkeypatch.setattr(
        train,
        "SegFormerUtae",
        SimpleNamespace(from_contract=lambda _: RefinementModel()),
    )
    monkeypatch.setattr(
        train,
        "WindowedSampleDataset",
        lambda **kw: SimpleNamespace(
            configure_supervision_split=lambda *a, **kw: None,
            ground_truth_pixels={(0, 0): 1},
            query_windows_for_pixel=lambda *a: [0],
        ),
    )
    monkeypatch.setattr(train, "assert_supervision_isolated", lambda *a: None)
    monkeypatch.setattr(train, "point_windows", lambda *a: {0: {1: 1}, 1: {1: 1}})
    monkeypatch.setattr(
        train,
        "supervision_summary",
        lambda *a: {"unique_class_counts": {"train": {"1": 1}, "validation": {"1": 1}}},
    )

    class Loader(list):
        def __init__(self):
            super().__init__(
                [
                    {
                        "features": torch.ones(1, 2, 1, 1),
                        "ground_truth": torch.ones(1, 1, 1, dtype=torch.long),
                        "ground_truth_mask": torch.ones(1, 1, 1, dtype=torch.bool),
                        "valid_mask": torch.ones(1, 1, 1, dtype=torch.bool),
                        "weak_label": torch.full((1, 1, 1), -1, dtype=torch.long),
                        "weak_label_mask": torch.zeros(1, 1, 1, dtype=torch.bool),
                    }
                ]
            )
            self.generator = torch.Generator().manual_seed(42)

    monkeypatch.setattr(train, "build_dataloader", lambda *a, **kw: Loader())
    baseline = report_from_confusion([[800, 0, 0], [100, 0, 0], [100, 0, 0]])
    candidate = report_from_confusion(
        [[797, 3, 0], [99, 1, 0], [100, 0, 0]]
        if accepted
        else [[780, 20, 0], [99, 1, 0], [99, 0, 1]]
    )
    calls = 0
    interrupt = True

    def evaluate(model, *a, **kw):
        nonlocal calls
        calls += 1
        if interrupt and calls == 5:
            raise KeyboardInterrupt
        model.eval()
        return baseline if calls == 1 else candidate

    monkeypatch.setattr(train, "_evaluate", evaluate)
    args = [
        str(run),
        "--data-config",
        str(data_path),
        "--train-config",
        str(train_path),
        "--device",
        "cpu",
        "--refine-heads",
        "--init-checkpoint",
        str(checkpoint),
        "--window-size",
        "2",
        "2",
        "--stride",
        "1",
        "1",
    ]
    interrupted, full = tmp_path / "interrupted", tmp_path / "full"
    with pytest.raises(KeyboardInterrupt):
        train.main([*args, "--output-dir", str(interrupted)])
    interrupt = False
    train.main([str(run), "--resume", str(interrupted / "last.pt"), "--device", "cpu"])
    calls = 0
    train.main([*args, "--output-dir", str(full)])
    recovered = torch.load(interrupted / "last.pt", weights_only=True)
    expected = torch.load(full / "last.pt", weights_only=True)
    assert recovered["next_epoch"] == expected["next_epoch"] == 3
    assert expected["best_epoch"] == (1 if accepted else 0)
    for key in ("model", "best_state"):
        for name in expected[key]:
            assert torch.equal(recovered[key][name], expected[key][name])
    assert recovered["train_log"]["epochs"] == expected["train_log"]["epochs"]
    assert recovered["scheduler"] == expected["scheduler"]
    final = torch.load(full / "model_full.pt", weights_only=True)["model"]
    assert any(not torch.equal(initial[k], expected["model"][k]) for k in initial)
    for name in final:
        if not accepted or not name.startswith("heads.experts."):
            assert torch.equal(final[name], initial[name]), name
        assert torch.equal(final[name], expected["best_state"][name])
    assert not list(full.glob("best_*.pt"))  # No unconstrained alternative outputs.
    log = json.loads((full / "train_log.json").read_text())
    assert log["head_refinement"]["baseline"]["accuracy"] == baseline["accuracy"]
    assert len(list(full.glob("training_epoch_*.json"))) == 3
