"""Regression coverage for held-out fused-map evaluation and abstentions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from data.sample_index import WindowedSampleDataset
from inference.point_evaluation import PredictionPointEvaluation, _hard_metrics


def _split(path, blocks):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 42,
                "block_size": [1, 1],
                "ratios": [0.5, 0, 0.5],
                "splits": {"train": [0], "test": [1, 2]},
                "blocks": blocks,
                "class_counts": {},
                "class_weights": {},
                "sampling_weights": {},
            }
        )
    )


@pytest.fixture
def setup(tmp_path):
    run = tmp_path / "run"
    experiment = tmp_path / "experiment"
    run.mkdir()
    experiment.mkdir()
    checkpoint = experiment / "model.pt"
    checkpoint.touch()
    (experiment / "train_log.json").write_text("{}")
    _split(run / "spatial_split.json", {f"0:{i}": "test" for i in range(5)})
    mapping = {
        "levels": [{"name": "ecosystem"}, {"name": "type"}, {"name": "subtype"}],
        "level_counts": [2, 3, 3],
        "classes": [
            {"alliance_code": 1, "level_codes": [1, 1, 1], "terminal_level": 2},
            {"alliance_code": 2, "level_codes": [1, 2, 2], "terminal_level": 3},
            {"alliance_code": 3, "level_codes": [2, 3, 3], "terminal_level": 3},
        ],
    }
    dataset = SimpleNamespace(
        evaluation_observations={
            (0, 0): ((1, 1, 1),),  # Leaf ends early; no artificial subtype target.
            (0, 1): ((1, -1, -1),),  # Partial label evaluated only at ecosystem level.
            (0, 2): ((1, 2, 2), (2, 3, 3), (2, 3, 3)),  # Mixed pixel.
            (0, 3): ((2, 3, 3),),  # NoData must remain in denominators.
            (0, 4): ((1, 2, 2),),
        }
    )
    return dataset, mapping, checkpoint, run


def test_partial_mixed_and_optional_depths_use_map_ancestors(setup, tmp_path):
    evaluator = PredictionPointEvaluation(*setup, {})
    probabilities = np.array(
        [
            [[0.8, 0.1, 0.1, 0, 0.1]],
            [[0.1, 0.7, 0.2, 0, 0.8]],
            [[0.1, 0.2, 0.7, 0, 0.1]],
        ],
        dtype=np.float32,
    )
    # Distributed shards have unequal accumulated weights, not equally weighted votes.
    denominator = np.array([[10, 10, 10, 0, 10]], dtype=np.float32)
    result = np.array([[1, 2, 3, -9999, 2]])
    evaluator.capture(
        0, 1, [probabilities * 9, probabilities], denominator, result, denominator > 0
    )
    # Later map edits must not affect saved decisions or original observations.
    result[:] = 3
    report = evaluator.write(tmp_path / "report.json")
    assert report["coverage"]["test_pixels"] == 5
    leaf = report["after_constraint"]["leaf"]
    assert leaf["labeled_pixels"] == 3
    assert leaf["accuracy"] == pytest.approx(2 / 3)
    assert leaf["accuracy_on_predicted_pixels"] == 1
    assert leaf["mixed_pixels"]["pixel_count"] == 1
    assert leaf["mixed_pixels"]["observed_label_agreement"] == pytest.approx(2 / 3)
    levels = report["after_constraint"]["levels"]
    assert levels["ecosystem"]["labeled_pixels"] == 4
    assert levels["subtype"]["labeled_pixels"] == 2
    assert report["point_predictions"][0]["after_prediction"] == 1
    np.testing.assert_allclose(
        report["point_predictions"][0]["fused_probabilities"], [0.8, 0.1, 0.1]
    )


def test_experiment_split_takes_precedence_over_source(setup):
    dataset, mapping, checkpoint, run = setup
    _split(checkpoint.parent / "spatial_split.json", {"0:0": "train", "0:1": "test"})
    evaluator = PredictionPointEvaluation(dataset, mapping, checkpoint, run, {})
    assert evaluator.by_row == {0: [(1, ((1, -1, -1),))]}


def test_training_on_test_skips_without_loading_source_split(setup, tmp_path):
    dataset, mapping, checkpoint, run = setup
    (checkpoint.parent / "train_log.json").write_text(
        '{"supervision_policy": {"train_on_test": true}}'
    )
    (run / "spatial_split.json").unlink()
    evaluator = PredictionPointEvaluation(dataset, mapping, checkpoint, run, {})
    report = evaluator.write(tmp_path / "skipped.json")
    assert report["status"] == "skipped_test_used_for_training"
    assert report["before_constraint"]["leaf"]["accuracy"] is None
    assert report["coverage"]["test_pixels"] == 0


@pytest.mark.parametrize(
    "blocks,status",
    [
        ({"0:0": "train"}, "skipped_no_test_blocks"),
        ({"8:8": "test"}, "skipped_no_test_points"),
    ],
)
def test_empty_test_is_explicit(setup, blocks, status):
    dataset, mapping, checkpoint, run = setup
    _split(run / "spatial_split.json", blocks)
    assert (
        PredictionPointEvaluation(dataset, mapping, checkpoint, run, {}).status
        == status
    )


def test_missing_split_fails_instead_of_evaluating_all_labels(setup):
    dataset, mapping, checkpoint, run = setup
    (run / "spatial_split.json").unlink()
    with pytest.raises(FileNotFoundError):
        PredictionPointEvaluation(dataset, mapping, checkpoint, run, {})


def test_window_limit_selects_points_without_changing_fusion(setup):
    dataset, mapping, checkpoint, run = setup
    dataset.ground_truth_pixels = {(0, 0): 1, (0, 4): 2}
    dataset.query_windows_for_pixel = lambda row, column: [1] if column == 0 else [2]
    evaluator = PredictionPointEvaluation(
        dataset, mapping, checkpoint, run, {"max_windows": 1}
    )
    assert list(evaluator.by_row) == [0]
    assert len(evaluator.by_row[0]) == 1
    assert evaluator.by_row[0][0][0] == 0


def test_abstentions_count_as_false_negatives():
    metrics = _hard_metrics([1, 2, 2], [1, None, 1], 2)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 1]]
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["accuracy_on_predicted_pixels"] == 0.5
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["per_class_recall"] == [1, 0]


def test_dataset_exposes_original_mixed_observations():
    dataset = object.__new__(WindowedSampleDataset)
    dataset._ground_truth_levels = {(0, 0): (1, -1), (0, 1): (1, 2)}
    dataset._mixed_observations = {(0, 0): [(1, 1), (1, 2), (1, 2)]}
    assert dataset.evaluation_observations == {
        (0, 0): ((1, 1), (1, 2), (1, 2)),
        (0, 1): ((1, 2),),
    }


def test_partial_constraint_conflict_requires_all_descendants_restricted(setup):
    evaluator = PredictionPointEvaluation(*setup, {})
    assert evaluator._restricted((1, 2, -1), {2})
    assert not evaluator._restricted((1, -1, -1), {2})
    assert evaluator._restricted((1, -1, -1), {1, 2})
