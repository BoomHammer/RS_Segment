from dataclasses import dataclass

import numpy as np
import pytest

from data.label_quality import (
    evaluate_label_quality,
    write_label_quality_report,
    write_label_quality_visualization,
)
from data.npc import NegativeCandidate, PointSeed
from inference.pointsam import (
    PointSAMPrediction,
    PointSAMRequest,
    run_pointsam_with_npc,
)


@dataclass
class FakePointSAM:
    request: PointSAMRequest | None = None

    def predict(self, request: PointSAMRequest) -> PointSAMPrediction:
        self.request = request
        return PointSAMPrediction(mask=np.ones((4, 5)), confidence=np.full((4, 5), 0.9))


def test_npc_points_are_passed_to_replaceable_backend() -> None:
    backend = FakePointSAM()
    seed = PointSeed(1, 2, 1, 1)
    negative = NegativeCandidate(2, 3, 1, 2, 2, 0.95)

    prediction = run_pointsam_with_npc(
        backend, np.zeros((3, 4, 5)), [seed], [negative], spatial_shape=(4, 5)
    )

    assert prediction.mask.shape == (4, 5)
    assert backend.request is not None
    assert backend.request.positive_points == (seed,)
    assert backend.request.negative_points == (negative,)


def test_quality_report_contains_metrics_and_writes_json(tmp_path) -> None:
    labels = np.array([[1, 1, 0, -1], [0, 2, 2, 0]], dtype=np.int32)
    seeds = [PointSeed(0, 0, 1, 1), PointSeed(1, 2, 2, 2), PointSeed(9, 9, 1, 1)]
    report = evaluate_label_quality(
        labels,
        seeds=seeds,
        alliance_to_formation={1: 1, 2: 2},
        negative_candidates=[NegativeCandidate(0, 3, 1, 2, 2, 0.9)],
    )
    path = write_label_quality_report(report, tmp_path / "quality.json")

    assert report["coverage"] == pytest.approx(4 / 8)
    assert report["invalid_pixel_count"] == 1
    assert report["sample_quality"]["invalid_samples"] == 1
    assert report["negative_candidate_count"] == 1
    assert report["class_distribution"]["1"]["alliance"] == "1"
    assert path.is_file()


def test_quality_visualization_writes_png(tmp_path) -> None:
    path = write_label_quality_visualization(
        np.array([[1, 0], [0, 2]], dtype=np.int32),
        tmp_path / "quality.png",
    )

    assert path.is_file()
