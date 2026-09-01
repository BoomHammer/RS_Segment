import numpy as np
import pytest

from data.npc import (
    NegativeCandidate,
    PointSeed,
    expand_positive_samples,
    filter_negative_candidates,
    generate_candidate_negatives,
)


def test_positive_expansion_masks_nodata_and_conflicts() -> None:
    seeds = [PointSeed(3, 3, 1, 1), PointSeed(3, 5, 2, 2)]
    valid = np.ones((7, 9), dtype=bool)
    valid[3, 3] = False

    labels = expand_positive_samples(seeds, valid.shape, radius=2, valid_mask=valid)

    assert labels[3, 3] == 0
    assert labels[3, 4] == -1
    assert labels[3, 1] == 1
    assert labels[3, 7] == 2


def test_negative_candidates_apply_hierarchy_and_confidence() -> None:
    probabilities = np.zeros((3, 9, 9), dtype=np.float32)
    probabilities[2, 3, 6] = 0.95
    probabilities[2, 3, 4] = 0.99
    probabilities[0, 4, 3] = 0.50
    seed = PointSeed(3, 3, 10, 1)
    mapping = {1: 10, 2: 10, 3: 20}
    labels = expand_positive_samples([seed], (9, 9), radius=1)

    candidates = generate_candidate_negatives(
        probabilities,
        [seed],
        labels,
        class_ids=[1, 2, 3],
        alliance_to_formation=mapping,
        protection_radius=1,
        search_radius=4,
        min_confidence=0.9,
    )

    assert [(item.row, item.column) for item in candidates] == [(3, 6)]
    assert candidates[0].predicted_formation_code == 20


def test_filter_keeps_top_k_per_anchor() -> None:
    candidates = [
        NegativeCandidate(1, 1, 2, 3, 4, 0.91),
        NegativeCandidate(1, 2, 2, 3, 5, 0.99),
        NegativeCandidate(1, 3, 2, 3, 6, 0.95),
        NegativeCandidate(2, 1, 7, 3, 4, 0.92),
    ]

    result = filter_negative_candidates(
        candidates, min_confidence=0.9, max_candidates_per_anchor=2
    )

    assert [(item.anchor_alliance_code, item.confidence) for item in result] == [
        (2, 0.99),
        (2, 0.95),
        (7, 0.92),
    ]


def test_invalid_probability_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="三维"):
        generate_candidate_negatives(
            np.ones((2, 4, 4, 1)),
            [PointSeed(1, 1, 1, 1)],
            np.zeros((4, 4), dtype=np.int32),
            class_ids=[1, 2],
            alliance_to_formation={1: 1, 2: 2},
        )
