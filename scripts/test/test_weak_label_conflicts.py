"""Weak-label conflict resolution must ignore candidate order and duplication."""

from itertools import permutations

import numpy as np
import pytest

from weak_label.conflicts import CandidateState


@pytest.mark.parametrize(
    "candidates, expected",
    [
        ([(1, 0.9), (2, 0.9), (3, 0.9)], -9999),
        ([(1, 0.9), (2, 0.9), (1, 0.9)], -9999),
        ([(1, 0.9), (2, 0.93), (1, 0.96)], -9999),
        ([(1, 0.8), (2, 0.8), (3, 0.95)], 3),
        ([(1, 0.86), (2, 0.9), (1, 0.98)], 1),
        ([(1, 0.8), (1, 0.81)], 1),
    ],
)
def test_every_candidate_order_matches_per_class_maximum(candidates, expected):
    by_class = {}
    for label, score in candidates:
        by_class[label] = max(by_class.get(label, -np.inf), score)
    ranked = sorted(by_class.items(), key=lambda item: (-item[1], item[0]))
    for ordered in permutations(candidates):
        state = CandidateState.empty((1, 1), -9999)
        for label, score in ordered:
            state.update(label, score)
        assert state.resolve(nodata=-9999, conflict_margin=0.05).item() == expected
        assert state.best_labels.item() == ranked[0][0]
        assert state.best_scores.item() == pytest.approx(ranked[0][1])
        if len(ranked) > 1:
            assert state.second_labels.item() == ranked[1][0]
            assert state.second_scores.item() == pytest.approx(ranked[1][1])
        else:
            assert state.second_labels.item() == -9999


def test_masked_candidates_and_arbitrary_sharding_match_full_class_oracle():
    rng = np.random.default_rng(19)
    shape = (3, 4)
    candidates = [
        (
            rng.integers(1, 6, size=shape),
            rng.integers(0, 11, size=shape).astype(np.float32) / 10,
            rng.random(shape) > 0.3,
        )
        for _ in range(21)
    ]
    all_scores = np.full((5, *shape), -np.inf, dtype=np.float32)
    for labels, scores, selected in candidates:
        for code in range(1, 6):
            all_scores[code - 1] = np.maximum(
                all_scores[code - 1],
                np.where(selected & (labels == code), scores, -np.inf),
            )
    ranking = np.argsort(-all_scores, axis=0, kind="stable")
    ranked_scores = np.take_along_axis(all_scores, ranking, axis=0)
    expected = np.where(
        ranked_scores[0] > ranked_scores[1] + 0.05, ranking[0] + 1, -9999
    )

    for shard_count in (1, 2, 4, 25):
        order = rng.permutation(len(candidates))
        shards = [CandidateState.empty(shape, -9999) for _ in range(shard_count)]
        for position, index in enumerate(order):
            shards[position % shard_count].update(*candidates[index])
        merged = CandidateState.empty(shape, -9999)
        for index in rng.permutation(shard_count):
            shard = shards[index]
            merged.update(shard.best_labels, shard.best_scores)
            merged.update(shard.second_labels, shard.second_scores)
        np.testing.assert_array_equal(
            merged.resolve(nodata=-9999, conflict_margin=0.05), expected
        )
        np.testing.assert_array_equal(merged.best_labels, ranking[0] + 1)
        np.testing.assert_array_equal(merged.second_labels, ranking[1] + 1)
        np.testing.assert_array_equal(merged.best_scores, ranked_scores[0])
        np.testing.assert_array_equal(merged.second_scores, ranked_scores[1])


def test_exact_margin_stays_ambiguous_and_invalid_candidates_are_ignored():
    state = CandidateState.empty((1, 3), -9999)
    state.update(1, [[0.875, 0.875, 0.875]], [[True, True, False]])
    state.update(2, [[0.75, np.nan, 0.5]], [[True, True, False]])
    state.update(-9999, [[1.0, 1.0, 1.0]])
    state.update(3, [[np.inf, -1.0, np.nan]])
    assert state.resolve(nodata=-9999, conflict_margin=0.125).tolist() == [
        [-9999, 1, -9999]
    ]


@pytest.mark.parametrize("margin", [-0.1, np.inf, np.nan])
def test_invalid_conflict_margin_is_rejected(margin):
    with pytest.raises(ValueError, match="conflict_margin"):
        CandidateState.empty((1, 1), -9999).resolve(
            nodata=-9999, conflict_margin=margin
        )
