"""Order-independent, bounded-memory aggregation of weak-label candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio


@dataclass
class CandidateState:
    """Keep the two highest per-class scores, even when the output is ambiguous."""

    best_labels: np.ndarray
    best_scores: np.ndarray
    second_labels: np.ndarray
    second_scores: np.ndarray

    @classmethod
    def empty(cls, shape, nodata):
        return cls(
            np.full(shape, nodata, dtype=np.int32),
            np.full(shape, -np.inf, dtype=np.float32),
            np.full(shape, nodata, dtype=np.int32),
            np.full(shape, -np.inf, dtype=np.float32),
        )

    def update(self, labels, scores, selected=None):
        """Merge one candidate per pixel; return where the best evidence changed."""
        labels = np.broadcast_to(np.asarray(labels, dtype=np.int32), self.best_labels.shape)
        scores = np.broadcast_to(np.asarray(scores, dtype=np.float32), self.best_scores.shape)
        valid = (labels > 0) & np.isfinite(scores) & (scores >= 0)
        if selected is not None:
            valid &= selected
        different = labels != self.best_labels
        replace_best = valid & (
            (scores > self.best_scores)
            | ((scores == self.best_scores) & (labels < self.best_labels))
        )
        replace_second = valid & different & ~replace_best & (
            (scores > self.second_scores)
            | ((scores == self.second_scores) & (labels < self.second_labels))
        )
        demote = replace_best & different
        self.second_labels[demote] = self.best_labels[demote]
        self.second_scores[demote] = self.best_scores[demote]
        self.second_labels[replace_second] = labels[replace_second]
        self.second_scores[replace_second] = scores[replace_second]
        self.best_labels[replace_best] = labels[replace_best]
        self.best_scores[replace_best] = scores[replace_best]
        return replace_best

    def resolve(self, *, nodata, conflict_margin):
        """Only emit a class that beats every other class by more than the margin."""
        if not np.isfinite(conflict_margin) or conflict_margin < 0:
            raise ValueError("conflict_margin 必须是非负有限数")
        accepted = (
            (self.best_labels > 0)
            & np.isfinite(self.best_scores)
            & (self.best_scores > self.second_scores + conflict_margin)
        )
        return np.where(accepted, self.best_labels, nodata).astype(np.int32)


def open_candidate_sources(stack, shards, reference):
    """Open both distinct-class candidates; resolved legacy rasters lose evidence."""
    sources = []
    for shard in shards:
        if not {"candidate_labels", "candidate_scores"} <= shard.keys():
            raise ValueError("旧版弱标签分片缺少两个类别的候选分数，请重新生成弱标签")
        pair = tuple(
            stack.enter_context(rasterio.open(shard[key]))
            for key in ("candidate_labels", "candidate_scores")
        )
        for source in pair:
            if (source.shape, source.transform, source.crs, source.count) != (
                reference.shape,
                reference.transform,
                reference.crs,
                2,
            ):
                raise ValueError("弱标签候选分片网格不一致或缺少两个候选波段")
        sources.append(pair)
    return sources


def merge_candidate_window(sources, window, *, nodata, conflict_margin):
    """Merge shard summaries without discarding locally ambiguous evidence."""
    state = CandidateState.empty((int(window.height), int(window.width)), nodata)
    for labels, scores in sources:
        for band in (1, 2):
            state.update(labels.read(band, window=window), scores.read(band, window=window))
    return state.resolve(nodata=nodata, conflict_margin=conflict_margin)
