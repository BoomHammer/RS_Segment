"""Classification heads used by the registered hierarchy ablations."""

from __future__ import annotations

import torch
from torch import nn

from models.architecture import HierarchicalHeads


def _aggregate_levels(leaf_probability, parents):
    levels = [leaf_probability]
    for edge in reversed(parents):
        edge = torch.as_tensor(edge, device=leaf_probability.device)
        parent = leaf_probability.new_zeros(
            leaf_probability.shape[0], int(edge.max()) + 1, *leaf_probability.shape[2:]
        )
        parent.index_add_(1, edge, levels[-1])
        levels.append(parent)
    return list(reversed(levels))


def _standard_output(levels):
    logs = [value.clamp_min(1e-8).log() for value in levels]
    output = {f"level_{depth}_logits": value for depth, value in enumerate(logs)}
    output.update(
        fine_logits=logs[-1],
        fine_probability=levels[-1],
        coarse_logits=logs[0],
        coarse_probability=levels[0],
    )
    return output


class FlatLeafHeads(nn.Module):
    """One leaf classifier; ancestor probabilities are derived, never predicted."""

    def __init__(self, channels, derived):
        super().__init__()
        self.classifier = nn.Conv2d(channels, derived["num_classes"], 1)
        self.parents = derived.get("level_parents", [derived["fine_to_coarse"]])

    def forward(self, features):
        leaf = self.classifier(features).softmax(1)
        output = _standard_output(_aggregate_levels(leaf, self.parents))
        output["head_mode"] = "flat"
        return output


class IndependentLevelHeads(nn.Module):
    """Predict every hierarchy level with an independent pointwise classifier."""

    def __init__(self, channels, derived):
        super().__init__()
        counts = derived.get("level_counts") or [
            max(derived["fine_to_coarse"]) + 1,
            derived["num_classes"],
        ]
        self.classifiers = nn.ModuleList(
            nn.Conv2d(channels, count, 1) for count in counts
        )

    def forward(self, features):
        probabilities = [
            classifier(features).softmax(1) for classifier in self.classifiers
        ]
        output = _standard_output(probabilities)
        output["head_mode"] = "independent"
        return output


class CRGNetHeads(nn.Module):
    """Shared features with the base and expanded classifiers from CRGNet."""

    def __init__(self, channels, derived):
        super().__init__()
        self.base = HierarchicalHeads.from_derived(channels, derived)
        self.expanded = HierarchicalHeads.from_derived(channels, derived)

    def forward(self, features):
        base = self.base(features)
        expanded = self.expanded(features)
        fine_probability = (base["fine_probability"] + expanded["fine_probability"]) / 2
        coarse_probability = (
            base["coarse_probability"] + expanded["coarse_probability"]
        ) / 2
        output = {
            **base,
            "fine_probability": fine_probability,
            "fine_logits": fine_probability.clamp_min(1e-8).log(),
            "coarse_probability": coarse_probability,
            "coarse_logits": coarse_probability.clamp_min(1e-8).log(),
            "base_output": base,
            "expanded_output": expanded,
            "head_mode": "crgnet",
        }
        for key, value in base.items():
            if key.startswith("level_") and key.endswith("_logits"):
                probability = (value.exp() + expanded[key].exp()) / 2
                output[key] = probability.clamp_min(1e-8).log()
        return output


class SAMDualHeads(CRGNetHeads):
    """E09.1: two conditional hierarchies with the same averaged inference API."""

    def forward(self, features):
        output = super().forward(features)
        output["head_mode"] = "sam_crgnet"
        return output


class HSSNHeads(nn.Module):
    """All-node multilabel scores and projection head following HSSN."""

    def __init__(self, channels, derived, projection_channels=256):
        super().__init__()
        self.counts = derived.get("level_counts") or [
            max(derived["fine_to_coarse"]) + 1,
            derived["num_classes"],
        ]
        self.parents = derived.get("level_parents", [derived["fine_to_coarse"]])
        self.classifier = nn.Conv2d(channels, sum(self.counts), 1)
        self.projection = nn.Sequential(
            nn.Conv2d(channels, projection_channels, 1),
            nn.ReLU(),
            nn.Conv2d(projection_channels, projection_channels, 1),
        )
        paths = []
        for leaf in range(self.counts[-1]):
            path = [leaf]
            for edge in reversed(self.parents):
                path.append(edge[path[-1]])
            paths.append(list(reversed(path)))
        self.paths = paths

    def forward(self, features):
        scores = self.classifier(features).sigmoid()
        offsets = [0]
        for count in self.counts:
            offsets.append(offsets[-1] + count)
        path_scores = torch.stack(
            [
                sum(scores[:, offsets[d] + node] for d, node in enumerate(path))
                for path in self.paths
            ],
            dim=1,
        )
        leaf = path_scores.softmax(1)
        output = _standard_output(_aggregate_levels(leaf, self.parents))
        output.update(
            hssn_scores=scores,
            hssn_embeddings=self.projection(features),
            head_mode="hssn",
        )
        return output


def build_experiment_heads(channels, derived, settings):
    mode = settings.get("classification_head")
    if mode == "flat":
        return FlatLeafHeads(channels, derived)
    if mode == "independent":
        return IndependentLevelHeads(channels, derived)
    if mode == "crgnet":
        return CRGNetHeads(channels, derived)
    if mode == "sam_crgnet":
        return SAMDualHeads(channels, derived)
    if mode == "hssn":
        return HSSNHeads(
            channels,
            derived,
            int(settings.get("hssn_projection_channels", 256)),
        )
    return HierarchicalHeads.from_derived(channels, derived)
