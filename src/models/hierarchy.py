"""Conditional classification for one or three root-to-leaf levels."""

from torch import nn


class MultiLevelHeads(nn.Module):
    """Share spatial features; add only pointwise classifiers for each edge."""

    def __init__(self, channels, counts, parents, head_factory=None):
        super().__init__()
        if not 1 <= len(counts) <= 3 or len(parents) != len(counts) - 1:
            raise ValueError("分类层级必须为 1、2 或 3 层")
        if any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("每层类别数必须为正整数")
        self.parents = [tuple(edge) for edge in parents]
        self.counts = counts
        factory = head_factory or (lambda size: nn.Conv2d(channels, size, 1))
        self.root = factory(counts[0])
        self.transitions = nn.ModuleList()
        for depth, edge in enumerate(self.parents):
            if (
                len(edge) != counts[depth + 1]
                or any(type(value) is not int for value in edge)
                or set(edge) != set(range(counts[depth]))
            ):
                raise ValueError("相邻层级父类映射必须完整、连续且每个父类有子类")
            self.transitions.append(
                nn.ModuleList(
                    factory(edge.count(parent)) for parent in range(counts[depth])
                )
            )

    def forward(self, features):
        root_logits = self.root(features)
        # Probability accumulation in FP32 avoids deep-path AMP underflow.
        log_probability = root_logits.float().log_softmax(1)
        output = {"level_0_logits": log_probability}
        for depth, (edge, experts) in enumerate(
            zip(self.parents, self.transitions, strict=True), 1
        ):
            child = log_probability.new_empty(
                (features.shape[0], len(edge), *log_probability.shape[-2:])
            )
            for parent, expert in enumerate(experts):
                indices = [i for i, value in enumerate(edge) if value == parent]
                child[:, indices] = (
                    expert(features).float().log_softmax(1)
                    + log_probability[:, parent : parent + 1]
                )
            log_probability = child
            output[f"level_{depth}_logits"] = child
        output.update(
            {
                "fine_logits": log_probability,
                "fine_probability": log_probability.exp(),
                "coarse_logits": root_logits,
                "coarse_probability": root_logits.float().softmax(1),
            }
        )
        return output
