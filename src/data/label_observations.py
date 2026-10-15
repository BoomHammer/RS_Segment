"""Sparse mixed-pixel observations, with one unit of evidence per pixel."""

import torch


def supervision_items(dataset):
    """Yield every observed class once per pixel for spatial split coverage."""
    if hasattr(dataset, "supervision_memberships"):
        for pixel, classes in dataset.supervision_memberships.items():
            for code in classes:
                yield pixel, code
    else:
        yield from getattr(
            dataset, "supervision_pixels", dataset.ground_truth_pixels
        ).items()


def mixed_level_observations(observations, target, valid, depth, active=None):
    """Select known conflicting labels; agreed labels use the dense target."""
    positions = observations[:, :3]
    labels = observations[:, depth + 3] - 1
    b, y, x = positions.unbind(1)
    keep = labels.ge(0) & valid[b, y, x] & target[b, y, x].lt(0)
    if active is not None:
        keep &= active[labels.clamp_min(0)]
    positions, labels = positions[keep], labels[keep]
    unique, inverse, counts = torch.unique(
        positions, dim=0, return_inverse=True, return_counts=True
    )
    weights = counts[inverse].float().reciprocal()
    return positions, labels, weights, unique, inverse
