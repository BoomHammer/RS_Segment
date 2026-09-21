import torch
from torch.nn import functional as F

from losses.focal import focal_cross_entropy


def test_zero_gamma_matches_weighted_cross_entropy() -> None:
    logits = torch.tensor([[[[2.0, -1.0]], [[-1.0, 2.0]]]])
    labels = torch.tensor([[[0, 1]]])
    weights = torch.tensor([1.0, 3.0])

    actual = focal_cross_entropy(logits, labels, gamma=0, weight=weights)
    expected = F.cross_entropy(logits, labels, weight=weights)

    torch.testing.assert_close(actual, expected)


def test_focal_loss_downweights_easy_examples_and_ignores_missing_labels() -> None:
    logits = torch.tensor([[[[5.0, 0.0]], [[0.0, 0.0]]]], requires_grad=True)
    labels = torch.tensor([[[0, -1]]])

    focal = focal_cross_entropy(logits, labels, gamma=2.0, ignore_index=-1)
    cross_entropy = focal_cross_entropy(logits, labels, gamma=0.0, ignore_index=-1)

    assert 0 < focal < cross_entropy
    focal.backward()
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_returns_differentiable_zero_for_empty_target() -> None:
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    labels = torch.full((1, 2, 2), -1)

    loss = focal_cross_entropy(logits, labels, ignore_index=-1)
    loss.backward()

    assert loss == 0
    assert torch.count_nonzero(logits.grad) == 0


def test_sample_mean_preserves_class_weight_in_single_point_gradient() -> None:
    logits = torch.tensor([[[[1.0]], [[0.0]]]], requires_grad=True)
    labels = torch.ones(1, 1, 1, dtype=torch.long)
    plain = focal_cross_entropy(logits, labels)
    weighted = focal_cross_entropy(
        logits,
        labels,
        weight=torch.tensor([1.0, 3.0]),
        weight_normalization="sample_mean",
    )
    plain_gradient = torch.autograd.grad(plain, logits, retain_graph=True)[0]
    weighted_gradient = torch.autograd.grad(weighted, logits)[0]
    torch.testing.assert_close(weighted, 3 * plain)
    torch.testing.assert_close(weighted_gradient, 3 * plain_gradient)


def test_accumulated_single_points_match_joint_weighted_batch() -> None:
    logits = torch.randn(4, 3, 1, 1, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 2]).reshape(4, 1, 1)
    weights = torch.tensor([0.5, 1.5, 2.0])
    options = {"weight": weights, "weight_normalization": "sample_mean"}
    joint = focal_cross_entropy(logits, labels, **options)
    accumulated = (
        sum(
            focal_cross_entropy(logits[i : i + 1], labels[i : i + 1], **options)
            for i in range(4)
        )
        / 4
    )
    joint_gradient = torch.autograd.grad(joint, logits, retain_graph=True)[0]
    accumulated_gradient = torch.autograd.grad(accumulated, logits)[0]
    torch.testing.assert_close(accumulated_gradient, joint_gradient)
