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
