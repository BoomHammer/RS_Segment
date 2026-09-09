"""Tests for synchronized spatial and continuous-feature augmentation."""

import torch

from data.augmentations import SynchronizedAugmentation


def _sample() -> dict[str, object]:
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    labels = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones((2, 2), dtype=torch.bool)
    split_mask = torch.tensor([[True, False], [False, False]])
    return {
        "dynamic": values.clone(),
        "static": values[0].clone(),
        "ground_truth": labels.clone(),
        "ground_truth_mask": mask.clone(),
        "weak_label": labels.clone(),
        "weak_label_mask": mask.clone(),
        "dynamic_valid_mask": mask.clone(),
        "static_valid_mask": mask.clone(),
        "valid_mask": mask.clone(),
        "supervision_split_mask": split_mask,
    }


def test_spatial_transform_is_shared_by_rasters_labels_and_masks() -> None:
    sample = _sample()
    augmented = SynchronizedAugmentation(
        horizontal_flip_probability=1.0,
        vertical_flip_probability=0.0,
        rotate_probability=0.0,
        seed=1,
    )(sample)
    expected_values = torch.tensor([[[[2.0, 1.0], [4.0, 3.0]]]])
    expected_labels = torch.tensor([[2, 1], [4, 3]])
    expected_split_mask = torch.tensor([[False, True], [False, False]])
    assert torch.equal(augmented["dynamic"], expected_values)
    assert torch.equal(augmented["static"], expected_values[0])
    assert torch.equal(augmented["ground_truth"], expected_labels)
    assert torch.equal(augmented["weak_label"], expected_labels)
    assert torch.equal(augmented["supervision_split_mask"], expected_split_mask)
    assert augmented["augmentation"]["horizontal_flip"] is True


def test_spectral_noise_does_not_change_labels_or_masks() -> None:
    sample = _sample()
    augmented = SynchronizedAugmentation(
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotate_probability=0.0,
        spectral_noise_std=0.1,
        seed=2,
    )(sample)
    assert not torch.equal(augmented["dynamic"], sample["dynamic"])
    assert torch.equal(augmented["ground_truth"], sample["ground_truth"])
    assert torch.equal(augmented["weak_label_mask"], sample["weak_label_mask"])
