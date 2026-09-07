"""Verify context geometry, augmentation and legacy prediction settings."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data.augmentations import SynchronizedAugmentation
from data.sample_index import WindowedSampleDataset


@pytest.mark.parametrize("origin", [0, 3])
@pytest.mark.parametrize("flip", [0.0, 1.0])
def test_halo_core_tracks_augmented_labels(origin, flip):
    dataset = object.__new__(WindowedSampleDataset)
    dataset.grid = SimpleNamespace(width=10, height=10)
    dataset.index = SimpleNamespace(iloc=[SimpleNamespace(row=origin, column=origin)])
    dataset.window_size = (4, 4)
    dataset.halo = (2, 2)
    dataset.transforms = SynchronizedAugmentation(
        horizontal_flip_probability=flip,
        vertical_flip_probability=flip,
        rotate_probability=1.0,
        seed=42,
    )

    def read(window):
        shape = (int(window.height), int(window.width))
        labels = torch.zeros(shape, dtype=torch.long)
        offset = min(origin, 2)
        labels[offset : offset + 4, offset : offset + 4] = 1
        return {
            "valid_mask": torch.ones(shape, dtype=torch.bool),
            "ground_truth": labels,
        }

    dataset._read_window = read
    sample = dataset[0]
    assert sample["core_mask"].sum() == 16
    assert torch.equal(sample["core_mask"], sample["ground_truth"].bool())
    assert sample["valid_mask"].sum() > 16
    assert sample["input_window"].width == (6 if origin == 0 else 8)


def test_prediction_uses_training_halo_and_legacy_zero(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "halo_predict", Path(__file__).parents[1] / "predict.py"
    )
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    checkpoint = tmp_path / "last.pt"
    metadata = tmp_path / "train_log.json"
    metadata.write_text(json.dumps({"halo": [32, 32]}))
    assert predict._load_training_halo(checkpoint) == (32, 32)
    metadata.write_text("{}")
    assert predict._load_training_halo(checkpoint) == (0, 0)
    metadata.write_text(json.dumps({"halo": [-1, 32]}))
    with pytest.raises(ValueError):
        predict._load_training_halo(checkpoint)
