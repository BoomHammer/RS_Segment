"""Check experiment isolation and the requested DNN/loss contract."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location(
    "pixel_dnn_ab", Path(__file__).parents[1] / "pixel_dnn_ab.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_training_only_scaler_and_missing_column():
    matrix = np.array([[1, np.nan], [3, np.nan], [9999, 8]], dtype=np.float32)
    mean, scale, count = MODULE.fit_scaler(matrix, [0, 1])
    np.testing.assert_array_equal(mean, [2, 0])
    np.testing.assert_array_equal(scale, [1, 1])
    np.testing.assert_array_equal(count, [2, 0])


def test_four_hidden_layers_l2_and_categorical_loss():
    model = MODULE.PixelDNN(7)
    assert [layer.out_features for layer in model.layers] == [256, 128, 64, 32, 32]
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.fill_(0.5)
            layer.bias.fill_(100)
    expected = 0.001 * 0.25 * sum(layer.weight.numel() for layer in model.layers)
    assert np.isclose(model.regularization().item(), expected)
    logits = torch.randn(5, 32)
    labels = torch.tensor([0, 4, 7, 9, 31])
    one_hot = torch.nn.functional.one_hot(labels, 32)
    categorical = -(one_hot * logits.softmax(-1).log()).sum(-1).mean()
    torch.testing.assert_close(
        torch.nn.functional.cross_entropy(logits, labels), categorical
    )


def test_pixel_split_uses_original_block_boundary():
    manifest = {
        "block_size": [10, 20],
        "blocks": {"0:0": "train", "0:1": "test", "1:0": "validation"},
    }
    result = MODULE.owners([9, 9, 10], [19, 20, 19], manifest)
    assert result.tolist() == ["train", "test", "validation"]
