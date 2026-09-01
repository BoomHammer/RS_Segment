import numpy as np
import pytest

from data.npc import NegativeCandidate, PointSeed
from inference.pointsam import PointSAMRequest
from inference.sam2_backend import build_sam2_prompts, prepare_sam2_image


def test_prepare_sam2_image_supports_chw_and_fixed_range() -> None:
    image = np.array([[[0, 1]], [[0.5, 1]], [[1, 2]]], dtype=np.float32)

    rgb = prepare_sam2_image(image, input_range=(0, 2))

    assert rgb.shape == (1, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 1].tolist() == [127, 127, 255]


def test_build_sam2_prompts_converts_row_column_to_xy() -> None:
    request = PointSAMRequest(
        image=np.zeros((4, 5, 3), dtype=np.uint8),
        positive_points=(PointSeed(2, 3, 1, 1),),
        negative_points=(NegativeCandidate(1, 4, 1, 2, 2, 0.9),),
    )

    coordinates, labels = build_sam2_prompts(request)

    assert coordinates.tolist() == [[3.0, 2.0], [4.0, 1.0]]
    assert labels.tolist() == [1, 0]


def test_prepare_sam2_image_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        prepare_sam2_image(np.full((3, 2, 2), np.nan))
