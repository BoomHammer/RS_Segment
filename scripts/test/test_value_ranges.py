from pathlib import Path

import numpy as np

from data.value_ranges import load_value_ranges, valid_and_scaled, value_range_for


def test_value_ranges_resolve_product_variants_and_scale(tmp_path: Path) -> None:
    path = tmp_path / "ranges.csv"
    path.write_text(
        "Data,Min,Max,Scale\n"
        "LST,7500,65535,0.02\n"
        "SR,-100,16000,0.0001\n"
        "ASPECT,0,360,\n",
        encoding="utf-8",
    )
    ranges = load_value_ranges(path)

    assert value_range_for("SR_B7", ranges) == ranges["sr"]
    assert value_range_for("DSM100aspect", ranges) == ranges["aspect"]
    values = valid_and_scaled(np.array([0, 7500, 10000]), ranges["lst"])
    assert np.isnan(values[0])
    np.testing.assert_allclose(values[1:], [150, 200])
