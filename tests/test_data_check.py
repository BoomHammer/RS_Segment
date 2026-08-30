from pathlib import Path

from rs_segment.config import AppConfig, DataConfig
from rs_segment.data_check import check_data


def test_check_data_accepts_expected_layout(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    (tmp_path / "raw").mkdir()
    config = AppConfig(
        data=DataConfig(
            root=tmp_path,
            labels=tmp_path / "labels",
            raw=tmp_path / "raw",
        )
    )

    assert check_data(config) == []
