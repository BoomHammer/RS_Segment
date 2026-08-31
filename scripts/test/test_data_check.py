from pathlib import Path

from config import AppConfig, DataConfig
from data_check import check_data


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
