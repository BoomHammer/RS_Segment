from pathlib import Path

from config import load_config


def test_load_config_resolves_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  root: data\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.data.root == (tmp_path / "data").resolve()
