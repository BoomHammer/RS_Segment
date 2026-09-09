from pathlib import Path

from config import load_config


def test_load_config_resolves_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  root: data\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.data.root == (tmp_path / "data").resolve()


def test_load_config_resolves_stage2_artifact_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n"
        "  stage2:\n"
        "    statistics_file: stats.json\n"
        "    split_file: split.json\n"
        "    value_range_file: ranges.csv\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.stage2["statistics_file"] == str(
        (tmp_path / "stats.json").resolve()
    )
    assert config.data.stage2["split_file"] == str((tmp_path / "split.json").resolve())
    assert config.data.stage2["value_range_file"] == str(
        (tmp_path / "ranges.csv").resolve()
    )
