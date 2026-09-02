import csv
import json
from pathlib import Path

from data.labels import (
    build_label_mapping,
    iter_encoded_labels,
    iter_label_rows,
    validate_label_mapping,
    validate_labels,
    write_label_artifacts,
)


def _write_labels(path: Path) -> None:
    fields = [
        "Index",
        "Alliance",
        "Formation",
        "Eng_Alliance",
        "Eng_Formation",
        "X",
        "Y",
    ]
    rows = [
        ["1", "小类A", "大类A", "Minor A", "Major A", "10", "20"],
        ["2", "小类A", "大类A", "Minor A", "Major A", "10", "20"],
        ["3", "小类B", "大类B", "Minor B", "Major B", "11", "21"],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


COLUMNS = {
    "x": "X",
    "y": "Y",
    "formation": "Eng_Formation",
    "alliance": "Eng_Alliance",
    "chn_formation": "Formation",
    "chn_alliance": "Alliance",
}


def test_label_mapping_and_streaming_validation(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(path)

    batches = list(iter_label_rows(path, label_columns=COLUMNS, batch_size=2))
    mapping = build_label_mapping(path, label_columns=COLUMNS)
    report = validate_labels(path, label_columns=COLUMNS, mapping=mapping, batch_size=2)

    assert [len(batch) for batch in batches] == [2, 1]
    assert mapping["minor_count"] == 2
    assert report.total_rows == report.valid_rows == 3
    assert report.duplicate_points == 1
    encoded = list(iter_encoded_labels(path, label_columns=COLUMNS, mapping=mapping))
    assert encoded[0][0].alliance_code == encoded[0][1].alliance_code
    assert encoded[0][0].alliance_code > 0


def test_label_mapping_validates_fine_to_coarse_contract() -> None:
    mapping = {
        "major_count": 1,
        "minor_count": 2,
        "classes": [
            {
                "formation_code": 1,
                "alliance_code": 1,
                "formation": "A",
                "alliance": "a",
            },
            {
                "formation_code": 1,
                "alliance_code": 2,
                "formation": "A",
                "alliance": "b",
            },
        ],
    }
    report = validate_label_mapping(mapping)
    assert report["valid"] is True
    assert report["alliance_to_formation"] == {"1": 1, "2": 1}


def test_label_artifacts_are_json(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(path)
    mapping, report = write_label_artifacts(
        path, output_dir=tmp_path, label_columns=COLUMNS
    )

    mapping_data = json.loads(mapping.read_text(encoding="utf-8"))
    assert mapping_data["major_count"] == 2
    assert mapping_data["output_nodata"] == -9999
    report_data = json.loads(report.read_text(encoding="utf-8"))
    assert report_data["invalid_rows"] == 0
    assert report_data["mapping_validation"]["valid"] is True
