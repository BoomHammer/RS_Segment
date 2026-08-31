from pathlib import Path

from data.filename_parser import parse_filename


def test_parse_date_product() -> None:
    metadata = parse_filename(Path("GPP230218.tif"))

    assert metadata.to_dict() == {
        "path": "GPP230218.tif",
        "filename": "GPP230218.tif",
        "category": "GPP",
        "temporal_resolution": "date",
        "date": "2023-02-18",
        "month": None,
        "band": None,
    }


def test_parse_monthly_product() -> None:
    metadata = parse_filename("SOIL2004.tif")

    assert metadata.month == "2020-04"
    assert metadata.temporal_resolution == "month"


def test_parse_modis_band() -> None:
    metadata = parse_filename("SR230805B6.tif")

    assert metadata.category == "SR"
    assert metadata.date == "2023-08-05"
    assert metadata.band == 6


def test_month_resolution_is_decided_by_digit_length() -> None:
    metadata = parse_filename("ANY2308.tif")

    assert metadata.category == "ANY"
    assert metadata.month == "2023-08"
