"""Resolve the configured SR bands used as SAM2 RGB channels."""

from __future__ import annotations

from pathlib import Path

from data.filename_parser import parse_filename


def validate_sam_bands(bands: object) -> tuple[int, int, int]:
    """Validate and normalize the three configured SAM2 input bands."""

    if not isinstance(bands, (list, tuple)) or len(bands) != 3:
        raise ValueError("weak_labels.sam_bands 必须包含三个波段编号")
    normalized = tuple(int(band) for band in bands)
    if any(band < 1 for band in normalized) or len(set(normalized)) != 3:
        raise ValueError("weak_labels.sam_bands 必须是三个互不重复的正整数")
    return normalized


def discover_sam_images(
    directory: str | Path, bands: object = (1, 4, 3)
) -> tuple[Path, Path, Path]:
    """Find the first complete SR date and return paths in configured order."""

    configured_bands = validate_sam_bands(bands)
    by_date: dict[str, dict[int, Path]] = {}
    for path in sorted(Path(directory).glob("*.tif")):
        try:
            metadata = parse_filename(path)
        except ValueError:
            continue
        if metadata.category != "SR" or metadata.date is None:
            continue
        if metadata.band in configured_bands:
            by_date.setdefault(metadata.date, {})[metadata.band] = path
    for date in sorted(by_date):
        paths = by_date[date]
        if all(band in paths for band in configured_bands):
            return tuple(paths[band] for band in configured_bands)  # type: ignore[return-value]
    band_text = "/".join(f"B{band}" for band in configured_bands)
    raise FileNotFoundError(f"未找到包含 SR {band_text} 的同日期影像")


def discover_sam_composites(
    directory: str | Path, composites: object
) -> tuple[tuple[Path, Path, Path], ...]:
    """Resolve every non-empty configured composite on one common SR date."""

    if not isinstance(composites, (list, tuple)):
        raise ValueError("weak_labels.CompositeBands1-5 必须是波段列表或空值")
    configured = [validate_sam_bands(bands) for bands in composites if bands]
    if not configured:
        raise ValueError("至少需要配置一个非空的 CompositeBands")

    by_date: dict[str, dict[int, Path]] = {}
    for path in sorted(Path(directory).glob("*.tif")):
        try:
            metadata = parse_filename(path)
        except ValueError:
            continue
        if metadata.category == "SR" and metadata.date is not None:
            by_date.setdefault(metadata.date, {})[metadata.band] = path
    for date in sorted(by_date):
        paths = by_date[date]
        if all(band in paths for bands in configured for band in bands):
            return tuple(tuple(paths[band] for band in bands) for bands in configured)  # type: ignore[return-value]
    band_text = ", ".join("/".join(map(str, bands)) for bands in configured)
    raise FileNotFoundError(f"未找到满足所有 SAM 组合的同日期 SR 影像: {band_text}")


def discover_sam_videos(
    directory: str | Path, composites: object
) -> tuple[tuple[tuple[Path, Path, Path], ...], int]:
    """Resolve complete temporal frames and return them with the July keyframe."""

    if not isinstance(composites, (list, tuple)):
        raise ValueError("CompositeBands1-5 必须是波段列表或空值")
    configured = [validate_sam_bands(bands) for bands in composites if bands]
    if not configured:
        raise ValueError("至少需要配置一个非空的 CompositeBands")
    by_date: dict[str, dict[int, Path]] = {}
    for path in sorted(Path(directory).glob("*.tif")):
        try:
            metadata = parse_filename(path)
        except ValueError:
            continue
        if metadata.category == "SR" and metadata.date is not None:
            by_date.setdefault(metadata.date, {})[metadata.band] = path
    dates = [
        date
        for date in sorted(by_date)
        if all(band in by_date[date] for bands in configured for band in bands)
    ]
    july_dates = [date for date in dates if date[5:7] == "07"]
    if not july_dates:
        raise FileNotFoundError("未找到包含所有配置波段的 7 月 SR 关键帧")
    keyframe_date = july_dates[0]
    frames = tuple(
        tuple(tuple(by_date[date][band] for band in bands) for bands in configured)
        for date in dates
    )
    return frames, dates.index(keyframe_date)
