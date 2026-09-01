"""Visualize the RGB window that is passed to SAM2 after preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rasterio.windows import Window

from config import load_config
from data.raster_alignment import aligned_raster, grid_from_config
from inference.sam2_backend import prepare_sam2_image


def _discover_rgb_images(directory: Path) -> tuple[Path, Path, Path]:
    """Find the first dynamic SR date with bands 1, 2, and 3."""

    by_date: dict[str, dict[int, Path]] = {}
    for path in sorted(directory.glob("SR*B[123].tif")):
        stem = path.stem
        date = stem[2:8]
        band = int(stem[-1])
        by_date.setdefault(date, {})[band] = path
    for date in sorted(by_date):
        bands = by_date[date]
        if all(band in bands for band in (1, 2, 3)):
            return bands[1], bands[2], bands[3]
    raise FileNotFoundError("未找到包含 SR B1/B2/B3 的同日期影像")


def _preview_window(
    width: int,
    height: int,
    window_size: tuple[int, int],
    row: int | None,
    column: int | None,
) -> Window:
    """Return a bounded window, centered by default."""

    window_width, window_height = window_size
    if window_width < 1 or window_height < 1:
        raise ValueError("window_size 必须是两个正整数")
    window_width = min(window_width, width)
    window_height = min(window_height, height)
    center_row = height // 2 if row is None else row
    center_column = width // 2 if column is None else column
    left = max(0, min(center_column - window_width // 2, width - window_width))
    top = max(0, min(center_row - window_height // 2, height - window_height))
    return Window(left, top, window_width, window_height)


def read_sam2_input_window(
    image_paths: tuple[Path, Path, Path],
    *,
    config_path: Path = Path("configs/data.yaml"),
    reference_raster: Path,
    window_size: tuple[int, int] = (1024, 1024),
    row: int | None = None,
    column: int | None = None,
    input_range: tuple[float, float] = (0.0, 10000.0),
) -> tuple[np.ndarray, Window]:
    """Read and convert one aligned window exactly as the SAM2 backend does."""

    config = load_config(config_path)
    grid = grid_from_config(config.data.target_grid, reference_raster)
    window = _preview_window(grid.width, grid.height, window_size, row, column)
    with (
        aligned_raster(image_paths[0], grid) as red,
        aligned_raster(image_paths[1], grid) as green,
        aligned_raster(image_paths[2], grid) as blue,
    ):
        arrays = [
            dataset.read(1, window=window, masked=True)
            for dataset in (red, green, blue)
        ]
    image = np.stack([array.filled(0) for array in arrays], axis=0)
    return prepare_sam2_image(image, input_range=input_range), window


def write_sam2_input_visualization(
    image_paths: tuple[Path, Path, Path],
    *,
    config_path: Path = Path("configs/data.yaml"),
    reference_raster: Path,
    output: Path,
    window_size: tuple[int, int] = (1024, 1024),
    row: int | None = None,
    column: int | None = None,
    input_range: tuple[float, float] = (0.0, 10000.0),
) -> Path:
    """Write RGB and per-channel previews of the actual SAM2 uint8 input."""

    from PIL import Image, ImageDraw

    image, window = read_sam2_input_window(
        image_paths,
        config_path=config_path,
        reference_raster=reference_raster,
        window_size=window_size,
        row=row,
        column=column,
        input_range=input_range,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = [Image.fromarray(image, mode="RGB")]
    panels.extend(
        Image.fromarray(image[:, :, index], mode="L").convert("RGB")
        for index in range(3)
    )
    labels = ("SAM2 RGB", "SAM2 R", "SAM2 G", "SAM2 B")
    label_height = 32
    canvas_size = (
        image.shape[1] * 2,
        (image.shape[0] + label_height) * 2,
    )
    canvas = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        left = (index % 2) * image.shape[1]
        top = (index // 2) * (image.shape[0] + label_height)
        canvas.paste(panel, (left, top + label_height))
        draw.text((left + 8, top + 8), label, fill="black")
    canvas.save(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可视化进入 SAM2 前的处理后影像")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--image", type=Path, nargs=3, metavar=("R", "G", "B"))
    parser.add_argument("--reference-raster", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/sam2_input_preview.png")
    )
    parser.add_argument("--window-size", type=int, nargs=2, default=(1024, 1024))
    parser.add_argument("--row", type=int)
    parser.add_argument("--column", type=int)
    parser.add_argument("--input-range", type=float, nargs=2, default=(0.0, 10000.0))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    image_paths = (
        tuple(args.image) if args.image else _discover_rgb_images(config.data.dynamic)
    )
    reference = args.reference_raster or image_paths[0]
    output = write_sam2_input_visualization(
        image_paths,
        config_path=args.config,
        reference_raster=reference,
        output=args.output,
        window_size=tuple(args.window_size),
        row=args.row,
        column=args.column,
        input_range=tuple(args.input_range),
    )
    print(f"SAM2 输入预览: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_preview_window_is_bounded() -> None:
    window = _preview_window(100, 80, (32, 24), row=0, column=99)

    assert (window.col_off, window.row_off) == (68, 0)
    assert (window.width, window.height) == (32, 24)
