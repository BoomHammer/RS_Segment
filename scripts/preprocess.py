"""Run label validation and raster statistics as one preprocessing job."""

from __future__ import annotations

import argparse
import json
from itertools import chain, islice
from pathlib import Path

from config import load_config
from data.labels import iter_encoded_labels
from data.raster_alignment import target_grid_from_raster
from data.weak_labels import WeakLabelGenerationConfig, generate_weak_labels
from inference.sam2_backend import SAM2Inferencer
from preprocessing import run_preprocessing


def _discover_rgb_images(directory: Path) -> tuple[Path, Path, Path]:
    """Find the first complete SR date with MOD09A1 RGB bands 1, 4, and 3."""

    by_date: dict[str, dict[int, Path]] = {}
    for path in sorted(directory.glob("SR*B[134].tif")):
        stem = path.stem
        date = stem[2:8]
        band = int(stem[-1])
        by_date.setdefault(date, {})[band] = path
    for date in sorted(by_date):
        bands = by_date[date]
        if all(band in bands for band in (1, 3, 4)):
            return bands[1], bands[4], bands[3]
    raise FileNotFoundError("未找到包含 SR B1/B4/B3 的同日期影像")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行完整数据预处理并生成统一产物目录")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--band", type=int, default=1)
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--nodata", type=float, default=None)
    parser.add_argument("--image", type=Path, nargs=3, metavar=("R", "G", "B"))
    parser.add_argument("--reference-raster", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input-range", type=float, nargs=2, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-weak-labels", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    weak_config = config.data.weak_labels
    run_dir = run_preprocessing(
        config,
        band=args.band,
        window_size=tuple(args.window_size or [1024, 1024]),
        nodata=args.nodata,
    )
    if not args.skip_weak_labels:
        image_paths = (
            tuple(args.image)
            if args.image
            else _discover_rgb_images(config.data.dynamic)
        )
        reference = args.reference_raster or image_paths[0]
        mapping_path = next(run_dir.glob("label_mapping_*.json"))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        record_batches = iter_encoded_labels(
            config.data.label_file,
            label_columns=config.data.label_columns,
            mapping=mapping,
            batch_size=4096,
        )
        records = chain.from_iterable(record_batches)
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise ValueError("max-samples 必须是正整数")
            records = islice(records, args.max_samples)
        grid = target_grid_from_raster(
            reference,
            target_crs=config.data.target_grid.get("crs", "EPSG:4326"),
            resolution=tuple(
                config.data.target_grid.get("resolution", [0.00225, 0.00225])
            ),
        )
        alliance_names = {
            int(item["alliance_code"]): str(item["alliance"])
            for item in mapping["classes"]
        }
        output = run_dir / "weak_labels.tif"
        inferencer = SAM2Inferencer(
            args.checkpoint
            or config.data.sam2_checkpoint
            or Path("SAM/sam2.1_hiera_small.pt"),
            device=args.device,
            input_range=(0.0, 255.0),
        )
        generate_weak_labels(
            records,
            grid=grid,
            image_paths=image_paths,
            inferencer=inferencer,
            output_path=output,
            quality_report_path=run_dir / "weak_labels_quality.json",
            quality_visualization_path=run_dir / "weak_labels_quality.png",
            alliance_names=alliance_names,
            config=WeakLabelGenerationConfig(
                window_size=tuple(
                    args.window_size or weak_config.get("input_window_size", [384, 384])
                ),
                label_radius=int(weak_config.get("label_radius", 48)),
                input_range=tuple(
                    args.input_range
                    or weak_config.get("input_range", [-100.0, 16000.0])
                ),
                stretch_percentiles=tuple(
                    weak_config.get("stretch_percentiles", [2.0, 98.0])
                ),
                medium_confidence=float(weak_config.get("medium_confidence", 0.70)),
                medium_label_radius=int(weak_config.get("medium_label_radius", 8)),
                fallback_radius=int(weak_config.get("fallback_radius", 4)),
                spectral_distance_threshold=float(
                    weak_config.get("spectral_distance_threshold", 0.12)
                ),
                min_confidence=(
                    args.min_confidence
                    if args.min_confidence > 0
                    else float(weak_config.get("mask_confidence", 0.85))
                ),
                logit_threshold=float(weak_config.get("logit_threshold", 0.5)),
                reject_boundary_touch=bool(
                    weak_config.get("reject_boundary_touch", True)
                ),
                conflict_margin=float(weak_config.get("conflict_margin", 0.05)),
                output_nodata=config.data.output_nodata,
            ),
        )
    print(f"预处理产物: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
