"""Generate PointSAM weak labels and all label-generation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import chain
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config  # noqa: E402
from data.labels import build_label_mapping, iter_encoded_labels  # noqa: E402
from data.raster_alignment import target_grid_from_raster  # noqa: E402
from inference.sam2_backend import SAM2Inferencer  # noqa: E402
from weak_label.generation import (  # noqa: E402
    WeakLabelGenerationConfig,
    generate_weak_labels,
)
from weak_label.sam_input import discover_sam_videos  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成流式 PointSAM 伪标签")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/weak_label.yaml"))
    parser.add_argument("--image", type=Path, nargs=3, metavar=("R", "G", "B"))
    parser.add_argument("--reference-raster", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--quality-visualization", type=Path)
    parser.add_argument("--window-size", type=int, nargs=2, default=None)
    parser.add_argument("--input-range", type=float, nargs=2, default=None)
    parser.add_argument("--mask-confidence", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.data_config)
    with args.config.open(encoding="utf-8") as stream:
        weak_config = dict((yaml.safe_load(stream) or {}).get("weak_label", {}))
    composite_names = (
        ["命令行影像"]
        if args.image
        else [
            f"CompositeBands{index}"
            for index in range(1, 6)
            if weak_config.get(f"CompositeBands{index}", [])
        ]
    )
    image_paths, keyframe_index = (
        (((tuple(args.image),),), 0)
        if args.image
        else discover_sam_videos(
            config.data.dynamic,
            [weak_config.get(f"CompositeBands{index}", []) for index in range(1, 6)],
        )
    )
    reference_raster = args.reference_raster or image_paths[0][0][0]
    grid = target_grid_from_raster(
        reference_raster,
        target_crs=config.data.target_grid.get("crs", "EPSG:4326"),
        resolution=tuple(config.data.target_grid.get("resolution", [0.00225, 0.00225])),
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.data.processed / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    label_mapping = build_label_mapping(
        config.data.label_file,
        label_columns=config.data.label_columns,
        schema=config.data.label_schema,
    )
    mapping = run_dir / "label_mapping.json"
    mapping.write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record_batches = iter_encoded_labels(
        config.data.label_file,
        label_columns=config.data.label_columns,
        mapping=label_mapping,
    )
    records = chain.from_iterable(record_batches)
    output = args.output or run_dir / "weak_labels.tif"
    report = args.quality_report or output.with_name("weak_labels_quality.json")
    visualization = args.quality_visualization or output.with_name(
        "weak_labels_quality.png"
    )
    inferencer = SAM2Inferencer(
        args.checkpoint
        or config.data.sam2_checkpoint
        or Path("third_party/SAM/sam2.1_hiera_small.pt"),
        device=args.device,
        input_range=(0.0, 255.0),
        use_video=True,
    )
    generate_weak_labels(
        records,
        grid=grid,
        image_paths=image_paths,
        inferencer=inferencer,
        output_path=output,
        quality_report_path=report,
        quality_visualization_path=visualization,
        config=WeakLabelGenerationConfig(
            window_size=tuple(
                args.window_size or weak_config.get("input_window_size", [384, 384])
            ),
            label_radius=int(weak_config.get("label_radius", 48)),
            input_range=tuple(
                args.input_range or weak_config.get("input_range", [-100.0, 16000.0])
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
            min_confidence=float(
                args.mask_confidence
                if args.mask_confidence is not None
                else weak_config.get("mask_confidence", 0.85)
            ),
            logit_threshold=float(weak_config.get("logit_threshold", 0.5)),
            reject_boundary_touch=bool(weak_config.get("reject_boundary_touch", True)),
            conflict_margin=float(weak_config.get("conflict_margin", 0.05)),
            mask_fusion=str(weak_config.get("mask_fusion", "intersection")),
            weighted_vote_threshold=float(
                weak_config.get("weighted_vote_threshold", 0.5)
            ),
            output_nodata=config.data.output_nodata,
        ),
        keyframe_index=keyframe_index,
        composite_names=composite_names,
    )
    print(f"弱标签: {output}")
    print(f"质量报告: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
