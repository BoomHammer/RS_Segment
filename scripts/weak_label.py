"""Generate PointSAM weak labels and all label-generation artifacts."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from datetime import datetime
from itertools import chain, zip_longest
from pathlib import Path

import numpy as np
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config  # noqa: E402
from data.labels import build_label_mapping, iter_encoded_labels  # noqa: E402
from data.raster_alignment import target_grid_from_raster  # noqa: E402
from distributed_runtime import (  # noqa: E402
    auto_launch,
    barrier,
    broadcast_object,
    finalize,
    gather_objects,
    initialize,
    shard_sequence,
)
from inference.sam2_backend import SAM2Inferencer  # noqa: E402
from weak_label.generation import (  # noqa: E402
    WeakLabelGenerationConfig,
    evaluate_label_quality_from_raster,
    generate_weak_labels,
)
from weak_label.quality import write_label_quality_visualization  # noqa: E402
from weak_label.sam_input import discover_sam_videos  # noqa: E402


def _load_shard_outcomes(shards):
    """Restore strided input order; CSV identifiers may be empty or nonnumeric."""
    groups = [
        json.loads(Path(shard["outcomes"]).read_text(encoding="utf-8"))
        for shard in shards
    ]
    return [item for row in zip_longest(*groups) for item in row if item is not None]


def _merge_weak_label_shards(
    shards: list[dict[str, str]],
    output: Path,
    *,
    nodata: int,
    conflict_margin: float,
) -> None:
    """Merge per-GPU labels using the same confidence/conflict semantics."""

    with rasterio.open(shards[0]["labels"]) as reference:
        profile = reference.profile.copy()
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as destination:
            label_sources = [
                rasterio.open(shard["labels"])  # noqa: SIM115
                for shard in shards
            ]
            score_sources = [
                rasterio.open(shard["scores"])  # noqa: SIM115
                for shard in shards
            ]
            try:
                for _, window in destination.block_windows(1):
                    shape = (int(window.height), int(window.width))
                    labels = np.full(shape, nodata, dtype=np.int32)
                    scores = np.full(shape, -np.inf, dtype=np.float32)
                    for label_source, score_source in zip(
                        label_sources, score_sources, strict=True
                    ):
                        candidate_labels = label_source.read(1, window=window)
                        candidate_scores = score_source.read(1, window=window)
                        selected = candidate_labels > 0
                        better = selected & (
                            candidate_scores > scores + conflict_margin
                        )
                        tied = (
                            selected
                            & np.isfinite(scores)
                            & (labels != candidate_labels)
                            & (np.abs(candidate_scores - scores) <= conflict_margin)
                        )
                        labels[better] = candidate_labels[better]
                        scores[better] = candidate_scores[better]
                        labels[tied] = nodata
                        scores[tied] = -np.inf
                    destination.write(labels, 1, window=window)
            finally:
                for source in [*label_sources, *score_sources]:
                    source.close()


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
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="从运行目录中的弱标签断点继续",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="将新生成的弱标签写入已有预处理目录（不作为断点恢复）",
    )
    args = parser.parse_args(argv)
    if args.resume is not None and args.run_dir is not None:
        parser.error("--resume 和 --run-dir 不能同时使用")
    launch_result = auto_launch(
        __file__, sys.argv[1:] if argv is None else argv, device=args.device
    )
    if launch_result is not None:
        return launch_result
    distributed = initialize(args.device)
    atexit.register(finalize, distributed)
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
    run_dir = None
    if distributed.is_main:
        if args.run_dir is not None:
            run_dir = args.run_dir.resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"预处理运行目录不存在: {run_dir}")
        elif args.resume is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = config.data.processed / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = args.resume.resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"弱标签运行目录不存在: {run_dir}")
    run_dir = Path(broadcast_object(str(run_dir) if run_dir else None, distributed))
    label_mapping = build_label_mapping(
        config.data.label_file,
        label_columns=config.data.label_columns,
        schema=config.data.label_schema,
    )
    mapping = run_dir / "label_mapping.json"
    if args.resume is None and distributed.is_main:
        mapping.write_text(
            json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    record_batches = iter_encoded_labels(
        config.data.label_file,
        label_columns=config.data.label_columns,
        mapping=label_mapping,
    )
    records = list(chain.from_iterable(record_batches))
    output = args.output or run_dir / "weak_labels.tif"
    report = args.quality_report or output.with_name("weak_labels_quality.json")
    visualization = args.quality_visualization or output.with_name(
        "weak_labels_quality.png"
    )
    inferencer = SAM2Inferencer(
        args.checkpoint
        or config.data.sam2_checkpoint
        or Path("third_party/SAM/sam2.1_hiera_small.pt"),
        device=distributed.device,
        input_range=(0.0, 255.0),
        use_video=True,
    )
    generation_config = WeakLabelGenerationConfig(
        window_size=tuple(
            args.window_size or weak_config.get("input_window_size", [384, 384])
        ),
        label_radius=int(weak_config.get("label_radius", 48)),
        input_range=tuple(
            args.input_range or weak_config.get("input_range", [-100.0, 16000.0])
        ),
        stretch_percentiles=tuple(weak_config.get("stretch_percentiles", [2.0, 98.0])),
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
        weighted_vote_threshold=float(weak_config.get("weighted_vote_threshold", 0.5)),
        output_nodata=config.data.output_nodata,
    )
    if not distributed.distributed:
        generate_weak_labels(
            records,
            grid=grid,
            image_paths=image_paths,
            inferencer=inferencer,
            output_path=output,
            quality_report_path=report,
            quality_visualization_path=visualization,
            config=generation_config,
            keyframe_index=keyframe_index,
            composite_names=composite_names,
            resume=args.resume is not None,
        )
        print(f"弱标签: {output}")
        print(f"质量报告: {report}")
        return 0
    shard_output = output.with_name(f".{output.name}.rank{distributed.rank}")
    shard_scores = output.with_name(f".{output.name}.rank{distributed.rank}.scores.tif")
    shard_outcomes = output.with_name(
        f".{output.name}.rank{distributed.rank}.outcomes.json"
    )
    generate_weak_labels(
        shard_sequence(records, distributed),
        grid=grid,
        image_paths=image_paths,
        inferencer=inferencer,
        output_path=shard_output,
        config=generation_config,
        keyframe_index=keyframe_index,
        composite_names=composite_names,
        resume=args.resume is not None,
        score_output_path=shard_scores,
        sample_outcomes_path=shard_outcomes,
    )
    shards = gather_objects(
        {
            "labels": str(shard_output),
            "scores": str(shard_scores),
            "outcomes": str(shard_outcomes),
        },
        distributed,
    )
    if distributed.is_main:
        _merge_weak_label_shards(
            shards,
            output,
            nodata=config.data.output_nodata,
            conflict_margin=generation_config.conflict_margin,
        )
        outcomes = _load_shard_outcomes(shards)
        quality = evaluate_label_quality_from_raster(output, sample_outcomes=outcomes)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with rasterio.open(output) as dataset:
            scale = min(1.0, 1024 / max(dataset.width, dataset.height))
            preview = dataset.read(
                1,
                out_shape=(
                    1,
                    max(1, int(dataset.height * scale)),
                    max(1, int(dataset.width * scale)),
                ),
                resampling=rasterio.enums.Resampling.nearest,
            )
            valid = preview != dataset.nodata
        write_label_quality_visualization(preview, visualization, valid_mask=valid)
        for shard in shards:
            Path(shard["labels"]).unlink(missing_ok=True)
            Path(shard["scores"]).unlink(missing_ok=True)
            Path(shard["outcomes"]).unlink(missing_ok=True)
        print(f"弱标签: {output}")
        print(f"质量报告: {report}")
    barrier(distributed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
