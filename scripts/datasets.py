"""Create and validate a training dataset in a weak-label run directory."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from config import load_config
from data.raster_alignment import grid_from_config
from data.sample_index import WindowedSampleDataset, build_sample_index
from data.spatial_split import build_spatial_split
from data.stage2 import (
    benchmark_dataset,
    validate_dataset_windows,
    validate_stage2_config,
    write_stage2_report,
)
from data_stats import compute_statistics


def _first_raster(*directories: Path) -> Path:
    for directory in directories:
        files = sorted(directory.glob("*.tif"))
        if files:
            return files[0]
    raise FileNotFoundError("dynamic/static 目录中没有可用的 GeoTIFF")


def _artifact(run: Path, pattern: str) -> Path | None:
    files = sorted(run.glob(pattern))
    return files[-1] if files else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="切分并验证训练数据集")
    parser.add_argument("run", type=Path, help="weak_label.py 生成的结果目录")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--train-config", type=Path, default=Path("configs/train.yaml")
    )
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"伪标签结果目录不存在: {run}")
    config = load_config(args.config)
    with args.train_config.open(encoding="utf-8") as stream:
        train_config = yaml.safe_load(stream) or {}
    weak_label = run / "weak_labels.tif"
    if not weak_label.exists():
        raise FileNotFoundError(f"伪标签结果目录缺少弱标签: {weak_label}")
    stage2 = copy.deepcopy(config.data.stage2)
    statistics = _artifact(run, "raster_stats_*.json")
    if statistics is None:
        payload = compute_statistics(
            config,
            band=1,
            window_size=tuple(stage2.get("statistics_window_size", (1024, 1024))),
            nodata=config.data.raster.get("nodata", -9999),
        )
        statistics = run / "raster_stats_stage2.json"
        statistics.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    mapping_path = _artifact(run, "label_mapping*.json")
    mapping = (
        json.loads(mapping_path.read_text(encoding="utf-8"))
        if mapping_path is not None
        else None
    )
    target_config = config.data.target_grid
    reference = target_config.get("reference_raster") or _first_raster(
        config.data.dynamic, config.data.static
    )
    grid = grid_from_config(target_config, reference)
    index_path = run / "sample_index.json"
    build_sample_index(
        dynamic_dir=config.data.dynamic,
        static_dir=config.data.static,
        target_grid=grid,
        label_file=config.data.label_file,
        weak_label_file=weak_label,
        label_crs=config.data.label_crs,
        output=index_path,
    )
    stage2["statistics_file"] = str(statistics) if statistics else None
    stage2["split_file"] = str(run / "spatial_split.json")
    window = dict(stage2.get("window", {}))
    dataset = WindowedSampleDataset(
        index_path,
        window_size=tuple(window.get("size", (256, 256))),
        stride=tuple(window.get("stride", window.get("size", (256, 256)))),
        label_columns=config.data.label_columns,
        label_mapping=mapping,
        statistics=statistics,
        nodata=config.data.raster.get("nodata", -9999),
        stage2=stage2,
    )
    split = dict(stage2.get("split", {}))
    build_spatial_split(
        dataset,
        block_size=tuple(split.get("block_size", (2048, 2048))),
        ratios=tuple(split.get("ratios", (0.8, 0.1, 0.1))),
        seed=int(split.get("seed", 42)),
        output=run / "spatial_split.json",
    )
    config_report = validate_stage2_config(stage2, index_path)
    window_report = validate_dataset_windows(
        dataset,
        max_windows=int(dict(stage2.get("validation", {})).get("max_windows", 2)),
        require_ground_truth=bool(
            dict(stage2.get("validation", {})).get("require_ground_truth", True)
        ),
    )
    validation_report = {**config_report, **window_report}
    write_stage2_report(validation_report, run / "stage2_validation.json")
    benchmark_config = dict(stage2.get("benchmark", {}))
    if benchmark_config.get("enabled", True):
        training = dict(stage2.get("training", {}))
        loader = dict(train_config.get("dataloader", {}))
        sampling = dict(stage2.get("sampling", {}))
        benchmark_report = benchmark_dataset(
            dataset,
            split_file=run / "spatial_split.json",
            split=sampling.get("split", "train"),
            batch_size=int(loader.get("batch_size", 1)),
            num_workers=int(loader.get("num_workers", 0)),
            pin_memory=bool(loader.get("pin_memory", True)),
            persistent_workers=bool(loader.get("persistent_workers", False)),
            prefetch_factor=int(loader.get("prefetch_factor", 2)),
            batches=int(benchmark_config.get("batches", 20)),
            warmup=int(benchmark_config.get("warmup", 3)),
            amp_dtype=str(training.get("amp_dtype", "bfloat16")),
            gradient_accumulation_steps=int(
                training.get("gradient_accumulation_steps", 1)
            ),
        )
    else:
        benchmark_report = {"enabled": False}
    write_stage2_report(benchmark_report, run / "stage2_benchmark.json")
    print(f"数据集产物: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
