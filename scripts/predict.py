"""Run memory-bounded, overlapping whole-grid inference from a checkpoint."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from config import load_config
from data.sample_index import WindowedSampleDataset, sample_collate_fn
from models.architecture import SegFormerUtae


def _artifact(run: Path, pattern: str) -> Path:
    files = sorted(run.glob(pattern))
    if not files:
        raise FileNotFoundError(f"找不到阶段2产物 {pattern}: {run}")
    return files[-1]


def _load_run(checkpoint: Path) -> Path:
    metadata_path = checkpoint.parent / "train_log.json"
    if not metadata_path.is_file():
        metadata_path = checkpoint.parent / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"checkpoint 目录缺少 train_log.json 或 run.json: {checkpoint.parent}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_run = metadata.get("source_run")
    if not source_run:
        raise ValueError(f"训练元数据缺少 source_run: {metadata_path}")
    run = Path(source_run).resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"阶段2 run 目录不存在: {run}")
    return run


def _load_training_window_config(
    checkpoint: Path,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Read window geometry from the checkpoint's training record."""

    metadata_path = checkpoint.parent / "train_log.json"
    if not metadata_path.is_file():
        metadata_path = checkpoint.parent / "run.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    data_config_path = checkpoint.parent / "data.yaml"
    if not data_config_path.is_file():
        data_config_path = Path("configs/data.yaml")
    data_config = load_config(data_config_path)
    window = dict(data_config.data.stage2.get("window", {}))
    window_size = tuple(metadata.get("window_size", window.get("size", (256, 256))))
    stride = tuple(metadata.get("stride", window.get("stride", window_size)))
    grid_offset = tuple(metadata.get("grid_offset", (0, 0)))
    if any(len(value) != 2 for value in (window_size, stride, grid_offset)):
        raise ValueError(f"训练窗口元数据格式错误: {metadata_path}")
    return window_size, stride, grid_offset


def _load_training_halo(checkpoint: Path) -> tuple[int, int]:
    """Use recorded context; legacy experiments retain zero halo."""
    for name in ("train_log.json", "run.json"):
        path = checkpoint.parent / name
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            halo = tuple(metadata.get("halo", (0, 0)))
            if len(halo) != 2 or any(
                type(value) is not int or value < 0 for value in halo
            ):
                raise ValueError(f"训练 halo 元数据格式错误: {path}")
            return halo
    return (0, 0)


def _gaussian_weights(height: int, width: int, sigma_scale: float) -> np.ndarray:
    """Create a 2-D Gaussian window for weighted overlap blending."""

    if height < 1 or width < 1 or sigma_scale <= 0:
        raise ValueError("高斯权重参数必须为正数")
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    sigma = max(float(sigma_scale), 1e-3)
    weights = np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / (2 * sigma**2))
    # Suppress the unreliable receptive-field edges. With 50% overlap, the
    # centre of a neighbouring window covers every interior boundary, while
    # this prevents edge-only predictions from drawing straight tile lines.
    return np.maximum(weights, 1e-6).astype(np.float32)


def _write_mapping(mapping: dict[str, Any], path: Path) -> None:
    fields = ["数字", "大类", "小类", "大类中文", "小类中文"]
    classes = sorted(
        mapping.get("classes", []), key=lambda item: int(item["alliance_code"])
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for item in classes:
            writer.writerow(
                [
                    int(item["alliance_code"]),
                    item["formation"],
                    item["alliance"],
                    item.get("formation_zh", ""),
                    item.get("alliance_zh", ""),
                ]
            )


def predict(
    checkpoint: str | Path,
    output: str | Path,
    mapping_output: str | Path,
    *,
    window_size: tuple[int, int] = (256, 256),
    stride: tuple[int, int] = (128, 128),
    halo: tuple[int, int] = (0, 0),
    grid_offset: tuple[int, int] = (0, 0),
    device: str | None = None,
    sigma_scale: float = 0.35,
    overwrite: bool = False,
    override_ground_truth: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    amp_enabled: bool = True,
    amp_dtype: str = "bfloat16",
    cpu_threads: int | None = None,
    output_compress: str = "deflate",
    output_predictor: int = 2,
    output_bigtiff: str = "IF_SAFER",
    write_row_block: int = 512,
) -> tuple[Path, Path]:
    """Predict the target grid while keeping raster/model tensors bounded."""

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型权重不存在: {checkpoint_path}")
    output_path = Path(output).resolve()
    mapping_path = Path(mapping_output).resolve()
    if (output_path.exists() or mapping_path.exists()) and not overwrite:
        raise FileExistsError("输出已存在；如需覆盖请指定 --overwrite")
    if (
        window_size[0] < 2
        or window_size[1] < 2
        or stride[0] < 1
        or stride[1] < 1
        or stride[0] >= window_size[0]
        or stride[1] >= window_size[1]
        or stride[0] * 2 > window_size[0]
        or stride[1] * 2 > window_size[1]
        or len(halo) != 2
        or min(halo) < 0
    ):
        raise ValueError("无缝推理要求正数窗口、非负 halo 且 stride 不得超过窗口的一半")
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError(
            "batch_size 必须为正数，num_workers 不能为负数，prefetch_factor 必须为正数"
        )
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers 需要 num_workers > 0")
    if cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError("cpu_threads 必须为正数")
        torch.set_num_threads(cpu_threads)

    run = _load_run(checkpoint_path)
    data_config_path = checkpoint_path.parent / "data.yaml"
    if not data_config_path.is_file():
        data_config_path = Path("configs/data.yaml")
    data_config = load_config(data_config_path)
    index_path = _artifact(run, "sample_index.json")
    mapping = json.loads(
        _artifact(run, "label_mapping*.json").read_text(encoding="utf-8")
    )
    statistics_path = _artifact(run, "raster_stats*.json")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = payload.get("contract") if isinstance(payload, dict) else None
    if contract is None:
        raise ValueError("checkpoint 缺少训练时保存的模型 contract")
    model = SegFormerUtae.from_contract(contract)
    model.load_state_dict(payload["model"])
    selected_device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device in {None, "", "auto"}
        else device
    )
    model.to(selected_device).eval()

    dataset = WindowedSampleDataset(
        index_path,
        window_size=window_size,
        stride=stride,
        grid_offset=grid_offset,
        halo=halo,
        label_columns=data_config.data.label_columns,
        label_mapping=mapping,
        statistics=statistics_path,
        nodata=data_config.data.raster.get("nodata", -9999),
        stage2=dict(data_config.data.stage2),
        use_weak_labels=False,
    )
    grid = dataset.grid
    ground_truth_by_row: dict[int, dict[int, int]] = {}
    for (pixel_row, pixel_column), code in dataset.ground_truth_pixels.items():
        ground_truth_by_row.setdefault(pixel_row, {})[pixel_column] = code
    classes = int(contract["derived"]["num_classes"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    temp_files: list[Path] = []
    scores: np.memmap | None = None
    weights: np.memmap | None = None
    try:
        score_handle, score_name = tempfile.mkstemp(
            prefix="predict_scores_", suffix=".dat", dir=output_path.parent
        )
        weight_handle, weight_name = tempfile.mkstemp(
            prefix="predict_weights_", suffix=".dat", dir=output_path.parent
        )
        import os

        os.close(score_handle)
        os.close(weight_handle)
        temp_files = [Path(score_name), Path(weight_name)]
        scores = np.memmap(
            temp_files[0],
            mode="w+",
            dtype=np.float32,
            shape=(classes, grid.height, grid.width),
        )
        weights = np.memmap(
            temp_files[1],
            mode="w+",
            dtype=np.float32,
            shape=(grid.height, grid.width),
        )
    except Exception:
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)
        raise
    scores[:] = 0
    weights[:] = 0
    use_amp = selected_device.type == "cuda" and amp_enabled
    amp_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if amp_dtype not in amp_dtype_map:
        raise ValueError("amp_dtype 必须是 bfloat16 或 float16")
    # Halo windows at the outer raster boundary can have different spatial
    # shapes after clipping. Keep them serial so the existing collator does
    # not pad spatial tensors and the 4090 memory budget stays predictable.
    effective_batch_size = 1 if any(halo) else batch_size
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
        collate_fn=sample_collate_fn,
    )
    denominator = result = valid = gaussian = probabilities = valid_masks = None
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc="全图预测", unit="batch"):
                windows = batch["window"]
                input_windows = batch["input_window"]
                batch = {
                    key: value.to(selected_device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype_map[amp_dtype],
                    enabled=use_amp,
                ):
                    probabilities = (
                        model(batch)["fine_probability"].float().cpu().numpy()
                    )
                valid_masks = batch["valid_mask"].cpu().numpy()
                for batch_index, window in enumerate(windows):
                    height, width = int(window.height), int(window.width)
                    input_window = input_windows[batch_index]
                    offset_y = int(window.row_off - input_window.row_off)
                    offset_x = int(window.col_off - input_window.col_off)
                    probability = probabilities[
                        batch_index,
                        :,
                        offset_y : offset_y + height,
                        offset_x : offset_x + width,
                    ]
                    valid = valid_masks[
                        batch_index,
                        offset_y : offset_y + height,
                        offset_x : offset_x + width,
                    ]
                    gaussian = _gaussian_weights(height, width, sigma_scale) * valid
                    row, column = int(window.row_off), int(window.col_off)
                    scores[:, row : row + height, column : column + width] += (
                        probability * gaussian
                    )
                    weights[row : row + height, column : column + width] += gaussian
        assert scores is not None and weights is not None
        scores.flush()
        weights.flush()
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise RuntimeError("滑窗融合权重出现非法值")
        profile = {
            "driver": "GTiff",
            "width": grid.width,
            "height": grid.height,
            "count": 1,
            "dtype": "int32",
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": data_config.data.output_nodata,
            "compress": output_compress,
            "predictor": output_predictor,
            "BIGTIFF": output_bigtiff,
        }
        with rasterio.open(output_path, "w", **profile) as destination:
            for row in range(0, grid.height, write_row_block):
                end = min(row + write_row_block, grid.height)
                denominator = weights[row:end]
                result = np.full(
                    (end - row, grid.width),
                    data_config.data.output_nodata,
                    dtype=np.int32,
                )
                valid = denominator > 0
                if valid.any():
                    result[valid] = (
                        scores[:, row:end].argmax(axis=0)[valid].astype(np.int32) + 1
                    )
                if override_ground_truth:
                    for pixel_row in range(row, end):
                        for column, code in ground_truth_by_row.get(
                            pixel_row, {}
                        ).items():
                            result[pixel_row - row, column] = code
                destination.write(
                    result, 1, window=Window(0, row, grid.width, end - row)
                )
        _write_mapping(mapping, mapping_path)
    finally:
        # Windows keeps the backing file locked while any memmap view exists.
        # Clear the last row/output views before closing and deleting the files.
        denominator = result = valid = gaussian = None
        probabilities = valid_masks = None
        if scores is not None and weights is not None:
            scores.flush()
            weights.flush()
            score_mmap = scores._mmap
            weight_mmap = weights._mmap
            del scores, weights
            score_mmap.close()
            weight_mmap.close()
        gc.collect()
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)
    return output_path, mapping_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="基于 checkpoint 生成目标网格植被图")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/predict.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mapping-output", type=Path, default=None)
    parser.add_argument("--sigma-scale", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    with args.config.resolve().open(encoding="utf-8") as stream:
        predict_config = dict((yaml.safe_load(stream) or {}).get("predict", {}))
    loader_config = dict(predict_config.get("dataloader", {}))
    amp_config = dict(predict_config.get("amp", {}))
    raster_config = dict(predict_config.get("output_raster", {}))
    checkpoint = args.checkpoint.resolve()
    window_size, stride, grid_offset = _load_training_window_config(checkpoint)
    output = args.output or predict_config.get("output")
    mapping_output = args.mapping_output or predict_config.get("mapping_output")
    stem = checkpoint.parent / f"vegetation_{checkpoint.stem}"
    output, mapping = predict(
        checkpoint,
        output or stem.with_suffix(".tif"),
        mapping_output or stem.with_suffix(".csv"),
        window_size=window_size,
        stride=stride,
        halo=_load_training_halo(checkpoint),
        grid_offset=grid_offset,
        device=args.device or predict_config.get("device"),
        sigma_scale=args.sigma_scale
        if args.sigma_scale != 0.35
        else float(predict_config.get("blending", {}).get("sigma_scale", 0.35)),
        overwrite=args.overwrite or bool(predict_config.get("overwrite", False)),
        override_ground_truth=bool(predict_config.get("override_ground_truth", False)),
        batch_size=args.batch_size or int(predict_config.get("batch_size", 1)),
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(loader_config.get("num_workers", 0))
        ),
        pin_memory=(
            args.pin_memory
            if args.pin_memory is not None
            else bool(loader_config.get("pin_memory", True))
        ),
        persistent_workers=bool(loader_config.get("persistent_workers", False)),
        prefetch_factor=int(loader_config.get("prefetch_factor", 2)),
        amp_enabled=(
            args.amp if args.amp is not None else bool(amp_config.get("enabled", True))
        ),
        amp_dtype=str(amp_config.get("dtype", "bfloat16")),
        cpu_threads=(
            None
            if predict_config.get("cpu_threads") is None
            else int(predict_config["cpu_threads"])
        ),
        output_compress=str(raster_config.get("compress", "deflate")),
        output_predictor=int(raster_config.get("predictor", 2)),
        output_bigtiff=str(raster_config.get("bigtiff", "IF_SAFER")),
        write_row_block=int(raster_config.get("write_row_block", 512)),
    )
    print(f"植被图: {output}")
    print(f"类别对照表: {mapping}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
