"""Train MAESTRO-S (or a legacy model) from a prepared run directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import yaml
from tqdm import tqdm

from config import load_config
from data.augmentations import SynchronizedAugmentation
from data.balanced_sampling import CappedClassSampler, ClassBalancedPointSampler
from data.overlap import overlapping_view
from data.sample_index import WindowedSampleDataset
from data.sampling import SpatialWeightedSampler, build_dataloader
from data.spatial_split import build_spatial_split, load_spatial_split
from data.training_policy import (
    assert_supervision_isolated,
    isolate_splits,
    point_windows,
    supervision_summary,
    training_class_weights,
)
from evaluation import evaluate_points as _evaluate
from losses.overlap import overlap_consistency_loss
from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.config import load_model_contract
from precision import resolve_amp_dtype, scaled_optimizer_step
from seed import seed_everything

METRICS_CSV_FIELDS = (
    "epoch",
    "train_ground_truth_loss",
    "train_ground_truth_accuracy",
    "train_weak_label_loss",
    "train_weak_label_accuracy",
    "validation_ground_truth_loss",
    "validation_ground_truth_accuracy",
    "validation_ground_truth_macro_f1",
)


def _initialize_weights(model, contract, path):
    """Start a new optimizer from compatible trained weights, never a resume."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    previous = payload["contract"]
    for key in ("architecture", "backbone", "temporal", "static", "fusion", "maestro"):
        if previous.get(key) != contract.get(key):
            raise ValueError(f"Initialization architecture mismatch: {key}")
    for key in ("dynamic_features", "static_features", "fine_to_coarse"):
        if previous["derived"].get(key) != contract["derived"].get(key):
            raise ValueError(f"Initialization input/label mismatch: {key}")
    model.load_state_dict(payload["model"], strict=True)
    return {
        "checkpoint": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "optimizer": "fresh",
    }


def _atomic_save(payload: dict, path: Path) -> None:
    """Replace a checkpoint only after its complete contents reach disk."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_metrics_csv(path: Path, row: dict[str, float | int]) -> None:
    """Append one validated epoch, without duplicating it after a resume."""

    has_header = path.is_file() and path.stat().st_size > 0
    if has_header:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            last_epoch = int(rows[-1]["epoch"])
            current_epoch = int(row["epoch"])
            if last_epoch == current_epoch:
                return
            if last_epoch > current_epoch:
                raise ValueError(
                    f"CSV 中最后一轮 {last_epoch} 晚于待写入轮次 {current_epoch}"
                )
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRICS_CSV_FIELDS)
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def _assert_resume_split_matches(path: Path, manifest: object) -> None:
    """Compare normalized manifests instead of raw JSON container types."""

    if path.is_file() and load_spatial_split(path) != manifest:
        raise ValueError("数据集空间划分已改变，旧断点不能用于新划分；请开始新实验")


def _source_supervision_loss(
    components: dict[str, torch.Tensor], source: str
) -> torch.Tensor | None:
    """Return one source's unweighted supervised loss across available heads."""

    direct = components.get(f"{source}_loss")
    if direct is not None:
        return direct
    fine = components.get(f"{source}_fine_loss")
    coarse = components.get(f"{source}_coarse_loss")
    if fine is None:
        return None
    return fine if coarse is None else fine + coarse


def _restrict_manifest_to_region(
    dataset: WindowedSampleDataset,
    manifest: object,
    fraction: float,
) -> object:
    """Keep windows whose centers fall inside the central region."""

    if not 0 < fraction <= 1:
        raise ValueError("region_fraction 必须位于 (0, 1]")
    width = int(dataset.grid.width * fraction)
    height = int(dataset.grid.height * fraction)
    left = (dataset.grid.width - width) // 2
    top = (dataset.grid.height - height) // 2
    right = left + width
    bottom = top + height

    def inside(index: int) -> bool:
        record = dataset.index.iloc[index]
        column = int(record.column)
        row = int(record.row)
        center_x = column + min(dataset.window_size[0], dataset.grid.width - column) / 2
        center_y = row + min(dataset.window_size[1], dataset.grid.height - row) / 2
        return left <= center_x < right and top <= center_y < bottom

    splits = {
        name: [index for index in indices if inside(index)]
        for name, indices in manifest.splits.items()
    }
    if not splits.get("train") or not splits.get("validation"):
        raise ValueError("中心区域筛选后 train 或 validation 集为空")
    return replace(manifest, splits=splits)


def _new_experiment_dir(root: Path) -> Path:
    """Create a unique experiment directory using the training start time."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / timestamp
    suffix = 1
    while output.exists():
        output = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output.mkdir(parents=True)
    return output


class ModelEMA:
    """Exponential moving average of model parameters and buffers."""

    def __init__(self, model: SegFormerUtae, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("ema.decay 必须位于 (0, 1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    def update(self, model: SegFormerUtae) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1 - self.decay
                )
            else:
                self.shadow[name].copy_(value.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def copy_to(self, model: SegFormerUtae) -> None:
        model.load_state_dict(self.shadow)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 MAESTRO-S 遥感分割模型")
    parser.add_argument("run", type=Path, help="datasets.py 生成的数据集目录")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, help="完整训练断点 last.pt 的路径")
    parser.add_argument("--init-checkpoint", type=Path, help="新实验的初始模型权重")
    parser.add_argument("--region-fraction", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-size", type=int, nargs=2, default=None)
    parser.add_argument("--stride", type=int, nargs=2, default=None)
    parser.add_argument("--halo", type=int, nargs=2, default=None)
    parser.add_argument("--grid-offset", type=int, nargs=2, default=(0, 0))
    args = parser.parse_args(argv)
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    run = args.run.resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"数据集目录不存在: {run}")
    resume = None
    if args.resume is not None:
        args.resume = args.resume.resolve()
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume.get("resume_version") != 1:
            raise ValueError("需要完整训练断点 last.pt，推理模型文件不能用于断点续训")
        if resume["source_run"] != str(run):
            raise ValueError("续训的数据集目录与断点不一致")
        if (
            args.output_dir is not None
            and args.output_dir.resolve() != args.resume.parent
        ):
            raise ValueError("续训必须使用断点所在的实验目录")
        args.output_dir = args.resume.parent
        args.train_config = args.output_dir / "train.yaml"
        args.config = args.output_dir / "model.yaml"
        args.data_config = args.output_dir / "data.yaml"
        for name, value in resume["window_options"].items():
            setattr(args, name, value)
        if args.epochs is None:
            args.epochs = resume["target_epochs"]
        elif args.epochs != resume["target_epochs"]:
            raise ValueError("断点续训须保持原目标 epochs，以保留原学习率计划")
    with args.train_config.open(encoding="utf-8") as stream:
        train_config = yaml.safe_load(stream) or {}
    training = dict(train_config.get("training", {}))
    seed_everything(int(training.get("seed", 42)))
    policy = dict(train_config.get("supervision_policy", {}))
    if resume is not None and not policy.get("fixed_spatial_supervision", False):
        raise ValueError("旧断点缺少固定空间监督隔离，不能继续用于正式训练；请重新训练")
    epochs = args.epochs if args.epochs is not None else int(training.get("epochs", 1))
    if epochs < 1:
        raise ValueError("epochs 必须是正整数")
    experiment_started_at = datetime.now().isoformat()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _new_experiment_dir(Path("experiments"))
    )
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"训练输出目录非空: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"model_{output.name}.pt"

    data_config = load_config(args.data_config)
    stage2 = data_config.data.stage2
    if "input_normalization" in train_config:
        stage2 = {**stage2, "normalization": dict(train_config["input_normalization"])}
    window = dict(stage2.get("window", {}))
    window_size = tuple(args.window_size or window.get("size", (256, 256)))
    window_stride = tuple(args.stride or window.get("stride", window_size))
    halo = tuple(args.halo or window.get("halo", (0, 0)))
    if resume is not None:
        halo = tuple(resume["train_log"].get("halo", (0, 0)))
    if len(halo) != 2 or any(type(value) is not int or value < 0 for value in halo):
        raise ValueError("halo 必须是两个非负整数")
    if (
        len(window_size) != 2
        or len(window_stride) != 2
        or min(window_size) < 1
        or min(window_stride) < 1
        or window_stride[0] * 2 > window_size[0]
        or window_stride[1] * 2 > window_size[1]
    ):
        raise ValueError("训练窗口必须为正数，且 stride 不得超过 window_size 的一半")
    mapping_path = next(iter(sorted(run.glob("label_mapping*.json"))), None)
    if mapping_path is None:
        raise FileNotFoundError(f"数据集目录缺少标签映射: {run}")
    statistics = next(iter(sorted(run.glob("raster_stats*.json"))), None)
    dataset_kwargs = {
        "index": run / "sample_index.json",
        "window_size": window_size,
        "stride": window_stride,
        "halo": halo,
        "grid_offset": tuple(args.grid_offset),
        "label_columns": data_config.data.label_columns,
        "label_mapping": json.loads(mapping_path.read_text(encoding="utf-8")),
        "statistics": statistics,
        "nodata": data_config.data.raster.get("nodata", -9999),
        "stage2": stage2,
    }
    if policy.get("disable_weak_labels", False):
        dataset_kwargs["use_weak_labels"] = False
    augmentation_config = dict(stage2.get("augmentation", {}))
    train_transforms = SynchronizedAugmentation(
        horizontal_flip_probability=float(
            augmentation_config.get("horizontal_flip_probability", 0.5)
        ),
        vertical_flip_probability=float(
            augmentation_config.get("vertical_flip_probability", 0.5)
        ),
        rotate_probability=float(augmentation_config.get("rotate_probability", 0.5)),
        spectral_noise_std=float(augmentation_config.get("spectral_noise_std", 0.0)),
        spectral_gain_std=float(augmentation_config.get("spectral_gain_std", 0.0)),
        seed=int(training.get("seed", 42)),
    )
    dataset = WindowedSampleDataset(
        **dataset_kwargs,
        transforms=train_transforms
        if augmentation_config.get("enabled", False)
        else None,
    )
    validation_dataset = WindowedSampleDataset(
        **{**dataset_kwargs, "use_weak_labels": False},
        transforms=None,
    )
    split_path = run / "spatial_split.json"
    if args.grid_offset != (0, 0):
        split = dict(stage2.get("split", {}))
        manifest = build_spatial_split(
            dataset,
            block_size=tuple(split.get("block_size", (2048, 2048))),
            ratios=tuple(split.get("ratios", (0.8, 0.1, 0.1))),
            seed=int(split.get("seed", 42)),
        )
    else:
        if not split_path.is_file():
            raise FileNotFoundError(f"数据集目录缺少空间划分文件: {split_path}")
        manifest = load_spatial_split(split_path)
    if args.region_fraction is not None:
        manifest = _restrict_manifest_to_region(dataset, manifest, args.region_fraction)
    if policy.get("isolate_spatial_splits", False):
        manifest = isolate_splits(dataset, manifest)
    experiment_split = output / "spatial_split.json"
    if resume is not None:
        _assert_resume_split_matches(experiment_split, manifest)
    supervision_audit = None
    if policy:
        supervision_audit = supervision_summary(dataset, manifest)
        (output / "supervision_audit.json").write_text(
            json.dumps(supervision_audit, indent=2), encoding="utf-8"
        )
        manifest.write(output / "spatial_split.json")
        if policy.get("require_validation_classes_in_train", False):
            counts = supervision_audit["unique_class_counts"]
            missing = set(counts["validation"]) - set(counts["train"])
            if missing:
                raise ValueError(
                    f"隔离后训练集缺少验证类别 {sorted(missing)}；"
                    "请检查监督审计。可用 --halo 0 0 保留更多实测点。"
                )
    train_indices = manifest.splits.get("train", [])
    validation_indices = manifest.splits.get("validation", [])
    assert_supervision_isolated(dataset, manifest)
    dataset.configure_supervision_split(manifest, "train", mask_weak_labels=True)
    validation_dataset.configure_supervision_split(
        manifest, "validation", mask_weak_labels=False
    )
    if policy.get("measured_windows_only", False):
        measured = point_windows(dataset, manifest, "train")
        train_indices = [index for index in train_indices if index in measured]
    if policy.get("skip_unlabeled_validation", False):
        measured = point_windows(validation_dataset, manifest, "validation")
        validation_indices = [
            index for index in validation_indices if index in measured
        ]
    if not train_indices:
        detail = (
            "；严格输入窗口隔离过滤掉了所有含训练点的窗口，"
            "请关闭 isolate_spatial_splits，监督标签仍会按空间块屏蔽"
            if policy.get("isolate_spatial_splits", False)
            else ""
        )
        raise ValueError(f"空间划分中的 train 集为空: {split_path}{detail}")
    if not validation_indices:
        raise ValueError(f"空间划分中的 validation 集为空: {split_path}")
    model_config = load_model_contract(args.config, run, stage2)
    model = SegFormerUtae.from_contract(model_config)
    pretrained_initialization = None
    weight_initialization = None
    if args.init_checkpoint is not None:
        weight_initialization = _initialize_weights(
            model, model_config, args.init_checkpoint
        )
    elif resume is None and hasattr(model, "initialize_pretrained"):
        pretrained_initialization = model.initialize_pretrained()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    loader_config = dict(train_config.get("dataloader", {}))
    configured_batch_size = int(loader_config.get("batch_size", 1))
    if configured_batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    effective_batch_size = 1 if any(halo) else configured_batch_size
    validation_batch_size = int(
        loader_config.get("validation_batch_size", configured_batch_size)
    )
    if validation_batch_size < 1:
        raise ValueError("validation_batch_size 必须为正整数")
    num_workers = int(loader_config.get("num_workers", 0))
    persistent_workers = bool(loader_config.get("persistent_workers", False))
    if num_workers == 0:
        persistent_workers = False
    sampling_config = dict(stage2.get("sampling", {}))
    if policy.get("class_balanced_point_sampling", False):
        train_sampler = ClassBalancedPointSampler(
            dataset,
            manifest,
            train_indices,
            seed=int(training.get("seed", 42)),
        )
        train_loader_indices = None
    elif policy.get("capped_class_sampling", False):
        train_sampler = CappedClassSampler(
            dataset,
            train_indices,
            supervision_audit["unique_class_counts"]["train"],
            manifest=manifest,
            extra_fraction=float(policy.get("extra_fraction", 0.25)),
            seed=int(training.get("seed", 42)),
        )
        train_loader_indices = None
    elif policy.get("measured_windows_only", False):
        # Sample measured windows uniformly, without applying class correction
        # twice through both the sampler and the loss.
        train_sampler = None
        train_loader_indices = train_indices
    elif sampling_config.get("strategy", "spatial_weighted") == "spatial_weighted":
        train_sampler = SpatialWeightedSampler(
            dataset,
            manifest=manifest,
            split="train",
            seed=int(training.get("seed", 42)),
        )
        train_loader_indices = None
    else:
        train_sampler = None
        train_loader_indices = train_indices
    loader = build_dataloader(
        dataset,
        indices=train_loader_indices,
        sampler=train_sampler,
        batch_size=effective_batch_size,
        num_workers=num_workers,
        pin_memory=bool(loader_config.get("pin_memory", True)),
        persistent_workers=persistent_workers,
        prefetch_factor=int(loader_config.get("prefetch_factor", 2)),
        drop_last=bool(loader_config.get("drop_last", False)),
        seed=int(training.get("seed", 42)),
    )
    validation_loader = build_dataloader(
        validation_dataset,
        indices=validation_indices,
        batch_size=validation_batch_size,
        # Validation must not keep a second large worker/pinned-memory pool
        # alive throughout the following training epoch. Apply these defaults
        # to old experiment snapshots as well as newly created runs.
        num_workers=int(
            loader_config.get("validation_num_workers", min(2, num_workers))
        ),
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1,
        drop_last=False,
        seed=int(training.get("seed", 42)),
    )
    optimizer_config = dict(train_config.get("optimizer", {}))
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps 必须是正整数")
    if any(halo):
        accumulation_steps *= configured_batch_size
        print(f"halo={halo}，单窗口训练，梯度累积={accumulation_steps}")
    parameters = model.parameters()
    if model_config.get("architecture") in {
        "segformer_utae_pretrained",
        "segformer_utae_static_ablation",
    }:
        encoder_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("static_encoder.") and parameter.requires_grad
        ]
        new_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("static_encoder.") and parameter.requires_grad
        ]
        parameters = [
            {"params": new_parameters},
            {
                "params": encoder_parameters,
                "lr": float(optimizer_config.get("pretrained_learning_rate", 1e-5)),
            },
        ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config.get("learning_rate", 1e-4)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0001)),
        betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
    )
    scheduler_config = dict(train_config.get("scheduler", {}))
    steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_ratio = float(scheduler_config.get("warmup_ratio", 0.05))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("scheduler.warmup_ratio 必须位于 [0, 1)")
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))
    if resume is not None:
        total_steps = resume["total_steps"]
        warmup_steps = resume["warmup_steps"]
        if total_steps != max(1, epochs * steps_per_epoch):
            remaining_epochs = max(epochs - int(resume["next_epoch"]), 0)
            total_steps = max(
                int(resume["optimizer_steps"]) + remaining_epochs * steps_per_epoch,
                int(resume["optimizer_steps"]) + 1,
            )
            print(
                "每轮训练样本数已改变；学习率计划已按剩余全量训练步数续接，"
                f"总优化步数调整为 {total_steps}"
            )

    def learning_rate(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    derived = model_config["derived"]
    class_weights = torch.ones(int(derived["num_classes"]), device=device)
    for code, weight in manifest.class_weights.items():
        class_index = int(code) - 1
        if 0 <= class_index < len(class_weights):
            class_weights[class_index] = float(weight)
    if policy.get("class_weighting") == "train_inverse_sqrt":
        class_weights = torch.tensor(
            training_class_weights(
                supervision_audit["unique_class_counts"]["train"],
                int(derived["num_classes"]),
            ),
            device=device,
        )
    if policy.get("class_weighting") == "none":
        class_weights = None
    supervision = dict(model_config.get("supervision", {}))
    if policy.get("disable_weak_labels", False):
        supervision["weak_label_weight"] = 0.0
    early_stopping = dict(training.get("early_stopping", {}))
    monitor = str(early_stopping.get("monitor", "val_loss"))
    if monitor not in {"val_loss", "val_accuracy", "val_macro_f1"}:
        raise ValueError("Unsupported early-stopping monitor: " + monitor)
    minimize_monitor = monitor == "val_loss"
    best_monitor_value = float("inf") if minimize_monitor else -float("inf")
    early_stopping_enabled = bool(early_stopping.get("enabled", True))
    patience = int(early_stopping.get("patience", 8))
    min_delta = float(early_stopping.get("min_delta", 1e-4))
    if patience < 1:
        raise ValueError("early_stopping.patience 必须是正整数")
    clipping = dict(train_config.get("gradient_clipping", {}))
    clipping_enabled = bool(clipping.get("enabled", True))
    max_norm = float(clipping.get("max_norm", 1.0))
    if clipping_enabled and max_norm <= 0:
        raise ValueError("gradient_clipping.max_norm 必须是正数")
    ema_config = dict(train_config.get("ema", {}))
    ema = (
        ModelEMA(model, float(ema_config.get("decay", 0.99)))
        if bool(ema_config.get("enabled", True))
        else None
    )
    amp_dtype = resolve_amp_dtype(device, str(training.get("amp_dtype", "auto")))
    amp_enabled = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    amp_name = str(amp_dtype).removeprefix("torch.") if amp_enabled else "none"
    print(f"计算精度: {amp_name}，梯度缩放: {scaler.is_enabled()}")
    overlap = dict(train_config.get("overlap_consistency", {}))
    overlap_weight = float(overlap.get("weight", 0.0))
    overlap_interval = int(overlap.get("every_n_batches", 4))
    if overlap_weight < 0 or overlap_interval < 1:
        raise ValueError("overlap weight 必须非负，every_n_batches 必须为正")
    metrics: list[dict[str, float | int]] = []
    train_log: dict[str, object] = {
        "status": "running",
        "source_run": str(run),
        "data_config": "data.yaml",
        "model_config": "model.yaml",
        "train_config": "train.yaml",
        "checkpoint": checkpoint_name,
        "window_size": list(window_size),
        "stride": list(window_stride),
        "grid_offset": list(args.grid_offset),
        "halo": list(halo),
        "started_at": experiment_started_at,
        "supervision_policy": policy,
        "pretrained_initialization": pretrained_initialization,
        "weight_initialization": weight_initialization,
        "supervision_audit": supervision_audit,
        "epochs": metrics,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "metric_protocol": "owned_unique_points_mean_probability_v1",
        "checkpoint_monitor": monitor,
    }
    if resume is None:
        shutil.copy2(args.train_config, output / "train.yaml")
        shutil.copy2(args.config, output / "model.yaml")
        # Absolute paths keep the snapshot valid in the experiment directory.
        data_payload = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
        for name in (
            "root",
            "labels",
            "raw",
            "dynamic",
            "static",
            "processed",
            "label_file",
            "sam2_checkpoint",
        ):
            value = getattr(data_config.data, name)
            data_payload.setdefault("data", {})[name] = str(value) if value else None
        data_payload["data"]["stage2"] = stage2
        (output / "data.yaml").write_text(
            yaml.safe_dump(data_payload, allow_unicode=True), encoding="utf-8"
        )
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    optimizer_steps = 0
    start_epoch = 0
    if resume is not None:
        if resume["contract"] != model_config:
            raise ValueError("模型配置或标签映射与断点不一致")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if scaler.is_enabled() and resume.get("grad_scaler"):
            scaler.load_state_dict(resume["grad_scaler"])
        scheduler.load_state_dict(resume["scheduler"])
        if ema is not None:
            for name, value in resume["ema"].items():
                ema.shadow[name].copy_(value)
        start_epoch = resume["next_epoch"]
        best_state = resume["best_state"]
        best_validation_loss = resume["best_validation_loss"]
        best_epoch = resume["best_epoch"]
        best_monitor_value = resume.get("best_monitor_value", best_validation_loss)
        stale_epochs = resume["stale_epochs"]
        optimizer_steps = resume["optimizer_steps"]
        train_log = resume["train_log"]
        metrics = train_log["epochs"]
        torch.set_rng_state(resume["rng"])
        if device.type == "cuda" and resume["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        loader.generator.set_state(resume["loader_rng"])
        validation_loader.generator.set_state(resume["validation_loader_rng"])
        train_transforms._generator.set_state(resume["augmentation_rng"])
        if train_sampler is not None:
            train_sampler.epoch = start_epoch
        print(f"已恢复断点，完成 {start_epoch} 轮，目标共 {epochs} 轮")
        del resume
    train_log["status"] = "running"
    train_log["precision"] = {
        "amp_dtype": amp_name,
        "gradient_scaling": scaler.is_enabled(),
    }
    (output / "train_log.json").write_text(
        json.dumps(train_log, indent=2), encoding="utf-8"
    )

    def optimizer_step() -> None:
        nonlocal optimizer_steps
        updated = scaled_optimizer_step(
            optimizer, scaler, max_norm if clipping_enabled else None
        )
        if updated:
            scheduler.step()
            optimizer_steps += 1
            if ema is not None:
                ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        if early_stopping_enabled and stale_epochs >= patience:
            break
        model.train()
        total = 0.0
        batches = 0
        correct_pixels = 0
        labeled_pixels = 0
        ground_truth_loss_sum = 0.0
        ground_truth_loss_batches = 0
        weak_label_loss_sum = 0.0
        weak_label_loss_batches = 0
        weak_label_correct_pixels = 0
        weak_label_pixels = 0
        overlap_loss_sum = 0.0
        overlap_batches = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            loader,
            desc=f"训练 Epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        batch_finished_at = perf_counter()
        for batch in progress:
            step_started_at = perf_counter()
            data_seconds = step_started_at - batch_finished_at
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_enabled,
            ):
                prediction = model(tensor_batch)
                if "core_mask" in tensor_batch:
                    tensor_batch["valid_mask"] = (
                        tensor_batch["valid_mask"] & tensor_batch["core_mask"]
                    )
                split_mask = tensor_batch.get("supervision_split_mask")
                if split_mask is not None and (
                    (tensor_batch["ground_truth_mask"] & ~split_mask).any()
                    or (tensor_batch["weak_label_mask"] & ~split_mask).any()
                ):
                    raise AssertionError("实际训练监督越过了训练空间归属边界")
                loss_components = combined_supervision_loss(
                    prediction,
                    tensor_batch,
                    ground_truth_weight=float(
                        supervision.get("ground_truth_weight", 1.0)
                    ),
                    weak_label_weight=float(supervision.get("weak_label_weight", 0.5)),
                    focal_gamma=float(supervision.get("focal_gamma", 0.0)),
                    fine_to_coarse=derived["fine_to_coarse"],
                    class_weights=class_weights,
                    weight_normalization=str(
                        supervision.get("weight_normalization", "weighted_mean")
                    ),
                    ignore_index=int(supervision.get("ignore_index", -1)),
                )
                loss = loss_components["loss"]
            ground_truth_mask = (
                tensor_batch["ground_truth_mask"] & tensor_batch["valid_mask"]
            )
            source_loss = None
            predicted = prediction["fine_logits"].argmax(dim=1)
            if ground_truth_mask.any():
                target = tensor_batch["ground_truth"] - 1
                correct_pixels += int(
                    (predicted[ground_truth_mask] == target[ground_truth_mask]).sum()
                )
                labeled_pixels += int(ground_truth_mask.sum())
                source_loss = _source_supervision_loss(loss_components, "ground_truth")
                if source_loss is not None:
                    ground_truth_loss_sum += float(source_loss.detach())
                    ground_truth_loss_batches += 1
            weak_label_mask = (
                tensor_batch.get("weak_label_mask", torch.zeros_like(ground_truth_mask))
                & tensor_batch["valid_mask"]
            )
            if weak_label_mask.any():
                weak_target = tensor_batch["weak_label"] - 1
                weak_label_correct_pixels += int(
                    (predicted[weak_label_mask] == weak_target[weak_label_mask]).sum()
                )
                weak_label_pixels += int(weak_label_mask.sum())
                source_loss = _source_supervision_loss(loss_components, "weak_label")
                if source_loss is not None:
                    weak_label_loss_sum += float(source_loss.detach())
                    weak_label_loss_batches += 1
            group_size = min(
                accumulation_steps,
                len(loader) - (batches // accumulation_steps) * accumulation_steps,
            )
            scaler.scale(loss / group_size).backward()
            if (
                overlap_weight > 0
                and epoch >= int(overlap.get("warmup_epochs", 3))
                and batches % overlap_interval == 0
            ):
                view, top, left = overlapping_view(
                    tensor_batch, int(overlap.get("crop_margin", 32))
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype or torch.float32,
                    enabled=amp_enabled,
                ):
                    cropped_prediction = model(view)
                    consistency = overlap_consistency_loss(
                        prediction["fine_logits"],
                        cropped_prediction["fine_logits"],
                        view["valid_mask"] & view["core_mask"],
                        top,
                        left,
                    )
                scaler.scale(overlap_weight * consistency / group_size).backward()
                overlap_loss_sum += float(consistency.detach())
                overlap_batches += 1
                del cropped_prediction, consistency, view
            if (batches + 1) % accumulation_steps == 0:
                optimizer_step()
            loss_value = float(loss.detach())
            total += loss_value
            batches += 1
            # Release the previous forward's outputs before fetching/transferring
            # another window. Backward has already consumed the saved tensors.
            del (
                ground_truth_mask,
                source_loss,
                loss,
                loss_components,
                prediction,
                predicted,
                tensor_batch,
            )
            status = {
                "loss": f"{loss_value:.5f}",
                "data": f"{data_seconds:.2f}s",
                "step": f"{perf_counter() - step_started_at:.2f}s",
            }
            if device.type == "cuda":
                status["VRAM"] = (
                    f"{torch.cuda.memory_allocated(device) / 2**30:.2f}/"
                    f"{torch.cuda.memory_reserved(device) / 2**30:.2f}G"
                )
            progress.set_postfix(status)
            batch_finished_at = perf_counter()
        if batches % accumulation_steps:
            optimizer_step()
        epoch_loss = total / max(batches, 1)
        epoch_accuracy = correct_pixels / max(labeled_pixels, 1)
        ground_truth_loss = ground_truth_loss_sum / max(ground_truth_loss_batches, 1)
        weak_label_loss = weak_label_loss_sum / max(weak_label_loss_batches, 1)
        weak_label_accuracy = weak_label_correct_pixels / max(weak_label_pixels, 1)
        current_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        if ema is not None:
            ema.copy_to(model)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        validation = _evaluate(model, validation_loader, device, amp_dtype=amp_name)
        # Inference and training have different allocation patterns. Do not
        # carry unused validation allocations into the next training epoch.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        (output / f"validation_epoch_{epoch + 1:03d}.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        model.load_state_dict(current_state)
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "ground_truth_loss": ground_truth_loss,
            "weak_label_loss": weak_label_loss,
            "weak_label_accuracy": weak_label_accuracy,
            "weak_label_pixels": weak_label_pixels,
            "overlap_loss": overlap_loss_sum / max(overlap_batches, 1),
            "overlap_batches": overlap_batches,
            "accuracy": epoch_accuracy,
            "labeled_pixels": labeled_pixels,
            "batches": batches,
            "val_loss": float(validation["loss"]),
            "val_accuracy": float(validation["accuracy"]),
            "val_labeled_pixels": int(validation["labeled_pixels"]),
            "val_macro_f1": validation.get("macro_f1"),
            "val_macro_recall": validation.get("macro_recall"),
            "val_unique_point_accuracy": validation.get("unique_point_accuracy"),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "pretrained_learning_rate": (
                float(optimizer.param_groups[1]["lr"])
                if len(optimizer.param_groups) > 1
                else None
            ),
            "optimizer_steps": optimizer_steps,
        }
        metrics.append(epoch_metrics)
        _append_metrics_csv(
            output / "epoch_metrics.csv",
            {
                "epoch": epoch + 1,
                "train_ground_truth_loss": ground_truth_loss,
                "train_ground_truth_accuracy": epoch_accuracy,
                "train_weak_label_loss": weak_label_loss,
                "train_weak_label_accuracy": weak_label_accuracy,
                "validation_ground_truth_loss": float(validation["loss"]),
                "validation_ground_truth_accuracy": float(validation["accuracy"]),
                "validation_ground_truth_macro_f1": float(validation["macro_f1"]),
            },
        )
        selected_state = (
            {name: value.detach().cpu().clone() for name, value in ema.shadow.items()}
            if ema is not None
            else current_state
        )
        if float(validation["loss"]) < best_validation_loss:
            best_validation_loss = float(validation["loss"])
            _atomic_save(
                {"model": selected_state, "contract": model_config},
                output / "best_loss.pt",
            )
        value = float(epoch_metrics[monitor])
        improved = (
            value < best_monitor_value - min_delta
            if minimize_monitor
            else value > best_monitor_value + min_delta
        )
        if improved:
            best_monitor_value = value
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = selected_state
        else:
            stale_epochs += 1
        if float(validation["accuracy"]) > max(
            (float(item["val_accuracy"]) for item in metrics[:-1]), default=-1.0
        ):
            accuracy_state = (
                {name: value.detach().cpu() for name, value in ema.shadow.items()}
                if ema is not None
                else current_state
            )
            _atomic_save(
                {"model": accuracy_state, "contract": model_config},
                output / "best_accuracy.pt",
            )
        unique_accuracy = validation.get("unique_point_accuracy")
        if unique_accuracy is not None and unique_accuracy > max(
            (
                item["val_unique_point_accuracy"]
                for item in metrics[:-1]
                if item.get("val_unique_point_accuracy") is not None
            ),
            default=-1.0,
        ):
            unique_state = (
                {name: value.detach().cpu() for name, value in ema.shadow.items()}
                if ema is not None
                else current_state
            )
            _atomic_save(
                {"model": unique_state, "contract": model_config},
                output / "best_unique_accuracy.pt",
            )
        train_log.update(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_validation_loss,
                "best_monitor_value": best_monitor_value,
                "optimizer_steps": optimizer_steps,
            }
        )
        _atomic_save(
            {
                "resume_version": 1,
                "source_run": str(run),
                "window_options": {
                    "window_size": window_size,
                    "stride": window_stride,
                    "grid_offset": tuple(args.grid_offset),
                    "region_fraction": args.region_fraction,
                },
                "target_epochs": epochs,
                "next_epoch": epoch + 1,
                "model": current_state,
                "contract": model_config,
                "optimizer": optimizer.state_dict(),
                "grad_scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "ema": {
                    name: value.detach().cpu() for name, value in ema.shadow.items()
                }
                if ema is not None
                else None,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "best_monitor_value": best_monitor_value,
                "stale_epochs": stale_epochs,
                "optimizer_steps": optimizer_steps,
                "train_log": train_log,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None,
                "loader_rng": loader.generator.get_state(),
                "validation_loader_rng": validation_loader.generator.get_state(),
                "augmentation_rng": train_transforms._generator.get_state(),
            },
            output / "last.pt",
        )
        (output / "train_log.json").write_text(
            json.dumps(train_log, indent=2), encoding="utf-8"
        )
        print(
            f"epoch {epoch + 1}/{epochs}: loss={epoch_loss:.5f}, "
            f"accuracy={epoch_accuracy:.4f}, "
            f"val_loss={validation['loss']:.5f}, "
            f"val_accuracy={validation['accuracy']:.4f} "
            f"({correct_pixels}/{labeled_pixels})"
        )
        if early_stopping_enabled and stale_epochs >= patience:
            print(f"early stopping: 连续 {patience} 轮验证集未改善")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = output / checkpoint_name
    cpu_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    _atomic_save({"model": cpu_state, "contract": model_config}, checkpoint)
    train_log.update(
        {
            "status": "completed",
            "best_epoch": best_epoch,
            "best_val_loss": best_validation_loss,
            "best_monitor_value": best_monitor_value,
            "optimizer_steps": optimizer_steps,
            "finished_at": datetime.now().isoformat(),
        }
    )
    (output / "train_log.json").write_text(
        json.dumps(train_log, indent=2), encoding="utf-8"
    )
    print(f"训练产物: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
