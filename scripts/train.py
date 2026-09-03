"""Train SegFormer-U-TAE from a prepared processed run directory."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from tqdm import tqdm

from config import load_config
from data.sample_index import WindowedSampleDataset
from data.sampling import build_dataloader
from data.spatial_split import load_spatial_split
from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.config import load_model_contract


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


def _evaluate(
    model: SegFormerUtae, loader: object, device: torch.device
) -> dict[str, float | int]:
    """Evaluate fine-label loss and accuracy on the validation split."""

    model.eval()
    loss_sum = 0.0
    correct_pixels = 0
    labeled_pixels = 0
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(tensor_batch)
            mask = tensor_batch["ground_truth_mask"] & tensor_batch["valid_mask"]
            target = (tensor_batch["ground_truth"] - 1).masked_fill(~mask, -1)
            if mask.any():
                pixel_loss = F.cross_entropy(
                    output["fine_logits"],
                    target,
                    reduction="none",
                    ignore_index=-1,
                )
                prediction = output["fine_logits"].argmax(dim=1)
                loss_sum += float(pixel_loss[mask].sum())
                correct_pixels += int((prediction[mask] == target[mask]).sum())
                labeled_pixels += int(mask.sum())
    return {
        "loss": loss_sum / max(labeled_pixels, 1),
        "accuracy": correct_pixels / max(labeled_pixels, 1),
        "labeled_pixels": labeled_pixels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 SegFormer-U-TAE 模型")
    parser.add_argument("run", type=Path, help="datasets.py 生成的数据集目录")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"数据集目录不存在: {run}")
    with args.train_config.open(encoding="utf-8") as stream:
        train_config = yaml.safe_load(stream) or {}
    training = dict(train_config.get("training", {}))
    epochs = args.epochs if args.epochs is not None else int(training.get("epochs", 1))
    if epochs < 1:
        raise ValueError("epochs 必须是正整数")
    experiment_started_at = datetime.now().isoformat()
    output = _new_experiment_dir(Path("experiments"))
    checkpoint_name = f"model_{output.name}.pt"

    data_config = load_config(args.data_config)
    stage2 = data_config.data.stage2
    window = dict(stage2.get("window", {}))
    mapping_path = next(iter(sorted(run.glob("label_mapping*.json"))), None)
    if mapping_path is None:
        raise FileNotFoundError(f"数据集目录缺少标签映射: {run}")
    statistics = next(iter(sorted(run.glob("raster_stats*.json"))), None)
    dataset = WindowedSampleDataset(
        run / "sample_index.json",
        window_size=tuple(window.get("size", (256, 256))),
        stride=tuple(window.get("stride", window.get("size", (256, 256)))),
        label_columns=data_config.data.label_columns,
        label_mapping=json.loads(mapping_path.read_text(encoding="utf-8")),
        statistics=statistics,
        nodata=data_config.data.raster.get("nodata", -9999),
        stage2=stage2,
    )
    split_path = run / "spatial_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"数据集目录缺少空间划分文件: {split_path}")
    manifest = load_spatial_split(split_path)
    train_indices = manifest.splits.get("train", [])
    validation_indices = manifest.splits.get("validation", [])
    if not train_indices:
        raise ValueError(f"空间划分中的 train 集为空: {split_path}")
    if not validation_indices:
        raise ValueError(f"空间划分中的 validation 集为空: {split_path}")
    model_config = load_model_contract(args.config, run)
    model = SegFormerUtae.from_contract(model_config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    loader_config = dict(stage2.get("dataloader", {}))
    num_workers = int(loader_config.get("num_workers", 0))
    persistent_workers = bool(loader_config.get("persistent_workers", False))
    if num_workers == 0:
        persistent_workers = False
    loader = build_dataloader(
        dataset,
        indices=train_indices,
        batch_size=int(loader_config.get("batch_size", 1)),
        num_workers=num_workers,
        pin_memory=bool(loader_config.get("pin_memory", True)),
        persistent_workers=persistent_workers,
        prefetch_factor=int(loader_config.get("prefetch_factor", 2)),
        drop_last=bool(loader_config.get("drop_last", False)),
        seed=int(training.get("seed", 42)),
    )
    validation_loader = build_dataloader(
        dataset,
        indices=validation_indices,
        batch_size=int(loader_config.get("batch_size", 1)),
        num_workers=num_workers,
        pin_memory=bool(loader_config.get("pin_memory", True)),
        persistent_workers=persistent_workers,
        prefetch_factor=int(loader_config.get("prefetch_factor", 2)),
        drop_last=False,
        seed=int(training.get("seed", 42)),
    )
    optimizer_config = dict(train_config.get("optimizer", {}))
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps 必须是正整数")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("learning_rate", 1e-4)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0001)),
        betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
    )
    scheduler_config = dict(train_config.get("scheduler", {}))
    total_steps = max(1, epochs * math.ceil(len(train_indices) / accumulation_steps))
    warmup_ratio = float(scheduler_config.get("warmup_ratio", 0.05))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("scheduler.warmup_ratio 必须位于 [0, 1)")
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))

    def learning_rate(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    derived = model_config["derived"]
    supervision = dict(model_config.get("supervision", {}))
    early_stopping = dict(training.get("early_stopping", {}))
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
    amp_enabled = device.type == "cuda"
    metrics: list[dict[str, float | int]] = []
    train_log: dict[str, object] = {
        "status": "running",
        "source_run": str(run),
        "data_config": "data.yaml",
        "model_config": "model.yaml",
        "train_config": "train.yaml",
        "checkpoint": checkpoint_name,
        "started_at": experiment_started_at,
        "epochs": metrics,
    }
    shutil.copy2(args.train_config, output / "train.yaml")
    shutil.copy2(args.config, output / "model.yaml")
    shutil.copy2(args.data_config, output / "data.yaml")
    (output / "train_log.json").write_text(
        json.dumps(train_log, indent=2), encoding="utf-8"
    )
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    optimizer_steps = 0

    def optimizer_step() -> None:
        nonlocal optimizer_steps
        if clipping_enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
        optimizer_steps += 1
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        batches = 0
        correct_pixels = 0
        labeled_pixels = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            loader,
            desc=f"训练 Epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
            ):
                prediction = model(tensor_batch)
                loss = combined_supervision_loss(
                    prediction,
                    tensor_batch,
                    ground_truth_weight=float(
                        supervision.get("ground_truth_weight", 1.0)
                    ),
                    weak_label_weight=float(supervision.get("weak_label_weight", 0.5)),
                    fine_to_coarse=derived["fine_to_coarse"],
                    ignore_index=int(supervision.get("ignore_index", -1)),
                )["loss"]
            ground_truth_mask = (
                tensor_batch["ground_truth_mask"] & tensor_batch["valid_mask"]
            )
            if ground_truth_mask.any():
                target = tensor_batch["ground_truth"] - 1
                predicted = prediction["fine_logits"].argmax(dim=1)
                correct_pixels += int(
                    (predicted[ground_truth_mask] == target[ground_truth_mask]).sum()
                )
                labeled_pixels += int(ground_truth_mask.sum())
            (loss / accumulation_steps).backward()
            if (batches + 1) % accumulation_steps == 0:
                optimizer_step()
            total += float(loss.detach().cpu())
            batches += 1
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.5f}")
        if batches % accumulation_steps:
            optimizer_step()
        epoch_loss = total / max(batches, 1)
        epoch_accuracy = correct_pixels / max(labeled_pixels, 1)
        current_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        if ema is not None:
            ema.copy_to(model)
        validation = _evaluate(model, validation_loader, device)
        model.load_state_dict(current_state)
        metrics.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_accuracy,
                "labeled_pixels": labeled_pixels,
                "batches": batches,
                "val_loss": float(validation["loss"]),
                "val_accuracy": float(validation["accuracy"]),
                "val_labeled_pixels": int(validation["labeled_pixels"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": optimizer_steps,
            }
        )
        if float(validation["loss"]) < best_validation_loss - min_delta:
            best_validation_loss = float(validation["loss"])
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = ema.state_dict() if ema is not None else current_state
        else:
            stale_epochs += 1
        train_log.update(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_validation_loss,
                "optimizer_steps": optimizer_steps,
            }
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
        print("训练循环完成，正在恢复验证集表现最好的 EMA 权重...", flush=True)
        model.load_state_dict(best_state)
    print("正在整理 CPU 权重并保存 checkpoint...", flush=True)
    checkpoint = output / checkpoint_name
    cpu_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    torch.save({"model": cpu_state, "contract": model_config}, checkpoint)
    print(f"checkpoint 已保存: {checkpoint}", flush=True)
    train_log.update(
        {
            "status": "completed",
            "best_epoch": best_epoch,
            "best_val_loss": best_validation_loss,
            "optimizer_steps": optimizer_steps,
            "finished_at": datetime.now().isoformat(),
        }
    )
    print("正在写入最终 train_log.json...", flush=True)
    (output / "train_log.json").write_text(
        json.dumps(train_log, indent=2), encoding="utf-8"
    )
    print(f"训练产物: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
