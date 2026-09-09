"""Evaluate a trained segmentation checkpoint on a spatial test split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from tqdm import tqdm

from config import load_config
from data.sample_index import WindowedSampleDataset
from data.sampling import build_dataloader
from data.spatial_split import load_spatial_split
from models.architecture import SegFormerUtae
from precision import resolve_amp_dtype


def _roc_curve(
    scores: torch.Tensor, positive: torch.Tensor
) -> tuple[dict[str, Any], float | None]:
    """Return a binary ROC curve and trapezoidal AUC."""

    positives = positive.sum()
    negatives = positive.numel() - positives
    if positives == 0 or negatives == 0:
        return {"fpr": [], "tpr": [], "thresholds": []}, None
    order = torch.argsort(scores, descending=True)
    sorted_positive = positive[order].float()
    sorted_scores = scores[order]
    true_positive = torch.cat((torch.zeros(1), sorted_positive.cumsum(0)))
    false_positive = torch.cat((torch.zeros(1), (1.0 - sorted_positive).cumsum(0)))
    tpr = true_positive / positives
    fpr = false_positive / negatives
    auc = float(torch.trapezoid(tpr, fpr))
    thresholds = torch.cat((sorted_scores[:1] + 1e-6, sorted_scores))
    return {
        "fpr": [float(value) for value in fpr],
        "tpr": [float(value) for value in tpr],
        "thresholds": [float(value) for value in thresholds],
    }, auc


def _classification_metrics(
    target: torch.Tensor, probabilities: torch.Tensor, classes: int
) -> dict[str, Any]:
    """Calculate common multiclass metrics from labeled pixels only."""

    prediction = probabilities.argmax(dim=1)
    encoded = target * classes + prediction
    confusion = torch.bincount(encoded, minlength=classes * classes).reshape(
        classes, classes
    )
    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    present = (support + predicted) > 0

    def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(
            denominator > 0, numerator / denominator, torch.zeros_like(numerator)
        )

    per_class_precision = ratio(true_positive, true_positive + false_positive)
    per_class_recall = ratio(true_positive, true_positive + false_negative)
    per_class_f1 = ratio(
        2 * true_positive, 2 * true_positive + false_positive + false_negative
    )
    per_class_iou = ratio(
        true_positive, true_positive + false_positive + false_negative
    )
    total = support.sum().clamp_min(1)
    micro_precision = true_positive.sum() / (
        true_positive.sum() + false_positive.sum()
    ).clamp_min(1)
    micro_recall = true_positive.sum() / (
        true_positive.sum() + false_negative.sum()
    ).clamp_min(1)
    micro_f1 = (
        2
        * micro_precision
        * micro_recall
        / (micro_precision + micro_recall).clamp_min(1e-8)
    )
    micro_iou = true_positive.sum() / (
        true_positive.sum() + false_positive.sum() + false_negative.sum()
    ).clamp_min(1)
    roc_curves = {}
    per_class_aucs: dict[str, float | None] = {}
    one_hot = torch.nn.functional.one_hot(target, classes).bool()
    for class_index in range(classes):
        curve, auc = _roc_curve(probabilities[:, class_index], one_hot[:, class_index])
        roc_curves[str(class_index + 1)] = curve
        per_class_aucs[str(class_index + 1)] = auc
    micro_curve, micro_auc = _roc_curve(probabilities.flatten(), one_hot.flatten())
    valid_aucs = [auc for auc in per_class_aucs.values() if auc is not None]
    return {
        "support": int(support.sum()),
        "accuracy": {
            "micro": float(true_positive.sum() / total),
            "macro": float(per_class_recall[present].mean()) if present.any() else 0.0,
        },
        "precision": {
            "micro": float(micro_precision),
            "macro": float(per_class_precision[present].mean())
            if present.any()
            else 0.0,
            "per_class": [float(value) for value in per_class_precision],
        },
        "recall": {
            "micro": float(micro_recall),
            "macro": float(per_class_recall[present].mean()) if present.any() else 0.0,
            "per_class": [float(value) for value in per_class_recall],
        },
        "f1_score": {
            "micro": float(micro_f1),
            "macro": float(per_class_f1[present].mean()) if present.any() else 0.0,
            "per_class": [float(value) for value in per_class_f1],
        },
        "iou": {
            "micro": float(micro_iou),
            "macro": float(per_class_iou[present].mean()) if present.any() else 0.0,
            "per_class": [float(value) for value in per_class_iou],
        },
        "mse": float((prediction.float() - target.float()).square().mean()),
        "mae": float((prediction.float() - target.float()).abs().mean()),
        "auc": {
            "micro": micro_auc,
            "macro": sum(valid_aucs) / len(valid_aucs) if valid_aucs else None,
            "per_class": per_class_aucs,
        },
        "roc": {"micro": micro_curve, "one_vs_rest": roc_curves},
        "confusion_matrix": confusion.tolist(),
    }


def _metrics(
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    fine_to_coarse: list[int],
    loss_sum: float,
) -> dict[str, Any]:
    """Build fine/coarse reports from all valid labeled pixels."""

    fine_classes = probabilities.shape[1]
    mapping = torch.tensor(fine_to_coarse, dtype=torch.long)
    coarse_probabilities = torch.zeros(
        targets.shape[0], int(mapping.max()) + 1, dtype=probabilities.dtype
    )
    coarse_probabilities.index_add_(1, mapping, probabilities)
    coarse_targets = mapping[targets]
    count = int(targets.numel())
    report = {
        "labeled_pixels": count,
        "loss": loss_sum / max(count, 1),
        "fine": _classification_metrics(targets, probabilities, fine_classes),
        "coarse": _classification_metrics(
            coarse_targets, coarse_probabilities, coarse_probabilities.shape[1]
        ),
        "metric_notes": {
            "mse_mae": "预测类别 ID 与真实类别 ID 的误差，类别 ID 为 0-based",
            "macro": "先按类别计算，再对测试集中出现或被预测的类别取平均",
            "micro": "先汇总所有有效实测像元，再计算整体指标",
            "auc": "多分类 One-vs-Rest；没有正负样本的类别 AUC 为 null",
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在空间测试集上评估分割模型")
    parser.add_argument(
        "checkpoint", type=Path, help="训练实验目录中的带时间戳 model_*.pt 路径"
    )
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint.resolve()
    experiment_dir = checkpoint_path.parent
    metadata_path = experiment_dir / "train_log.json"
    if not metadata_path.is_file():
        metadata_path = experiment_dir / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"checkpoint 所在目录缺少 train_log.json: {experiment_dir}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_run = metadata.get("source_run")
    if not source_run:
        raise ValueError(f"训练日志缺少 source_run: {metadata_path}")
    run = Path(source_run).resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"训练数据目录不存在: {run}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型权重不存在: {checkpoint_path}")
    split_path = run / "spatial_split.json"
    mapping_path = next(iter(sorted(run.glob("label_mapping*.json"))), None)
    if not split_path.is_file() or mapping_path is None:
        raise FileNotFoundError("数据集目录缺少 spatial_split.json 或标签映射")

    data_config_path = experiment_dir / str(metadata.get("data_config", "data.yaml"))
    if not data_config_path.is_file():
        data_config_path = Path("configs/data.yaml")
    data_config = load_config(data_config_path)
    train_config_path = experiment_dir / str(metadata.get("train_config", "train.yaml"))
    if not train_config_path.is_file():
        train_config_path = Path("configs/train.yaml")
    with train_config_path.open(encoding="utf-8") as stream:
        train_config = yaml.safe_load(stream) or {}
    stage2 = dict(data_config.data.stage2)
    window = dict(stage2.get("window", {}))
    statistics = next(iter(sorted(run.glob("raster_stats*.json"))), None)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    dataset = WindowedSampleDataset(
        run / "sample_index.json",
        window_size=tuple(window.get("size", (256, 256))),
        stride=tuple(window.get("stride", window.get("size", (256, 256)))),
        label_columns=data_config.data.label_columns,
        label_mapping=mapping,
        statistics=statistics,
        nodata=data_config.data.raster.get("nodata", -9999),
        stage2=stage2,
        use_weak_labels=False,
    )
    manifest = load_spatial_split(split_path)
    indices = manifest.splits[args.split]
    if args.max_windows is not None:
        if args.max_windows < 1:
            raise ValueError("max-windows 必须是正整数")
        indices = indices[: args.max_windows]
    if not indices:
        raise ValueError(f"{args.split} 划分没有窗口")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = payload.get("contract") if isinstance(payload, dict) else None
    if contract is None:
        raise ValueError("checkpoint 缺少训练时保存的模型 contract")
    model = SegFormerUtae.from_contract(contract)
    model.load_state_dict(payload["model"])
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device).eval()
    amp_dtype = resolve_amp_dtype(
        device, str(train_config.get("training", {}).get("amp_dtype", "auto"))
    )
    loader_config = dict(train_config.get("dataloader", {}))
    num_workers = int(loader_config.get("num_workers", 0))
    loader = build_dataloader(
        dataset,
        indices=indices,
        batch_size=int(loader_config.get("batch_size", 1)),
        num_workers=num_workers,
        pin_memory=bool(loader_config.get("pin_memory", True)),
        persistent_workers=bool(loader_config.get("persistent_workers", False))
        and num_workers > 0,
        prefetch_factor=int(loader_config.get("prefetch_factor", 2)),
    )
    fine_to_coarse = list(contract["derived"]["fine_to_coarse"])
    loss_sum = 0.0
    targets: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"测试 ({args.split})", unit="batch"):
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
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
                loss_sum += float(pixel_loss[mask].sum())
                targets.append(target[mask].cpu())
                probabilities.append(
                    output["fine_logits"].softmax(dim=1).permute(0, 2, 3, 1)[mask].cpu()
                )

    if not targets:
        raise ValueError("测试划分中没有有效的实测标签像元")

    report = _metrics(
        torch.cat(targets),
        torch.cat(probabilities),
        fine_to_coarse=fine_to_coarse,
        loss_sum=loss_sum,
    )
    report.update(
        {
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "source_run": str(run),
            "created_at": datetime.now().isoformat(),
        }
    )
    report_path = experiment_dir / "test_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"测试结果: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
