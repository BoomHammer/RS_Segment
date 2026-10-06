"""Check a real window, AMP backward, overlap loss and optimizer/EMA memory."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from config import load_config
from data.overlap import overlapping_view
from data.sample_index import WindowedSampleDataset, sample_collate_fn
from losses.overlap import overlap_consistency_loss
from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.config import load_model_contract
from precision import resolve_amp_dtype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = load_config(args.data_config)
    stage = config.data.stage2
    mapping = json.loads(
        next(args.run.glob("label_mapping*.json")).read_text(encoding="utf-8")
    )
    dataset = WindowedSampleDataset(
        args.run / "sample_index.json",
        window_size=tuple(stage["window"]["size"]),
        stride=tuple(stage["window"]["stride"]),
        halo=tuple(stage["window"]["halo"]),
        label_columns=config.data.label_columns,
        label_mapping=mapping,
        statistics=next(args.run.glob("raster_stats*.json")),
        stage2=stage,
        use_weak_labels=False,
    )
    row, column = next(iter(dataset.ground_truth_pixels))
    candidates = dataset.index[
        (dataset.index.row <= row)
        & (dataset.index.column <= column)
        & (dataset.index.row + dataset.window_size[1] > row)
        & (dataset.index.column + dataset.window_size[0] > column)
    ]
    selected = candidates.iloc[-1]
    choice = next(
        i
        for i, item in enumerate(dataset.index.itertuples())
        if item.row == selected.row and item.column == selected.column
    )
    batch = sample_collate_fn([dataset[choice]])
    dataset.close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    batch["valid_mask"] &= batch["core_mask"]
    contract = load_model_contract(args.model_config, args.run, stage)
    if contract.get("architecture") != "anysat":
        raise ValueError("This check expects an AnySat model configuration")
    model = SegFormerUtae.from_contract(contract)
    initialization = model.initialize_pretrained()
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema = {name: p.detach().clone() for name, p in model.state_dict().items()}
    dtype = resolve_amp_dtype(device, "auto")
    scaler = torch.amp.GradScaler(
        device.type, enabled=dtype == torch.float16, init_scale=128
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with torch.autocast(
        device.type, dtype=dtype or torch.float32, enabled=dtype is not None
    ):
        output = model(batch)
        losses = combined_supervision_loss(
            output,
            batch,
            fine_to_coarse=contract["derived"]["fine_to_coarse"],
            level_parents=contract["derived"].get("level_parents"),
        )
    assert torch.isfinite(losses["loss"]) and losses["loss"] > 0
    scaler.scale(losses["loss"]).backward()
    view, top, left = overlapping_view(batch, 32)
    with torch.autocast(
        device.type, dtype=dtype or torch.float32, enabled=dtype is not None
    ):
        second = model(view)
        overlap = overlap_consistency_loss(
            output["fine_logits"],
            second["fine_logits"],
            view["valid_mask"] & view["core_mask"],
            top,
            left,
        )
    scaler.scale(0.1 * overlap).backward()
    scaler.unscale_(optimizer)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            ema[name].lerp_(value, 0.01)
    del output, losses, second, overlap, view
    model.eval()
    with (
        torch.no_grad(),
        torch.autocast(
            device.type, dtype=dtype or torch.float32, enabled=dtype is not None
        ),
    ):
        output = model(batch)
    assert torch.isfinite(output["fine_logits"]).all()
    if device.type == "cuda":
        torch.cuda.synchronize()
    report = {
        "status": "passed",
        "architecture": contract["architecture"],
        "initialization": initialization,
        "sensors": [spec["name"] for spec in model.specs],
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "precision": str(dtype),
        "dynamic_shape": list(batch["dynamic"].shape),
        "static_shape": list(batch["static"].shape),
        "classes": contract["derived"]["num_classes"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
        if device.type == "cuda"
        else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
        if device.type == "cuda"
        else None,
        "checks": [
            "real_window",
            "hierarchical_loss",
            "finite_gradients",
            "overlap_backward",
            "adamw_step",
            "ema",
            "eval_forward",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
