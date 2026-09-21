"""Real-window forward/backward checks before launching branch comparisons."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch

from data.balanced_sampling import ClassBalancedPointSampler
from data.overlap import overlapping_view
from data.sample_index import sample_collate_fn
from data.training_policy import point_windows, training_class_weights
from losses.overlap import overlap_consistency_loss
from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae
from models.config import load_model_contract
from precision import resolve_amp_dtype
from prepare_branch_labels import dataset_for
from run_accuracy_experiments import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    _, _, dataset, manifest = dataset_for(output)
    dataset.use_weak_labels = False
    dataset.configure_supervision_split(manifest, "train", mask_weak_labels=True)
    indices = sorted(
        set(manifest.splits["train"]) & point_windows(dataset, manifest, "train").keys()
    )
    choice = next(iter(ClassBalancedPointSampler(dataset, manifest, indices, seed=42)))
    batch = sample_collate_fn([dataset[choice]])
    dataset.close()
    batch = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    batch["valid_mask"] &= batch["core_mask"]
    # A synthetic weak mask checks the separate gradient path without SAM labels.
    batch["weak_label"] = batch["ground_truth"].clone()
    batch["weak_label_mask"] = batch["ground_truth_mask"].clone()
    dtype = resolve_amp_dtype(torch.device("cuda"), "auto")
    results = []
    for job in read_json(output / "campaign.json")["jobs"]:
        torch.manual_seed(42)
        contract = load_model_contract(
            output / "settings" / f"{job['name']}_model.yaml", output / "dataset"
        )
        model = SegFormerUtae.from_contract(contract)
        if hasattr(model, "initialize_pretrained"):
            model.initialize_pretrained()
        model.cuda().train()
        weights = torch.tensor(
            training_class_weights(
                manifest.class_counts["train"], contract["derived"]["num_classes"]
            ),
            device="cuda",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ema = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        scaler = torch.amp.GradScaler(
            "cuda", enabled=dtype == torch.float16, init_scale=128
        )
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(4):
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    prediction = model(batch)
                    losses = combined_supervision_loss(
                        prediction,
                        batch,
                        fine_to_coarse=contract["derived"]["fine_to_coarse"],
                        weak_label_weight=0.5,
                        focal_gamma=2.0,
                        class_weights=weights,
                        weight_normalization="sample_mean",
                    )
                assert losses["weak_label_fine_loss"].item() > 0
                assert torch.isfinite(losses["loss"])
                scaler.scale(losses["loss"] / 4).backward()
                if micro == 0:
                    view, top, left = overlapping_view(batch, 32)
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                        second = model(view)
                        overlap = overlap_consistency_loss(
                            prediction["fine_logits"],
                            second["fine_logits"],
                            view["valid_mask"] & view["core_mask"],
                            top,
                            left,
                        )
                    scaler.scale(0.05 * overlap / 4).backward()
                    del second, view, overlap
            scaler.unscale_(optimizer)
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        results.append(
            {
                "name": job["name"],
                "loss": float(losses["loss"]),
                "seconds": perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "precision": str(dtype),
                "dynamic_shape": list(batch["dynamic"].shape),
                "finite_backward": True,
                "optimizer_steps": 2,
                "accumulation_steps": 4,
                "overlap_and_ema_memory_checked": True,
            }
        )
        print(results[-1], flush=True)
        del prediction, losses, optimizer, scaler, grads, model, ema
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output / "preflight.json", {"results": results, "status": "passed"})


if __name__ == "__main__":
    main()
