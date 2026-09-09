"""Export spatially split pixel CSVs and run the controlled SAM/ground-truth DNN."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rasterio.windows import Window
from torch import nn

from config import load_config
from data.raster_alignment import aligned_raster
from data.sample_index import WindowedSampleDataset

SPLITS = ("train", "validation", "test")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def owners(rows, cols, manifest):
    height, width = manifest["block_size"]
    return np.array(
        [
            manifest["blocks"][f"{r // height}:{c // width}"]
            for r, c in zip(rows, cols, strict=True)
        ]
    )


def prepare(experiment, output, workers):
    """Read bounded VRT windows; cache only labelled pixels, never full imagery."""
    log = json.loads((experiment / "train_log.json").read_text())
    run = Path(log["source_run"])
    config = load_config(experiment / "data.yaml")
    mapping = json.loads(next(run.glob("label_mapping*.json")).read_text())
    dataset = WindowedSampleDataset(
        run / "sample_index.json",
        label_columns=config.data.label_columns,
        label_mapping=mapping,
        statistics=next(run.glob("raster_stats*.json")),
        stage2=config.data.stage2,
        use_weak_labels=False,
    )
    manifest = json.loads((experiment / "spatial_split.json").read_text())
    write_json(output / "spatial_split.json", manifest)
    write_json(output / "label_mapping.json", mapping)
    grid = dataset.grid
    pseudo = {}
    weak_path = dataset.sample_index.weak_label["path"]
    with aligned_raster(weak_path, grid) as raster:
        for r in range(0, grid.height, 512):
            for c in range(0, grid.width, 512):
                window = Window(
                    c, r, min(512, grid.width - c), min(512, grid.height - r)
                )
                values = raster.read(1, window=window, masked=True).filled(-1)
                rr, cc = np.where((values >= 1) & (values <= mapping["minor_count"]))
                pseudo.update(
                    {
                        (int(y + r), int(x + c)): int(values[y, x])
                        for y, x in zip(rr, cc, strict=True)
                    }
                )
    truth = dataset.ground_truth_pixels
    coords = sorted(pseudo.keys() | truth.keys())
    rows, cols = np.asarray(coords).T
    split = owners(rows, cols, manifest)
    labels_a = np.array([pseudo.get(p, -1) for p in coords], dtype=np.int64)
    labels_b = np.array([truth.get(p, -1) for p in coords], dtype=np.int64)
    np.savez(
        output / "pixels.npz", rows=rows, cols=cols, split=split, A=labels_a, B=labels_b
    )
    assets = dataset._dynamic_assets + dataset._static_assets
    names = [f"{a.role}__{Path(a.path).stem}__band{a.band or 1}" for a in assets]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate feature names")
    write_json(
        output / "features.json",
        {"names": names, "assets": [asdict(a) for a in assets]},
    )
    matrix = np.lib.format.open_memmap(
        output / "features.npy",
        mode="w+",
        dtype="float32",
        shape=(len(coords), len(assets)),
    )
    # Group sparse pixels into small windows, with one open VRT per feature.
    tile_ids = (rows // 256) * ((grid.width + 255) // 256) + cols // 256
    order = np.argsort(tile_ids, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(tile_ids[order])) + 1)
    windows = [
        (
            indices,
            int(cols[indices].min()),
            int(rows[indices].min()),
            int(np.ptp(cols[indices])) + 1,
            int(np.ptp(rows[indices])) + 1,
        )
        for indices in groups
    ]

    def extract(column, asset):
        with aligned_raster(asset.path, grid, resampling=asset.resampling) as raster:
            band = asset.band if asset.count > 1 and asset.band else 1
            for indices, x, y, w, h in windows:
                values = raster.read(band, window=Window(x, y, w, h), masked=True)
                sampled = (
                    values[rows[indices] - y, cols[indices] - x]
                    .astype(np.float32)
                    .filled(np.nan)
                )
                sampled[~np.isfinite(sampled) | (sampled == dataset.nodata)] = np.nan
                if asset.nodata is not None:
                    sampled[sampled == asset.nodata] = np.nan
                matrix[indices, column] = sampled
        return column

    print(
        f"Extracting {len(coords)} unique pixels, "
        f"{len(assets)} features, {len(windows)} windows",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(extract, i, a) for i, a in enumerate(assets)]
        for n, future in enumerate(as_completed(pending), 1):
            column = future.result()
            if n % 10 == 0 or n == len(assets):
                print(f"Features {n}/{len(assets)}: {names[column]}", flush=True)
    matrix.flush()
    metadata = {
        "source_experiment": str(experiment.resolve()),
        "source_run": str(run),
        "classes": mapping["minor_count"],
        "configured_ratios": manifest["ratios"],
        "split_policy": (
            "Original spatial block ownership; pixels are never randomly re-split."
        ),
        "counts": {},
        "features": len(assets),
        "unique_pixels": len(coords),
    }
    for name, labels in (("A", labels_a), ("B", labels_b)):
        selected = np.flatnonzero(labels > 0)
        metadata["counts"][name] = {
            s: int(np.sum((labels > 0) & (split == s))) for s in SPLITS
        }
        with (output / f"{name}_pixels.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["row", "column", "x", "y", "split", "label", *names])
            for i in selected:
                x, y = grid.transform * (int(cols[i]) + 0.5, int(rows[i]) + 0.5)
                writer.writerow(
                    [
                        rows[i],
                        cols[i],
                        x,
                        y,
                        split[i],
                        labels[i],
                        *[
                            format(float(v), ".9g") if np.isfinite(v) else ""
                            for v in matrix[i]
                        ],
                    ]
                )
    overlap = (labels_a > 0) & (labels_b > 0)
    metadata["pseudo_agreement_at_prompt_pixels"] = {
        s: {
            "n": int(np.sum(overlap & (split == s))),
            "accuracy": float(
                np.mean(
                    labels_a[overlap & (split == s)] == labels_b[overlap & (split == s)]
                )
            ),
        }
        for s in SPLITS
    }
    metadata["caveat"] = (
        "SAM labels were generated using ground-truth prompts, including held-out "
        "regions. Agreement at prompt pixels is not independent label-quality "
        "evidence. Block ownership prevents same-pixel leakage but cannot certify "
        "seed provenance across boundaries."
    )
    write_json(output / "dataset.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


class PixelDNN(nn.Module):
    """Four hidden Dense layers and a categorical softmax classification head."""

    def __init__(self, features, classes=32, activation="relu", four_hidden=True):
        super().__init__()
        widths = [features, 256, 128, 64, 32]
        if four_hidden:
            widths.append(classes)
        elif classes != 32:
            raise ValueError(
                "The requested fourth layer has 32 outputs, requiring 32 classes"
            )
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:], strict=True)
        )
        self.activation = nn.ReLU() if activation == "relu" else nn.Softmax(dim=-1)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def regularization(self):
        # Keras kernel_regularizer=l2(0.001), excluding bias; not AdamW decay.
        return 0.001 * sum(layer.weight.square().sum() for layer in self.layers)


def fit_scaler(matrix, indices):
    values = np.asarray(matrix[indices], dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    mean = np.divide(
        np.where(finite, values, 0).sum(axis=0),
        count,
        out=np.zeros(values.shape[1]),
        where=count > 0,
    )
    variance = np.divide(
        np.where(finite, (values - mean) ** 2, 0).sum(axis=0),
        count,
        out=np.zeros_like(mean),
        where=count > 0,
    )
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1
    return mean.astype(np.float32), scale.astype(np.float32), count


@torch.no_grad()
def evaluate(model, x, y, indices):
    model.eval()
    logits = torch.cat([model(x[batch]) for batch in indices.split(2048)])
    target = y[indices]
    prediction = logits.argmax(1)
    classes = logits.shape[1]
    confusion = (
        torch.bincount(target * classes + prediction, minlength=classes**2)
        .reshape(classes, classes)
        .cpu()
        .numpy()
    )
    support = confusion.sum(axis=1)
    recall = np.divide(
        confusion.diagonal(), support, out=np.zeros(classes), where=support > 0
    )
    return {
        "n": len(indices),
        "accuracy": float((prediction == target).float().mean()),
        "cross_entropy": float(nn.functional.cross_entropy(logits, target)),
        "macro_recall_present_classes": float(recall[support > 0].mean()),
        "confusion_matrix": confusion.tolist(),
        "recall": recall.tolist(),
        "support": support.tolist(),
    }, prediction.cpu().numpy() + 1


def train(output, activation, four_hidden, epochs, batch_size, seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    metadata = json.loads((output / "dataset.json").read_text())
    matrix = np.load(output / "features.npy", mmap_mode="r")
    pixels = np.load(output / "pixels.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    results = {}
    for name in ("A", "B"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        labels = pixels[name]
        selected = {
            s: np.flatnonzero((labels > 0) & (pixels["split"] == s)) for s in SPLITS
        }
        truth = {
            s: np.flatnonzero((pixels["B"] > 0) & (pixels["split"] == s))
            for s in SPLITS
        }
        if any(len(i) == 0 for i in selected.values()):
            raise ValueError("Empty split")
        mean, scale, count = fit_scaler(matrix, selected["train"])
        normal = (np.asarray(matrix) - mean) / scale
        normal[~np.isfinite(normal)] = 0
        x = torch.tensor(normal, device=device)
        y = torch.tensor(labels - 1, device=device)
        gt = torch.tensor(pixels["B"] - 1, device=device)
        indices = {s: torch.tensor(i, device=device) for s, i in selected.items()}
        gt_indices = {s: torch.tensor(i, device=device) for s, i in truth.items()}
        model = PixelDNN(
            matrix.shape[1], metadata["classes"], activation, four_hidden
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        folder = output / name
        folder.mkdir(exist_ok=True)
        np.savez(
            folder / "scaler.npz", mean=mean, scale=scale, observed_train_counts=count
        )
        history, best, best_epoch, best_state = [], -1, None, None
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            shuffled = indices["train"][
                torch.randperm(len(indices["train"]), device=device)
            ]
            for batch in shuffled.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    nn.functional.cross_entropy(model(x[batch]), y[batch])
                    + model.regularization()
                )
                loss.backward()
                optimizer.step()
            train_metrics, _ = evaluate(model, x, y, indices["train"])
            val_metrics, _ = evaluate(model, x, y, indices["validation"])
            gt_val, _ = evaluate(model, x, gt, gt_indices["validation"])
            regularization = float(model.regularization().detach())
            record = {
                "epoch": epoch,
                "train_accuracy": train_metrics["accuracy"],
                "train_ce": train_metrics["cross_entropy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_ce": val_metrics["cross_entropy"],
                "gt_val_accuracy": gt_val["accuracy"],
                "gt_val_ce": gt_val["cross_entropy"],
                "l2_penalty": regularization,
                "train_loss": train_metrics["cross_entropy"] + regularization,
                "val_loss": val_metrics["cross_entropy"] + regularization,
            }
            history.append(record)
            # Common clean validation labels select both models; test is untouched.
            if gt_val["accuracy"] > best:
                best, best_epoch = gt_val["accuracy"], epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"{name} epoch {epoch}: train={record['train_accuracy']:.4f} "
                    f"val={record['val_accuracy']:.4f} GT_val={best:.4f} (best)",
                    flush=True,
                )
            write_json(folder / "history.json", history)
        torch.save(
            {
                "model": model.state_dict(),
                "mean": torch.from_numpy(mean),
                "scale": torch.from_numpy(scale),
                "activation": activation,
                "four_hidden": four_hidden,
            },
            folder / "last.pt",
        )
        result = {
            "best_epoch": best_epoch,
            "selection": (
                "maximum shared ground-truth validation accuracy; earliest tie"
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "epochs": epochs,
            "optimizer_steps": epochs
            * ((len(indices["train"]) + batch_size - 1) // batch_size),
            "final": {},
            "best": {},
        }
        for state_name in ("final", "best"):
            if state_name == "best":
                model.load_state_dict(best_state)
            for s in ("validation", "test"):
                result[state_name][s], _ = evaluate(model, x, y, indices[s])
                result[state_name][f"ground_truth_{s}"], pred = evaluate(
                    model, x, gt, gt_indices[s]
                )
                with (folder / f"{state_name}_ground_truth_{s}_predictions.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["row", "column", "label", "prediction"])
                    for i, prediction in zip(truth[s], pred, strict=True):
                        writer.writerow(
                            [
                                pixels["rows"][i],
                                pixels["cols"][i],
                                pixels["B"][i],
                                prediction,
                            ]
                        )
        torch.save(
            {
                "model": best_state,
                "mean": torch.from_numpy(mean),
                "scale": torch.from_numpy(scale),
                "activation": activation,
                "four_hidden": four_hidden,
            },
            folder / "best_loss.pt",
        )
        write_json(folder / "metrics.json", result)
        results[name] = result
    write_json(
        output / "results.json",
        {
            "settings": {
                "activation": activation,
                "four_hidden": four_hidden,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "optimizer": "Adam",
                "lr": 0.001,
                "l2_kernel": 0.001,
                "device": device,
                "normalization": (
                    "per-feature mean/std fitted on each arm's training pixels; "
                    "missing -> training mean"
                ),
                "output": (
                    "softmax; cross_entropy(logits, integer_target) equals "
                    "categorical_crossentropy(one_hot, softmax(logits))"
                ),
            },
            "results": results,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=Path, default=Path("experiments/20260907_233429")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/pixel_dnn_ab_20260908")
    )
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--activation", choices=("relu", "softmax"), default="relu")
    parser.add_argument(
        "--four-hidden", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        if (args.output / "dataset.json").exists():
            raise FileExistsError(
                "Prepared dataset exists; use --stage train or a new output directory"
            )
        prepare(args.experiment, args.output, args.workers)
    if args.stage in ("train", "all"):
        train(
            args.output,
            args.activation,
            args.four_hidden,
            args.epochs,
            args.batch_size,
            args.seed,
        )


if __name__ == "__main__":
    main()
