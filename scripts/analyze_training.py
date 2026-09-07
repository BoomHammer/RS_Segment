"""Plot training trajectories and export per-class confusion diagnostics."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    log = json.loads((args.experiment / "train_log.json").read_text(encoding="utf-8"))
    epochs = log["epochs"]
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    x = [item["epoch"] for item in epochs]
    for key, label in [
        ("accuracy", "Train"),
        ("val_accuracy", "Validation windows"),
        ("val_unique_point_accuracy", "Validation unique points"),
    ]:
        axes[0].plot(x, [item[key] for item in epochs], label=label)
    axes[0].axhline(0.5, color="gray", linestyle="--", label="Target 0.5")
    axes[0].set(title="Accuracy", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [item["val_loss"] for item in epochs], color="tab:orange")
    axes[1].set(title="Validation cross entropy")
    axes[2].plot(x, [item["val_macro_recall"] for item in epochs], color="tab:green")
    axes[2].set(title="Validation macro recall", ylim=(0, 0.5))
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.savefig(args.output / "training_curves.png", dpi=180)
    plt.close(figure)
    best = max(epochs, key=lambda item: item["val_accuracy"])
    validation = json.loads(
        (args.experiment / f"validation_epoch_{best['epoch']:03d}.json").read_text(
            encoding="utf-8"
        )
    )
    confusion = np.array(validation["confusion_matrix"])
    mapping_path = next(Path(log["source_run"]).glob("label_mapping*.json"))
    classes = sorted(
        json.loads(mapping_path.read_text(encoding="utf-8"))["classes"],
        key=lambda item: int(item["alliance_code"]),
    )
    summary = {
        "best_window_accuracy": best,
        "best_unique_accuracy": max(
            epochs, key=lambda item: item["val_unique_point_accuracy"]
        ),
        "best_validation_loss": min(epochs, key=lambda item: item["val_loss"]),
        "zero_recall_class_count": int((confusion.diagonal() == 0).sum()),
        "predicted_class_count": int((confusion.sum(0) > 0).sum()),
        "mean_epoch_minutes_including_validation_and_saving": (
            datetime.fromisoformat(log["finished_at"])
            - datetime.fromisoformat(log["started_at"])
        ).total_seconds()
        / 60
        / len(epochs),
        "classes": [
            {
                "code": i + 1,
                "name": item["alliance"],
                "train_unique_points": log["supervision_audit"]["unique_class_counts"][
                    "train"
                ].get(str(i + 1), 0),
                "validation_occurrences": int(confusion[i].sum()),
                "predicted_occurrences": int(confusion[:, i].sum()),
                "recall": float(confusion[i, i] / max(confusion[i].sum(), 1)),
                "most_frequent_prediction": classes[int(confusion[i].argmax())][
                    "alliance"
                ],
            }
            for i, item in enumerate(classes)
        ],
    }
    (args.output / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
