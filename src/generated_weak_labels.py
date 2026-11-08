"""Persist the dynamic weak supervision used by E08 and E09."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _windows(values) -> np.ndarray:
    return np.asarray(
        [
            [
                int(window.col_off),
                int(window.row_off),
                int(window.width),
                int(window.height),
            ]
            for window in values
        ],
        dtype=np.int32,
    )


class GeneratedWeakLabelStore:
    """Keep the latest epoch's batch-level labels without retaining GPU tensors."""

    def __init__(self, output: Path, *, rank: int, experiment: str) -> None:
        self.directory = (
            output / "generated_weak_labels" / "latest" / f"rank_{rank:03d}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment

    def write_batch(self, batch: dict, *, epoch: int, batch_index: int) -> None:
        mask = batch["weak_label_mask"].detach().bool().cpu()
        labels = batch["weak_label"].detach().to(torch.int16).cpu()
        labels.masked_fill_(~mask, 0)
        payload = {
            "labels": labels.numpy(),
            "mask": mask.numpy(),
            "window": _windows(batch["window"]),
            "input_window": _windows(batch["input_window"]),
            "epoch": np.asarray(epoch, dtype=np.int32),
            "batch": np.asarray(batch_index, dtype=np.int32),
        }
        if "self_training_label" in batch:
            self_training_mask = batch["self_training_label_mask"].detach().bool().cpu()
            self_training = batch["self_training_label"].detach().to(torch.int16).cpu()
            self_training.masked_fill_(~self_training_mask, 0)
            payload["self_training_labels"] = self_training.numpy()
            payload["self_training_mask"] = self_training_mask.numpy()
        path = self.directory / f"batch_{batch_index:06d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)

    def finish_epoch(self, *, epoch: int, batches: int) -> None:
        manifest = {
            "schema_version": 1,
            "experiment": self.experiment,
            "epoch": epoch,
            "batches": batches,
            "label_encoding": "project class code; 0 means absent",
            "spatial_orientation": "exact tensors consumed after training augmentation",
        }
        path = self.directory / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)
