"""Compare bounded DataLoader throughput on identical real training windows."""

import argparse
import ctypes
import gc
import json
import os
import time
from pathlib import Path

import torch

from config import load_config
from data.balanced_sampling import ClassBalancedPointSampler
from data.sample_index import WindowedSampleDataset
from data.sampling import build_dataloader
from data.spatial_split import load_spatial_split
from data.training_policy import point_windows


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("load", ctypes.c_uint32),
        ("total", ctypes.c_uint64),
        ("available", ctypes.c_uint64),
        ("page_total", ctypes.c_uint64),
        ("page_available", ctypes.c_uint64),
        ("virtual_total", ctypes.c_uint64),
        ("virtual_available", ctypes.c_uint64),
        ("extended_available", ctypes.c_uint64),
    ]


def memory_gib():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot read physical memory status")
    return (status.total - status.available) / 2**30, status.available / 2**30


def make_dataset(campaign, handles):
    config = load_config(campaign / "settings/data.yaml")
    run = campaign / "dataset"
    stage2 = {**config.data.stage2, "io": {"max_open_rasters": handles}}
    return WindowedSampleDataset(
        run / "sample_index.json",
        window_size=(256, 256),
        stride=(128, 128),
        halo=(32, 32),
        label_columns=config.data.label_columns,
        label_mapping=json.loads(
            next(run.glob("label_mapping*.json")).read_text(encoding="utf-8")
        ),
        statistics=next(run.glob("raster_stats*.json")),
        stage2=stage2,
        use_weak_labels=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    os.environ["GDAL_CACHEMAX"] = "128"
    manifest = load_spatial_split(args.campaign / "dataset/spatial_split.json")
    results = []
    for workers, handles in [(2, 0), (2, 384), (4, 384), (6, 384)]:
        dataset = make_dataset(args.campaign, handles)
        dataset.configure_supervision_split(manifest, "train", mask_weak_labels=True)
        measured = point_windows(dataset, manifest, "train")
        sampler = ClassBalancedPointSampler(
            dataset,
            manifest,
            [i for i in manifest.splits["train"] if i in measured],
            seed=42,
        )
        warmup = 6
        # Identical windows and order for every configuration.
        anchors = list(sampler)[: warmup + args.batches]
        loader = build_dataloader(
            dataset,
            sampler=anchors,
            num_workers=workers,
            batch_size=1,
            persistent_workers=True,
            prefetch_factor=1,
            pin_memory=True,
        )
        start = time.perf_counter()
        iterator = iter(loader)
        measured_start = None
        peak_memory = 0
        minimum_available = float("inf")
        for i in range(len(anchors)):
            batch = next(iterator)
            # Include pinning, IPC and host-to-device transfer.
            dynamic = batch["dynamic"].to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            used, available = memory_gib()
            peak_memory = max(peak_memory, used)
            minimum_available = min(minimum_available, available)
            del dynamic, batch
            if i + 1 == warmup:
                measured_start = time.perf_counter()
        elapsed = time.perf_counter() - measured_start
        result = {
            "workers": workers,
            "max_open_rasters": handles,
            "measured_batches": args.batches,
            "seconds": elapsed,
            "batches_per_second": args.batches / elapsed,
            "including_startup_seconds": time.perf_counter() - start,
            "system_peak_used_gib": peak_memory,
            "system_min_available_gib": minimum_available,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        del iterator, loader, dataset
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
