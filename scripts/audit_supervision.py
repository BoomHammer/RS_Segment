"""Audit an experiment's measured supervision without loading raster windows."""

import argparse
import json
from pathlib import Path

from config import load_config
from data.sample_index import WindowedSampleDataset
from data.sampling import SpatialWeightedSampler
from data.spatial_split import load_spatial_split
from data.training_policy import isolate_splits, point_windows, supervision_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--halo", type=int, nargs=2, default=None)
    args = parser.parse_args()
    log = json.loads((args.experiment / "train_log.json").read_text(encoding="utf-8"))
    run = Path(log["source_run"])
    config = load_config(args.experiment / "data.yaml")
    dataset = WindowedSampleDataset(
        run / "sample_index.json",
        window_size=tuple(log["window_size"]),
        stride=tuple(log["stride"]),
        halo=tuple(args.halo or log.get("halo", (0, 0))),
        grid_offset=tuple(log.get("grid_offset", (0, 0))),
        label_columns=config.data.label_columns,
        label_mapping=json.loads(
            next(run.glob("label_mapping*.json")).read_text(encoding="utf-8")
        ),
        stage2=config.data.stage2,
    )
    manifest = load_spatial_split(run / "spatial_split.json")
    sampler = SpatialWeightedSampler(dataset, manifest=manifest)
    measured = point_windows(dataset)
    probability = sum(
        float(weight)
        for index, weight in zip(sampler.indices, sampler.weights, strict=True)
        if index in measured
    ) / float(sampler.weights.sum())
    report = {
        "experiment": str(args.experiment),
        "halo": list(dataset.halo),
        "original": supervision_summary(dataset, manifest),
        "isolated": supervision_summary(dataset, isolate_splits(dataset, manifest)),
        "original_measured_window_probability": probability,
        "note": "Counts precede raster validity masking; points repeat across windows.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
