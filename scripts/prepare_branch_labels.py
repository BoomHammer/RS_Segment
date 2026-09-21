"""Audit supervision coverage and generate traceable training-only PointSAM labels."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
import yaml

from config import load_config
from data.balanced_sampling import ClassBalancedPointSampler
from data.labels import iter_encoded_labels
from data.raster_alignment import aligned_raster, locate_points
from data.sample_index import WindowedSampleDataset, _window
from data.spatial_split import load_spatial_split
from data.training_policy import point_owner, point_windows
from run_accuracy_experiments import read_json, write_json


def dataset_for(output):
    config = load_config(output / "settings/data.yaml")
    data = output / "dataset"
    mapping = read_json(next(data.glob("label_mapping*.json")))
    stage = config.data.stage2
    dataset = WindowedSampleDataset(
        data / "sample_index.json",
        window_size=tuple(stage["window"]["size"]),
        stride=tuple(stage["window"]["stride"]),
        halo=tuple(stage["window"]["halo"]),
        label_columns=config.data.label_columns,
        label_mapping=mapping,
        statistics=next(data.glob("raster_stats*.json")),
        stage2=stage,
    )
    return config, mapping, dataset, load_spatial_split(data / "spatial_split.json")


def training_records(config, mapping, dataset, manifest):
    """Match the dataset's last-record-per-pixel rule before selecting train seeds."""
    records = [
        r
        for batch in iter_encoded_labels(
            config.data.label_file,
            label_columns=config.data.label_columns,
            mapping=mapping,
        )
        for r in batch
    ]
    locations = locate_points(
        [(r.x, r.y) for r in records], point_crs="EPSG:4326", grid=dataset.grid
    )
    unique = {
        (int(loc["row"]), int(loc["column"])): record
        for record, loc in zip(records, locations, strict=True)
        if loc["inside"]
    }
    selected = [
        (pixel, record)
        for pixel, record in unique.items()
        if point_owner(*pixel, manifest) == "train"
    ]
    assert len(selected) == sum(manifest.class_counts["train"].values())
    for pixel, record in selected:
        assert dataset.ground_truth_pixels[pixel] == record.alliance_code
    return selected


def audit_coverage(path, dataset, manifest):
    """Count unique labels, valid inputs, eligible cores and epoch-one occurrences."""
    rows, columns, labels, valids = [], [], [], []
    # Match the loader's nearest-neighbor alignment (the old raster is one
    # column wider than its sample index, with the same pixel transform).
    with aligned_raster(path, dataset.grid, resampling="nearest") as source:
        for _, window in source.block_windows(1):
            values = source.read(1, window=window)
            rr, cc = np.nonzero(values > 0)
            if not len(rr):
                continue
            valid = np.zeros(values.shape, dtype=bool)
            for asset in [*dataset._static_assets, *dataset._dynamic_assets]:
                valid |= np.isfinite(dataset._read_asset(asset, window))
                if valid[rr, cc].all():
                    break
            rows.extend((rr + int(window.row_off)).tolist())
            columns.extend((cc + int(window.col_off)).tolist())
            labels.extend(values[rr, cc].tolist())
            valids.extend(valid[rr, cc].tolist())
    rows, columns, labels, valids = map(np.asarray, (rows, columns, labels, valids))
    owned = np.asarray(
        [
            point_owner(int(r), int(c), manifest) == "train"
            for r, c in zip(rows, columns, strict=True)
        ],
        dtype=bool,
    )
    eligible = sorted(
        set(manifest.splits["train"]) & point_windows(dataset, manifest, "train").keys()
    )

    def coverage(indices):
        counts = np.zeros(len(rows), dtype=np.int32)
        for index in indices:
            record = dataset.index.iloc[index]
            w = _window(
                int(record.row), int(record.column), dataset.window_size, dataset.grid
            )
            counts += (
                (rows >= w.row_off)
                & (rows < w.row_off + w.height)
                & (columns >= w.col_off)
                & (columns < w.col_off + w.width)
            )
        return counts

    reachable = coverage(eligible) > 0
    sampler = ClassBalancedPointSampler(dataset, manifest, eligible, seed=42)
    occurrences = coverage([i for i, _ in sampler])
    active = owned & valids
    return {
        "source": str(path),
        "unique_labeled_pixels": len(rows),
        "train_owned_pixels": int(owned.sum()),
        "excluded_other_splits": int((~owned).sum()),
        "train_all_inputs_invalid": int((owned & ~valids).sum()),
        "train_valid_pixels": int(active.sum()),
        "reachable_train_valid_pixels": int((active & reachable).sum()),
        "unreachable_train_valid_pixels": int((active & ~reachable).sum()),
        "epoch1_unique_used_pixels": int((active & (occurrences > 0)).sum()),
        "epoch1_loss_occurrences": int(occurrences[active].sum()),
        "eligible_core_windows": len(eligible),
        "per_class_train_valid": dict(Counter(map(int, labels[active]))),
        "per_class_epoch1_unique_used": dict(
            Counter(map(int, labels[active & (occurrences > 0)]))
        ),
        "protocol": "seed42 point sampler; owned core; any selected input valid",
    }


def restrict_to_training(raw, provenance, output, output_provenance, manifest):
    """Discard held-out/unassigned blocks, keeping source IDs aligned with labels."""
    bw, bh = manifest.block_size
    with rasterio.open(raw) as source, rasterio.open(provenance) as seeds:
        with (
            rasterio.open(output, "w", **source.profile) as target,
            rasterio.open(output_provenance, "w", **seeds.profile) as target_seeds,
        ):
            for _, window in source.block_windows(1):
                values = source.read(1, window=window)
                ids = seeds.read(1, window=window)
                rr = (np.arange(values.shape[0]) + int(window.row_off)) // bh
                cc = (np.arange(values.shape[1]) + int(window.col_off)) // bw
                owned = np.zeros(values.shape, dtype=bool)
                for r in np.unique(rr):
                    for c in np.unique(cc):
                        if manifest.blocks.get(f"{r}:{c}") == "train":
                            owned |= (rr[:, None] == r) & (cc[None, :] == c)
                values[~owned] = source.nodata
                ids[~owned] = seeds.nodata
                assert np.array_equal(values > 0, ids > 0)
                target.write(values, 1, window=window)
                target_seeds.write(ids, 1, window=window)


def generate(output):
    config, mapping, dataset, manifest = dataset_for(output)
    artifacts = output / "labels"
    artifacts.mkdir(exist_ok=True)
    old_index = read_json(
        Path(read_json(output / "campaign.json")["historical_dataset"])
        / "sample_index.json"
    )
    audit = audit_coverage(Path(old_index["weak_label"]["path"]), dataset, manifest)
    write_json(artifacts / "historical_coverage.json", audit)
    print("Historical supervision audit:", json.dumps(audit), flush=True)
    selected = training_records(config, mapping, dataset, manifest)
    write_json(
        artifacts / "training_prompts.json",
        [
            {"seed_id": i + 1, "row": pixel[0], "column": pixel[1], **asdict(record)}
            for i, (pixel, record) in enumerate(selected)
        ],
    )
    # scripts/weak_label.py shares the package name; prefer this snapshot's src.
    source_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_root))
    from inference.sam2_backend import SAM2Inferencer
    from weak_label.generation import WeakLabelGenerationConfig, generate_weak_labels
    from weak_label.sam_input import discover_sam_videos

    sam_provenance = read_json(output / "settings/sam_provenance.json")
    if (
        hashlib.sha256(config.data.sam2_checkpoint.read_bytes()).hexdigest()
        != (sam_provenance["checkpoint_sha256"])
    ):
        raise ValueError("SAM checkpoint differs from the frozen provenance")

    weak = yaml.safe_load(
        (output / "settings/weak_label.yaml").read_text(encoding="utf-8")
    )["weak_label"]
    frames, keyframe = discover_sam_videos(
        config.data.dynamic, [weak.get(f"CompositeBands{i}", []) for i in range(1, 6)]
    )
    print(
        f"Training-only prompts: {len(selected)}, video frames: {len(frames)}",
        flush=True,
    )
    settings = WeakLabelGenerationConfig(
        window_size=tuple(weak["input_window_size"]),
        label_radius=weak["label_radius"],
        medium_label_radius=weak["medium_label_radius"],
        fallback_radius=weak["fallback_radius"],
        input_range=tuple(weak["input_range"]),
        stretch_percentiles=tuple(weak["stretch_percentiles"]),
        medium_confidence=weak["medium_confidence"],
        min_confidence=weak["mask_confidence"],
        spectral_distance_threshold=weak["spectral_distance_threshold"],
        logit_threshold=weak["logit_threshold"],
        reject_boundary_touch=weak["reject_boundary_touch"],
        conflict_margin=weak["conflict_margin"],
        mask_fusion=weak["mask_fusion"],
        weighted_vote_threshold=weak["weighted_vote_threshold"],
    )
    inferencer = SAM2Inferencer(
        config.data.sam2_checkpoint, device="cuda", use_video=True
    )
    import torch

    with torch.inference_mode():
        generate_weak_labels(
            [record for _, record in selected],
            grid=dataset.grid,
            image_paths=frames,
            inferencer=inferencer,
            output_path=artifacts / "generated.tif",
            config=settings,
            provenance_path=artifacts / "generated_seed_ids.tif",
            sample_outcomes_path=artifacts / "prompt_outcomes.json",
            quality_report_path=artifacts / "generation_quality.json",
            keyframe_index=keyframe,
        )
    restrict_to_training(
        artifacts / "generated.tif",
        artifacts / "generated_seed_ids.tif",
        output / "dataset/weak_labels.tif",
        artifacts / "training_seed_ids.tif",
        manifest,
    )
    new_audit = audit_coverage(output / "dataset/weak_labels.tif", dataset, manifest)
    write_json(artifacts / "training_coverage.json", new_audit)
    if (
        new_audit["excluded_other_splits"]
        or new_audit["epoch1_unique_used_pixels"] == 0
    ):
        raise ValueError("Training pseudo labels failed ownership/coverage validation")
    dataset.configure_supervision_split(manifest, "train", mask_weak_labels=True)
    indices = sorted(
        set(manifest.splits["train"]) & point_windows(dataset, manifest, "train").keys()
    )
    # Check actual loader tensors before any long training job is allowed.
    sampler = ClassBalancedPointSampler(dataset, manifest, indices, seed=42)
    checked = []
    for selection in list(sampler)[:3]:
        sample = dataset[selection]
        mask = sample["weak_label_mask"] & sample["valid_mask"] & sample["core_mask"]
        assert not (mask & ~sample["supervision_split_mask"]).any()
        checked.append(int(mask.sum()))
    if not sum(checked):
        raise ValueError("No pseudo supervision in the initial loader batches")
    write_json(artifacts / "loader_check.json", {"weak_pixels_first_three": checked})
    dataset.close()
    paths = [
        output / "dataset/weak_labels.tif",
        *artifacts.glob("*.json"),
        *artifacts.glob("*.tif"),
    ]
    write_json(
        output / "label_artifacts.json",
        {
            "status": "ready",
            "training_seed_count": len(selected),
            "sha256": {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output.resolve())


if __name__ == "__main__":
    main()
