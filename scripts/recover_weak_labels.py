"""Verify completed rank rasters and recover reports without rerunning SAM."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from scripts.weak_label import _load_shard_outcomes

from weak_label.conflicts import merge_candidate_window, open_candidate_sources
from weak_label.generation import evaluate_label_quality_from_raster
from weak_label.quality import write_label_quality_visualization


def recover(run: Path, ranks: int, conflict_margin: float) -> dict:
    """Check every pixel against retained rank outputs, preserving all inputs."""
    shards = [
        {
            "labels": run / f".weak_labels.tif.rank{rank}",
            "candidate_labels": run / f".weak_labels.tif.rank{rank}.candidates.tif",
            "candidate_scores": run / f".weak_labels.tif.rank{rank}.candidate_scores.tif",
            "outcomes": run / f".weak_labels.tif.rank{rank}.outcomes.json",
        }
        for rank in range(ranks)
    ]
    if any(
        not shard[key].is_file()
        for shard in shards
        for key in ("candidate_labels", "candidate_scores")
    ):
        raise ValueError("候选分片缺失；旧版单类别分片无法恢复冲突证据，请重新生成弱标签")
    outcomes = _load_shard_outcomes(shards)
    if not outcomes or any(item.get("status") == "pending" for item in outcomes):
        raise ValueError("Missing or unfinished sample outcomes")
    mapping = json.loads((run / "label_mapping.json").read_text(encoding="utf-8"))
    codes = {int(item["alliance_code"]) for item in mapping["classes"]}
    mismatch = 0
    with ExitStack() as stack:
        final = stack.enter_context(rasterio.open(run / "weak_labels.tif"))
        sources = open_candidate_sources(stack, shards, final)
        for _, window in final.block_windows(1):
            actual = final.read(1, window=window)
            if not set(np.unique(actual)) <= codes | {final.nodata}:
                raise ValueError("Final raster has unknown class codes")
            labels = merge_candidate_window(
                sources, window, nodata=final.nodata, conflict_margin=conflict_margin
            )
            mismatch += int(np.count_nonzero(actual != labels))
        if mismatch:
            raise ValueError(f"Final raster differs from rank merge: {mismatch} pixels")
        scale = min(1.0, 1024 / max(final.shape))
        preview = final.read(
            1,
            out_shape=(
                max(1, int(final.height * scale)),
                max(1, int(final.width * scale)),
            ),
            resampling=rasterio.enums.Resampling.nearest,
        )
        valid = preview != final.nodata
    report = evaluate_label_quality_from_raster(
        run / "weak_labels.tif", sample_outcomes=outcomes
    )
    report["recovery_audit"] = {
        "all_raster_blocks_read": True,
        "rank_count": ranks,
        "merge_mismatch_pixels": mismatch,
        "conflict_margin": conflict_margin,
        "missing_class_codes": sorted(
            codes - set(map(int, report["class_distribution"]))
        ),
        "whole_grid_labeled_fraction": report["labeled_pixels"]
        / report["total_pixels"],
    }
    (run / "weak_labels_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_label_quality_visualization(
        preview, run / "weak_labels_quality.png", valid_mask=valid
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--conflict-margin", type=float, required=True)
    args = parser.parse_args()
    if args.ranks < 1 or args.conflict_margin < 0:
        parser.error("ranks must be positive and conflict-margin nonnegative")
    report = recover(args.run.resolve(), args.ranks, args.conflict_margin)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("labeled_pixels", "sample_quality", "recovery_audit")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
