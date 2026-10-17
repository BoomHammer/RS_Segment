"""Evaluate held-out points from fused map decisions before label overrides."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from data.spatial_split import load_spatial_split
from data.training_policy import point_owner, point_windows


def _hard_metrics(targets, predictions, classes):
    """Include abstentions as false negatives and report coverage explicitly."""
    confusion = np.zeros((classes, classes + 1), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target - 1, prediction - 1 if prediction else classes] += 1
    support = confusion.sum(axis=1)
    predicted = confusion[:, :classes].sum(axis=0)
    tp = np.diag(confusion[:, :classes])

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)

    precision = divide(tp, predicted)
    recall = divide(tp, support)
    f1 = divide(2 * tp, support + predicted)
    iou = divide(tp, support + predicted - tp)
    present = support + predicted > 0
    total = int(support.sum())
    covered = int(predicted.sum())
    return {
        "labeled_pixels": total,
        "predicted_pixels": covered,
        "nodata_pixels": total - covered,
        "coverage": covered / total if total else None,
        "accuracy": int(tp.sum()) / total if total else None,
        "accuracy_on_predicted_pixels": int(tp.sum()) / covered if covered else None,
        "macro_precision": float(precision[present].mean()) if present.any() else None,
        "macro_recall": float(recall[present].mean()) if present.any() else None,
        "macro_f1": float(f1[present].mean()) if present.any() else None,
        "macro_iou": float(iou[present].mean()) if present.any() else None,
        "class_support": support.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_iou": iou.tolist(),
        "confusion_matrix": confusion.tolist(),
        "confusion_columns": [*range(1, classes + 1), "NoData"],
    }


class PredictionPointEvaluation:
    """Keep only sparse point probabilities; never allocate a full probability map."""

    def __init__(self, dataset, mapping, checkpoint, run, config):
        checkpoint, run = Path(checkpoint), Path(run)
        self.checkpoint = checkpoint
        self.mapping = mapping
        self.class_paths = {
            int(item["alliance_code"]): tuple(
                item.get("level_codes")
                or (item.get("formation_code", 1), int(item["alliance_code"]))
            )
            for item in mapping["classes"]
        }
        self._restricted_observations = {}
        self.records = []
        self.by_row = {}
        self.status = "ok"
        self.split_path = None
        self.max_windows = config.get("max_windows")
        if self.max_windows is not None and (
            type(self.max_windows) is not int or self.max_windows < 1
        ):
            raise ValueError("evaluation.max_windows 必须是正整数或 null")
        metadata_path = checkpoint.parent / "train_log.json"
        if not metadata_path.is_file():
            metadata_path = checkpoint.parent / "run.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("supervision_policy", {}).get("train_on_test", False):
            self.status = "skipped_test_used_for_training"
            return
        self.split_path = checkpoint.parent / "spatial_split.json"
        if not self.split_path.is_file():
            self.split_path = run / "spatial_split.json"
        manifest = load_spatial_split(self.split_path)
        if "test" not in manifest.blocks.values():
            self.status = "skipped_no_test_blocks"
            return
        selected_windows = None
        if self.max_windows is not None:
            measured = point_windows(dataset, manifest, "test")
            selected_windows = set(
                [index for index in manifest.splits["test"] if index in measured][
                    : self.max_windows
                ]
            )
        for (row, column), observations in sorted(
            dataset.evaluation_observations.items()
        ):
            if point_owner(row, column, manifest) != "test":
                continue
            if selected_windows is not None and not selected_windows.intersection(
                dataset.query_windows_for_pixel(row, column)
            ):
                continue
            self.by_row.setdefault(row, []).append((column, observations))
        if not self.by_row:
            self.status = "skipped_no_test_points"

    def capture(
        self,
        row,
        end,
        score_shards,
        denominator,
        result,
        valid,
        constraint=None,
        elevation=None,
        dem_allowed=None,
        dem_missing=None,
    ):
        """Snapshot both decisions before the caller applies any ground truth."""
        points = [
            (r, c, observations)
            for r in range(row, end)
            for c, observations in self.by_row.get(r, [])
        ]
        if not points:
            return
        rows = np.array([point[0] for point in points])
        columns = np.array([point[1] for point in points])
        local_rows = rows - row
        weights = denominator[local_rows, columns]
        probabilities = np.zeros((len(points), score_shards[0].shape[0]), np.float32)
        for shard in score_shards:
            probabilities += shard[:, rows, columns].T
        before_predictions = probabilities.argmax(axis=1) + 1
        np.divide(
            probabilities,
            weights[:, None],
            out=probabilities,
            where=weights[:, None] > 0,
        )
        for index, (r, c, observations) in enumerate(points):
            lr = r - row
            covered = bool(weights[index] > 0)
            probability = probabilities[index]
            if covered and not np.isfinite(probability).all():
                raise ValueError("测试点融合概率出现非有限值")
            before = int(before_predictions[index]) if covered else None
            after = int(result[lr, c]) if valid[lr, c] else None
            missing = bool(dem_missing[lr, c]) if constraint else False
            allowed = bool(dem_allowed[lr, c]) if constraint else True
            restricted_scores = probability.copy()
            if constraint is not None and not allowed:
                restricted_scores[np.array(sorted(constraint.class_codes)) - 1] = 0
            conflict = (
                constraint is not None
                and not missing
                and not allowed
                and any(
                    self._restricted(codes, constraint.class_codes)
                    for codes in observations
                )
            )
            self.records.append(
                {
                    "row": r,
                    "column": c,
                    "observations": [list(codes) for codes in observations],
                    "before_prediction": before,
                    "after_prediction": after,
                    "fused_probabilities": probability.tolist() if covered else None,
                    "constrained_scores": restricted_scores.tolist() if after else None,
                    "dem_m": float(elevation[lr, c])
                    if constraint and not missing
                    else None,
                    "dem_missing": missing,
                    "label_constraint_conflict": bool(conflict),
                }
            )

    def _restricted(self, codes, class_codes):
        """A partial label conflicts only if all compatible leaves are forbidden."""
        key = (tuple(codes), frozenset(class_codes))
        if key not in self._restricted_observations:
            compatible = {
                leaf
                for leaf, path in self.class_paths.items()
                if all(
                    code <= 0 or code == expected
                    for code, expected in zip(codes, path, strict=True)
                )
            }
            self._restricted_observations[key] = (
                bool(compatible) and compatible <= class_codes
            )
        return self._restricted_observations[key]

    def _summary(self, key):
        items = sorted(self.mapping["classes"], key=lambda item: item["alliance_code"])
        paths = self.class_paths
        depths = {
            int(item["alliance_code"]): item.get(
                "terminal_level", len(paths[int(item["alliance_code"])])
            )
            for item in items
        }
        names = [level["name"] for level in self.mapping.get("levels", [])]
        counts = self.mapping.get("level_counts")
        if not names:
            names = ["coarse", "fine"]
        if counts is None:
            counts = [
                max(path[i] for path in paths.values()) for i in range(len(names))
            ]

        def summarize_level(level, classes):
            targets, predictions, mixed_agreements = [], [], []
            mixed_covered = 0
            for record in self.records:
                observed = [
                    codes[-1] if level is None else codes[level]
                    for codes in record["observations"]
                    if (codes[-1] if level is None else codes[level]) > 0
                    and (level is None or codes[-1] < 1 or depths[codes[-1]] > level)
                ]
                if not observed:
                    continue
                leaf = record[key]
                prediction = leaf
                if level is not None:
                    prediction = (
                        paths[leaf][level] if leaf and depths[leaf] > level else None
                    )
                if len(set(observed)) == 1:
                    targets.append(observed[0])
                    predictions.append(prediction)
                else:
                    mixed_agreements.append(
                        Counter(observed)[prediction] / len(observed)
                    )
                    mixed_covered += prediction is not None
            report = _hard_metrics(targets, predictions, classes)
            report["mixed_pixels"] = {
                "pixel_count": len(mixed_agreements),
                "predicted_pixels": mixed_covered,
                "observed_label_agreement": float(np.mean(mixed_agreements))
                if mixed_agreements
                else None,
            }
            return report

        return {
            "leaf": summarize_level(None, len(items)),
            "levels": {
                name: summarize_level(level, counts[level])
                for level, name in enumerate(names)
            },
        }

    def write(self, path, constraint=None):
        report = {
            "status": self.status,
            "checkpoint": str(self.checkpoint),
            "split": "test",
            "split_file": str(self.split_path) if self.split_path else None,
            "metric_protocol": "gaussian_fused_map_before_ground_truth_override_v1",
            "partial_evaluation": self.max_windows is not None,
            "max_windows": self.max_windows,
            "constraint": constraint.tags() if constraint else None,
            "metric_notes": {
                "accuracy": "NoData counts as incorrect; conditional accuracy "
                "and coverage are also reported",
                "levels": "Ancestors of the selected map leaf; "
                "not independent head argmax",
                "probabilities": "Original fused model probabilities; constrained "
                "scores mask prohibited classes without renormalization",
                "mixed": "Observed label frequencies, not vegetation area fractions",
            },
            "coverage": {
                "test_pixels": len(self.records),
                "before_predicted_pixels": sum(
                    r["before_prediction"] is not None for r in self.records
                ),
                "after_predicted_pixels": sum(
                    r["after_prediction"] is not None for r in self.records
                ),
                "dem_missing_pixels": sum(r["dem_missing"] for r in self.records),
                "label_constraint_conflicts": sum(
                    r["label_constraint_conflict"] for r in self.records
                ),
            },
            "before_constraint": self._summary("before_prediction"),
            "after_constraint": self._summary("after_prediction"),
            "point_predictions": self.records,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return report
