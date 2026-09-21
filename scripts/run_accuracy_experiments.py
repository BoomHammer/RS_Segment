"""Run a sequential, snapshotted real-point accuracy experiment campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def prepare(root, output, run):
    """Freeze code, labels, split, metadata and settings before launching work."""
    output.mkdir(parents=True, exist_ok=False)
    code = output / "code"
    for folder in ("src", "scripts"):
        for source in (root / folder).rglob("*.py"):
            destination = code / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    data = output / "dataset"
    data.mkdir()
    for pattern in ("label_mapping*.json", "raster_stats*.json", "spatial_split.json"):
        for source in run.glob(pattern):
            shutil.copy2(source, data / source.name)
    index = read_json(run / "sample_index.json")
    labels = Path(index["ground_truth"]["path"])
    shutil.copy2(labels, data / labels.name)
    index["ground_truth"]["path"] = str(data / labels.name)
    # This phase intentionally excludes historical, all-prompt weak labels.
    index["weak_label"] = None
    write_json(data / "sample_index.json", index)
    settings = output / "settings"
    settings.mkdir()
    # The previous experiment has absolute raw-data and value-range paths.
    payload = yaml.safe_load(
        (root / "experiments/20260910_074604/data.yaml").read_text(encoding="utf-8")
    )
    payload["data"]["label_file"] = str(data / labels.name)
    payload["data"]["processed"] = str(data)
    payload["data"]["stage2"]["split_file"] = str(data / "spatial_split.json")
    payload["data"]["stage2"]["training"]["amp_dtype"] = "auto"
    write_yaml(settings / "data.yaml", payload)
    train = yaml.safe_load((root / "configs/train.yaml").read_text(encoding="utf-8"))
    train["training"].update(epochs=30, amp_dtype="auto", seed=42)
    train["training"]["early_stopping"].update(
        monitor="val_accuracy", patience=8, min_delta=0.0
    )
    train["dataloader"].update(
        batch_size=2,
        validation_batch_size=1,
        num_workers=2,
        validation_num_workers=1,
        prefetch_factor=1,
    )
    train["supervision_policy"].update(
        disable_weak_labels=True, class_weighting="train_inverse_sqrt"
    )
    # Same overlap regularizer for every arm; a second forward only every 8 steps.
    train["overlap_consistency"] = {
        "weight": 0.05,
        "every_n_batches": 8,
        "warmup_epochs": 3,
        "crop_margin": 32,
    }
    models = {}
    for name, filename in (
        ("lite", "diagnostic_model_main.yaml"),
        ("full", "diagnostic_model_full_groupnorm.yaml"),
    ):
        model = yaml.safe_load(
            (root / "configs" / filename).read_text(encoding="utf-8")
        )
        model["model"]["regularization"]["dropout"] = 0.1
        model["model"]["supervision"].update(
            weak_label_weight=0.0, focal_gamma=2.0, weight_normalization="sample_mean"
        )
        if "pretrained" in model["model"]:
            model["model"]["pretrained"]["path"] = str(
                root / "third_party/pretrained/mit-b1"
            )
        models[name] = model
    specifications = [
        ("lite_control", "lite", "weighted_mean"),
        ("lite_weighted", "lite", "sample_mean"),
        ("full_weighted", "full", "sample_mean"),
    ]
    jobs = []
    for name, architecture, normalization in specifications:
        model = copy.deepcopy(models[architecture])
        model["model"]["supervision"]["weight_normalization"] = normalization
        write_yaml(settings / f"{name}_model.yaml", model)
        write_yaml(settings / f"{name}_train.yaml", train)
        jobs.append({"name": name, "architecture": architecture, "seed": 42})
    sources = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (code, data, settings)
        for p in folder.rglob("*")
        if p.is_file()
    }
    plan = {
        "created_at": datetime.now().isoformat(),
        "workspace": str(root),
        "source_run": str(run),
        "status": "prepared",
        "jobs": jobs,
        "selection": "validation accuracy, then macro F1; never test metrics",
        "next": "CE on best weighted architecture; repeat selected arm at seeds 43/44",
        "weak_labels": "disabled; clean training-only pseudo labels belong to phase 2",
        "source_sha256": sources,
    }
    write_json(output / "campaign.json", plan)
    (output / "logs").mkdir()
    (output / "README.md").write_text(
        "# Accuracy campaign, phase 1\n\n"
        "Sequential GPU jobs; AMP auto; fixed 815/270/273 split.\n"
        "Real training points only. Historical pseudo labels are excluded.\n"
        "All arms retain halo32/stride128, augmentation and overlap consistency.\n"
        "Control vs corrected weighting, then lightweight vs full GroupNorm model, "
        "then CE vs Focal on the best weighted architecture.\n"
        "Select on validation accuracy; macro F1 breaks ties. "
        "Repeat the selected configuration at seeds 43 and 44. "
        "Only then evaluate all three seeds on test, without selecting a test winner.\n"
        "Data rasters are streamed from original paths; code and small metadata are "
        "snapshotted. campaign.json records hashes.\n\n"
        "Live progress: state.json and runs/*/train_log.json. "
        "Logs: logs/*.log. Results: results.json. "
        "No accuracy target is assumed achieved by launching these runs.\n",
        encoding="utf-8",
    )


def execute(root, output, arguments, log_name):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(output / "code/src"),
        PYTHONUNBUFFERED="1",
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        GDAL_CACHEMAX="128",
    )
    command = [sys.executable, "-u", *map(str, arguments)]
    with (output / "logs" / f"{log_name}.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        child = subprocess.Popen(
            command, cwd=root, env=env, stdout=stream, stderr=stream
        )
        write_json(
            output / "state.json",
            {
                "status": "running",
                "task": log_name,
                "child_pid": child.pid,
                "runner_pid": os.getpid(),
                "updated_at": datetime.now().isoformat(),
            },
        )
        returncode = child.wait()
    if returncode:
        raise RuntimeError(f"{log_name} failed ({returncode}); inspect its log")


def train_job(root, output, job):
    name = job["name"]
    destination = output / "runs" / name
    history = destination / "train_log.json"
    if history.exists() and read_json(history).get("status") == "completed":
        return
    arguments = [
        output / "code/scripts/train.py",
        output / "dataset",
        "--data-config",
        output / "settings/data.yaml",
        "--config",
        output / "settings" / f"{name}_model.yaml",
        "--train-config",
        output / "settings" / f"{name}_train.yaml",
        "--output-dir",
        destination,
        "--device",
        "cuda",
    ]
    if (destination / "last.pt").exists():
        arguments.extend(["--resume", destination / "last.pt"])
    elif job.get("init_checkpoint"):
        arguments.extend(["--init-checkpoint", job["init_checkpoint"]])
    execute(root, output, arguments, name)


def score(output, name):
    epochs = read_json(output / "runs" / name / "train_log.json")["epochs"]
    # Checkpoint selection takes the earliest epoch at maximum accuracy.
    best = max(epochs, key=lambda e: e["val_accuracy"])
    return best["val_accuracy"], best["val_macro_f1"]


def clone_job(output, original, name, *, gamma=None, seed=42):
    settings = output / "settings"
    model = yaml.safe_load(
        (settings / f"{original}_model.yaml").read_text(encoding="utf-8")
    )
    train = yaml.safe_load(
        (settings / f"{original}_train.yaml").read_text(encoding="utf-8")
    )
    if gamma is not None:
        model["model"]["supervision"]["focal_gamma"] = gamma
    train["training"]["seed"] = seed
    write_yaml(settings / f"{name}_model.yaml", model)
    write_yaml(settings / f"{name}_train.yaml", train)
    return {"name": name, "seed": seed, "derived_from": original}


def run_campaign(root, output):
    plan = read_json(output / "campaign.json")
    # Fail rather than silently mixing code/data revisions on restart.
    for relative, expected in plan["source_sha256"].items():
        if hashlib.sha256((output / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Campaign snapshot changed: {relative}")
    for job in plan["jobs"]:
        train_job(root, output, job)
    weighted = max(("lite_weighted", "full_weighted"), key=lambda n: score(output, n))
    ce = clone_job(output, weighted, "selected_ce", gamma=0.0)
    train_job(root, output, ce)
    names = [job["name"] for job in plan["jobs"]] + [ce["name"]]
    selected = max(names, key=lambda n: score(output, n))
    write_json(
        output / "selection.json",
        {
            "selected": selected,
            "selection_uses_test": False,
            "validation_scores": {name: score(output, name) for name in names},
        },
    )
    repeats = [selected]
    for seed in (43, 44):
        job = clone_job(output, selected, f"selected_seed{seed}", seed=seed)
        train_job(root, output, job)
        repeats.append(job["name"])
    # Verify standalone validation agrees before opening the test split.
    for name in repeats:
        destination = output / "runs" / name
        checkpoint = destination / "best_accuracy.pt"
        execute(
            root,
            output,
            [
                output / "code/scripts/test.py",
                checkpoint,
                "--split",
                "validation",
                "--output",
                destination / "validation_verified.json",
                "--device",
                "cuda",
            ],
            f"{name}_validation",
        )
        report = read_json(destination / "validation_verified.json")
        if abs(report["accuracy"] - score(output, name)[0]) > 1e-6:
            raise ValueError(f"Standalone validation disagrees for {name}")
    for name in repeats:
        destination = output / "runs" / name
        execute(
            root,
            output,
            [
                output / "code/scripts/test.py",
                destination / "best_accuracy.pt",
                "--split",
                "test",
                "--output",
                destination / "test_unique_points.json",
                "--device",
                "cuda",
            ],
            f"{name}_test",
        )
    results = []
    for name in repeats:
        test = read_json(output / "runs" / name / "test_unique_points.json")
        results.append(
            {
                "name": name,
                "validation": score(output, name),
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_points": test["unique_point_count"],
            }
        )
    write_json(
        output / "results.json",
        {
            "selected": selected,
            "runs": results,
            "test_accuracy_mean": sum(r["test_accuracy"] for r in results)
            / len(results),
            "phase2": "Generate training-only pseudo labels before weak-label trials",
        },
    )
    write_json(output / "state.json", {"status": "completed", "runs": repeats})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run", type=Path, default=Path("data/processed/20260909_190125")
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not args.resume:
        prepare(root, output, args.run.resolve())
    if args.prepare_only:
        return 0
    try:
        run_campaign(root, output)
    except Exception as error:
        write_json(output / "state.json", {"status": "failed", "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
