"""Compare three loss choices from one fixed validation-selected checkpoint."""

import argparse
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from run_accuracy_experiments import (
    clone_job,
    execute,
    prepare,
    read_json,
    score,
    train_job,
    write_json,
    write_yaml,
)


def prepare_followup(root, output, original, benchmark):
    prepare(root, output, original / "dataset")
    # Prefer throughput with at least 5 GiB system RAM left during the benchmark.
    measurements = read_json(benchmark)
    candidates = [r for r in measurements if r["system_min_available_gib"] >= 5]
    best = max(candidates, key=lambda r: r["batches_per_second"])
    settings = output / "settings"
    data = yaml.safe_load((settings / "data.yaml").read_text(encoding="utf-8"))
    data["data"]["stage2"]["io"] = {"max_open_rasters": best["max_open_rasters"]}
    write_yaml(settings / "data.yaml", data)
    initial = output / "initial.pt"
    shutil.copy2(original / "runs/lite_control/best_accuracy.pt", initial)
    shutil.copy2(benchmark, output / "throughput.json")
    train = yaml.safe_load(
        (settings / "lite_weighted_train.yaml").read_text(encoding="utf-8")
    )
    train["training"].update(epochs=12)
    train["training"]["early_stopping"].update(patience=5)
    train["optimizer"]["learning_rate"] = 3e-5
    train["dataloader"].update(
        num_workers=best["workers"], validation_num_workers=min(2, best["workers"])
    )
    train["overlap_consistency"]["warmup_epochs"] = 0
    jobs = []
    for name, normalization, gamma in [
        ("weighted_focal", "sample_mean", 2.0),
        ("weighted_ce", "sample_mean", 0.0),
        ("continued_control", "weighted_mean", 2.0),
    ]:
        model = yaml.safe_load(
            (settings / "lite_weighted_model.yaml").read_text(encoding="utf-8")
        )
        model["model"]["supervision"].update(
            weight_normalization=normalization, focal_gamma=gamma
        )
        write_yaml(settings / f"{name}_model.yaml", model)
        write_yaml(settings / f"{name}_train.yaml", train)
        jobs.append({"name": name, "seed": 42, "init_checkpoint": str(initial)})
    plan = read_json(output / "campaign.json")
    plan.update(
        jobs=jobs,
        phase="loss_finetuning",
        initial_checkpoint=str(initial),
        initial_validation_accuracy=138 / 270,
        initial_validation_macro_f1=0.2309848666,
        throughput_selected=best,
        next="Select using validation; repeat improved winner at seeds 43/44",
        seed_interpretation=(
            "Fine-tuning randomness only; all arms share the seed42 initial model"
        ),
    )
    plan["source_sha256"] = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("code", "dataset", "settings")
        for p in (output / folder).rglob("*")
        if p.is_file()
    }
    plan["source_sha256"]["initial.pt"] = hashlib.sha256(
        initial.read_bytes()
    ).hexdigest()
    write_json(output / "campaign.json", plan)
    (output / "README.md").write_text(
        "# Follow-up: class weighting and throughput\n\n"
        "The original queue was stopped on user request after 13 complete epochs. "
        "Its best validation checkpoint (epoch 10, accuracy 138/270 = 0.511111, "
        "macro F1 0.230985) initializes EVERY arm with a fresh optimizer.\n\n"
        "Three arms: sample-mean weighted Focal; sample-mean weighted CE; "
        "weighted-mean Focal continuation control. All use learning rate 3e-5, "
        "12 epochs maximum, patience 5, AMP auto, fixed split and true labels only. "
        "Halo32, stride128 and overlap consistency remain enabled.\n\n"
        "Reader/worker settings are selected by throughput.json. "
        "The reader pool reuses VRT metadata and stays bounded per worker. "
        "No resampling or pixel precision is changed.\n\n"
        "Selection uses validation only. If any arm improves over the initial "
        "accuracy, repeat the selected arm at seeds 43 and 44; these are fine-tuning "
        "seed repeats with a common initial model, not independent full trainings. "
        "Otherwise stop without opening test. No automatic full-map prediction.\n",
        encoding="utf-8",
    )


def run(root, output):
    plan = read_json(output / "campaign.json")
    for relative, expected in plan["source_sha256"].items():
        if hashlib.sha256((output / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot changed: {relative}")
    for job in plan["jobs"]:
        train_job(root, output, job)
        write_json(
            output / "interim_results.json",
            {
                item["name"]: score(output, item["name"])
                for item in plan["jobs"]
                if (output / "runs" / item["name"] / "train_log.json").is_file()
            },
        )
    selected = max((j["name"] for j in plan["jobs"]), key=lambda n: score(output, n))
    selection = {
        "selected": selected,
        "validation_scores": {
            j["name"]: score(output, j["name"]) for j in plan["jobs"]
        },
        "uses_test": False,
    }
    write_json(output / "selection.json", selection)
    if score(output, selected)[0] <= plan["initial_validation_accuracy"] + 1e-6:
        write_json(
            output / "state.json",
            {
                "status": "completed_no_improvement",
                "selection": selection,
                "test_evaluated": False,
            },
        )
        return
    repeats = [selected]
    for seed in (43, 44):
        job = clone_job(output, selected, f"selected_seed{seed}", seed=seed)
        job["init_checkpoint"] = plan["initial_checkpoint"]
        train_job(root, output, job)
        repeats.append(job["name"])
    for name in repeats:
        destination = output / "runs" / name
        execute(
            root,
            output,
            [
                output / "code/scripts/test.py",
                destination / "best_accuracy.pt",
                "--split",
                "validation",
                "--output",
                destination / "validation_verified.json",
            ],
            f"{name}_validation",
        )
        verified = read_json(destination / "validation_verified.json")
        if abs(verified["accuracy"] - score(output, name)[0]) > 1e-6:
            raise ValueError(f"Validation mismatch: {name}")
    results = []
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
            ],
            f"{name}_test",
        )
        test = read_json(destination / "test_unique_points.json")
        results.append(
            {
                "name": name,
                "validation": score(output, name),
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
            }
        )
    write_json(
        output / "results.json",
        {
            "results": results,
            "seed_interpretation": plan["seed_interpretation"],
            "test_accuracy_mean": sum(r["test_accuracy"] for r in results)
            / len(results),
        },
    )
    write_json(output / "state.json", {"status": "completed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--original", type=Path, default=Path("experiments/accuracy_20260918")
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("experiments/throughput_20260918.json")
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not args.resume:
        prepare_followup(
            root, output, args.original.resolve(), args.benchmark.resolve()
        )
    if args.prepare_only:
        return 0
    try:
        write_json(
            output / "state.json",
            {
                "status": "starting",
                "runner_pid": os.getpid(),
                "started_at": datetime.now().isoformat(),
            },
        )
        run(root, output)
    except Exception as error:
        write_json(output / "state.json", {"status": "failed", "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
