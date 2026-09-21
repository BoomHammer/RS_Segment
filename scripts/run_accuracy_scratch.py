"""Prepare a scratch loss comparison; training requires an explicit --run."""

import argparse
import copy
import hashlib
from datetime import datetime
from pathlib import Path

import yaml

from run_accuracy_experiments import (
    prepare,
    read_json,
    score,
    train_job,
    write_json,
    write_yaml,
)


def prepare_scratch(root, output, previous):
    prepare(root, output, previous / "dataset")
    settings = output / "settings"
    # Inherit the measured I/O settings, but keep the new snapshot's data paths.
    data = yaml.safe_load((settings / "data.yaml").read_text(encoding="utf-8"))
    old_data = yaml.safe_load(
        (previous / "settings/data.yaml").read_text(encoding="utf-8")
    )
    data["data"]["stage2"]["io"] = old_data["data"]["stage2"]["io"]
    write_yaml(settings / "data.yaml", data)
    model = yaml.safe_load(
        (previous / "settings/weighted_ce_model.yaml").read_text(encoding="utf-8")
    )
    train = yaml.safe_load(
        (previous / "settings/weighted_ce_train.yaml").read_text(encoding="utf-8")
    )
    train["training"].update(epochs=30, seed=42, amp_dtype="auto")
    train["training"]["early_stopping"].update(patience=8)
    train["optimizer"]["learning_rate"] = 1e-4
    train["overlap_consistency"]["warmup_epochs"] = 3
    jobs = []
    for name, weighting, gamma in (
        ("scratch_ce", "none", 0.0),
        ("scratch_weighted_ce", "train_inverse_sqrt", 0.0),
        ("scratch_weighted_focal", "train_inverse_sqrt", 2.0),
    ):
        arm_model, arm_train = copy.deepcopy(model), copy.deepcopy(train)
        arm_model["model"]["supervision"].update(
            weight_normalization="sample_mean", focal_gamma=gamma
        )
        arm_train["supervision_policy"]["class_weighting"] = weighting
        write_yaml(settings / f"{name}_model.yaml", arm_model)
        write_yaml(settings / f"{name}_train.yaml", arm_train)
        jobs.append({"name": name, "seed": 42})
    # Remove unused settings created by the general snapshot preparer.
    for name in ("lite_control", "lite_weighted", "full_weighted"):
        for kind in ("train", "model"):
            (settings / f"{name}_{kind}.yaml").unlink()
    plan = read_json(output / "campaign.json")
    plan.update(
        jobs=jobs,
        phase="scratch_loss_comparison",
        initialization="independent scratch runs, same seed; no checkpoint loading",
        next="Review validation results before authorizing repeats or test evaluation",
        reference_validation_accuracy=138 / 270,
        target_correct_points=149,
        auto_test=False,
    )
    plan["source_sha256"] = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("code", "dataset", "settings")
        for p in (output / folder).rglob("*")
        if p.is_file()
    }
    write_json(output / "campaign.json", plan)
    write_json(output / "state.json", {"status": "prepared", "training_started": False})
    (output / "README.md").write_text(
        "# Scratch loss comparison\n\n"
        "Prepared only. Training requires --run. See PLAN.md for the experiment "
        "rationale and exact launch command.\n\n"
        "Three seed42 arms: unweighted CE, train-weighted CE, train-weighted Focal. "
        "All start from scratch with the same architecture and training budget. "
        "AMP auto; 30 epochs maximum, patience8 on validation accuracy, LR1e-4. "
        "Same owned-point split; no weak labels; halo32/stride128 retained. "
        "No automatic test evaluation, seed expansion, or full-map prediction.\n",
        encoding="utf-8",
    )


def run(root, output):
    plan = read_json(output / "campaign.json")
    if plan["phase"] != "scratch_loss_comparison":
        raise ValueError("Expected a scratch loss comparison snapshot")
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
    write_json(
        output / "state.json",
        {
            "status": "completed_validation_only",
            "finished_at": datetime.now().isoformat(),
            "test_evaluated": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("experiments/accuracy_followup_20260918"),
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not args.run:
        prepare_scratch(root, output, args.previous.resolve())
        return 0
    try:
        run(root, output)
    except Exception as error:
        write_json(output / "state.json", {"status": "failed", "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
