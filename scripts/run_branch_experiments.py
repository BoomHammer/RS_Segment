"""Prepare and run training-only pseudo-label branch ablations sequentially."""

import argparse
import copy
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from run_accuracy_experiments import (
    execute,
    read_json,
    score,
    train_job,
    write_json,
    write_yaml,
)
from run_accuracy_scratch import prepare_scratch


def prepare_campaign(root, output):
    prepare_scratch(root, output, root / "experiments/accuracy_followup_20260918")
    settings = output / "settings"
    data_config = yaml.safe_load((settings / "data.yaml").read_text(encoding="utf-8"))
    data_config["data"]["sam2_checkpoint"] = str(
        root / "third_party/SAM/sam2.1_hiera_small.pt"
    )
    write_yaml(settings / "data.yaml", data_config)
    base_model = yaml.safe_load(
        (settings / "scratch_weighted_focal_model.yaml").read_text(encoding="utf-8")
    )
    base_train = yaml.safe_load(
        (settings / "scratch_weighted_focal_train.yaml").read_text(encoding="utf-8")
    )
    base_model["model"]["supervision"]["weak_label_weight"] = 0.5
    base_train["supervision_policy"]["disable_weak_labels"] = False
    base_train["optimizer"]["pretrained_learning_rate"] = 1e-5
    jobs = []
    for name, architecture in (
        ("A_light_pseudo", "segformer_utae"),
        ("B_replace_dynamic", "segformer_utae_dynamic_ablation"),
        ("C_replace_static", "segformer_utae_static_ablation"),
    ):
        model = copy.deepcopy(base_model)
        model["model"]["architecture"] = architecture
        if name.startswith("B"):
            model["model"]["temporal"].update(frame_chunk_size=2, normalization="group")
        if name.startswith("C"):
            model["model"]["pretrained"] = {
                "path": str(root / "third_party/pretrained/mit-b1"),
                "revision": "13ddceec4e8bdf401e7cd7acf5aebc526222518c",
                "freeze_stages": 2,
            }
        write_yaml(settings / f"{name}_model.yaml", model)
        write_yaml(settings / f"{name}_train.yaml", base_train)
        jobs.append({"name": name, "seed": 42})
    for name in ("scratch_ce", "scratch_weighted_ce", "scratch_weighted_focal"):
        for kind in ("train", "model"):
            (settings / f"{name}_{kind}.yaml").unlink()
    shutil.copy2(root / "configs/weak_label.yaml", settings / "weak_label.yaml")
    shutil.copy2(
        root / "third_party/SAM/provenance.json", settings / "sam_provenance.json"
    )
    index = read_json(output / "dataset/sample_index.json")
    historical = root / "data/processed/20260909_190125"
    index["weak_label"] = read_json(historical / "sample_index.json")["weak_label"]
    index["weak_label"]["path"] = str(output / "dataset/weak_labels.tif")
    write_json(output / "dataset/sample_index.json", index)
    plan = read_json(output / "campaign.json")
    plan.update(
        jobs=jobs,
        phase="single_branch_with_training_only_pseudo_labels",
        historical_dataset=str(historical),
        initialization="scratch seed42; identical shared tensors; replacement seed1042",
        next="Analyze A/B/C validation before further training; no automatic test",
        weak_labels="training prompts and blocks only; source IDs saved",
        selection="earliest maximum validation accuracy; report same-epoch macro F1",
    )
    plan["source_sha256"] = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("code", "dataset", "settings")
        for p in (output / folder).rglob("*")
        if p.is_file()
    }
    write_json(output / "campaign.json", plan)
    (output / "README.md").write_text(
        "# Single-branch experiments with training-only pseudo labels\n\n"
        "Order: real-window GPU preflight; historical coverage audit; train-only SAM "
        "generation and ownership/loader checks; A, B, C sequential training.\n\n"
        "A retains the lightweight architecture. B replaces its dynamic encoder with "
        "GroupNorm U-TAE plus stride/channel adapters. C replaces its static encoder "
        "with pretrained MiT-B1 (first two stages frozen, LR1e-5). "
        "All keep the same lightweight fusion/decoder/heads and shared initialization. "
        "All use train-weighted sample-mean Focal, pseudo weight0.5, dropout0.1, "
        "LR1e-4, seed42, max30 epochs, patience8, AMP auto, effective batch4, "
        "halo32/stride128, overlap0.05/every8/warmup3, workers6/validation2.\n\n"
        "A shares the recent scratch-weighted-Focal settings except enabling clean "
        "pseudo labels. Historical 0.54 runs used different pseudo-label provenance "
        "and are context, not an exact causal control for the new label generation. "
        "This is not a repeated simple-model capability test or a loss sweep.\n\n"
        "state.json/logs track progress. preflight.json records actual memory. "
        "labels/*coverage.json separates unique pixels from repeated occurrences. "
        "prompt_outcomes.json and training_seed_ids.tif trace label sources. "
        "label_artifacts.json hashes generated supervision before training. "
        "No automatic extra seeds, test evaluation, or map generation.\n",
        encoding="utf-8",
    )


def verify(output, hashes):
    for name, expected in hashes.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Experiment artifact changed: {name}")


def run(root, output):
    plan = read_json(output / "campaign.json")
    if plan["phase"] != "single_branch_with_training_only_pseudo_labels":
        raise ValueError("Wrong experiment type")
    verify(output, plan["source_sha256"])
    for artifact, script, name in (
        ("preflight.json", "check_branch_training.py", "preflight"),
        ("label_artifacts.json", "prepare_branch_labels.py", "training_pseudo_labels"),
    ):
        if not (output / artifact).exists():
            execute(
                root,
                output,
                [output / "code/scripts" / script, "--output", output],
                name,
            )
    labels = read_json(output / "label_artifacts.json")
    if (
        labels["status"] != "ready"
        or read_json(output / "preflight.json")["status"] != "passed"
    ):
        raise ValueError("Pre-training checks did not pass")
    for job in plan["jobs"]:
        verify(output, labels["sha256"])
        train_job(root, output, job)
        history = read_json(output / "runs" / job["name"] / "train_log.json")
        if not history["epochs"] or any(
            e["weak_label_pixels"] == 0 for e in history["epochs"]
        ):
            raise ValueError("Missing pseudo supervision during training")
        write_json(
            output / "interim_results.json",
            {
                item["name"]: score(output, item["name"])
                for item in plan["jobs"]
                if (output / "runs" / item["name"] / "train_log.json").exists()
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not args.run:
        prepare_campaign(root, output)
        return
    try:
        run(root, output)
    except Exception as error:
        write_json(output / "state.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
