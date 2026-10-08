"""Run data preparation, training, evaluation, and whole-grid prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import load_config
from experiment_options import add_experiment_arguments, experiment_arguments


def _run(root: Path, module: str, arguments: list[str]) -> None:
    # Inherit stdout/stderr so every stage keeps its original logs and
    # tqdm progress bars. ``-u`` prevents buffering when launched by uv.
    command = [sys.executable, "-u", "-m", f"scripts.{module}", *arguments]
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=root, check=True)


def _latest_run(processed: Path, before: set[Path]) -> Path:
    candidates = sorted(
        (path for path in processed.iterdir() if path.is_dir() and path not in before),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError("预处理未生成新的 data/processed run 目录")
    return candidates[-1]


def _resume_source_run(resume: Path) -> Path:
    """Read the prepared dataset path recorded alongside a training checkpoint."""
    metadata_path = resume.parent / "train_log.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"续训实验目录缺少 train_log.json: {resume.parent}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_run = metadata.get("source_run")
    if not source_run:
        raise ValueError(f"训练日志缺少 source_run: {metadata_path}")
    run = Path(source_run).resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"续训所需的数据集目录不存在: {run}")
    return run


def _experiment_checkpoint(experiment: Path) -> Path:
    """Return the inference checkpoint produced by a training run."""
    metadata_path = experiment / "train_log.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_name = metadata.get("checkpoint")
    checkpoint = experiment / checkpoint_name if checkpoint_name else None
    if checkpoint is None or not checkpoint.is_file():
        checkpoints = sorted(experiment.glob("model_*.pt"))
        checkpoint = checkpoints[-1] if checkpoints else None
    if checkpoint is None:
        raise RuntimeError(f"训练未生成推理 checkpoint: {experiment}")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="一键完成数据准备、弱标签、数据集、训练、测试和全图预测"
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--predict-config", type=Path, default=Path("configs/predict.yaml")
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从 experiments/<时间戳>/last.pt 续训，并在完成后测试和预测",
    )
    parser.add_argument(
        "--retrain",
        type=Path,
        default=None,
        help="使用已有 data/processed run 从头训练，并在完成后测试和预测",
    )
    add_experiment_arguments(parser)
    args = parser.parse_args(argv)
    if experiment_arguments(args) and args.retrain is None:
        parser.error("实验参数仅用于 --retrain；续训自动恢复实验设置")
    if args.resume is not None and args.retrain is not None:
        parser.error("--resume 和 --retrain 不能同时使用")
    root = Path.cwd().resolve()
    resume = args.resume.resolve() if args.resume is not None else None
    retrain = args.retrain.resolve() if args.retrain is not None else None
    if resume is not None:
        if not resume.is_file() or resume.name != "last.pt":
            raise FileNotFoundError("--resume 必须指向实验目录中的 last.pt")
        run = _resume_source_run(resume)
        experiment = resume.parent
        train_args = [str(run), "--resume", str(resume)]
    elif retrain is not None:
        if not retrain.is_dir():
            raise NotADirectoryError(f"重新训练所需的数据集目录不存在: {retrain}")
        run = retrain
        experiments = root / "experiments"
        before_experiments = (
            {path.resolve() for path in experiments.iterdir()}
            if experiments.is_dir()
            else set()
        )
        config = args.data_config.resolve()
        train_args = [
            str(run),
            "--data-config",
            str(config),
            "--config",
            str(args.model_config.resolve()),
            "--train-config",
            str(args.train_config.resolve()),
        ]
    else:
        config = args.data_config.resolve()
        processed = load_config(config).data.processed
        experiments = root / "experiments"
        before_experiments = (
            {path.resolve() for path in experiments.iterdir()}
            if experiments.is_dir()
            else set()
        )
        before = (
            {path.resolve() for path in processed.iterdir()}
            if processed.is_dir()
            else set()
        )
        _run(
            root,
            "preprocess",
            ["--config", str(config), "--skip-weak-labels"],
        )
        run = _latest_run(processed, before)
        weak_label_args = [
            "--data-config",
            str(config),
            "--run-dir",
            str(run),
        ]
        if args.device is not None:
            weak_label_args.extend(["--device", args.device])
        _run(root, "weak_label", weak_label_args)
        _run(
            root,
            "datasets",
            [
                str(run),
                "--config",
                str(config),
                "--train-config",
                str(args.train_config.resolve()),
            ],
        )
        train_args = [
            str(run),
            "--data-config",
            str(config),
            "--config",
            str(args.model_config.resolve()),
            "--train-config",
            str(args.train_config.resolve()),
        ]
    if args.epochs is not None:
        train_args.extend(["--epochs", str(args.epochs)])
    if args.device is not None:
        train_args.extend(["--device", args.device])
    train_args.extend(experiment_arguments(args))
    _run(root, "train", train_args)
    if resume is None:
        experiment_candidates = sorted(
            (
                path
                for path in (root / "experiments").iterdir()
                if path.is_dir() and path.resolve() not in before_experiments
            ),
            key=lambda path: path.stat().st_mtime,
        )
        if not experiment_candidates:
            raise RuntimeError("训练未生成新的 experiments 实验目录")
        experiment = experiment_candidates[-1]
    checkpoint = _experiment_checkpoint(experiment)
    test_args = [str(checkpoint)]
    if args.device is not None:
        test_args.extend(["--device", args.device])
    if args.max_windows is not None:
        test_args.extend(["--max-windows", str(args.max_windows)])
    metadata = json.loads((experiment / "train_log.json").read_text(encoding="utf-8"))
    merged_test = metadata.get("supervision_policy", {}).get("train_on_test", False)
    if not (args.train_on_test or merged_test):
        _run(root, "test", test_args)
    predict_args = [str(checkpoint), "--config", str(args.predict_config.resolve())]
    if args.device is not None:
        predict_args.extend(["--device", args.device])
    _run(root, "predict", predict_args)
    print(f"\n工作流完成（{datetime.now().isoformat()}）: {experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
