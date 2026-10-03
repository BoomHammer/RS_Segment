"""Train or resume a model and automatically run whole-grid prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(root: Path, module: str, arguments: list[str]) -> None:
    """Run one project stage while preserving its live output."""
    command = [sys.executable, "-u", "-m", f"scripts.{module}", *arguments]
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=root, check=True)


def _latest_experiment(root: Path, before: set[Path]) -> Path:
    experiments = root / "experiments"
    candidates = sorted(
        (
            path
            for path in experiments.iterdir()
            if path.is_dir() and path.resolve() not in before
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError("训练未生成新的 experiments 实验目录")
    return candidates[-1]


def _checkpoint(experiment: Path) -> Path:
    metadata = experiment / "train_log.json"
    if metadata.is_file():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        configured = payload.get("checkpoint")
        if configured:
            candidate = experiment / str(configured)
            if candidate.is_file():
                return candidate
    candidates = sorted(experiment.glob("model_*.pt"))
    if not candidates:
        raise RuntimeError(f"训练未生成推理 checkpoint: {experiment}")
    return candidates[-1]


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--predict-config", type=Path, default=Path("configs/predict.yaml")
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)


def _train_and_predict(
    root: Path,
    run: Path,
    train_args: list[str],
    predict_config: Path,
    device: str | None,
) -> Path:
    before = (
        {path.resolve() for path in (root / "experiments").iterdir()}
        if (root / "experiments").is_dir()
        else set()
    )
    _run(root, "train", [str(run), *train_args])
    if "--resume" in train_args:
        experiment = Path(train_args[train_args.index("--resume") + 1]).resolve().parent
    elif "--output-dir" in train_args:
        experiment = Path(train_args[train_args.index("--output-dir") + 1]).resolve()
    else:
        experiment = _latest_experiment(root, before)
    checkpoint = _checkpoint(experiment)
    predict_args = [str(checkpoint), "--config", str(predict_config.resolve())]
    if device is not None:
        predict_args.extend(["--device", device])
    _run(root, "predict", predict_args)
    print(f"\n训练和预测完成（{datetime.now().isoformat()}）: {experiment}")
    return checkpoint


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练模型并自动完成全图预测")
    parser.add_argument("run", type=Path, help="datasets.py 生成的数据集目录")
    _common_parser(parser)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    run = args.run.resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"数据集目录不存在: {run}")
    train_args = [
        "--data-config",
        str(args.data_config.resolve()),
        "--config",
        str(args.model_config.resolve()),
        "--train-config",
        str(args.train_config.resolve()),
    ]
    if args.epochs is not None:
        train_args.extend(["--epochs", str(args.epochs)])
    if args.device is not None:
        train_args.extend(["--device", args.device])
    _train_and_predict(root, run, train_args, args.predict_config, args.device)
    return 0


def resume_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="续训模型并自动完成全图预测")
    parser.add_argument("resume", type=Path, help="训练实验目录中的 last.pt")
    parser.add_argument(
        "--predict-config", type=Path, default=Path("configs/predict.yaml")
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    resume = args.resume.resolve()
    if not resume.is_file():
        raise FileNotFoundError(f"续训断点不存在: {resume}")
    metadata_path = resume.parent / "train_log.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"断点目录缺少 train_log.json: {resume.parent}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_run = metadata.get("source_run")
    if not source_run:
        raise ValueError(f"训练日志缺少 source_run: {metadata_path}")
    run = Path(source_run).resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"训练数据目录不存在: {run}")
    train_args = ["--resume", str(resume)]
    if args.device is not None:
        train_args.extend(["--device", args.device])
    _train_and_predict(root, run, train_args, args.predict_config, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
