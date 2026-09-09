"""Run data preparation, training, evaluation, and whole-grid prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import load_config


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
        "--resume", type=Path, default=None, help="last.pt；跳过准备数据并续训后预测"
    )
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    config = args.data_config.resolve()
    if args.resume is not None:
        resume = args.resume.resolve()
        metadata_path = resume.parent / "train_log.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"断点目录缺少 train_log.json: {resume.parent}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run = Path(metadata["source_run"]).resolve()
        if not run.is_dir():
            raise NotADirectoryError(f"训练数据目录不存在: {run}")
    else:
        processed = load_config(config).data.processed
        before = (
            {path.resolve() for path in processed.iterdir()}
            if processed.is_dir()
            else set()
        )
        _run(root, "preprocess", ["--config", str(config)])
        run = _latest_run(processed, before)
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
    if args.resume is not None:
        train_args.extend(["--resume", str(args.resume.resolve())])
    _run(root, "train", train_args)
    experiments = root / "experiments"
    experiment_candidates = sorted(
        (path for path in experiments.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not experiment_candidates:
        raise RuntimeError("训练未生成 experiments 实验目录")
    experiment = experiment_candidates[-1]
    checkpoint = next(iter(sorted(experiment.glob("model_*.pt"))), None)
    if checkpoint is None:
        raise RuntimeError(f"训练未生成 checkpoint: {experiment}")
    test_args = [str(checkpoint)]
    if args.device is not None:
        test_args.extend(["--device", args.device])
    if args.max_windows is not None:
        test_args.extend(["--max-windows", str(args.max_windows)])
    _run(root, "test", test_args)
    predict_args = [str(checkpoint), "--config", str(args.predict_config.resolve())]
    if args.device is not None:
        predict_args.extend(["--device", args.device])
    _run(root, "predict", predict_args)
    print(f"\n工作流完成（{datetime.now().isoformat()}）: {experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
