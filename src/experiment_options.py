"""Shared command-line controls for experiments on prepared datasets."""

MODEL_CHOICES = {
    "segformer-utae": "segformer_utae_pretrained",
    "utae": "utae",
    "segformer": "segformer",
    "maestro": "maestro_s",
    "anysat": "anysat",
    "tsvit": "tsvit_adapted",
    "segnext": "segnext_tiny",
    "unet": "unet_resnet34",
    "lightweight": "lightweight_dual_branch",
}

EXPERIMENT_CHOICES = tuple(f"E{number:02d}" for number in range(8, 16))


def add_experiment_arguments(parser):
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None)
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENT_CHOICES,
        default=None,
        help="运行 E08-E15 的固定实验变体（自动使用 SegFormer+U-TAE）",
    )
    parser.add_argument(
        "--no-pseudo-labels", action="store_true", help="仅使用真实标签监督"
    )
    parser.add_argument(
        "--train-on-test",
        action="store_true",
        help="合并已有 train/test，保留 validation，跳过最终测试",
    )


def experiment_arguments(args):
    result = []
    if args.model is not None:
        result.extend(["--model", args.model])
    if args.experiment is not None:
        result.extend(["--experiment", args.experiment])
    for name in ("no_pseudo_labels", "train_on_test"):
        if getattr(args, name):
            result.append("--" + name.replace("_", "-"))
    return result
