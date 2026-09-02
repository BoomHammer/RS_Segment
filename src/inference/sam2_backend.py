"""SAM2 image-predictor backend for the project PointSAM protocol."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference.pointsam import PointSAMPrediction, PointSAMRequest


def _quiet_tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
    """Disable SAM2's per-frame progress bars; outer progress remains visible."""

    return iterable


def prepare_sam2_image(
    image: np.ndarray,
    *,
    channels: Sequence[int] = (0, 1, 2),
    input_range: tuple[float, float] = (0.0, 255.0),
) -> np.ndarray:
    """Convert CHW/HWC imagery to RGB uint8 without per-image normalization."""

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("SAM2 输入影像必须是三维 CHW 或 HWC 数组")
    if len(channels) != 3 or len(set(channels)) != 3:
        raise ValueError("channels 必须包含三个不重复的波段索引")
    if array.shape[0] >= max(channels) + 1 and array.shape[-1] != 3:
        selected = array[list(channels), :, :].transpose(1, 2, 0)
    elif array.shape[-1] >= max(channels) + 1:
        selected = array[:, :, list(channels)]
    else:
        raise ValueError("channels 超出输入影像波段范围")
    if not np.isfinite(selected).all():
        raise ValueError("SAM2 输入影像不能包含 NaN 或 Inf")
    lower, upper = (float(value) for value in input_range)
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("input_range 必须是两个有限且递增的数")
    scaled = (selected.astype(np.float32) - lower) * (255.0 / (upper - lower))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def build_sam2_prompts(
    request: PointSAMRequest,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert project row/column prompts to SAM2 x/y coordinates and labels."""

    points: list[tuple[float, float]] = []
    labels: list[int] = []
    for seed in request.positive_points:
        points.append((float(seed.column), float(seed.row)))
        labels.append(1)
    for candidate in request.negative_points:
        points.append((float(candidate.column), float(candidate.row)))
        labels.append(0)
    if not points:
        raise ValueError("PointSAM 至少需要一个正样本或负样本点")
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


class SAM2Inferencer:
    """Concrete SAM2 backend implementing PointSAMInferencer."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config_file: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str | torch.device | None = None,
        channels: Sequence[int] = (0, 1, 2),
        input_range: tuple[float, float] = (0.0, 255.0),
        multimask_output: bool = True,
        use_amp: bool = True,
        use_video: bool = False,
    ) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM2 权重不存在: {checkpoint_path}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但当前 PyTorch 不可用 CUDA")
        self.channels = tuple(int(channel) for channel in channels)
        self.input_range = input_range
        self.multimask_output = multimask_output
        self.use_amp = use_amp and self.device.type == "cuda"
        if use_video:
            try:
                from sam2.build_sam import build_sam2_video_predictor
            except ImportError as error:
                raise RuntimeError("当前 SAM2 版本不支持视频预测器") from error
            self.model = build_sam2_video_predictor(
                config_file=config_file,
                ckpt_path=str(checkpoint_path),
                device=str(self.device),
                apply_postprocessing=False,
            )
            self.video_predictor = self.model
            self._disable_video_progress()
        else:
            self.model = build_sam2(
                config_file=config_file,
                ckpt_path=str(checkpoint_path),
                device=str(self.device),
                mode="eval",
            )
            self.video_predictor = None
        self.predictor = SAM2ImagePredictor(self.model)

    def _disable_video_progress(self) -> None:
        """Silence tqdm calls inside the installed SAM2 video implementation."""

        for name, module in tuple(sys.modules.items()):
            if name.startswith("sam2") and module is not None:
                if "tqdm" in vars(module):
                    module.tqdm = _quiet_tqdm

    def _autocast(self) -> Any:
        if not self.use_amp:
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.bfloat16
        if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
            dtype = torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def predict(self, request: PointSAMRequest) -> PointSAMPrediction:
        """Run SAM2 on one RGB or multispectral window."""

        image = prepare_sam2_image(
            np.asarray(request.image),
            channels=self.channels,
            input_range=self.input_range,
        )
        point_coords, point_labels = build_sam2_prompts(request)
        with self._autocast():
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=self.multimask_output,
                return_logits=True,
            )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        mask_logits = np.asarray(masks, dtype=np.float32)
        if (
            mask_logits.ndim != 3
            or not len(scores)
            or mask_logits.shape[0] != len(scores)
        ):
            raise RuntimeError("SAM2 返回的 mask 与 score 形状不一致")
        best = int(np.argmax(scores))
        confidence = np.full(mask_logits.shape[1:], np.clip(scores[best], 0, 1))
        return PointSAMPrediction(
            mask=mask_logits[best] > 0,
            confidence=confidence,
            mask_logits=mask_logits[best],
        )

    def predict_video(
        self,
        images: Sequence[np.ndarray],
        keyframe_index: int,
        seed: Any,
    ) -> list[PointSAMPrediction]:
        """Prompt the July keyframe and propagate masks in both directions."""

        if self.video_predictor is None:
            raise RuntimeError("SAM2Inferencer 未启用视频模式")
        if not images or not 0 <= keyframe_index < len(images):
            raise ValueError("视频帧或 keyframe_index 无效")
        from tempfile import TemporaryDirectory

        from PIL import Image

        with TemporaryDirectory(prefix="sam2_video_") as directory:
            for index, image in enumerate(images):
                Image.fromarray(prepare_sam2_image(image)).save(
                    Path(directory) / f"{index:05d}.jpg",
                    format="JPEG",
                )
            state = self.video_predictor.init_state(video_path=directory)
            self.video_predictor.reset_state(state)
            points = np.asarray([[seed.column, seed.row]], dtype=np.float32)
            labels = np.asarray([1], dtype=np.int32)
            propagated: dict[int, PointSAMPrediction] = {}
            with self._autocast():
                _, object_ids, masks = self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=keyframe_index,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )
                propagated[keyframe_index] = _video_prediction_for_object(
                    masks, object_ids, seed
                )
                for reverse in (False, True):
                    for frame_idx, ids, frame_masks in (
                        self.video_predictor.propagate_in_video(
                            state,
                            start_frame_idx=keyframe_index,
                            reverse=reverse,
                        )
                    ):
                        propagated[frame_idx] = _video_prediction_for_object(
                            frame_masks, ids, seed
                        )
            return [propagated[index] for index in sorted(propagated)]


def _video_prediction_for_object(
    masks: Any, object_ids: Any, seed: Any
) -> PointSAMPrediction:
    """Extract object 1's logits and point confidence from a video result."""

    ids = np.asarray(object_ids).reshape(-1).tolist()
    mask_array = (
        masks.detach().float().cpu().numpy()
        if torch.is_tensor(masks)
        else np.asarray(masks)
    )
    object_index = ids.index(1)
    logits = np.squeeze(
        np.asarray(mask_array[object_index], dtype=np.float32)
    )
    if logits.ndim != 2:
        raise RuntimeError(
            f"SAM2 视频 mask 形状异常，目标对象 mask 应为二维，实际为 {logits.shape}"
        )
    row = min(max(int(seed.row), 0), logits.shape[0] - 1)
    column = min(max(int(seed.column), 0), logits.shape[1] - 1)
    confidence = float(1.0 / (1.0 + np.exp(-logits[row, column])))
    return PointSAMPrediction(
        mask=logits > 0,
        confidence=np.full(logits.shape, confidence, dtype=np.float32),
        mask_logits=logits,
    )


def default_sam2_paths(project_root: str | Path) -> tuple[Path, str]:
    """Return the repository's SAM2.1 small checkpoint and config name."""

    root = Path(project_root).resolve()
    return root / "SAM" / "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"
