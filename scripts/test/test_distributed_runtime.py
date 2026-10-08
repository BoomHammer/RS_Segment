"""CPU tests for deterministic sharding and distributed result merging."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from scripts.weak_label import _load_shard_outcomes, _merge_weak_label_shards

import distributed_runtime
from distributed_runtime import DistributedContext, DistributedSamplerAdapter
from evaluation import merge_point_prediction_shards


class _Sampler(torch.utils.data.Sampler[int]):
    def __init__(self) -> None:
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        return iter(range(5))

    def __len__(self) -> int:
        return 5


def _context(rank: int, world_size: int = 2) -> DistributedContext:
    return DistributedContext(rank, rank, world_size, torch.device("cpu"))


@pytest.mark.parametrize("configured, expected", [(None, "1"), ("6", "6")])
def test_auto_launch_sets_quiet_omp_default(
    tmp_path, monkeypatch, configured, expected
) -> None:
    if configured is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", configured)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    captured = {}

    def run(command, *, check, env):
        captured.update(command=command, check=check, env=env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(distributed_runtime.subprocess, "run", run)
    result = distributed_runtime.auto_launch(tmp_path / "stage.py", ["--example"])

    assert result == 0
    assert captured["env"]["OMP_NUM_THREADS"] == expected
    assert "torch.distributed.run" in captured["command"]


def test_sampler_adapter_pads_equal_ddp_steps() -> None:
    first = list(DistributedSamplerAdapter(_Sampler(), _context(0)))
    second = list(DistributedSamplerAdapter(_Sampler(), _context(1)))

    assert first == [0, 2, 4]
    assert second == [1, 3, 0]


def test_outcomes_restore_input_order_without_numeric_ids(tmp_path):
    records = [{"record_index": value} for value in ("", "site-A", None, "2", "")]
    shards = []
    for rank in range(2):
        path = tmp_path / f"rank{rank}.json"
        path.write_text(json.dumps(records[rank::2]), encoding="utf-8")
        shards.append({"outcomes": path})
    assert _load_shard_outcomes(shards) == records


def test_sampler_adapter_forwards_resume_epoch() -> None:
    sampler = _Sampler()
    adapter = DistributedSamplerAdapter(sampler, _context(0))
    adapter.epoch = 7

    assert sampler.epoch == 7
    assert adapter.epoch == 7


def test_point_shards_restore_weighted_overlap_average() -> None:
    shards = [
        {
            "leaf": {(3, 4): (torch.tensor([1.0, 0.0]), 1, 0)},
            "levels": {},
        },
        {
            "leaf": {(3, 4): (torch.tensor([0.0, 2.0]), 2, 0)},
            "levels": {},
        },
    ]

    (targets, probabilities, positions, occurrences), levels = (
        merge_point_prediction_shards(shards)
    )

    assert targets.tolist() == [0]
    assert positions == [(3, 4)]
    assert occurrences == 3
    assert torch.allclose(probabilities, torch.tensor([[1 / 3, 2 / 3]]))
    assert levels == {}


def test_module_is_available_from_project_pythonpath() -> None:
    assert Path(__file__).is_file()


def test_weak_label_shards_merge_scores_and_conflicts(tmp_path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "width": 2,
        "height": 1,
        "count": 1,
        "dtype": "int32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 1, 1),
        "nodata": -9999,
    }
    shards = []
    for rank, (labels, scores) in enumerate(
        [([[1, 1]], [[0.9, 0.7]]), ([[2, 2]], [[0.6, 0.68]])]
    ):
        label_path = tmp_path / f"labels_{rank}.tif"
        score_path = tmp_path / f"scores_{rank}.tif"
        with rasterio.open(label_path, "w", **profile) as destination:
            destination.write(np.asarray(labels, dtype=np.int32), 1)
        with rasterio.open(
            score_path, "w", **{**profile, "dtype": "float32", "nodata": -1.0}
        ) as destination:
            destination.write(np.asarray(scores, dtype=np.float32), 1)
        shards.append({"labels": str(label_path), "scores": str(score_path)})

    output = tmp_path / "merged.tif"
    _merge_weak_label_shards(shards, output, nodata=-9999, conflict_margin=0.05)

    with rasterio.open(output) as dataset:
        assert dataset.read(1).tolist() == [[1, -9999]]
