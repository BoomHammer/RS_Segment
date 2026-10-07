"""Bounded profiling must not alter samples or synchronize when disabled."""

import json

import pytest
import torch

from performance import (
    TrainingProfiler,
    profile_collate,
    profile_sample,
    worker_stage,
)


class ToyDataset(torch.utils.data.Dataset):
    profile_steps = 2

    def __len__(self):
        return 4

    @profile_sample
    def __getitem__(self, index):
        return {"dynamic": self.read(index)}

    @worker_stage("read_s")
    def read(self, index):
        return torch.full((1, 1, 2, 2), float(index))


@profile_collate
def collate(samples):
    return {"dynamic": torch.stack([sample["dynamic"] for sample in samples])}


def test_disabled_no_sync_or_files(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("disabled profiler must not synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    profiler = TrainingProfiler(tmp_path, 0, torch.device("cuda:0"), 0)
    profiler.begin(1, 1, 0, None)
    profiler.mark("forward_s")
    profiler.finish()
    assert not list(tmp_path.iterdir())
    dataset = ToyDataset()
    dataset.profile_steps = 0
    assert "_performance" not in collate([dataset[0]])


def test_bounded_rank_logs(tmp_path):
    for rank in (0, 1):
        profiler = TrainingProfiler(tmp_path, rank, torch.device("cpu"), 2)
        for batch in range(4):
            profiler.begin(1, batch + 1, 0.5, {"samples": []})
            profiler.mark("forward_s")
            profiler.finish()
        records = [json.loads(line) for line in profiler.path.read_text().splitlines()]
        assert len(records) == 2
        assert all(row["rank"] == rank for row in records)
        assert all(row["iteration_s"] >= 0.5 for row in records)
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_cuda_sync_uses_only_local_device_and_stops(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", calls.append)
    device = torch.device("cuda:1")
    profiler = TrainingProfiler(tmp_path, 1, device, 1)
    for step in range(2):
        profiler.begin(1, step + 1, 0.0, None)
        profiler.mark("forward_s")
        profiler.finish()
    assert calls == [device, device, device]


@pytest.mark.parametrize("workers", [0, 2])
def test_worker_metadata_survives_collation(workers):
    loader = torch.utils.data.DataLoader(
        ToyDataset(),
        batch_size=1,
        num_workers=workers,
        collate_fn=collate,
        **({"multiprocessing_context": "spawn"} if workers else {}),
    )
    batches = list(loader)
    for index, batch in enumerate(batches):
        torch.testing.assert_close(
            batch["dynamic"], torch.full((1, 1, 1, 2, 2), float(index))
        )
    profile = batches[0]["_performance"]
    assert profile["collate_s"] >= 0
    assert profile["samples"][0]["seconds"]["read_s"] >= 0
    assert profile["samples"][0]["seconds"]["sample_total_s"] >= 0
    if not workers:
        assert "_performance" not in batches[-1]


def test_exception_resets_sample_context():
    class Broken(ToyDataset):
        @profile_sample
        def __getitem__(self, index):
            raise ValueError("broken")

    with pytest.raises(ValueError, match="broken"):
        Broken()[0]
    dataset = ToyDataset()
    dataset.profile_steps = 0
    assert "_performance" not in dataset[0]
