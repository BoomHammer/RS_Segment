"""Opt-in, bounded training diagnostics; no distributed collectives."""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from time import perf_counter

import torch

_sample_timings: ContextVar[dict | None] = ContextVar("sample_timings", default=None)


def worker_stage(name):
    """Accumulate inclusive wall time of a dataset method in the current sample."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            timings = _sample_timings.get()
            if timings is None:
                return function(*args, **kwargs)
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timings[name] = timings.get(name, 0.0) + perf_counter() - started

        return wrapped

    return decorate


def profile_sample(function):
    """Collect only the first configured samples per worker, including transforms."""

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        remaining = getattr(self, "profile_steps", 0)
        if remaining <= 0:
            return function(self, *args, **kwargs)
        self.profile_steps = remaining - 1
        timings = {}
        token = _sample_timings.set(timings)
        started = perf_counter()
        try:
            result = function(self, *args, **kwargs)
            timings["sample_total_s"] = perf_counter() - started
            result["_performance"] = {"pid": os.getpid(), "seconds": timings}
            return result
        finally:
            _sample_timings.reset(token)

    return wrapped


def profile_collate(function):
    """Keep worker metadata on CPU and measure batching separately."""

    @wraps(function)
    def wrapped(samples, *args, **kwargs):
        profiles = [
            sample["_performance"] for sample in samples if "_performance" in sample
        ]
        if not profiles:
            return function(samples, *args, **kwargs)
        started = perf_counter()
        result = function(samples, *args, **kwargs)
        result["_performance"] = {
            "samples": profiles,
            "collate_s": perf_counter() - started,
            "dynamic_shape": list(result["dynamic"].shape),
        }
        return result

    return wrapped


class TrainingProfiler:
    """Per-rank JSONL wall timings, with local CUDA synchronization at boundaries.

    Stages include CPU work and GPU/distributed waits, NOT just GPU kernel time.
    Worker timings are inclusive and overlap main-process prefetch/compute.
    """

    def __init__(self, directory: Path, rank: int, device: torch.device, steps: int):
        if steps < 0:
            raise ValueError("profile_steps must be nonnegative")
        self.steps = steps
        self.rank = rank
        self.device = device
        self.count = 0
        self.active = False
        self.path = directory / f"performance_rank{rank}_pid{os.getpid()}.jsonl"

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def begin(self, epoch, batch, data_seconds, worker):
        self.active = self.count < self.steps
        if not self.active:
            return
        self._sync()
        self.record = {
            "rank": self.rank,
            "pid": os.getpid(),
            "epoch": epoch,
            "batch": batch,
            "device": str(self.device),
            "timing_mode": "synchronized_wall_including_ddp_wait",
            "data_wait_s": data_seconds,
            "worker": worker,
        }
        self.started = self.last = perf_counter()

    def mark(self, name):
        if self.active:
            self._sync()
            now = perf_counter()
            self.record[name] = now - self.last
            self.last = now

    def finish(self):
        if not self.active:
            return
        self.mark("cleanup_s")
        self.record["step_s"] = self.last - self.started
        self.record["iteration_s"] = self.record["data_wait_s"] + self.record["step_s"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.record) + "\n")
        self.count += 1
        print(f"[PERF rank={self.rank}] " + json.dumps(self.record), flush=True)
        if self.count == self.steps:
            print(f"[PERF rank={self.rank}] complete: {self.path}", flush=True)
        self.active = False
