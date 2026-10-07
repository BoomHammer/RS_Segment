"""Small, opt-in distributed runtime shared by command-line stages."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

T = TypeVar("T")


def wrap_training_model(model, context):
    """Allow unused branches in supervised and overlap-only backward passes."""
    if not context.distributed:
        return model
    cuda = context.device.type == "cuda"
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[context.local_rank] if cuda else None,
        output_device=context.local_rank if cuda else None,
        find_unused_parameters=True,
    )


def _explicit_single_device(device: str | None) -> bool:
    if device is None or device in {"", "auto", "cuda"}:
        return False
    return True


def auto_launch(
    script: str | Path,
    arguments: list[str],
    *,
    device: str | None = None,
) -> int | None:
    """Relaunch one process per visible GPU when the user used a plain command.

    A non-``None`` result means the parent launcher ran the real command and the
    caller should return that exit code. Existing torchrun jobs and explicit
    single-device requests are left untouched.
    """

    if (
        "LOCAL_RANK" in os.environ
        or _explicit_single_device(device)
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        return None
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={torch.cuda.device_count()}",
        str(Path(script).resolve()),
        *arguments,
    ]
    return subprocess.run(command, check=False).returncode


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Rank metadata and the CUDA device owned by this process."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize(device: str | None = None) -> DistributedContext:
    """Initialize torch.distributed only when launched with multiple ranks."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU execution requires CUDA")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        selected = torch.device("cuda", local_rank)
    else:
        selected = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device in {None, "", "auto"}
            else device
        )
    return DistributedContext(rank, local_rank, world_size, selected)


def finalize(context: DistributedContext) -> None:
    """Close the process group created for this command."""

    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def broadcast_object(value: T, context: DistributedContext, source: int = 0) -> T:
    if not context.distributed:
        return value
    values = [value if context.rank == source else None]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def gather_objects(value: T, context: DistributedContext) -> list[T] | None:
    """Gather Python objects on rank zero without replicating them everywhere."""

    if not context.distributed:
        return [value]
    gathered = [None] * context.world_size if context.is_main else None
    dist.gather_object(value, gathered, dst=0)
    return gathered


def shard_sequence(values: list[T], context: DistributedContext) -> list[T]:
    """Return a non-padding, strided shard suitable for inference/evaluation."""

    return values[context.rank :: context.world_size]


class DistributedSamplerAdapter(Sampler):
    """Shard any deterministic project sampler into equal-length DDP streams."""

    def __init__(self, sampler: Sampler, context: DistributedContext) -> None:
        self.sampler = sampler
        self.rank = context.rank
        self.world_size = context.world_size
        self.num_samples = (len(sampler) + self.world_size - 1) // self.world_size

    @property
    def epoch(self) -> int:
        return int(getattr(self.sampler, "epoch", 0))

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.sampler.epoch = value

    def __iter__(self):
        values = list(self.sampler)
        required = self.num_samples * self.world_size
        if values and len(values) < required:
            values.extend(values[: required - len(values)])
        return iter(values[self.rank : required : self.world_size])

    def __len__(self) -> int:
        return self.num_samples


class EpochRandomSampler(Sampler):
    """Shuffle a fixed subset reproducibly from an explicit epoch counter."""

    def __init__(self, indices, seed: int = 42) -> None:
        self.indices = list(indices)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        return iter(self.indices[position] for position in order)

    def __len__(self) -> int:
        return len(self.indices)
