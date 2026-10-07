"""Exercise the real model's supervised/overlap backward sequence on two ranks."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.overlap import overlapping_view
from distributed_runtime import DistributedContext, wrap_training_model
from losses.overlap import overlap_consistency_loss
from losses.supervision import combined_supervision_loss
from models.architecture import SegFormerUtae


def _worker(rank, rendezvous, fixed):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(10)
        contract = {
            "architecture": "lightweight_dual_branch",
            "derived": {
                "dynamic_features_count": 1,
                "static_features_count": 1,
                "num_classes": 3,
                "fine_to_coarse": [0, 1, 1],
            },
            "temporal": {"projection_channels": 4},
            "static": {"projection_channels": 4},
            "fusion": {"output_channels": 8},
        }
        raw = SegFormerUtae.from_contract(contract)
        names = [
            name
            for name, parameter in raw.named_parameters()
            if parameter.requires_grad
        ]
        assert names[6:8] == ["adapter.fusion.0.weight", "adapter.fusion.0.bias"]
        model = (
            wrap_training_model(
                raw, DistributedContext(rank, rank, 2, torch.device("cpu"))
            )
            if fixed
            else torch.nn.parallel.DistributedDataParallel(raw)
        )
        optimizer = torch.optim.SGD(raw.parameters(), lr=0.01)
        torch.manual_seed(20 + rank)
        batch = {
            "dynamic": torch.randn(1, 2, 1, 32, 32),
            "static": torch.randn(1, 1, 32, 32),
            "dynamic_mask": torch.ones(1, 2, 1, dtype=torch.bool),
            "dynamic_time_mask": torch.ones(1, 2, dtype=torch.bool),
            "time_encoding": torch.zeros(1, 2, 3),
            "ground_truth": torch.full((1, 32, 32), rank + 1),
            "weak_label": torch.ones(1, 32, 32, dtype=torch.long),
        }
        for key in (
            "valid_mask",
            "core_mask",
            "dynamic_valid_mask",
            "static_valid_mask",
            "ground_truth_mask",
            "weak_label_mask",
        ):
            batch[key] = torch.ones(1, 32, 32, dtype=torch.bool)
        # Four microbatches, two optimizer steps, intermittent second backward.
        for step in range(4):
            output = model(batch)
            loss = combined_supervision_loss(output, batch, fine_to_coarse=[0, 1, 1])[
                "loss"
            ]
            (loss / 2).backward()
            if step % 2 == 0:
                view, top, left = overlapping_view(batch, 8)
                cropped = model(view)
                consistency = overlap_consistency_loss(
                    output["fine_logits"],
                    cropped["fine_logits"],
                    view["valid_mask"] & view["core_mask"],
                    top,
                    left,
                    border=2,
                )
                (0.1 * consistency / 2).backward()
            assert raw.adapter.fusion[0].weight.grad is None
            if step % 2 == 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for parameter in raw.parameters():
            expected = parameter.detach().clone()
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_two_rank_supervision_overlap_and_accumulation(tmp_path):
    mp.spawn(
        _worker, args=((tmp_path / "ddp_init").as_uri(), True), nprocs=2, join=True
    )


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_original_ddp_configuration_reproduces_reduction_error(tmp_path):
    with pytest.raises(
        mp.ProcessRaisedException, match="Expected to have finished reduction"
    ):
        mp.spawn(
            _worker, args=((tmp_path / "ddp_init").as_uri(), False), nprocs=2, join=True
        )
