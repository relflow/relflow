"""Automatic collection geometry may differ between real DDP ranks."""

from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel

import relflow as rf


class TrainingStep(torch.nn.Module):
    """Expose the native model step as the DDP forward/backward boundary."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        result = self.model.training_step(batch, batch_idx=0)
        assert result is not None
        return result["loss"]


def observations(length, outcome):
    return pa.Table.from_pylist(
        [
            {
                "events": [
                    {"amount": float(index + 1), "merchant": "known", "target": float(2 * index + 3)}
                    for index in range(length)
                ],
                "outcome": outcome,
            }
        ],
        schema=pa.schema(
            [
                (
                    "events",
                    pa.list_(
                        pa.struct([("amount", pa.float64()), ("merchant", pa.string()), ("target", pa.float64())])
                    ),
                ),
                ("outcome", pa.bool_()),
            ]
        ),
    )


def distributed_lengths(rank, directory):
    torch.set_num_threads(1)
    torch.manual_seed(149)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            reduction=None,
            amount=rf.Number,
            merchant=rf.Category(p_unavailable=0),
            target=rf.Number(mask=True),
        ),
        outcome=rf.Boolean(mask=True),
    )
    # Reserve vocabulary before DDP; this test isolates geometry from resource growth.
    model.encode(observations(1, True), strata="train")
    distributed.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        wrapped = DistributedDataParallel(TrainingStep(model), find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        group = optimizer.param_groups[0]
        parameter_list = group["params"]
        identities = [id(parameter) for parameter in model.parameters()]
        schema = model.schema.model_dump()
        decoder = model.nodes["/events/target"].decoder
        target = decoder.regression.weight
        queries = decoder.pool.queries
        before = target.detach().clone()
        snapshots = []
        lengths = (0, 5, 0) if rank == 0 else (3, 0, 0)
        for step, length in enumerate(lengths):
            optimizer.zero_grad(set_to_none=True)
            batch = model.encode(observations(length, bool(rank)), strata="train")
            assert batch["/events/target"].state.shape == (1, 1, length)
            assert rf.Category.vocabulary(model, "/events/merchant") == ("known",)
            loss = wrapped(batch)
            assert torch.isfinite(loss)
            loss.backward()
            if step < 2:
                assert target.grad is not None and target.grad.square().sum() > 0
                assert queries.grad is not None and torch.isfinite(queries.grad).all()
            else:
                assert target.grad is None
                assert queries.grad is None
            gradient = None if target.grad is None else target.grad.detach().clone()
            optimizer.step()
            assert optimizer.param_groups[0] is group
            assert group["params"] is parameter_list
            assert identities == [id(parameter) for parameter in model.parameters()]
            assert identities == [id(parameter) for parameter in parameter_list]
            assert queries.shape == (1, 8)
            assert model.schema.model_dump() == schema
            snapshots.append(
                {
                    "length": length,
                    "loss": loss.item(),
                    "count": rf.Number.normalization(model, "/events/amount")["count"],
                    "mean": rf.Number.normalization(model, "/events/amount")["mean"],
                    "target_gradient": gradient,
                    "target_weights": target.detach().clone(),
                    "optimizer_step": optimizer.state[target]["step"].item(),
                }
            )
        assert not torch.equal(before, target)
        torch.save(snapshots, Path(directory) / f"rank-{rank}.pt")
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not distributed.is_available() or not distributed.is_gloo_available(), reason="requires CPU Gloo")
def test_automatic_lengths_can_differ_and_swap_between_ddp_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    multiprocessing.spawn(distributed_lengths, args=(str(tmp_path),), nprocs=2, join=True)

    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    assert [[snapshot["length"] for snapshot in result] for result in ranks] == [[0, 5, 0], [3, 0, 0]]
    for result in ranks:
        assert [snapshot["count"] for snapshot in result] == [4, 9, 9]
        assert [snapshot["optimizer_step"] for snapshot in result] == [1, 2, 2]
        assert result[-1]["mean"] == pytest.approx(22 / 9)
    for left, right in zip(*ranks, strict=True):
        torch.testing.assert_close(left["target_weights"], right["target_weights"], rtol=0, atol=0)
        torch.testing.assert_close(left["target_gradient"], right["target_gradient"], rtol=0, atol=0)
