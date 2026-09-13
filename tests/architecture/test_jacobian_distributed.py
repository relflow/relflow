from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import lightning.pytorch as lit
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from lightning.fabric.plugins.environments import TorchElasticEnvironment
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset


class Objectives(lit.LightningModule):
    """Local task gradients agree while the globally averaged task gradients conflict."""

    def __init__(self):
        super().__init__()
        self.point = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
        self.optimizer = None
        self.expected_gradients = []

    def training_step(self, batch, batch_idx):
        from torchjd.aggregation import UPGrad, UPGradWeighting

        from relflow.architecture.jacobian import combine

        scale = 1.0 + batch_idx / 10
        local = (
            self.point.new_tensor([[2.0, 0.0], [0.0, 1.0]] if self.global_rank == 0 else [[0.0, 1.0], [-2.0, 0.0]])
            * scale
        )
        averaged = self.point.new_tensor([[1.0, 0.5], [-1.0, 0.5]]) * scale
        preferences = self.point.new_ones(2)
        expected = UPGrad(pref_vector=preferences)(averaged)
        wrong = UPGradWeighting(pref_vector=preferences)(torch.eye(2, dtype=self.point.dtype) * 2.5 * scale**2)
        assert (expected - averaged.T @ wrong).abs().max() > 0.5

        loss = combine(self, list(local @ self.point + 3.0), self.point.sum() * 0.0)
        local_gradient = torch.autograd.grad(loss, self.point, retain_graph=True)[0]
        weights = torch.linalg.solve(local.T, local_gradient)
        received = [torch.empty_like(weights) for _ in range(2)]
        distributed.all_gather(received, weights)
        torch.testing.assert_close(received[0], received[1], rtol=0, atol=1e-12)
        torch.testing.assert_close(averaged.T @ weights, expected, rtol=1e-12, atol=1e-12)
        self.expected_gradients.append(expected)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def train_distributed(rank: int, directory: str, accumulation: int) -> None:
    torch.set_num_threads(1)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
    )
    distributed.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        from relflow.architecture.jacobian import gramian

        # One parameter is locally private on both ranks but belongs to
        # different objectives. Its cross-objective product must be retained.
        parameters = [torch.zeros(size, dtype=torch.float64) for size in (2, 1, 3)]
        rows = (
            [(torch.tensor([1.0, 2.0], dtype=torch.float64), torch.tensor([2.0]), None), (None, None, None)]
            if rank == 0
            else [(None, None, None), (torch.tensor([-1.0, 3.0], dtype=torch.float64), None, None)]
        )
        averaged = torch.tensor([[0.5, 1.0, 1.0], [-0.5, 1.5, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(gramian(rows, parameters, 1.0), averaged @ averaged.T, rtol=0, atol=0)

        # Flush shared and private buckets, then reinsert both after another
        # task. Local support differs, but every rank must reduce in one order.
        specifications = [
            (524_288, (1.0, None), (None, 2.0)),
            (1_048_577, (2.0, None), (None, None)),
            (3, (None, None), (None, 2.0)),
            (3, (None, None), (4.0, None)),
            (3, (2.0, -1.0), (None, 3.0)),
            (17, (None, None), (None, None)),
        ]
        parameters = [torch.zeros(size) for size, _, _ in specifications]
        rows = [
            tuple(
                None if values[rank][task] is None else torch.full((size,), values[rank][task])
                for size, *values in specifications
            )
            for task in range(2)
        ]
        expected = torch.zeros(2, 2)
        for size, left, right in specifications:
            averaged = torch.tensor([(a or 0.0) + (b or 0.0) for a, b in zip(left, right, strict=True)]) / 2
            expected.add_(size * torch.outer(averaged, averaged))
        torch.testing.assert_close(gramian(rows, parameters, 1.0), expected, rtol=0, atol=0)
        del parameters, rows

        model = Objectives()
        trainer = lit.Trainer(
            accelerator="cpu",
            devices=2,
            strategy=DDPStrategy(
                process_group_backend="gloo",
                cluster_environment=TorchElasticEnvironment(),
            ),
            precision="64-true",
            max_epochs=1,
            accumulate_grad_batches=accumulation,
            use_distributed_sampler=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, DataLoader(TensorDataset(torch.arange(4)), batch_size=1))

        expected = model.point.new_tensor([0.2, -0.3]) - (
            0.1 * torch.stack(model.expected_gradients).sum(dim=0) / accumulation
        )
        torch.testing.assert_close(model.point, expected, rtol=1e-12, atol=1e-12)
        received = [torch.empty_like(model.point) for _ in range(2)]
        distributed.all_gather(received, model.point.detach())
        torch.testing.assert_close(received[0], received[1], rtol=0, atol=1e-12)
        assert trainer.global_step == 4 // accumulation
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("accumulation", [1, 2])
def test_ddp_averages_task_jacobians_before_upgrad_and_accumulates_updates(tmp_path, accumulation):
    pytest.importorskip("torchjd")
    pytest.importorskip("quadprog")
    if not distributed.is_available() or not distributed.is_gloo_available():
        pytest.skip("CPU distributed Jacobian descent requires Gloo")

    multiprocessing.spawn(train_distributed, args=(str(tmp_path), accumulation), nprocs=2, join=True)
