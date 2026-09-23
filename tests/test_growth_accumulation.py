"""DDP growth must retain the global gradient accumulated before replacement."""

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import lightning.pytorch as lit
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn.parallel import DistributedDataParallel

pytestmark = pytest.mark.skipif(
    not distributed.is_available() or not distributed.is_gloo_available(), reason="requires CPU Gloo"
)

ACCUMULATION = 6
ROW_PARAMETERS = ("rows", "rows_before", "rows_after", "rows_unused")


class Conditional(lit.LightningModule):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.before = nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.unused = nn.Parameter(torch.ones(1, dtype=torch.float64))

    def forward(self, first, rank):
        loss = self.anchor.sum()
        if first and rank == 0:
            loss = loss + self.before.sum()
        return loss


class GrowthModel(lit.LightningModule):
    def __init__(self):
        super().__init__()
        names = (*ROW_PARAMETERS, "before", "after", "alternating", "rank_zero", "rank_one", "cancel", "unused")
        for index, name in enumerate(names):
            shape = (2, 3) if name in ROW_PARAMETERS else (3,)
            values = torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(shape)
            self.register_parameter(name, nn.Parameter(values * 0.013 + 0.1 + index * 0.07))

    def forward(self, microbatch, rank, size):
        used = {
            "rows": True,
            "rows_before": microbatch == 0 and rank == 0,
            "rows_after": microbatch == ACCUMULATION - 1 and rank == 1,
            "rows_unused": False,
            "before": microbatch == 0 and rank == 0,
            "after": microbatch == ACCUMULATION - 1 and rank == 1,
            "alternating": microbatch % 2 == rank,
            "rank_zero": rank == 0,
            "rank_one": rank == 1,
            "cancel": microbatch == 0,
            "unused": False,
        }
        terms = []
        for index, (name, parameter) in enumerate(self.named_parameters()):
            if not used[name]:
                continue
            if name == "cancel":
                terms.append(parameter.sum() * (1 if rank == 0 else -1))
                continue
            values = parameter[size - 1] if parameter.ndim == 2 else parameter
            scale = 0.3 + rank * 0.2 + microbatch * 0.11 + index * 0.04
            target = 0.1 - rank * 0.13 + microbatch * 0.017
            terms.append((values * scale - target).square().sum())
        return sum(terms)


def make_optimizer(model, kind):
    groups = [
        {"params": [p for name, p in model.named_parameters() if name in ROW_PARAMETERS], "lr": 0.021},
        {"params": [p for name, p in model.named_parameters() if name not in ROW_PARAMETERS], "lr": 0.013},
    ]
    if kind == "adamw":
        return torch.optim.AdamW(groups, weight_decay=0.07, amsgrad=True, foreach=False)
    return torch.optim.SGD(groups, momentum=0.9, weight_decay=0.07, foreach=False)


@torch.no_grad()
def grow(model, optimizer, size):
    """Test-local row replacement, independent of RelFlow's binding and Resize."""
    for name in ROW_PARAMETERS:
        previous = getattr(model, name)
        if previous.shape[0] == size:
            continue
        data = torch.arange(size * 3, dtype=torch.float64).reshape(size, 3) * 0.009 + 0.17
        data[: previous.shape[0]].copy_(previous)
        expanded = nn.Parameter(data)
        if previous.grad is not None:
            expanded.grad = torch.zeros_like(expanded)
            expanded.grad[: previous.shape[0]].copy_(previous.grad)
        for group in optimizer.param_groups:
            group["params"][:] = [expanded if p is previous else p for p in group["params"]]
        state = optimizer.state.pop(previous, None)
        if state is not None:
            migrated = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.shape == previous.shape:
                    migrated[key] = torch.zeros_like(expanded)
                    migrated[key][: previous.shape[0]].copy_(value)
                else:
                    migrated[key] = value
            optimizer.state[expanded] = migrated
        setattr(model, name, expanded)


def check_gradients(model, reference):
    for (name, actual), (other_name, expected) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other_name
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            # The reference preallocates this window's final capacity before any backward.
            torch.testing.assert_close(actual.grad, expected.grad[: actual.shape[0]], atol=1e-11, rtol=1e-11)


def check_state(model, optimizer, reference, reference_optimizer):
    for (name, actual), (other_name, expected) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        state = optimizer.state.get(actual, {})
        reference_state = reference_optimizer.state.get(expected, {})
        assert state.keys() == reference_state.keys(), name
        for key in state:
            torch.testing.assert_close(state[key], reference_state[key], atol=1e-11, rtol=1e-11)


def accumulation_worker(rank, directory, bucket_view, optimizer_kind):
    from relflow.architecture.binding import synchronize_gradients

    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        # Without synchronization, rank 1 retains None even with bucket views enabled.
        conditional = Conditional()
        wrapped = DistributedDataParallel(conditional, find_unused_parameters=True, gradient_as_bucket_view=bucket_view)
        with wrapped.no_sync():
            wrapped(True, rank).backward()
        synchronize_gradients(conditional)
        torch.testing.assert_close(conditional.before.grad, torch.tensor([0.5], dtype=torch.float64))
        assert conditional.unused.grad is None
        old = wrapped
        old._remove_autograd_hooks()
        wrapped = DistributedDataParallel(conditional, find_unused_parameters=True, gradient_as_bucket_view=bucket_view)
        wrapped(False, rank).backward()
        torch.testing.assert_close(conditional.before.grad, torch.tensor([0.5], dtype=torch.float64))
        assert conditional.unused.grad is None
        wrapped._remove_autograd_hooks()

        model, reference = GrowthModel(), GrowthModel()
        optimizer = make_optimizer(model, optimizer_kind)
        reference_optimizer = make_optimizer(reference, optimizer_kind)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
        reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.85)
        groups = list(optimizer.param_groups)
        parameter_lists = [group["params"] for group in groups]
        wrapped = DistributedDataParallel(
            model, find_unused_parameters=True, gradient_as_bucket_view=bucket_view, bucket_cap_mb=0.0001
        )
        retired = []
        replacements = 0
        for length, growth_points in ((6, ()), (6, (1, 2, 4, 5)), (3, (1, 2))):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            size = model.rows.shape[0]
            capacities = []
            for microbatch in range(length):
                size += microbatch in growth_points
                capacities.append(size)
            # This reference never replaces parameters or migrates gradients mid-window.
            grow(reference, reference_optimizer, capacities[-1])
            for microbatch, size in enumerate(capacities):
                if microbatch in growth_points:
                    old = wrapped
                    old._check_reducer_finalized()
                    synchronize_gradients(model)
                    check_gradients(model, reference)
                    old._remove_autograd_hooks()
                    retired.append(old)  # Hook cleanup must work even while old wrappers remain alive.
                    grow(model, optimizer, size)
                    check_gradients(model, reference)
                    wrapped = DistributedDataParallel(
                        model,
                        find_unused_parameters=True,
                        gradient_as_bucket_view=bucket_view,
                        bucket_cap_mb=0.0001,
                    )
                    check_gradients(model, reference)
                    replacements += 1
                context = wrapped.no_sync() if microbatch < length - 1 else nullcontext()
                with context:
                    (wrapped(microbatch, rank, size) / ACCUMULATION).backward()
                sum(reference(microbatch, r, size) / (2 * ACCUMULATION) for r in range(2)).backward()

            check_gradients(model, reference)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.37)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=0.37)
            optimizer.step()
            reference_optimizer.step()
            scheduler.step()
            reference_scheduler.step()
            check_state(model, optimizer, reference, reference_optimizer)
            assert scheduler.state_dict() == reference_scheduler.state_dict()
            assert scheduler.optimizer is optimizer
            for index, group in enumerate(optimizer.param_groups):
                assert group is groups[index]
                assert group["params"] is parameter_lists[index]
                assert group["lr"] == reference_optimizer.param_groups[index]["lr"]
        assert replacements == 6
        assert scheduler.last_epoch == 3
        wrapped._remove_autograd_hooks()
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("bucket_view", [False, True], ids=["separate_gradients", "bucket_views"])
@pytest.mark.parametrize("optimizer_kind", ["sgd", "adamw"])
def test_growth_preserves_accumulation_across_repeated_ddp_replacement(
    tmp_path, monkeypatch, bucket_view, optimizer_kind
):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    multiprocessing.spawn(accumulation_worker, args=(str(tmp_path), bucket_view, optimizer_kind), nprocs=2, join=True)
