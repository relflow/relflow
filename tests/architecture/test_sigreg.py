from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

import relflow as rf
from relflow.architecture import runtime
from relflow.architecture.sigreg import regularize, statistic
from relflow.structs.enums import Strata


def directions(width: int, count: int = 32) -> torch.Tensor:
    values = torch.randn(width, count, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    return values / values.norm(dim=0, keepdim=True)


KNOTS = torch.linspace(0.0, 3.0, 17, dtype=torch.float64)


def nested(sigreg: rf.SIGReg | None = None) -> rf.Model:
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        sigreg=sigreg,
        amount=rf.Number,
        items=rf.Branch(length=4, sku=rf.Category, qty=rf.Number),
        label=rf.Boolean(mask=True),
    )


def table() -> pa.Table:
    return pa.table(
        {
            "amount": [float(index) for index in range(8)],
            "items": [[{"sku": f"s{index % 3}", "qty": float(index)}] * (index % 4 + 1) for index in range(8)],
            "label": [index % 2 == 0 for index in range(8)],
        }
    )


def test_statistic_separates_gaussian_from_collapsed_and_anisotropic_samples():
    generator = torch.Generator().manual_seed(1)
    gaussian = torch.randn(4096, 8, generator=generator, dtype=torch.float64)
    basis = directions(8)

    reference = statistic(gaussian, basis, KNOTS)
    assert reference < 2.0
    assert statistic(torch.ones_like(gaussian), basis, KNOTS) > 100 * reference
    assert statistic(gaussian * 3.0, basis, KNOTS) > 100 * reference
    assert statistic(gaussian + 2.0, basis, KNOTS) > 100 * reference
    shuffled = gaussian[torch.randperm(4096, generator=generator)]
    torch.testing.assert_close(statistic(shuffled, basis, KNOTS), reference)


def test_statistic_without_a_distribution_is_a_connected_zero():
    single = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    value = statistic(single, directions(8), KNOTS)

    assert value.item() == 0.0
    value.backward()
    assert single.grad is not None


@pytest.mark.parametrize("values", [{"weight": 0.0}, {"projections": 0}, {"knots": 1}, {"weight": True}])
def test_configuration_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        rf.SIGReg(**values)


def test_session_policy_is_not_part_of_the_schema():
    model = nested(rf.SIGReg(weight=0.1))

    assert model.sigreg == rf.SIGReg(weight=0.1)
    assert "sigreg" not in model.schema.model_dump()


def test_regularize_covers_embedded_leaves_and_branches_and_repeats():
    model = nested(rf.SIGReg())
    inputs = model.encode(table(), strata=Strata.train)
    routed = {}
    runtime.forward(model, inputs, strata=Strata.train, routed=routed)
    plan = runtime.execution(model)

    assert set(routed) == {"/amount", "/items/sku", "/items/qty", "/label", "/items", "/"}
    first = regularize(model, routed, plan, batch_idx=0)
    torch.testing.assert_close(first, regularize(model, routed, plan, batch_idx=0), rtol=0, atol=0)
    assert first.isfinite() and first > 0
    assert regularize(model, routed, plan, batch_idx=1) != first

    # The skipped target never carries a token, so it neither contributes nor dilutes the mean.
    without = {address: parcel for address, parcel in routed.items() if address != "/label"}
    torch.testing.assert_close(regularize(model, without, plan, batch_idx=0), first, rtol=0, atol=0)


def test_regularize_ignores_absent_tokens():
    model = nested(rf.SIGReg())
    routed = {}
    runtime.forward(model, model.encode(table(), strata=Strata.train), strata=Strata.train, routed=routed)
    plan = runtime.execution(model)
    reference = regularize(model, routed, plan, batch_idx=0)

    items = routed["/items/sku"]
    items.payload = items.payload.masked_fill(~items.present.unsqueeze(-1), 1.0e6)
    torch.testing.assert_close(regularize(model, routed, plan, batch_idx=0), reference)


def test_training_step_adds_weighted_regularization_and_reaches_every_encoder(monkeypatch):
    baseline = nested()
    enabled = deepcopy(baseline)
    enabled.sigreg = rf.SIGReg(weight=0.5)
    losses = []
    for candidate in (baseline, enabled):
        monkeypatch.setattr(candidate, "track", lambda names, value, **kwargs: value)
        candidate.zero_grad()
        losses.append(candidate.training_step(candidate.encode(table(), strata=Strata.train), 0)["loss"])

    routed = {}
    runtime.forward(enabled, enabled.encode(table(), strata=Strata.train), strata=Strata.train, routed=routed)
    penalty = regularize(enabled, routed, runtime.execution(enabled), batch_idx=0) * 0.5
    torch.testing.assert_close(losses[1], losses[0] + penalty)

    losses[1].backward()
    for address in ("/amount", "/items/sku", "/items/qty"):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in enabled.nodes[address].embedder.parameters()
        ), address
    for address in ("/items", "/"):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in enabled.nodes[address].encoder.parameters()
        ), address


def test_disabled_policy_keeps_loss_identical(monkeypatch):
    baseline = nested()
    toggled = deepcopy(baseline)
    toggled.sigreg = rf.SIGReg()
    toggled.sigreg = None
    values = []
    for candidate in (baseline, toggled):
        monkeypatch.setattr(candidate, "track", lambda names, value, **kwargs: value)
        values.append(candidate.training_step(candidate.encode(table(), strata=Strata.train), 0)["loss"].detach())
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)


def test_prediction_does_not_regularize(monkeypatch):
    model = nested(rf.SIGReg())

    def forbidden(*args, **kwargs):
        raise AssertionError("prediction must not evaluate SIGReg")

    monkeypatch.setattr("relflow.architecture.sigreg.regularize", forbidden)
    assert model.predict(table()).height == 8


@pytest.mark.integration
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("jacobian", [False, True])
def test_fit_logs_regularization(compiled, jacobian):
    if jacobian:
        pytest.importorskip("torchjd")
        pytest.importorskip("quadprog")
    import polars as pl

    model = nested(rf.SIGReg())
    model.jacobian = jacobian
    if compiled:
        model.compile(encoders=True, backend="eager")
    frame = pl.from_arrow(table())
    datamodule = rf.PolarsDataModule(
        model=model, train=frame, validate=frame, num_workers=0, persistent_workers=False, pin_memory=False
    )
    model.fit(
        datamodule,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        batch_size=8,
        accelerator="cpu",
        enable_progress_bar=False,
        rollback=False,
        enable_checkpointing=False,
        logger=False,
    )
    metrics = model.trainer.callback_metrics
    assert metrics["sigreg/train"].isfinite()
    assert metrics["sigreg/validate"].isfinite()


def ranks(rank: int, directory: str, embeddings: torch.Tensor, basis: torch.Tensor) -> None:
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        # Uneven shards, including an empty one on the second rank, then even shards.
        for local in (embeddings if rank == 0 else embeddings[:0], embeddings[rank::2]):
            local = local.clone().requires_grad_(True)
            value = statistic(local, basis, KNOTS)
            value.backward()
            torch.save((value.detach(), local.grad), Path(directory) / f"rank{rank}-{local.shape[0]}.pt")
    finally:
        distributed.destroy_process_group()


@pytest.mark.integration
def test_distributed_statistic_matches_the_global_batch(tmp_path):
    embeddings = torch.randn(64, 8, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    basis = directions(8)
    multiprocessing.spawn(ranks, args=(str(tmp_path), embeddings, basis), nprocs=2, join=True)

    reference_input = embeddings.clone().requires_grad_(True)
    reference = statistic(reference_input, basis, KNOTS)
    reference.backward()
    reference_gradient = reference_input.grad
    assert reference_gradient is not None

    even = [torch.load(tmp_path / f"rank{rank}-32.pt") for rank in (0, 1)]
    for value, _ in even:
        torch.testing.assert_close(value, reference)
    # Every rank's loss is the global statistic, so the collective's backward
    # sums both ranks' cotangents; DDP's gradient average then restores the
    # global gradient for each rank's rows.
    torch.testing.assert_close(even[0][1] / 2, reference_gradient[0::2])
    torch.testing.assert_close(even[1][1] / 2, reference_gradient[1::2])

    full, empty = torch.load(tmp_path / "rank0-64.pt"), torch.load(tmp_path / "rank1-0.pt")
    torch.testing.assert_close(full[0], reference)
    torch.testing.assert_close(empty[0], reference)
