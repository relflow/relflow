"""Prefetched enum IDs and quantiles change live state only when consumed."""

import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture.binding import bind
from relflow.architecture.runtime import ModelRuntime


@pytest.mark.parametrize("workers", [0, pytest.param(1, marks=pytest.mark.integration)])
def test_prefetch_preserves_enum_ids_and_consumes_each_quantile_delta_once(workers):
    labels = ("pending", "complete", "rejected")
    configured = rf.Model.xs(
        d_model=8,
        batch_size=2,
        status=rf.Enum(values=labels, p_unavailable=0, mask=True),
        amount=rf.Quantile(mask=True),
    )
    data = rf.ArrowDataModule(
        model=configured,
        train=pa.table(
            {
                "status": ["rejected", "complete", "pending", "pending", "complete", "rejected"],
                "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        ),
        shuffle=False,
        num_workers=workers,
        persistent_workers=False,
        pin_memory=False,
    )
    batches = list(data.train_dataloader())
    assert len(batches) == 3
    assert rf.Enum.counts(configured, "/status") == dict.fromkeys(labels, 0)
    assert rf.Quantile.normalization(configured, "/amount")["count"] == 0
    for batch, expected in zip(batches, ([2, 1], [0, 0], [1, 2]), strict=True):
        assert batch.tensors["/status"].targets[rf.TensorKey.content].flatten().tolist() == expected

    consumed = bind(configured, batches[0], rf.Strata.train)
    ModelRuntime.learn(configured, consumed.observations, strata=rf.Strata.train)
    assert not consumed.bindings
    assert rf.Enum.counts(configured, "/status") == {"pending": 0, "complete": 1, "rejected": 1}
    assert rf.Enum.vocabulary(configured, "/status") == labels
    assert rf.Quantile.normalization(configured, "/amount")["count"] == 2
    assert bind(configured, consumed, rf.Strata.train) is consumed
    assert rf.Quantile.normalization(configured, "/amount")["count"] == 2

    pending = batches[1].tensors["/amount"].targets[rf.TensorKey.content].clone()
    second = bind(configured, batches[1], rf.Strata.train)
    ModelRuntime.learn(configured, second.observations, strata=rf.Strata.train)
    assert rf.Quantile.normalization(configured, "/amount")["count"] == 4
    assert not torch.equal(second.tensors["/amount"].targets[rf.TensorKey.content], pending)
    assert rf.Enum.counts(configured, "/status") == {"pending": 2, "complete": 1, "rejected": 1}
    assert rf.Enum.vocabulary(configured, "/status") == labels
