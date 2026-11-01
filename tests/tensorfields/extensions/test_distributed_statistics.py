"""Independent extension statistics share ranks while retaining fresh state."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

import relflow as rf
from relflow.architecture.binding import bind
from relflow.data.iterables import encode
from relflow.tensorfields.extensions.category import Reconstruction
from relflow.tensorfields.extensions.number import GlobalOnlineNormalizer, moments
from relflow.tensorfields.extensions.quantile import Synchronize

pytestmark = pytest.mark.integration


def distributed_statistics(rank: int, directory: str) -> None:
    # Rank-local history predates synchronization and must be replaced by rank zero's.
    model = rf.Model.xs(d_model=8, value=rf.Quantile)
    model.encode(pa.table({"value": [float(rank * 100)]}), strata="train")
    distributed.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        Synchronize().on_fit_start(SimpleNamespace(), model)
        assert rf.Quantile.normalization(model, "/value")["maximum"] == 0.0
        state = model.interprocess_encoding_context[rf.Address("value")]
        source = pa.table({"value": pa.array([None, None] if rank == 0 else [10.0, 20.0], type=pa.float64())})
        batch = encode(source, model.schema, rf.Strata.train, model.interprocess_encoding_context)
        assert state.snapshot().count == 1
        bound = bind(model, batch, rf.Strata.train)
        assert not bound.bindings
        assert state.snapshot().count == 3
        assert bind(model, bound, rf.Strata.train) is bound
        assert state.snapshot().count == 3

        snapshots = [None, None]
        distributed.all_gather_object(snapshots, state.snapshot().serialize())
        assert snapshots[0] == snapshots[1]
        summary = rf.Quantile.normalization(model, "/value")
        assert summary["minimum"] == 0.0
        assert summary["median"] == pytest.approx(10.0)
        assert summary["maximum"] == 20.0

        # An all-null step still participates, without remerging shared history.
        model.encode(pa.table({"value": pa.array([None], type=pa.float64())}), strata="train")
        assert state.snapshot().count == 3
        model.encode(pa.table({"value": [1e9]}), strata="validate")
        assert state.snapshot().count == 3
        assert rf.Quantile.normalization(model, "/value")["maximum"] == 20.0

        for empty in (False, True):
            normalizer = GlobalOnlineNormalizer()
            local = [1_000_000.0] if rank == 0 else [1_000_002.0, 1_000_004.0, 1_000_006.0]
            if empty and rank == 0:
                local = []
            normalizer.learn(moments(torch.tensor(local, dtype=torch.float32)))
            assert normalizer.count.item() == (3 if empty else 4)
            assert normalizer.mean.item() == (1_000_004.0 if empty else 1_000_003.0)
            assert normalizer.var.item() == pytest.approx(8 / 3 if empty else 5.0)
            assert normalizer.mean.dtype == normalizer.var.dtype == torch.float64
            normalizer.learn(moments(torch.empty(0)))
            assert normalizer.count.item() == (3 if empty else 4)

        metric = Reconstruction([2])
        metric.update(
            torch.tensor([2, 3, 1, 2]) if rank == 0 else torch.tensor([0, 7, 0, 0]),
            torch.tensor([6.0, 4.0]) if rank == 0 else torch.zeros(2),
        )
        result = metric.compute()
        assert result["targets.known"] == 2
        assert result["targets.unavailable"] == 8
        assert result["coverage.content"] == pytest.approx(0.2)
        assert result["accuracy.content"] == 0.5
        assert result["accuracy.all"] == pytest.approx(0.1)
        assert result["nll.content"] == 2
        assert result["loss.content"] == 3
        metric.reset()
        assert not metric.counts.any()
        assert not metric.sums.any()
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not distributed.is_available() or not distributed.is_gloo_available(), reason="requires CPU Gloo")
def test_quantiles_moments_and_reconstruction_reduce_with_empty_ranks(tmp_path):
    multiprocessing.spawn(distributed_statistics, args=(str(tmp_path),), nprocs=2, join=True)
