"""Ranks merge only consumed local deltas into one Quantile distribution."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

import relflow as rf
from relflow.architecture.binding import bind
from relflow.data.iterables import encode
from relflow.tensorfields.extensions.quantile import Synchronize


def distributed_quantiles(rank: int, directory: str) -> None:
    model = rf.Model.xs(value=rf.Quantile)
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
    finally:
        distributed.destroy_process_group()


def test_quantile_digest_merges_across_ranks_with_empty_observations(tmp_path):
    multiprocessing.spawn(distributed_quantiles, args=(str(tmp_path),), nprocs=2, join=True)
