"""Index twigs place branch records in geometry by their own values."""

import datetime as dt

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.data.datasets.arrow import ship
from relflow.data.iterables import encode, geometry
from relflow.data.ragged import coalesce
from relflow.structs.experiment import Schema


def schema(index: rf.Index | type[rf.Index] = rf.Index, **options) -> Schema:
    return Schema.from_tree(d_model=8, n_layers=1, n_heads=2, events=rf.Branch(at=index, amount=rf.Number, **options))


def positions(source: pa.Table, configured: Schema, strata: rf.Strata = rf.Strata.predict) -> np.ndarray:
    return coalesce(source, configured, strata).positions["/events"]


def test_integer_indexes_count_from_each_collections_smallest_value_in_any_order():
    source = pa.table(
        {
            "events": [
                [{"at": 105, "amount": 1.0}, {"at": 100, "amount": 2.0}, {"at": 112, "amount": 3.0}],
                None,
                [],
                [{"at": -7, "amount": 1.0}, None, {"at": -3, "amount": 2.0}],
            ]
        }
    )

    placed = positions(source, schema())

    assert placed.dtype == np.float32
    assert placed.tolist() == [[[5, 0, 12]], [[0, 0, 0]], [[0, 0, 0]], [[0, 0, 4]]]


@pytest.mark.parametrize("zone", [None, "UTC", "America/New_York"])
def test_timestamps_count_granularity_units_whatever_their_resolution_or_zone(zone):
    start = dt.datetime(2024, 3, 9, 12)
    times = [start, start + dt.timedelta(hours=36), start + dt.timedelta(minutes=90)]
    for unit in ("s", "ms", "us", "ns"):
        values = pa.array(times, type=pa.timestamp(unit, tz=zone))
        events = pa.StructArray.from_arrays([values, pa.array([1.0, 2.0, 3.0])], names=["at", "amount"])
        source = pa.table({"events": pa.ListArray.from_arrays(pa.array([0, 3]), events)})

        assert positions(source, schema(rf.Index(granularity="hours"))).tolist() == [[[0.0, 36.0, 1.5]]]
        assert positions(source, schema()).tolist() == [[[0.0, 1.5, 0.0625]]]


def test_dates_default_to_days():
    days = [dt.date(2024, 1, 1), dt.date(2024, 1, 15), dt.date(2023, 12, 25)]
    for datatype in (pa.date32(), pa.date64()):
        events = pa.StructArray.from_arrays(
            [pa.array(days, type=datatype), pa.array([1.0, 2.0, 3.0])], names=["at", "amount"]
        )
        source = pa.table({"events": pa.ListArray.from_arrays(pa.array([0, 3]), events)})

        assert positions(source, schema()).tolist() == [[[7.0, 21.0, 0.0]]]
        assert positions(source, schema(rf.Index(granularity="weeks"))).tolist() == [[[1.0, 3.0, 0.0]]]


def test_index_reads_its_query_and_crops_with_its_records():
    source = pa.table({"events": [[{"started": value, "amount": 1.0} for value in (40, 10, 30, 20)]]})

    placed = positions(source, schema(rf.Index(query="started"), length=2, overflow="tail"))

    assert placed.tolist() == [[[10.0, 0.0]]]


def test_index_rejects_missing_values_unsupported_types_and_granular_integers():
    with pytest.raises(ValueError, match=r"Index '/events/at' has 1 null or absent value\(s\) among 2"):
        positions(pa.table({"events": [[{"at": 1, "amount": 1.0}, {"at": None, "amount": 2.0}]]}), schema())
    with pytest.raises(TypeError, match="reads Arrow double; an index takes integers, dates, or timestamps"):
        positions(pa.table({"events": [[{"at": 1.5, "amount": 1.0}]]}), schema())
    with pytest.raises(ValueError, match="whose values count their own units; remove granularity='days'"):
        positions(pa.table({"events": [[{"at": 1, "amount": 1.0}]]}), schema(rf.Index(granularity="days")))
    unsigned = pa.table(
        {
            "events": pa.array(
                [[{"at": 2**64 - 1, "amount": 1.0}]],
                type=pa.list_(pa.struct([("at", pa.uint64()), ("amount", pa.float64())])),
            )
        }
    )
    with pytest.raises(ValueError, match="beyond the int64 range"):
        positions(unsigned, schema())


def test_nested_indexes_position_each_grandchild_collection_independently():
    configured = Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        orders=rf.Branch(items=rf.Branch(step=rf.Index, price=rf.Number), total=rf.Number),
    )
    source = pa.table(
        {
            "orders": [
                [
                    {"total": 1.0, "items": [{"step": 4, "price": 1.0}, {"step": 9, "price": 2.0}]},
                    {"total": 2.0, "items": [{"step": 2, "price": 3.0}]},
                ]
            ]
        }
    )

    resolved = coalesce(source, configured, rf.Strata.predict)

    assert set(resolved.positions) == {"/orders/items"}
    assert resolved.positions["/orders/items"].tolist() == [[[[0.0, 5.0], [0.0, 0.0]]]]


def test_encoded_geometry_carries_float32_positions_only_for_indexed_branches_through_shipping():
    configured = Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(at=rf.Index, amount=rf.Number),
        plain=rf.Branch(amount=rf.Number),
    )
    source = pa.table(
        {
            "events": [[{"at": 3, "amount": 1.0}, {"at": 1, "amount": 2.0}]],
            "plain": [[{"amount": 1.0}]],
        }
    )
    resolved = coalesce(source, configured, rf.Strata.predict)
    axes = geometry(resolved, rows=1)
    assert axes["/events"].positions.dtype == torch.float32
    assert axes["/events"].positions.tolist() == [[[2.0, 0.0]]]
    assert axes["/plain"].positions is None and axes["/"].positions is None

    model = rf.Model(d_model=8, n_layers=1, n_heads=2, events=rf.Branch(at=rf.Index, amount=rf.Number))
    encoded = encode(source, model.schema, rf.Strata.predict, model.interprocess_encoding_context)
    shipped = ship(encoded).tensors[rf.Address()]
    assert shipped["/events"].positions.tolist() == [[[2.0, 0.0]]]
    assert shipped["/"].positions is None
