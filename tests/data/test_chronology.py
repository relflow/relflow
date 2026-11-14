"""Branch time positions come from raw Arrow values and stay relative to the first row."""

import datetime

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.data import ragged
from relflow.data.iterables import encode, mock
from relflow.structs.enums import Strata
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address


def configured(*, step=None, length=None, overflow=None, nested=False, time_index="ts") -> Schema:
    ett = rf.Branch(
        length=length, overflow=overflow, time_index=time_index, time_step=step, ts=rf.Number, value=rf.Number
    )
    if nested:
        ett = rf.Branch(
            length=2, readings=rf.Branch(length=3, time_index="ts", time_step=step, ts=rf.Number, value=rf.Number)
        )
    return Schema.from_tree(d_model=8, n_layers=1, n_heads=2, ett=ett)


def table(rows, *, datatype=pa.float64()) -> pa.Table:
    item = pa.struct([pa.field("ts", datatype), pa.field("value", pa.float64())])
    records = [None if record is None else [{"ts": ts, "value": 1.0} for ts in record] for record in rows]
    return pa.table({"ett": pa.array(records, type=pa.list_(item))})


def times(source: pa.Table, schema: Schema, address: str = "/ett") -> np.ndarray:
    return ragged.coalesce(source, schema, Strata.predict).times[Address(address)]


def test_numeric_time_positions_are_relative_and_default_to_one_unit():
    assert times(table([[10.0, 12.0, 15.0]]), configured()).tolist() == [[[0.0, 2.0, 5.0]]]


def test_numeric_time_step_divides_positions():
    assert times(table([[10.0, 12.0, 15.0]]), configured(step=2.0)).tolist() == [[[0.0, 1.0, 2.5]]]


def test_equal_times_are_allowed():
    assert times(table([[5.0, 5.0, 6.0]]), configured()).tolist() == [[[0.0, 0.0, 1.0]]]


def test_decreasing_time_raises_with_branch_and_record():
    with pytest.raises(ValueError, match=r"branch '/ett'.*'ts' decreases in record 1 at row 2"):
        times(table([[0.0, 1.0], [0.0, 3.0, 1.0]]), configured())


def test_null_time_in_present_row_raises():
    with pytest.raises(ValueError, match=r"branch '/ett'.*'ts' is null in 1 present row"):
        times(table([[0.0, None, 2.0]]), configured())


def test_null_row_record_is_skipped_not_an_error():
    assert times(table([[0.0, 1.0], None, []]), configured()).tolist() == [
        [[0.0, 1.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ]


def test_timestamp_requires_time_step():
    stamps = [datetime.datetime(2026, 1, 1), datetime.datetime(2026, 1, 2)]
    with pytest.raises(ValueError, match=r"branch '/ett'.*timestamp\[us\].*set time_step=timedelta"):
        times(table([stamps], datatype=pa.timestamp("us")), configured())


def test_numeric_column_rejects_timedelta_step():
    with pytest.raises(ValueError, match=r"branch '/ett'.*double.*time_step=<number>"):
        times(table([[0.0, 1.0]]), configured(step=datetime.timedelta(hours=1)))


def test_timezone_aware_timestamps_count_elapsed_time_across_dst():
    zone = pa.timestamp("us", tz="Europe/Berlin")
    berlin = [
        datetime.datetime(2026, 3, 29, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=1))),
        datetime.datetime(2026, 3, 29, 4, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
    ]
    step = datetime.timedelta(hours=1)
    local = times(table([berlin], datatype=zone), configured(step=step))
    assert local.tolist() == [[[0.0, 3.0]]]
    utc = [moment.astimezone(datetime.timezone.utc) for moment in berlin]
    assert times(table([utc], datatype=pa.timestamp("us", tz="UTC")), configured(step=step)).tolist() == local.tolist()


def test_epoch_seconds_stay_exact_in_float32():
    start = 1_700_000_000
    column = pa.array([start, start + 1, start + 2], type=pa.timestamp("s"))
    item = pa.struct([pa.field("ts", pa.timestamp("s")), pa.field("value", pa.float64())])
    source = pa.table({"ett": pa.array([[{"ts": s, "value": 1.0} for s in column.to_pylist()]], type=pa.list_(item))})
    positions = torch.from_numpy(times(source, configured(step=datetime.timedelta(seconds=1)))).to(torch.float32)
    assert positions.tolist() == [[[0.0, 1.0, 2.0]]]


def test_date_and_duration_columns_read_as_seconds():
    days = [datetime.date(2026, 1, 1), datetime.date(2026, 1, 3)]
    step = datetime.timedelta(days=1)
    assert times(table([days], datatype=pa.date32()), configured(step=step)).tolist() == [[[0.0, 2.0]]]
    spans = [datetime.timedelta(milliseconds=0), datetime.timedelta(milliseconds=1500)]
    assert times(
        table([spans], datatype=pa.duration("ms")), configured(step=datetime.timedelta(seconds=1))
    ).tolist() == [[[0.0, 1.5]]]


def test_positions_are_relative_per_instance_in_a_repeated_parent():
    reading = pa.struct([pa.field("ts", pa.float64()), pa.field("value", pa.float64())])
    station = pa.struct([pa.field("readings", pa.list_(reading))])
    stations = [
        [
            {"readings": [{"ts": 0.0, "value": 1.0}, {"ts": 1.0, "value": 1.0}, {"ts": 2.0, "value": 1.0}]},
            {"readings": [{"ts": 100.0, "value": 1.0}, {"ts": 101.0, "value": 1.0}, {"ts": 102.0, "value": 1.0}]},
        ]
    ]
    source = pa.table({"ett": pa.array(stations, type=pa.list_(station))})
    result = times(source, configured(nested=True), address="/ett/readings")
    assert result.shape == (1, 1, 2, 3)
    assert result[0, 0].tolist() == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]


def test_tail_overflow_positions_start_at_first_retained_row():
    schema = configured(length=2, overflow="tail")
    assert times(table([[0.0, 1.0, 5.0]]), schema).tolist() == [[[0.0, 4.0]]]


def encoded(source: pa.Table, schema: Schema):
    return encode(source, schema, Strata.predict, {})


def test_plain_branch_geometry_has_no_times():
    schema = Schema.from_tree(d_model=8, n_layers=1, n_heads=2, ett=rf.Branch(ts=rf.Number, value=rf.Number))
    assert encoded(table([[0.0, 1.0]]), schema).tensors[Address()][Address("/ett")].times is None


def test_timed_geometry_carries_float64_times_and_mock_is_empty():
    schema = configured()
    geometry = encoded(table([[0.0, 2.0], [1.0, 4.0]]), schema).tensors[Address()][Address("/ett")]
    assert geometry.times.dtype == torch.float64 and geometry.times.shape == geometry.slots.shape
    assert mock(schema, 2)[Address()][Address("/ett")].times.shape == (2, 1, 0)


def test_times_survive_clone_device_transfer_and_row_selection():
    batch = encoded(table([[0.0, 2.0], [1.0, 4.0]]), configured()).tensors[Address()]
    expected = batch[Address("/ett")].times
    assert torch.equal(batch.clone()[Address("/ett")].times, expected)
    assert torch.equal(batch.to("cpu")[Address("/ett")].times, expected)
    assert torch.equal(batch[torch.tensor([1])][Address("/ett")].times, expected[1:])
