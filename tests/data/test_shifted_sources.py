"""Leaves can read earlier records of their own collection without datatype dispatch."""

import uuid
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Component, Tokens
from relflow.structs.tree import Operand
from relflow.tensorfields.base import TENSORFIELDS, Extension
from relflow.tensorfields.extensions import number

P, N, V = Tokens.padded, Tokens.null, Tokens.valued


@pytest.fixture
def Previous():
    """A third-party request that reads its source ``lag`` records earlier, with no derive component."""
    extension = Extension(
        f"previous_{uuid.uuid4().hex[:8]}", types=(int | float,), callbacks=number.number.callback_factories
    )

    @extension.register(Component.Request)
    class Request(number.Numeric):
        type: str = extension.name
        lag: int = 1

        def operands(self) -> tuple[Operand, ...]:
            return tuple(replace(operand, shift=self.lag) for operand in super().operands())

    for role, component in number.number.components.items():
        if role != Component.Request:
            extension.register(role, component)
    try:
        yield Request
    finally:
        TENSORFIELDS.pop(extension.name, None)


def items(*values):
    return [None if value == "hole" else {"value": value} for value in values]


def events(*collections, item=pa.struct([("value", pa.float64())])):
    return pa.table({"events": pa.array(collections, type=pa.list_(item))})


@pytest.mark.parametrize(
    ("lag", "content", "state"),
    [
        (1, [[[0, 1, 2], [0, 4, 0]], [[0, 0, 0], [0, 0, 0]]], [[[P, V, V], [P, V, P]], [[P, P, P], [P, P, P]]]),
        (2, [[[0, 0, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], [[[P, P, V], [P, P, P]], [[P, P, P], [P, P, P]]]),
    ],
)
def test_shift_reads_earlier_records_within_each_parent_collection(Previous, lag, content, state):
    model = rf.Model(
        preset="small",
        d_model=8,
        groups=rf.Branch(length=2, events=rf.Branch(length=3, previous=Previous(query="value", lag=lag))),
    )
    group = pa.struct([("events", pa.list_(pa.struct([("value", pa.float64())])))])
    source = pa.table(
        {
            "groups": pa.array(
                [[{"events": items(1.0, 2.0, 3.0)}, {"events": items(4.0, 5.0)}], [None, {"events": items(6.0)}]],
                type=pa.list_(group),
            )
        }
    )

    field = model.encode(source)["/groups/events/previous"]

    assert field.content.reshape(2, 2, 3).tolist() == content
    assert field.state.reshape(2, 2, 3).tolist() == state


def test_null_items_and_null_values_keep_their_positions(Previous):
    model = rf.Model(preset="small", d_model=8, events=rf.Branch(length=5, previous=Previous(query="value")))

    field = model.encode(events(items(1.0, "hole", 3.0, None, 5.0)))["/events/previous"]

    # Slot 2 follows a null item and slot 4 follows a null value; neither borrows 1.0 or 3.0.
    assert field.state.reshape(-1).tolist() == [P, P, P, V, N]
    assert field.content.reshape(-1).tolist() == [0, 0, 0, 3, 0]


@pytest.mark.parametrize("overflow", ["head", "tail", "random"])
def test_each_retained_record_reads_its_source_predecessor_before_cropping(Previous, overflow):
    schema = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(length=3, overflow=overflow, value=rf.Number, previous=Previous(query="value")),
    ).schema
    source = events(*[items(*(float(row * 100 + index) for index in range(8))) for row in range(4)])

    fields = coalesce(source, schema, rf.Strata.train, seed=11, epoch=1, ordinal=2).fields

    value, previous = fields["/events/value"].pristine, fields["/events/previous"].pristine
    values = np.asarray(value.values)
    reachable = values % 100 > 0
    assert previous.placement.equals(value.placement.filter(pa.array(reachable)))
    np.testing.assert_array_equal(np.asarray(previous.values), values[reachable] - 1)


def test_shifted_values_do_not_depend_on_batch_boundaries_or_arrow_chunks(Previous):
    model = rf.Model(preset="small", d_model=8, events=rf.Branch(length=3, previous=Previous(query="value")))
    # Dictionary members keep Arrow chunks separate, so lookups cross a chunked candidate pool.
    item = pa.struct([("value", pa.float64()), ("label", pa.dictionary(pa.int8(), pa.string()))])
    collections = [items(1.0, 2.0, 3.0), [], None, items(4.0, "hole", 6.0), items(7.0)]
    source = events(*collections, item=item)
    chunked = pa.concat_tables([events(*collections[:2], item=item), events(*collections[2:], item=item)])

    whole = model.encode(source)["/events/previous"]
    split = model.encode(chunked)["/events/previous"]

    torch.testing.assert_close(split.content, whole.content)
    torch.testing.assert_close(split.state, whole.state)
    for row in range(source.num_rows):
        alone = model.encode(source.slice(row, 1))["/events/previous"]
        width = alone.state.shape[-1]
        torch.testing.assert_close(alone.state[0], whole.state[row][..., :width])
        torch.testing.assert_close(alone.content[0], whole.content[row][..., :width])


@pytest.mark.parametrize("shift", [-1, True, 1.0, "1", 2**63])
def test_operand_shift_must_address_an_int64_position(shift):
    with pytest.raises(ValueError, match="'value' shift must be a non-negative integer below 2\\*\\*63"):
        Operand("value", shift)


def test_shifts_require_a_local_query_inside_a_repeated_branch(Previous):
    with pytest.raises(ValueError, match="'/previous'.*model root holds one record"):
        rf.Model(preset="small", d_model=8, previous=Previous(query="value"))
    with pytest.raises(ValueError, match="'/events/previous'.*ancestor branch"):
        rf.Model(preset="small", d_model=8, events=rf.Branch(length=2, previous=Previous(query="../value")))


def test_extensions_without_derive_read_exactly_one_operand():
    extension = Extension(f"pair_{uuid.uuid4().hex[:8]}", types=(int | float,))

    @extension.register(Component.Request)
    class Request(number.Numeric):
        type: str = extension.name

        def operands(self) -> tuple[Operand, ...]:
            (current,) = super().operands()
            return (replace(current, shift=1), current)

    for role, component in number.number.components.items():
        if role != Component.Request:
            extension.register(role, component)
    try:
        with pytest.raises(ValueError, match="'/events/pair'.*no derive component"):
            rf.Model(preset="small", d_model=8, events=rf.Branch(length=2, pair=Request(query="value")))
    finally:
        TENSORFIELDS.pop(extension.name, None)
