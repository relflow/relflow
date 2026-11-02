"""Prediction collections follow retained structure rather than dense batch width."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
import relflow.architecture.output as output
from relflow.architecture.output import OutputEntry, Projection, projection
from relflow.data.arrow import Geometry
from relflow.data.iterables import geometry
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address

EVENT = pa.struct([("value", pa.float64())])


def project(source: pa.Table, configured: Schema, address: str) -> Projection:
    request = configured.requests[Address(address)]
    branches = tuple(node.address for node in request.path if node.type == "branch" and node.address != Address())
    entry = cast(
        OutputEntry,
        SimpleNamespace(address=Address(address), axes=request.shape[1:], branches=branches, passthrough=False),
    )
    resolved = coalesce(source, configured, Strata.predict)
    prediction = Prediction(
        address=Address(address),
        payload=TensorDict({}, batch_size=[source.num_rows]),
        geometry=geometry(resolved.layouts, rows=source.num_rows),
        batch_size=[source.num_rows],
    )
    return projection(entry, prediction, source.num_rows)


@pytest.mark.parametrize("length", [None, 8])
def test_missing_empty_null_items_and_missing_targets_keep_distinct_geometry(length):
    configured = Schema.from_tree(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(length=length, value=rf.Number(mask=True))
    )
    source = pa.table({"events": pa.array([None, [], [None], [{"value": None}]], type=pa.list_(EVENT))})
    projected = project(source, configured, "/events/value")

    assert projected.size == 4
    assert projected.indices.tolist() == [-1, 3]
    assert projected.rows.tolist() == [2, 3]
    assert projected.parents[0].tolist() == [0, 0]
    assert projected.outputs.tolist() == [0, 0]
    assert projected.axes[0].offsets.tolist() == [0, 0, 0, 1, 2]
    assert projected.axes[0].valid.tolist() == [False, True, True, True]
    assert projected.axes[0].slots.tolist() == [False, True]


def test_bounded_empty_output_does_not_depend_on_neighbor_batch_width():
    configured = Schema.from_tree(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(length=3, value=rf.Number(mask=True))
    )
    empty = pa.table({"events": pa.array([[]], type=pa.list_(EVENT))})
    mixed = pa.table({"events": pa.array([[], [{"value": 1.0}]], type=pa.list_(EVENT))})
    alone = project(empty, configured, "/events/value")
    batched = project(mixed, configured, "/events/value")

    assert alone.size == 0
    assert alone.axes[0].offsets.tolist() == [0, 0]
    assert alone.axes[0].valid.tolist() == [True]
    assert alone.axes[0].slots.tolist() == []
    assert batched.indices.tolist() == [1]
    assert batched.axes[0].offsets.tolist() == [0, 0, 1]


@pytest.mark.parametrize("length", [None, 8])
def test_nested_null_structure_and_local_parent_ordinals(length):
    configured = Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        parents=rf.Branch(length=length, children=rf.Branch(length=length, value=rf.Number(mask=True))),
    )
    parent = pa.struct([("children", pa.list_(EVENT))])
    source = pa.table(
        {
            "parents": pa.array(
                [
                    [None, {"children": None}, {"children": []}, {"children": [None, {"value": None}]}],
                    None,
                    [],
                    [{"children": [{"value": 1.0}]}],
                ],
                type=pa.list_(parent),
            )
        }
    )
    projected = project(source, configured, "/parents/children/value")

    assert projected.size == 32
    assert projected.indices.tolist() == [-1, 7, 24]
    assert projected.rows.tolist() == [0, 0, 3]
    assert [parent.tolist() for parent in projected.parents] == [[3, 3, 0], [0, 1, 0]]
    assert projected.axes[0].offsets.tolist() == [0, 4, 4, 4, 5]
    assert projected.axes[0].slots.tolist() == [False, True, True, True, True]
    assert projected.axes[1].offsets.tolist() == [0, 0, 0, 0, 2, 3]
    assert projected.axes[1].valid.tolist() == [False, False, True, True, True]
    assert projected.axes[1].slots.tolist() == [False, True, True]


@pytest.mark.parametrize("length", [None, 3])
def test_zero_width_axes_preserve_typed_collections(length):
    configured = Schema.from_tree(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(length=length, value=rf.Number(mask=True))
    )
    source = pa.table({"events": pa.array([[], None], type=pa.list_(EVENT))})
    projected = project(source, configured, "/events/value")

    assert projected.size == 0
    assert projected.indices.size == projected.rows.size == projected.outputs.size == 0
    assert projected.parents[0].size == 0
    assert projected.axes[0].offsets.tolist() == [0, 0, 0]
    assert projected.axes[0].valid.tolist() == [True, False]
    assert projected.axes[0].slots.size == 0


def test_zero_rows_keep_every_nested_axis():
    configured = Schema.from_tree(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(items=rf.Branch(value=rf.Number(mask=True)))
    )
    event = pa.struct([("items", pa.list_(EVENT))])
    source = pa.table({"events": pa.array([], type=pa.list_(event))})
    projected = project(source, configured, "/events/items/value")

    assert projected.size == 0
    assert len(projected.axes) == len(projected.parents) == 2
    assert all(axis.offsets.tolist() == [0] for axis in projected.axes)
    assert all(axis.valid.size == axis.slots.size == 0 for axis in projected.axes)


def test_pooled_outputs_exist_for_empty_and_missing_own_collections():
    configured = Schema.from_tree(d_model=8, n_layers=1, n_heads=2, events=rf.Branch(value=rf.Number))
    source = pa.table({"events": pa.array([None, []], type=pa.list_(EVENT))})
    resolved = coalesce(source, configured, Strata.predict)
    entry = cast(OutputEntry, SimpleNamespace(address=Address("/events"), axes=(2,), branches=(), passthrough=False))
    prediction = Prediction(
        address=Address("/events"),
        payload=TensorDict({}, batch_size=[2]),
        geometry=geometry(resolved.layouts, rows=2),
        batch_size=[2],
    )

    projected = projection(entry, prediction, 2)

    assert projected.indices.tolist() == [0, 1, 2, 3]
    assert projected.axes[0].offsets.tolist() == [0, 2, 4]
    assert projected.axes[0].valid.tolist() == [True, True]
    assert projected.rows.tolist() == [0, 0, 1, 1]
    assert projected.parents == ()
    assert projected.outputs.tolist() == [0, 1, 0, 1]
    assert projected.outputs.dtype == np.int32


def test_nested_pooled_outputs_keep_parent_ordinals_without_null_ancestor_outputs():
    configured = Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        parents=rf.Branch(children=rf.Branch(value=rf.Number)),
    )
    parent = pa.struct([("children", pa.list_(EVENT))])
    source = pa.table({"parents": pa.array([[{"children": None}, None, {"children": []}]], type=pa.list_(parent))})
    resolved = coalesce(source, configured, Strata.predict)
    entry = cast(
        OutputEntry,
        SimpleNamespace(
            address=Address("/parents/children"), axes=(None, 2), branches=(Address("/parents"),), passthrough=False
        ),
    )
    prediction = Prediction(
        address=Address("/parents/children"),
        payload=TensorDict({}, batch_size=[1]),
        geometry=geometry(resolved.layouts, rows=1),
        batch_size=[1],
    )

    projected = projection(entry, prediction, 1)

    assert projected.size == 6
    assert projected.indices.tolist() == [0, 1, 4, 5]
    assert projected.rows.tolist() == [0, 0, 0, 0]
    assert projected.parents[0].tolist() == [0, 0, 2, 2]
    assert projected.outputs.tolist() == [0, 1, 0, 1]
    assert projected.axes[0].offsets.tolist() == [0, 3]
    assert projected.axes[0].slots.tolist() == [True, False, True]
    assert projected.axes[1].offsets.tolist() == [0, 2, 2, 4]
    assert projected.axes[1].valid.tolist() == [True, False, True]


def test_pass_through_positions_restore_holes_without_reordering_tokens():
    entry = cast(OutputEntry, SimpleNamespace(address=Address(), axes=(None,), branches=(), passthrough=True))
    prediction = Prediction(
        address=Address(),
        payload=TensorDict({}, batch_size=[1]),
        positions=torch.tensor([[0, 2, 1]]),
        spans=torch.tensor([3]),
        slots=torch.tensor([[True, True, False]]),
        batch_size=[1],
    )

    projected = projection(entry, prediction, 1)

    assert projected.size == 3
    assert projected.indices.tolist() == [0, -1, 1]
    assert projected.axes[0].offsets.tolist() == [0, 3]
    assert projected.axes[0].slots.tolist() == [True, False, True]
    assert projected.outputs.tolist() == [0, 1, 2]


def test_singleton_pass_through_keeps_explicit_output_index():
    entry = cast(OutputEntry, SimpleNamespace(address=Address(), axes=(), branches=(), passthrough=True))
    prediction = Prediction(
        address=Address(),
        payload=TensorDict({}, batch_size=[2]),
        positions=torch.tensor([[0], [0]]),
        spans=torch.tensor([1, 1]),
        slots=torch.tensor([[True], [True]]),
        batch_size=[2],
    )

    projected = projection(entry, prediction, 2)

    assert projected.size == 2
    assert projected.indices.tolist() == [0, 1]
    assert projected.axes[0].offsets.tolist() == [0, 1, 2]
    assert projected.rows.tolist() == [0, 1]
    assert projected.parents == ()
    assert projected.outputs.tolist() == [0, 0]


def test_scalar_projection_keeps_root_rows_without_collection_metadata():
    entry = cast(OutputEntry, SimpleNamespace(address=Address("/answer"), axes=(), branches=(), passthrough=False))
    prediction = Prediction(address=Address("/answer"), payload=TensorDict({}, batch_size=[2]), batch_size=[2])

    projected = projection(entry, prediction, 2)

    assert projected.size == 2
    assert projected.indices.tolist() == [0, 1]
    assert projected.axes == projected.parents == ()
    assert projected.rows.tolist() == [0, 1]
    assert projected.outputs.tolist() == [0, 0]


def test_structural_axes_require_geometry_even_when_bounded():
    entry = cast(
        OutputEntry,
        SimpleNamespace(address=Address("/events/value"), axes=(3,), branches=(Address("/events"),), passthrough=False),
    )
    prediction = Prediction(address=Address("/events/value"), payload=TensorDict({}, batch_size=[1]), batch_size=[1])

    with pytest.raises(ValueError, match="requires batch geometry for branch /events"):
        projection(entry, prediction, 1)


@pytest.fixture
def bounded_prediction():
    branch = Geometry(
        lengths=torch.tensor([[1]]),
        present=torch.tensor([[True]]),
        valid=torch.tensor([[True]]),
        slots=torch.tensor([[[True]]]),
        batch_size=[1],
    )
    entry = cast(
        OutputEntry,
        SimpleNamespace(address=Address("/events/value"), axes=(3,), branches=(Address("/events"),), passthrough=False),
    )
    prediction = Prediction(
        address=Address("/events/value"),
        payload=TensorDict({}, batch_size=[1]),
        geometry=TensorDict(cast(Any, {"/events": branch}), batch_size=[1]),
        batch_size=[1],
    )
    return entry, prediction, branch


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"lengths": torch.tensor([[-1]])}, ValueError, "lengths must be within its extent"),
        ({"lengths": torch.tensor([[4]])}, ValueError, "lengths must be within its extent"),
        ({"lengths": torch.tensor([[1.0]])}, TypeError, "lengths must be a torch.int64"),
        ({"valid": torch.tensor([[1]])}, TypeError, "valid must be a torch.bool"),
        ({"lengths": torch.tensor([[1, 1]])}, ValueError, "lengths must have shape"),
        ({"slots": torch.tensor([[True]])}, ValueError, "slots must have shape"),
        ({"slots": torch.tensor([[[True, False, False, False]]])}, ValueError, "extent must be at most 3"),
        ({"lengths": torch.tensor([[0]])}, ValueError, "slots must lie within collection lengths"),
        ({"valid": torch.tensor([[False]])}, ValueError, "validity must agree with presence and collection lengths"),
        ({"present": torch.tensor([[False]])}, ValueError, "validity must agree with presence and collection lengths"),
    ],
)
def test_writer_rejects_invalid_geometry_before_projecting(bounded_prediction, changes, error, message):
    entry, prediction, branch = bounded_prediction
    for name, value in changes.items():
        setattr(branch, name, value)

    with pytest.raises(error, match=rf"prediction geometry for /events {message}"):
        projection(entry, prediction, 1)


def test_child_geometry_cannot_exist_under_a_null_parent_item():
    configured = Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        parents=rf.Branch(children=rf.Branch(value=rf.Number)),
    )
    parent = pa.struct([("children", pa.list_(EVENT))])
    source = pa.table({"parents": pa.array([[None, {"children": [{"value": 1.0}]}]], type=pa.list_(parent))})
    resolved = coalesce(source, configured, Strata.predict)
    metadata = geometry(resolved.layouts, rows=1)
    cast(Geometry, metadata["/parents/children"]).present[0, 0, 0] = True
    entry = cast(
        OutputEntry,
        SimpleNamespace(
            address=Address("/parents/children/value"),
            axes=(None, None),
            branches=(Address("/parents"), Address("/parents/children")),
            passthrough=False,
        ),
    )
    prediction = Prediction(
        address=entry.address, payload=TensorDict({}, batch_size=[1]), geometry=metadata, batch_size=[1]
    )

    with pytest.raises(ValueError, match="/parents/children cannot exist outside its parent slots"):
        projection(entry, prediction, 1)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"spans": torch.tensor([-1])}, ValueError, "spans must be non-negative"),
        ({"spans": torch.tensor([2**31])}, OverflowError, "spans exceed the Arrow int32 offset limit"),
        ({"spans": torch.tensor([3.0])}, TypeError, "spans must be a torch.int64"),
        ({"spans": torch.tensor([[3]])}, ValueError, "spans must have shape"),
        ({"positions": torch.tensor([[[0, 2, 1]]])}, ValueError, "positions must start with parent shape"),
        ({"slots": torch.tensor([[True, True]])}, ValueError, "slots must match positions shape"),
        ({"positions": torch.tensor([[-1, 2, -9]])}, ValueError, "selected positions must lie within their spans"),
        ({"positions": torch.tensor([[0, 3, -9]])}, ValueError, "selected positions must lie within their spans"),
        ({"positions": torch.tensor([[0, 0, -9]])}, ValueError, "selected positions must be unique within each parent"),
    ],
)
def test_writer_rejects_invalid_pass_through_positions(changes, error, message):
    entry = cast(OutputEntry, SimpleNamespace(address=Address(), axes=(None,), branches=(), passthrough=True))
    prediction = Prediction(
        address=Address(),
        payload=TensorDict({}, batch_size=[1]),
        positions=torch.tensor([[0, 2, -9]]),
        spans=torch.tensor([3]),
        slots=torch.tensor([[True, True, False]]),
        batch_size=[1],
    )
    for name, value in changes.items():
        setattr(prediction, name, value)

    with pytest.raises(error, match=rf"pass-through prediction at / {message}"):
        projection(entry, prediction, 1)


def test_unselected_pass_through_positions_do_not_change_logical_holes():
    entry = cast(OutputEntry, SimpleNamespace(address=Address(), axes=(None,), branches=(), passthrough=True))
    prediction = Prediction(
        address=Address(),
        payload=TensorDict({}, batch_size=[1]),
        positions=torch.tensor([[0, 2, -9]]),
        spans=torch.tensor([3]),
        slots=torch.tensor([[True, True, False]]),
        batch_size=[1],
    )

    assert projection(entry, prediction, 1).indices.tolist() == [0, -1, 1]


def test_collection_offset_overflow_is_rejected_before_expanding_coordinates(bounded_prediction, monkeypatch):
    entry, prediction, branch = bounded_prediction
    branch.lengths = torch.tensor([[3]])
    branch.slots = torch.tensor([[[True, True, True]]])
    monkeypatch.setattr(output, "LIST_MAX", 2)

    def reject_expansion(*args, **kwargs):
        pytest.fail("overflow reached coordinate expansion")

    monkeypatch.setattr(output.np, "repeat", reject_expansion)

    with pytest.raises(
        OverflowError, match="prediction at /events/value collections exceed the Arrow int32 offset limit"
    ):
        projection(entry, prediction, 1)
