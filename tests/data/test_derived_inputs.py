"""Third-party computed fields share structural sources without datatype dispatch."""

import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

import relflow as rf
from relflow.data.query import MissingFieldError, Projection, scope
from relflow.data.ragged import coalesce
from relflow.structs.enums import Component, Tokens
from relflow.structs.experiment import Schema
from relflow.tensorfields.base import TENSORFIELDS, Extension
from relflow.tensorfields.extensions import number


@pytest.fixture
def arithmetic():
    extension = Extension(
        f"computed_{uuid.uuid4().hex[:8]}", types=(int | float,), callbacks=number.number.callback_factories
    )

    @extension.register(Component.Request)
    class Request(number.Numeric):
        type: str = extension.name
        constant: float = 0.0

    @extension.register(Component.derive)
    def add(inputs, *, request, address, rows):
        values = pa.repeat(pa.scalar(request.constant), rows)
        present = pa.repeat(pa.scalar(True), rows)
        for item in inputs:
            values = pc.add(values, item.values)
            present = pc.and_(present, item.present)
        return Projection(values=values, present=present)

    for role, component in number.number.components.items():
        if role != Component.Request:
            extension.register(role, component)
    try:
        yield extension, Request
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_two_sources_derive_before_numeric_observation_and_target_masking(arithmetic):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        total=Request(query=("left", "right"), mask=rf.Mask(rate=1.0, reconstruct=True, dropout=False)),
    )
    encoded = model.encode(pa.table({"left": [2.0, 7.0], "right": [3.0, 11.0]}), strata="train")["/total"]

    assert encoded.state.reshape(-1).tolist() == [Tokens.masked, Tokens.masked]
    assert encoded.targets[rf.TensorKey.content].reshape(-1).tolist() == [5.0, 18.0]
    assert encoded.targets[rf.TensorKey.state].reshape(-1).tolist() == [Tokens.valued, Tokens.valued]
    normalizer = model.nodes["/total"].embedder.normalizers["raw"]
    assert normalizer.count.item() == 2
    assert normalizer.mean.item() == 11.5


def test_source_presence_and_null_values_remain_distinct(arithmetic):
    _, Request = arithmetic
    model = rf.Model(preset="small", d_model=8, total=Request(query=("left.value", "right")))
    source = pa.Table.from_pylist(
        [
            {"left": {"value": 2.0}, "right": 3.0},
            {"left": {"value": None}, "right": 4.0},
            {"left": None, "right": 5.0},
        ]
    )
    field = model.encode(source)["/total"]

    assert field.state.reshape(-1).tolist() == [Tokens.valued, Tokens.null, Tokens.padded]
    assert field.content.reshape(-1).tolist() == [5.0, 0.0, 0.0]
    strict = rf.Model(preset="small", d_model=8, total=Request(query=("left.value", "right"), nullable=False))
    with pytest.raises(ValueError, match="/total.*nullable=False.*1 null"):
        strict.encode(source)


def test_explicit_empty_query_receives_retained_row_count_for_constants(arithmetic):
    _, Request = arithmetic
    model = rf.Model(preset="small", d_model=8, total=Request(query=(), constant=9.5))
    field = model.encode(pa.table({"unreferenced": [1, 2, 3]}))["/total"]

    assert field.state.reshape(-1).tolist() == [Tokens.valued] * 3
    assert field.content.reshape(-1).tolist() == [9.5] * 3


@pytest.mark.parametrize("query", [None, "source", ("source",)])
def test_default_single_and_one_element_tuple_queries_preserve_source_order(arithmetic, query):
    _, Request = arithmetic
    options = {} if query is None else {"query": query}
    model = rf.Model(preset="small", d_model=8, total=Request(constant=1.5, **options))
    field = model.encode(pa.table({"total": [1.0, 2.0], "source": [4.0, 8.0]}))["/total"]

    assert field.content.reshape(-1).tolist() == ([2.5, 3.5] if query is None else [5.5, 9.5])
    restored = Schema.model_validate_json(model.schema.model_dump_json())
    assert restored.requests["/total"].query == model.schema.requests["/total"].query


def test_derive_capability_is_read_from_the_live_registry(arithmetic):
    extension, Request = arithmetic
    derive = extension.components.pop(Component.derive)
    request = Request(query=("left", "right"))
    with pytest.raises(ValueError, match="/total.*no derive component"):
        rf.Model(preset="small", d_model=8, total=request)
    extension.register(Component.derive, derive)
    model = rf.Model(preset="small", d_model=8, total=request)

    assert model.encode(pa.table({"left": [2.0], "right": [5.0]}))["/total"].content.item() == 7.0


@pytest.mark.parametrize("query", [("left", "right"), ()])
def test_extensions_without_derive_reject_query_tuples(query):
    with pytest.raises(ValueError, match="no derive component"):
        rf.Model(preset="small", d_model=8, total=rf.Number(query=query))


@pytest.mark.parametrize(
    "expression,depth,path", [("value", 0, "value"), ("../value", 1, "value"), ("../../data.value", 2, "data.value")]
)
def test_source_scope_preserves_the_native_query_after_ancestor_prefixes(expression, depth, path):
    assert scope(expression) == (depth, path)


def test_ancestor_scope_cannot_escape_selected_model_root():
    with pytest.raises(ValueError, match="ascends 1 owner branches.*only 0 ancestor"):
        rf.Model(preset="small", d_model=8, value=rf.Number(query="../value"))
    with pytest.raises(ValueError, match="ascends 2 owner branches.*only 1 ancestor"):
        rf.Model(preset="small", d_model=8, events=rf.Branch(length=2, value=rf.Number(query="../../value")))
    with pytest.raises(ValueError):
        rf.Branch(length=2, query="../events", value=rf.Number)
    with pytest.raises(ValueError):
        rf.Mask(query="../selected")


def test_selected_root_is_the_upper_bound_for_ancestor_queries(arithmetic):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        query="payload",
        events=rf.Branch(length=2, total=Request(query=("../base", "value"))),
    )
    source = pa.Table.from_pylist(
        [
            {"base": 1000.0, "payload": {"base": 10.0, "events": [{"value": 2.0}]}},
            {"base": 2000.0, "payload": None},
            {"base": 3000.0, "payload": {"base": 20.0, "events": [{"value": 3.0}]}},
        ]
    )

    field = model.encode(source)["/events/total"]
    assert field.content.reshape(-1).tolist() == [12.0, 0.0, 23.0]
    assert field.state.reshape(-1).tolist() == [Tokens.valued, Tokens.padded, Tokens.valued]


def test_single_source_extensions_can_broadcast_ancestors_without_derive():
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(length=2, overflow="tail", source=rf.Number(query="../context.value")),
    )
    source = pa.Table.from_pylist(
        [
            {"context": {"value": 3.0}, "events": [{}, {}, {}]},
            {"context": {"value": None}, "events": [{}, None]},
            {"context": None, "events": [{}]},
        ]
    )
    field = model.encode(source)["/events/source"]

    assert field.content.tolist() == [[[3.0, 3.0]], [[0.0, 0.0]], [[0.0, 0.0]]]
    assert field.state.tolist() == [
        [[Tokens.valued, Tokens.valued]],
        [[Tokens.null, Tokens.padded]],
        [[Tokens.padded, Tokens.padded]],
    ]


def test_nested_tail_crops_and_null_items_preserve_ancestor_identity_and_chunking(arithmetic):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        groups=rf.Branch(
            length=2,
            overflow="tail",
            events=rf.Branch(
                length=2,
                overflow="tail",
                local=Request(query=("../bias", "value")),
                root=Request(query=("../../root_bias", "value")),
            ),
        ),
    )
    source = pa.Table.from_pylist(
        [
            {
                "root_bias": 1000.0,
                "groups": [
                    {"bias": 10.0, "events": [{"value": 1.0}]},
                    None,
                    {"bias": 20.0, "events": [{"value": 3.0}, None, {"value": 5.0}]},
                ],
            },
            {
                "root_bias": 2000.0,
                "groups": [
                    {"bias": 30.0, "events": [{"value": 6.0}]},
                    {"bias": 40.0, "events": [None, {"value": 8.0}, {"value": 9.0}]},
                    None,
                ],
            },
        ]
    )
    chunked = pa.concat_tables([source.slice(0, 1), source.slice(1, 1)])
    first, second = model.encode(source), model.encode(chunked)

    assert first["/groups/events/local"].content.tolist() == [[[[0.0, 0.0], [0.0, 25.0]]], [[[48.0, 49.0], [0.0, 0.0]]]]
    assert first["/groups/events/root"].content.tolist() == [
        [[[0.0, 0.0], [0.0, 1005.0]]],
        [[[2008.0, 2009.0], [0.0, 0.0]]],
    ]
    for address in ("/groups/events/local", "/groups/events/root"):
        torch.testing.assert_close(first[address].content, second[address].content)
        torch.testing.assert_close(first[address].state, second[address].state)


def test_random_windows_share_parent_coordinates_across_sources_and_arrow_chunks(arithmetic):
    _, Request = arithmetic
    schema = rf.Model(
        preset="small",
        d_model=8,
        groups=rf.Branch(
            length=2,
            overflow="random",
            events=rf.Branch(
                length=3,
                overflow="random",
                value=rf.Number,
                local=Request(query=("../bias", "value")),
                root=Request(query=("../../root_bias", "value")),
            ),
        ),
    ).schema
    source = pa.Table.from_pylist(
        [
            {
                "root_bias": float(row * 10000),
                "groups": [
                    {
                        "bias": float(row * 1000 + group * 10),
                        "events": [
                            None if index == 3 else {"value": float(row * 10000 + group * 100 + index)}
                            for index in range(8)
                        ],
                    }
                    for group in range(5)
                ],
            }
            for row in range(4)
        ]
    )
    options = dict(seed=7, epoch=2, ordinal=3)
    resolved = coalesce(source, schema, rf.Strata.train, **options)
    chunked = coalesce(pa.concat_tables([source.slice(0, 2), source.slice(2)]), schema, rf.Strata.train, **options)
    raw = resolved.fields["/groups/events/value"].pristine
    values = np.asarray(raw.values)
    rows = (values // 10000).astype(int)
    groups = ((values % 10000) // 100).astype(int)
    expected = {"local": values + rows * 1000 + groups * 10, "root": values + rows * 10000}
    for name in ("local", "root"):
        field = resolved.fields[f"/groups/events/{name}"].pristine
        assert field.placement.equals(raw.placement)
        np.testing.assert_array_equal(np.asarray(field.values), expected[name])
        other = chunked.fields[f"/groups/events/{name}"].pristine
        assert other.values.equals(field.values)
        assert other.state.equals(field.state)


@pytest.mark.parametrize("collections", [[None, [], [None], [{"value": 2.0}]], [None, [], [None]], [None, []], []])
def test_derived_sources_preserve_empty_and_null_collection_geometry(arithmetic, collections):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(length=2, total=Request(query=("../base", "value"))),
    )
    item = pa.struct([("value", pa.float64())])
    source = pa.table(
        {
            "base": pa.array([10.0] * len(collections), type=pa.float64()),
            "events": pa.array(collections, type=pa.list_(item)),
        }
    )
    encoded = model.encode(source)
    field = encoded["/events/total"]
    width = max((len(items or []) for items in collections), default=0)

    assert tuple(field.state.shape) == (len(collections), 1, width)
    if len(collections) == 4:
        assert field.state.reshape(-1).tolist() == [Tokens.padded, Tokens.padded, Tokens.padded, Tokens.valued]
        assert field.content.reshape(-1).tolist() == [0.0, 0.0, 0.0, 12.0]
    else:
        assert field.state.eq(Tokens.padded).all()
    axis = encoded["/"]["/events"]
    assert axis.valid.reshape(-1).tolist() == [items is not None for items in collections]


@pytest.mark.parametrize("fault", ["result", "array", "count", "presence_type", "presence_null"])
def test_derive_result_validation_owns_shape_and_presence_invariants(arithmetic, fault):
    extension, Request = arithmetic

    def invalid(inputs, *, request, address, rows):
        if fault == "result":
            return pa.array([1.0] * rows)
        values = [1.0] * rows if fault == "array" else pa.array([1.0] * (rows - (fault == "count")))
        present = pa.array([True] * rows)
        if fault == "presence_type":
            present = pa.array([1] * rows)
        if fault == "presence_null":
            present = pa.array([True, None])
        return Projection(values=values, present=present)

    extension.components[Component.derive] = invalid
    model = rf.Model(preset="small", d_model=8, total=Request(query=("left", "right")))
    with pytest.raises((TypeError, ValueError), match="derived field '/total'"):
        model.encode(pa.table({"left": [2.0, 3.0], "right": [4.0, 5.0]}))


@pytest.mark.parametrize("source", [{"context": [1.0, 2.0]}, {"context": [1.0, 2.0], "left": [3.0, 4.0]}])
def test_full_reconstruction_derived_targets_predict_without_required_source_columns(arithmetic, source):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        context=rf.Number,
        total=Request(query=("left", "future"), mask=True),
    ).eval()
    inputs = pa.table(source)
    encoded = model.encode(inputs, strata="predict")["/total"]
    output = model.predict(inputs)

    assert encoded.present.eq(False).all()
    assert encoded.inferred.all()
    assert output.height == 2
    assert "total" in output["predictions"].struct.fields


def test_target_vacancy_preserves_derived_target_poison_invariance(arithmetic):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        context=rf.Number,
        total=Request(query=("left", "future"), mask=True),
    ).eval()
    source = pa.table({"context": [1.0, 2.0], "left": [3.0, 4.0]})
    missing = model.predict(source)
    poisoned = model.predict(source.append_column("future", pa.array([1e20, float("nan")])))

    assert missing.equals(poisoned)


def test_inherited_full_reconstruction_keeps_branch_geometry_without_derived_target_sources(arithmetic):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        context=rf.Number,
        events=rf.Branch(
            length=2,
            overflow="tail",
            mask=True,
            total=Request(query=("left", "future")),
        ),
    ).eval()
    item = pa.struct([("left", pa.float64())])
    source = pa.table(
        {
            "context": [1.0, 2.0, 3.0],
            "events": pa.array([[{"left": 1.0}, None, {"left": 3.0}], [], None], type=pa.list_(item)),
        }
    )
    encoded = model.encode(source, strata="predict")
    field = encoded["/events/total"]

    assert tuple(field.state.shape) == (3, 1, 2)
    assert field.inferred.tolist() == [[[False, True]], [[False, False]], [[False, False]]]
    axis = encoded["/"]["/events"]
    assert axis.lengths.reshape(-1).tolist() == [2, 0, 0]
    assert axis.valid.reshape(-1).tolist() == [True, True, False]
    output = model.predict(source)["predictions"].to_list()
    assert len(output[0]["events"]) == 2
    assert output[1]["events"] == []
    assert output[2]["events"] is None


@pytest.mark.parametrize("strata", ["train", "validate", "test"])
def test_missing_derived_target_sources_remain_strict_outside_prediction(arithmetic, strata):
    _, Request = arithmetic
    model = rf.Model(preset="small", d_model=8, total=Request(query=("left", "future"), mask=True))
    with pytest.raises(MissingFieldError, match="future"):
        model.encode(pa.table({"left": [3.0]}), strata=strata)


@pytest.mark.parametrize(
    "mask",
    [False, rf.Mask(query="selected", reconstruct=True), rf.Mask(rate=0.5, reconstruct=True)],
)
def test_missing_input_or_conditional_derived_sources_do_not_become_vacant(arithmetic, mask):
    _, Request = arithmetic
    model = rf.Model(preset="small", d_model=8, total=Request(query=("left", "future"), mask=mask))
    with pytest.raises(MissingFieldError, match="future"):
        model.encode(pa.table({"left": [3.0], "selected": [True]}), strata="predict")


def test_target_vacancy_does_not_hide_derived_type_errors(arithmetic):
    _, Request = arithmetic
    model = rf.Model(preset="small", d_model=8, total=Request(query=("left", "future"), mask=True))
    with pytest.raises(pa.ArrowException):
        model.encode(pa.table({"left": [3.0], "future": ["not numeric"]}), strata="predict")


def test_canonical_full_rate_reconstruction_retains_masked_input_tokens_without_sources(arithmetic):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        total=Request(query=("left", "future"), mask=rf.Mask(rate=1.0, reconstruct=True)),
    )
    field = model.encode(pa.table({"context": [1.0, 2.0]}), strata="predict")["/total"]

    assert field.present.all()
    assert field.inferred.all()
    assert field.state.eq(Tokens.masked).all()


def test_explicit_nonderived_target_query_keeps_its_existing_missing_source_error():
    model = rf.Model(preset="small", d_model=8, total=rf.Number(query="future", mask=True))
    with pytest.raises(MissingFieldError, match="future"):
        model.encode(pa.table({"left": [3.0]}), strata="predict")
