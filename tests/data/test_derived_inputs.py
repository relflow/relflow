"""Third-party computed fields share structural sources without datatype dispatch."""

import uuid
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch
from tensordict import tensorclass

import relflow as rf
from relflow.data.query import MissingFieldError, Projection, compile, scope
from relflow.data.ragged import coalesce
from relflow.structs.enums import Component, Tokens
from relflow.structs.experiment import Schema
from relflow.tensorfields.base import TENSORFIELDS, Extension, resolve
from relflow.tensorfields.extensions import number


@pytest.fixture
def arithmetic():
    extension = Extension(
        f"computed_{uuid.uuid4().hex[:8]}",
        types=(int | float,),
        queries=slice(0, None),
        callbacks=number.number.callback_factories,
    )

    @extension.register(Component.Request)
    class Request(number.Numeric):
        type: str = extension.name
        constant: float = 0.0

    @extension.register(Component.TensorField)
    @tensorclass
    class TensorField(number.TensorField):
        @classmethod
        def new(cls, sources, *, schema, context):
            request = schema.requests[sources.address]
            if sources.vacant:
                return super().new(sources, schema=schema, context=context)
            rows = sources.rows
            values = pa.repeat(pa.scalar(request.constant), rows)
            present = pa.repeat(pa.scalar(True), rows)
            for item in sources.inputs:
                values = pc.add(values, item.values)
                present = pc.and_(present, item.present)
            return super().new(
                replace(sources, inputs=(Projection(values, present),)),
                schema=schema,
                context=context,
            )

    for role, component in number.number.components.items():
        if role not in {Component.Request, Component.TensorField}:
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


def test_source_resolution_keeps_observations_before_masking():
    schema = Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        value=rf.Number(mask=rf.Mask(rate=1.0, reconstruct=True, dropout=False)),
    )
    sources = coalesce(pa.table({"value": [2.0, 7.0]}), schema, rf.Strata.train).fields["/value"]
    original = rf.Context(salt=23)

    projection, resolved, observation, binding = resolve(sources, schema=schema, context=original)
    inputs, targets = projection.split()

    assert resolved == original and binding == ()
    assert observation is not None
    assert observation[rf.TensorKey.content]["mean"].item() == 4.5
    assert inputs.dense.reshape(-1).tolist() == [Tokens.masked, Tokens.masked]
    assert targets.values.to_pylist() == [2.0, 7.0]


def test_transformed_values_are_observed_before_tensorization(monkeypatch):
    model = rf.Model(
        preset="small",
        d_model=8,
        elapsed=rf.Duration(
            query=("start", "end"),
            granularity="seconds",
            mask=rf.Mask(rate=1.0, reconstruct=True, dropout=False),
        ),
    )
    source = pa.table(
        {
            "start": pa.array([0, 100], type=pa.timestamp("s")),
            "end": pa.array([60, 220], type=pa.timestamp("s")),
        }
    )
    events = []
    inside_observer = False
    original_observe = number.number.own_components[Component.observe]
    original_encode = number.encode

    def observe(field, *, address, schema, state, learn):
        nonlocal inside_observer
        events.append(("observe", field.values.to_pylist()))
        inside_observer = True
        try:
            result = original_observe(field, address=address, schema=schema, state=state, learn=learn)
        finally:
            inside_observer = False
        events.append(("observed",))
        return result

    def encode(field, *, address):
        if not inside_observer:
            events.append(("encode", field.dense.reshape(-1).tolist()))
        return original_encode(field, address=address)

    monkeypatch.setitem(number.number.own_components, Component.observe, observe)
    monkeypatch.setattr(number, "encode", encode)
    field = model.encode(source, strata="train")["/elapsed"]

    assert events[:3] == [("observe", [60.0, 120.0]), ("observed",), ("encode", [Tokens.masked, Tokens.masked])]
    assert field.targets[rf.TensorKey.content].reshape(-1).tolist() == [60.0, 120.0]


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


def test_query_count_is_read_from_the_live_extension_registration(arithmetic):
    extension, Request = arithmetic
    extension.queries = 1
    request = Request(query=("left", "right"))
    with pytest.raises(ValueError, match="/total.*expects queries 1, got 2"):
        rf.Model(preset="small", d_model=8, total=request)
    extension.queries = slice(0, None)
    model = rf.Model(preset="small", d_model=8, total=request)

    assert model.encode(pa.table({"left": [2.0], "right": [5.0]}))["/total"].content.item() == 7.0


@pytest.mark.parametrize(
    ("allowed", "accepted", "rejected"),
    [
        (2, (2,), (0, 1, 3)),
        ((0, 2), (0, 2), (1, 3)),
        (slice(1, 4), (1, 2, 3), (0, 4)),
        (slice(0, None, 2), (0, 2, 4), (1, 3)),
    ],
)
def test_extension_query_counts_accept_int_tuples_and_slices(arithmetic, allowed, accepted, rejected):
    extension, Request = arithmetic
    extension.queries = allowed
    for count in accepted:
        model = rf.Model(preset="small", d_model=8, total=Request(query=tuple(f"v{i}" for i in range(count))))
        assert len(model.schema.requests["/total"].queries()) == count
    for count in rejected:
        with pytest.raises(ValueError, match="expects queries"):
            rf.Model(preset="small", d_model=8, total=Request(query=tuple(f"v{i}" for i in range(count))))


@pytest.mark.parametrize("query", [("left", "right"), ()])
def test_single_source_extensions_reject_other_source_counts(query):
    with pytest.raises(ValueError, match="expects queries 1"):
        rf.Model(preset="small", d_model=8, total=rf.Number(query=query))


@pytest.mark.parametrize(
    "expression,depth,path", [("value", 0, "value"), ("../value", 1, "value"), ("../../data.value", 2, "data.value")]
)
def test_source_scope_preserves_the_native_query_after_ancestor_prefixes(expression, depth, path):
    owner, parsed = scope(expression)
    assert owner == depth
    assert parsed == compile(path)


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


def test_absolute_queries_align_named_ancestors_with_their_descendants(arithmetic):
    _, Request = arithmetic
    model = rf.Model(
        preset="small",
        d_model=8,
        query="payload",
        groups=rf.Branch(
            query="data.groups",
            length=2,
            overflow="tail",
            events=rf.Branch(
                query="activity.events",
                length=2,
                overflow="tail",
                relative=Request(query=("../bias", "value")),
                absolute=Request(query=("/groups/bias", "/groups/events/value")),
                root=Request(query=("/root_bias", "value")),
                quoted=Request(query=('/groups/["a/b"]', "value")),
            ),
        ),
    )
    source = pa.Table.from_pylist(
        [
            {
                "root_bias": 9999.0,
                "payload": {
                    "root_bias": 100.0,
                    "data": {
                        "groups": [
                            {"bias": 10.0, "a/b": 3.0, "activity": {"events": [{"value": 1.0}]}},
                            {"bias": 20.0, "a/b": 4.0, "activity": {"events": [{"value": 2.0}, {"value": 3.0}]}},
                        ]
                    },
                },
            },
            {
                "root_bias": 8888.0,
                "payload": {
                    "root_bias": 200.0,
                    "data": {"groups": [{"bias": 30.0, "a/b": 5.0, "activity": {"events": [{"value": 4.0}]}}]},
                },
            },
        ]
    )
    encoded = model.encode(source)
    relative = encoded["/groups/events/relative"]
    absolute = encoded["/groups/events/absolute"]
    torch.testing.assert_close(relative.content, absolute.content)
    torch.testing.assert_close(relative.state, absolute.state)
    assert absolute.content.reshape(2, 2, 2).tolist() == [[[11.0, 0.0], [22.0, 23.0]], [[34.0, 0.0], [0.0, 0.0]]]
    assert encoded["/groups/events/root"].content.reshape(2, 2, 2).tolist() == [
        [[101.0, 0.0], [102.0, 103.0]],
        [[204.0, 0.0], [0.0, 0.0]],
    ]
    assert encoded["/groups/events/quoted"].content.reshape(2, 2, 2).tolist() == [
        [[4.0, 0.0], [6.0, 7.0]],
        [[9.0, 0.0], [0.0, 0.0]],
    ]
    chunked = model.encode(pa.concat_tables([source.slice(0, 1), source.slice(1, 1)]))
    for name in ("relative", "absolute", "root", "quoted"):
        address = f"/groups/events/{name}"
        torch.testing.assert_close(encoded[address].content, chunked[address].content)
        torch.testing.assert_close(encoded[address].state, chunked[address].state)


@pytest.mark.parametrize("query", ["/other/value", "/groups/other/value", "/groups//value"])
def test_absolute_queries_reject_non_ancestor_branches(query):
    with pytest.raises(ValueError, match="absolute"):
        rf.Model(preset="small", groups=rf.Branch(events=rf.Branch(value=rf.Number(query=query))), d_model=8)


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
    raw = resolved.fields["/groups/events/value"].single().pristine
    values = np.asarray(raw.values)
    rows = (values // 10000).astype(int)
    groups = ((values % 10000) // 100).astype(int)
    expected = {"local": values + rows * 1000 + groups * 10, "root": values + rows * 10000}
    extension, _ = arithmetic
    for name in ("local", "root"):
        address = rf.Address(f"/groups/events/{name}")
        fields = []
        for batch in (resolved, chunked):
            sources = batch.fields[address]
            field, _, _ = extension.TensorField.new(
                sources,
                schema=schema,
                context=rf.Context(),
            )
            fields.append(field)
        field, other = fields
        np.testing.assert_array_equal(field.content.reshape(-1).numpy()[raw.placement.to_numpy()], expected[name])
        torch.testing.assert_close(other.content, field.content)
        torch.testing.assert_close(other.state, field.state)


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
def test_source_selection_validates_shape_and_presence_invariants(arithmetic, fault):
    _, Request = arithmetic

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

    model = rf.Model(preset="small", d_model=8, total=Request(query=("left", "right")))
    sources = coalesce(pa.table({"left": [2.0, 3.0], "right": [4.0, 5.0]}), model.schema, rf.Strata.train).fields[
        "/total"
    ]
    output = invalid(sources.inputs, request=None, address=None, rows=2)
    with pytest.raises((TypeError, ValueError), match="field '/total'"):
        number.TensorField.new(
            replace(sources, inputs=(output,)),
            schema=model.schema,
            context=rf.Context(),
        )


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


def test_explicit_number_target_query_can_be_missing_during_prediction():
    model = rf.Model(preset="small", d_model=8, total=rf.Number(query="future", mask=True))
    field = model.encode(pa.table({"left": [3.0]}), strata="predict")["/total"]
    assert field.inferred.all()
    assert not field.present.any()
