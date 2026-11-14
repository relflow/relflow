"""Named presets resolve defaults without replacing explicit schema intent."""

import inspect

import pyarrow as pa
import pytest

import relflow as rf
from relflow.tensorfields.base import TENSORFIELDS
from tests.architecture.test_extension_contract import build_extension

PROFILES = [
    ("small", 128, 2, 1, 4, 2, 1, 4),
    ("medium", 64, 2, 2, 8, 4, 1, 1),
]


@pytest.mark.parametrize("name,width,root_depth,branch_depth,heads,ffnx,root_outputs,branch_outputs", PROFILES)
def test_factories_apply_presets_to_root_nested_branches_and_leaves(
    name, width, root_depth, branch_depth, heads, ffnx, root_outputs, branch_outputs
):
    model = rf.Model(
        preset=name,
        d_model=16,
        amount=rf.Number,
        orders=rf.Branch(
            length=3,
            overflow="tail",
            merchant=rf.Category,
            items=rf.Branch(length=2, quantity=rf.Number),
        ),
        total=rf.Number(mask=True, objective="huber"),
        label=rf.Category(mask=True, topk=[3]),
    )

    # Full default widths are covered without module allocation in test_preset_resolution.
    assert model.schema.d_model == 16
    assert model.preset.d_model == width
    assert model.preset.name == name
    assert model.preset.revision == 1
    assert model.schema.fields.name is None
    assert model.schema.fields.address == rf.Address()
    assert model.schema.fields.length == 1
    assert model.schema.branches["/orders"].length == 3
    assert model.schema.branches["/orders"].overflow == "tail"
    assert model.schema.branches["/orders/items"].length == 2
    for address, branch in model.schema.branches.items():
        assert branch.n_layers == (root_depth if address == rf.Address() else branch_depth)
        assert branch.n_heads == (heads if address == rf.Address() else 4)
        assert branch.ffnx == (ffnx if address == rf.Address() else 8)
        assert branch.attention == "mha"
        assert branch.dropout == 0.0
        assert branch.reduction == rf.Attention(n_outputs=root_outputs if address == rf.Address() else branch_outputs)
    for request in model.schema.requests.values():
        assert request.pooling == "query"
        assert request.n_heads == 4
        assert request.ffnx == 8
        assert request.n_linear == 1
        assert request.dropout == 0.0
        assert request.decoder_position is None
    assert model.schema.requests["/total"].objective == "huber"
    assert model.schema.requests["/label"].topk == [3]
    serialized = model.schema.model_dump(mode="json")
    assert rf.Schema.model_validate(serialized).model_dump(mode="json") == serialized


def test_constructor_exposes_typed_preset_and_architecture_options():
    parameters = inspect.signature(rf.Model).parameters
    for option in (
        "preset",
        "d_model",
        "n_layers",
        "n_heads",
        "fields",
        "attention",
        "reduction",
        "dropout",
        "batch_size",
        "optimizer",
        "scheduler",
    ):
        assert parameters[option].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[option].annotation is not inspect.Parameter.empty
    assert parameters["preset"].default is None


def test_generated_preset_methods_and_factory_are_removed():
    assert all(not hasattr(rf.Model, name) for name in ("xs", "sm", "md", "lg", "xl"))
    assert all(not hasattr(rf.presets, name) for name in ("XS", "SM", "MD", "LG", "XL"))
    assert not hasattr(rf.presets, "factory")
    assert not hasattr(rf.Model, "with_preset")


def test_explicit_old_defaults_and_none_survive_nested_binding():
    model = rf.Model(
        preset="medium",
        d_model=32,
        n_layers=1,
        n_heads=4,
        attention=None,
        reduction=None,
        dropout=None,
        events=rf.Branch(
            n_layers=1,
            n_heads=4,
            dropout=None,
            attention=None,
            reduction=None,
            items=rf.Branch(
                n_layers=1,
                n_heads=4,
                attention="mha",
                reduction=rf.Attention(),
                value=rf.Number(n_heads=4, dropout=None, decoder_position=None, n_linear=1),
            ),
        ),
    )

    for address in ("/", "/events", "/events/items"):
        branch = model.schema.branches[address]
        assert branch.n_layers == 1
        assert branch.n_heads == 4
    for address in ("/", "/events"):
        branch = model.schema.branches[address]
        assert branch.attention is None
        assert branch.reduction is None
        assert branch.dropout is None
    items = model.schema.branches["/events/items"]
    assert items.attention == "mha"
    assert items.reduction == rf.Attention()
    assert items.dropout == 0.0
    leaf = model.schema.requests["/events/items/value"]
    assert leaf.n_heads == 4
    assert leaf.dropout is None
    assert leaf.decoder_position is None
    assert leaf.n_linear == 1


@pytest.mark.parametrize("reduction", [rf.Attention(), rf.Attention(n_outputs=3, n_heads=2), rf.Mean(), None])
def test_explicit_reduction_replaces_the_whole_profile(reduction):
    model = rf.Model(
        preset="medium",
        d_model=32,
        reduction=reduction,
        events=rf.Branch(reduction=reduction, value=rf.Number),
    )

    assert model.schema.fields.reduction == reduction
    assert model.schema.branches["/events"].reduction == reduction


def test_root_and_branch_overrides_do_not_change_other_scopes():
    model = rf.Model(
        preset="medium",
        d_model=32,
        n_layers=1,
        n_heads=2,
        attention="mqa",
        dropout=0.1,
        events=rf.Branch(
            n_layers=1,
            n_heads=4,
            dropout=0.2,
            items=rf.Branch(value=rf.Number),
            label=rf.Category(mask=True, pooling="mean", n_linear=2, decoder_position=False),
        ),
    )

    root = model.schema.fields
    assert (root.n_layers, root.n_heads, root.attention, root.dropout) == (1, 2, "mqa", 0.1)
    branch = model.schema.branches["/events"]
    assert (branch.n_layers, branch.n_heads, branch.dropout) == (1, 4, 0.2)
    nested = model.schema.branches["/events/items"]
    assert (nested.n_layers, nested.n_heads, nested.attention, nested.dropout) == (2, 4, "mha", 0.0)
    for request in model.schema.requests.values():
        assert request.n_heads == 4
        assert request.dropout == 0.0
    label = model.schema.requests["/events/label"]
    assert label.pooling == "mean"
    assert label.n_linear == 2
    assert label.decoder_position is False
    assert root.reduction.n_heads is None
    assert root.reduction.dropout is None
    assert branch.reduction.n_heads is None
    assert branch.reduction.dropout is None


def test_reusing_definitions_does_not_capture_another_presets_defaults():
    value = rf.Number(objective="mse", mask=True)
    events = rf.Branch(length=3, records=rf.Branch(length=2, value=value))
    original = events.model_dump(mode="json")
    supplied = [(node, node.model_fields_set.copy()) for node in (events, *events.descendants, value)]

    small = rf.Model(preset="small", events=events)
    large = rf.Model(preset="medium", d_model=32, events=events)

    assert small.schema.branches["/events"].reduction.n_outputs == 4
    assert large.schema.branches["/events"].reduction.n_outputs == 1
    assert small.schema.fields.ffnx == 2
    assert large.schema.fields.ffnx == 4
    assert events.model_dump(mode="json") == original
    assert value.name is None
    assert value.parent is None
    for node, fields in supplied:
        assert node.model_fields_set == fields


def test_factory_constructs_subclass_and_forwards_runtime_configuration():
    class Specialized(rf.Model):
        pass

    optimizer = rf.adamw(learning_rate=1e-3)
    scheduler = {"interval": "epoch"}
    model = Specialized(
        preset="small",
        fields={"length": rf.Number, "preset": rf.Number},
        profile=rf.Number,
        cls=rf.Number,
        size=rf.Number,
        md=rf.Number,
        batch_size=7,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert isinstance(model, Specialized)
    assert model.batch_size == 7
    assert model.optimizer is optimizer
    assert model.scheduler is scheduler
    assert set(model.schema.requests) == {"/length", "/preset", "/profile", "/cls", "/size", "/md"}


@pytest.mark.parametrize("name", [profile[0] for profile in PROFILES])
def test_factory_rejects_an_already_resolved_schema(name):
    schema = rf.Schema.from_tree(d_model=8, n_layers=1, n_heads=2, value=rf.Number)

    with pytest.raises(TypeError, match=r"Model\(schema="):
        rf.Model(preset=name, schema=schema)


def test_ordinary_constructor_preserves_required_dimensions_and_no_policy():
    with pytest.raises(TypeError, match="d_model, n_layers, n_heads"):
        rf.Model(value=rf.Number)

    model = rf.Model(d_model=8, n_layers=1, n_heads=2, value=rf.Number)
    restored = rf.Model(schema=model.schema)

    assert model.preset is None
    assert restored.preset is None
    assert model.schema.requests["/value"].n_heads == 4
    assert model.schema.requests["/value"].dropout is None


def test_width_override_validates_nested_explicit_heads():
    with pytest.raises(ValueError, match=r"/events"):
        rf.Model(preset="small", d_model=32, events=rf.Branch(n_heads=6, value=rf.Number))


@pytest.mark.parametrize("name", [profile[0] for profile in PROFILES])
def test_common_leaf_defaults_apply_to_late_registered_extensions(name):
    extension, Request = build_extension()
    try:
        model = rf.Model(
            preset=name,
            d_model=32,
            events=rf.Branch(value=Request(mask=True, family="bytes")),
        )

        request = model.schema.requests["/events/value"]
        assert isinstance(request, Request)
        assert request.family == "bytes"
        assert request.n_heads == 4
        assert request.dropout == 0.0
        assert request.pooling == "query"
        assert request.mask == rf.Number(mask=True).mask
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_small_predicts_nested_targets_without_padding_and_separate_root_embedding():
    model = rf.Model(
        preset="small", embed=True, events=rf.Branch(length=2, value=rf.Number, answer=rf.Number(mask=True))
    )
    inputs = pa.Table.from_pylist([{"events": [{"value": 1.0}]}, {"events": [{"value": 2.0}, {"value": 3.0}]}])

    result = pa.table(model.predict(inputs))

    assert result.num_rows == 2
    predictions = result["predictions"].combine_chunks()
    representations = result["representations"].combine_chunks().values
    assert representations.field("embedding").type == pa.list_(pa.field("item", pa.float32()), 128)
    assert representations.field("path").to_pylist() == [[], []]
    events = predictions.field("events")
    assert pa.types.is_large_list(events.type)
    assert events.offsets.to_pylist() == [0, 1, 3]
    assert events.values.field("answer").field("inferred").to_pylist() == [True, True, True]
