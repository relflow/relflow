"""Size policies resolve once without changing reusable schema definitions."""

from typing import Literal

import pydantic
import pytest

import relflow as rf
from relflow.presets import LG, MD, SM, XL, XS, Preset
from relflow.presets.base import resolve, subtree


@pytest.mark.parametrize(
    ("preset", "width", "heads", "root_layers", "branch_layers", "root_outputs", "branch_outputs"),
    [
        (XS, 64, 2, 1, 1, 1, 1),
        (SM, 128, 4, 2, 1, 2, 2),
        (MD, 256, 8, 3, 2, 4, 4),
        (LG, 384, 8, 4, 2, 8, 4),
        (XL, 512, 8, 6, 3, 16, 8),
    ],
)
def test_profiles_resolve_each_scope_at_every_depth(
    preset, width, heads, root_layers, branch_layers, root_outputs, branch_outputs
):
    schema = resolve(
        preset,
        value=rf.Number,
        target=rf.Category(mask=True),
        items=rf.Branch(length=3, records=rf.Branch(length=5, value=rf.Number)),
    )

    assert schema.d_model == width
    for address, branch in schema.branches.items():
        root = address == rf.Address()
        assert branch.n_layers == (root_layers if root else branch_layers)
        assert branch.n_heads == heads
        assert branch.attention == "mha"
        assert branch.dropout == 0.0
        assert branch.reduction == rf.Attention(n_outputs=root_outputs if root else branch_outputs)
    for leaf in schema.requests.values():
        assert (leaf.n_heads, leaf.pooling, leaf.n_linear, leaf.dropout, leaf.decoder_position) == (
            heads,
            "query",
            1,
            0.0,
            None,
        )
    assert schema.fields.length == 1
    assert schema.branches[rf.Address("items")].length == 3
    assert schema.branches[rf.Address("items", "records")].length == 5
    assert rf.Schema.model_validate_json(schema.model_dump_json()).model_dump() == schema.model_dump()


@pytest.mark.parametrize("reduction", [None, rf.Mean(), rf.Attention()])
def test_explicit_defaults_and_none_remain_local_atomic_overrides(reduction):
    schema = resolve(
        XL,
        d_model=64,
        n_layers=1,
        n_heads=2,
        attention=None,
        dropout=None,
        reduction=reduction,
        items=rf.Branch(
            n_layers=1,
            n_heads=4,
            attention=None,
            dropout=None,
            reduction=reduction,
            target=rf.Number(mask=True, n_heads=4, n_linear=1, dropout=None, decoder_position=None),
            implicit=rf.Number,
        ),
        implicit=rf.Branch(value=rf.Number),
    )

    for address in (rf.Address(), rf.Address("items")):
        branch = schema.branches[address]
        assert branch.n_layers == 1
        assert branch.attention is None
        assert branch.dropout is None
        assert branch.reduction == reduction
    assert schema.fields.n_heads == 2
    assert schema.branches[rf.Address("items")].n_heads == 4
    assert schema.branches[rf.Address("implicit")].n_layers == 3
    assert schema.branches[rf.Address("implicit")].n_heads == 8
    target = schema.requests[rf.Address("items", "target")]
    assert (target.n_heads, target.n_linear, target.dropout, target.decoder_position) == (4, 1, None, None)
    assert schema.requests[rf.Address("items", "implicit")].n_heads == 8


def test_reused_nested_definitions_remain_unresolved_and_independent():
    leaf = rf.Category(topk=[2], mask=True)
    definition = rf.Branch(length=3, child=rf.Branch(target=leaf))
    before = definition.model_dump()
    supplied = definition.model_fields_set.copy()
    small = resolve(XS, first=definition, second=definition)
    large = resolve(XL, first=definition)

    assert definition.model_dump() == before
    assert definition.model_fields_set == supplied
    assert definition.name is None and definition.parent is None
    assert leaf.name is None and leaf.parent is None
    assert small.branches[rf.Address("first", "child")].n_heads == 2
    assert large.branches[rf.Address("first", "child")].n_heads == 8
    small.requests[rf.Address("first", "child", "target")].topk.append(3)
    assert small.requests[rf.Address("second", "child", "target")].topk == [2]
    assert large.requests[rf.Address("first", "child", "target")].topk == [2]
    assert leaf.topk == [2]


def test_late_extension_uses_common_profile_without_losing_extension_fields():
    extension = rf.Extension(name="preset_extension", types=(float,))
    try:

        @extension.register
        class Request(rf.RequestBase):
            type: Literal["preset_extension"] = "preset_extension"
            scale: float = pydantic.Field(default=1.0, gt=0)
            fields: int = 7

        schema = resolve(MD, custom=Request(scale=3, fields=9), items=rf.Branch(custom=Request))
        request = schema.requests[rf.Address("custom")]
        assert isinstance(request, Request)
        assert (request.scale, request.fields, request.n_heads) == (3, 9, 8)
        assert schema.requests[rf.Address("items", "custom")].n_heads == 8
        assert rf.Schema.model_validate_json(schema.model_dump_json()).model_dump() == schema.model_dump()
    finally:
        rf.TENSORFIELDS.pop("preset_extension", None)


def test_profile_words_remain_available_as_ordinary_child_names():
    schema = resolve(XS, preset=rf.Number, size=rf.Number, md=rf.Number)
    assert list(schema.requests) == [rf.Address("preset"), rf.Address("size"), rf.Address("md")]


def test_policy_snapshot_is_frozen_and_can_default_new_nodes_after_restore():
    restored = Preset.model_validate_json(XL.model_dump_json())
    assert restored == XL
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        restored.d_model = 64
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        restored.branch.n_heads = 4
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        restored.branch.reduction.n_outputs = 1

    definition = rf.Branch(value=rf.Number, child=rf.Branch(target=rf.Category(n_heads=4)))
    live = subtree(XL, "added", definition)
    loaded = subtree(restored, "added", definition)
    assert live.model_dump() == loaded.model_dump()
    assert loaded.n_heads == 8
    assert loaded.fields[0].n_heads == 8
    assert loaded.fields[1].fields[0].n_heads == 4


@pytest.mark.parametrize(
    ("options", "address"),
    [
        ({"d_model": 30, "value": rf.Number}, "/"),
        (
            {"d_model": 24, "n_heads": 4, "items": rf.Branch(n_heads=10, value=rf.Number)},
            "/items",
        ),
        ({"d_model": 24, "target": rf.Number(mask=True, n_heads=10)}, "/target"),
        ({"reduction": rf.Attention(n_heads=6), "value": rf.Number}, "/"),
    ],
)
def test_effective_attention_geometry_reports_address_and_values(options, address):
    with pytest.raises(ValueError) as error:
        resolve(MD, **options)
    message = str(error.value)
    assert repr(address) in message
    assert "d_model=" in message and "n_heads=" in message
    assert "override" in message


def test_head_geometry_only_applies_to_actual_attention_modules():
    schema = resolve(
        XS,
        d_model=1,
        attention=None,
        reduction=rf.Mean(),
        input=rf.Number,
        target=rf.Number(mask=True, pooling="mean"),
    )
    assert schema.d_model == 1

    scalar = resolve(
        XS,
        d_model=2,
        attention=None,
        reduction=rf.Attention(position=False),
        target=rf.Number(mask=True),
    )
    assert scalar.d_model == 2
    with pytest.raises(ValueError, match="leaf '/items/target'.*at least 2 dimensions"):
        resolve(
            XS,
            d_model=2,
            attention=None,
            reduction=rf.Mean(),
            items=rf.Branch(length=2, attention=None, reduction=None, target=rf.Number(mask=True)),
        )
