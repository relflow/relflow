"""Reusable definitions retain explicit options separately from ordinary defaults."""

from itertools import count
from typing import Literal

import pydantic
import pytest

import relflow as rf
from relflow.structs.experiment import bind_tree_field

BRANCH_OPTIONS = {"attention", "n_layers", "n_heads", "reduction", "dropout"}
LEAF_OPTIONS = {"pooling", "n_heads", "n_linear", "dropout", "decoder_position"}


def test_branch_omissions_retain_ordinary_defaults_without_becoming_explicit():
    definition = rf.Branch(value=rf.Number)
    bound = bind_tree_field("items", definition)

    for branch in (definition, bound):
        assert branch.model_fields_set.isdisjoint(BRANCH_OPTIONS)
        assert branch.attention == rf.AttentionMode.mha
        assert branch.n_layers == 1
        assert branch.n_heads == 4
        assert branch.reduction == rf.Attention()
        assert branch.dropout is None
        assert branch.fields[0].model_fields_set.isdisjoint(LEAF_OPTIONS)


@pytest.mark.parametrize(
    "option, value",
    [
        ("attention", "mha"),
        ("attention", None),
        ("n_layers", 1),
        ("n_heads", 4),
        ("reduction", rf.Attention()),
        ("reduction", rf.Mean()),
        ("reduction", None),
        ("dropout", None),
        ("dropout", 0.0),
    ],
)
def test_branch_binding_preserves_explicit_defaults_and_none(option, value):
    definition = rf.Branch(value=rf.Number, **{option: value})
    bound = bind_tree_field("items", definition)

    for branch in (definition, bound):
        assert branch.model_fields_set & BRANCH_OPTIONS == {option}
        assert getattr(branch, option) == value


def test_binding_nested_definitions_preserves_leaf_intent_and_original_parents():
    leaf = rf.Number(pooling="query", n_heads=4, n_linear=1, dropout=None, decoder_position=None)
    inner = rf.Branch(n_layers=1, value=leaf)
    outer = rf.Branch(length=3, entries=inner, category=rf.Category)
    original = outer.model_dump(mode="python")
    left = bind_tree_field("left", outer)
    right = bind_tree_field("right", outer)

    for branch in (outer, left, right):
        assert branch.model_fields_set.isdisjoint(BRANCH_OPTIONS)
        entries, category = branch.fields
        assert entries.model_fields_set & BRANCH_OPTIONS == {"n_layers"}
        assert entries.fields[0].model_fields_set & LEAF_OPTIONS == LEAF_OPTIONS
        assert category.model_fields_set.isdisjoint(LEAF_OPTIONS)
        assert entries.parent is branch
        assert entries.fields[0].parent is entries

    assert outer.model_dump(mode="python") == original
    assert outer.parent is None
    assert inner.parent is None
    assert leaf.parent is None
    assert left.fields[0] is not right.fields[0]
    assert left.fields[0].fields[0] is not right.fields[0].fields[0]


def test_extension_subclasses_keep_custom_defaults_and_intent_when_bound():
    extension = rf.Extension(name="field_intent", types=(float,))
    generations = count()
    try:

        @extension.register
        class Request(rf.RequestBase):
            type: Literal["field_intent"] = "field_intent"
            generation: int = pydantic.Field(default_factory=lambda: next(generations))
            scales: list[float] = pydantic.Field(default_factory=lambda: [1.0])

        class Specialized(Request):
            units: str = "seconds"

        definition = Specialized(units="days", n_heads=4)
        branch = rf.Branch(value=definition)
        bound = bind_tree_field("items", branch).fields[0]

        assert type(bound) is Specialized
        assert bound.generation == definition.generation == 0
        assert bound.model_fields_set == definition.model_fields_set | {"name"}
        assert bound.units == "days"
        assert bound.model_fields_set & LEAF_OPTIONS == {"n_heads"}
        bound.scales.append(2.0)
        assert definition.scales == branch.fields[0].scales == [1.0]
    finally:
        rf.TENSORFIELDS.pop("field_intent", None)
