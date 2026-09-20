"""Parents bind independent, named nodes from reusable field definitions."""

import inspect
from typing import Literal

import pydantic
import pytest
import torch

import relflow as rf


def test_parent_mappings_accept_names_that_collide_with_configuration():
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        fields={"name": rf.Number, "d_model": rf.Number, "fields": rf.Number},
        items=rf.Branch(
            length=3,
            fields={"length": rf.Number, "mask": rf.Number, "description": rf.Number},
            name=rf.Category(),
        ),
    )
    assert list(model.schema.requests) == [
        "/name",
        "/d_model",
        "/fields",
        "/items/length",
        "/items/mask",
        "/items/description",
        "/items/name",
    ]
    assert model.schema.branches["/items"].length == 3


def test_reused_definitions_bind_independent_nodes_without_changing_templates():
    category = rf.Category(topk=[2], mask=True)
    branch = rf.Branch(length=3, left=category, right=category)
    model = rf.Model(d_model=16, n_layers=1, n_heads=4, first=branch, second=branch)

    assert category.name is None and category.parent is None
    assert branch.name is None and branch.parent is None
    assert all(child.parent is branch for child in branch.fields)
    nodes = list(model.schema.requests.values())
    assert len({id(node) for node in nodes}) == 4
    assert all(node.mask == category.mask for node in nodes)
    nodes[0].topk.append(3)
    assert all(node.topk == [2] for node in nodes[1:])
    assert category.topk == [2]
    assert all(child.topk == [2] for child in branch.fields)


def test_mapping_keyword_collisions_do_not_bind_or_mutate_definitions():
    field = rf.Number()
    with pytest.raises(ValueError, match="duplicate field name.*amount"):
        rf.Branch(fields={"amount": field}, amount=field)
    assert field.name is None and field.parent is None


def test_extend_binds_reserved_names_and_keeps_rejected_mutations_atomic():
    field = rf.Number()
    model = rf.Model(d_model=16, n_layers=1, n_heads=4, items=rf.Branch(value=field))
    selected = rf.where("name") == "items"
    model.extend(selected, fields={"include_root": field}, score=field)
    assert "/items/include_root" in model.schema.requests
    assert "/items/score" in model.schema.requests
    before = model.schema.model_dump()
    with pytest.raises(ValueError, match="duplicate field name"):
        model.extend(selected, value=field, other=field)
    assert model.schema.model_dump() == before
    assert field.name is None and field.parent is None


def test_named_checkpoint_restoration_preserves_masks_aliases_and_weights(tmp_path):
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        items=rf.Branch(length=3, label=rf.Category(mask=True)),
    )
    restored = rf.Model.load(model.save(tmp_path / "named.ckpt"))
    assert restored.schema.model_dump() == model.schema.model_dump()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
    request = restored.schema.requests["/items/label"]
    assert request.parent is restored.schema.branches["/items"]
    assert request.mask[0].skip


def test_late_extension_fields_keep_the_same_binding_and_validation_contract():
    extension = rf.Extension(name="parent_named", types=(float,))
    try:

        @extension.register
        class Request(rf.RequestBase):
            type: Literal["parent_named"] = "parent_named"
            scale: float = pydantic.Field(default=1, gt=0)
            fields: int = 7

        definition = Request(scale=3, fields=9, mask=0.25)
        schema = rf.Schema.from_tree(d_model=16, n_layers=1, n_heads=4, custom=definition)
        restored = rf.Schema.model_validate_json(schema.model_dump_json())
        assert restored.requests["/custom"].scale == 3
        assert restored.requests["/custom"].fields == 9
        assert restored.requests["/custom"].mask == definition.mask
        assert definition.name is None
        with pytest.raises(TypeError, match="names come from the parent"):
            Request(name="custom")
        with pytest.raises(ValueError, match="scale"):
            Request.model_validate({"name": "custom", "scale": 0})
        assert "name" not in inspect.signature(Request).parameters
    finally:
        rf.TENSORFIELDS.pop("parent_named", None)


@pytest.mark.parametrize("constructor", [rf.Number, rf.Category, rf.Branch])
def test_public_signatures_and_display_support_anonymous_definitions(constructor):
    assert "name" not in inspect.signature(constructor).parameters
    definition = constructor()
    assert f"[{definition.type}]" in str(definition)
    assert definition.name is None


def test_serialized_validation_still_checks_internal_names():
    with pytest.raises(ValueError, match="name may contain"):
        rf.Number.model_validate({"name": "invalid/name"})
    with pytest.raises(ValueError, match="duplicate field name"):
        rf.Branch.model_validate({"name": "items", "fields": [{"type": "number", "name": "value"}] * 2})
    with pytest.raises(ValueError, match="children require names"):
        rf.Branch.model_validate({"name": "items", "fields": [{"type": "number"}]})
    with pytest.raises(ValueError, match="children require names"):
        rf.Branch(fields={None: rf.Number})
