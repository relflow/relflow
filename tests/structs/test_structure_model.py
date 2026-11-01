from typing import Literal

import pydantic
import pytest

import relflow as rf
from relflow.structs.enums import Overflow
from relflow.structs.experiment import Schema
from relflow.structs.structure import Branch
from relflow.tensorfields.extensions.category import Request as Category


def schema_payload() -> dict:
    return {
        "d_model": 16,
        "fields": {
            "type": "branch",
            "description": "root branch docs",
            "dropout": 0.1,
            "length": 2,
            "fields": [
                {
                    "name": "branch",
                    "type": "branch",
                    "description": "branch docs",
                    "length": 4,
                    "fields": [
                        {
                            "name": "category_leaf",
                            "type": "category",
                            "description": "category docs",
                        }
                    ],
                }
            ],
        },
    }


def test_branch_accepts_keyword_children():
    branch = Branch(category_leaf=Category())

    assert branch.fields[0].name == "category_leaf"


def test_branch_accepts_mapping_and_keyword_children():
    branch = Branch(category_leaf=Category(), fields={"length": Category})

    assert [field.name for field in branch.fields] == ["length", "category_leaf"]


def test_branch_rejects_children_that_are_not_definitions():
    with pytest.raises(TypeError, match="tree field 'unexpected'.*must be a Branch, Leaf, or Leaf class"):
        Branch(category_leaf=Category(), unexpected=0.1)


def test_schema_derives_branches_requests_and_shapes():
    structure = Schema.model_validate(schema_payload())

    assert "/" in structure.branches
    assert "/branch" in structure.branches
    assert "/branch/category_leaf" in structure.requests
    assert structure.branches["/"].length == 1
    assert structure.branches["/"].overflow == Overflow.error
    assert structure.shapes["/branch/category_leaf"] == (1, 4)
    assert structure.overflows("/branch/category_leaf") == (Overflow.error, Overflow.error, Overflow.head)


def test_branch_accepts_overflow_policy():
    branch = Branch(category_leaf=Category(), length=4, overflow="tail")

    assert branch.overflow == Overflow.tail


def test_branch_defaults_to_unbounded_collections():
    branch = Branch(value=rf.Number)

    assert branch.length is None
    assert branch.overflow is None
    assert Branch.model_validate_json(branch.model_dump_json()).model_dump() == branch.model_dump()


@pytest.mark.parametrize("length", [0, -1, True, 2.5, "3"])
def test_branch_limits_require_positive_integers(length):
    with pytest.raises(ValueError, match="length"):
        Branch(length=length, value=rf.Number)


@pytest.mark.parametrize("overflow", ["head", "tail", "random", "error"])
def test_unbounded_branches_cannot_select_overflow(overflow):
    with pytest.raises(ValueError, match="unbounded branch.*finite length"):
        Branch(overflow=overflow, value=rf.Number)
    with pytest.raises(ValueError, match="unbounded branch.*finite length"):
        Branch.model_validate({"name": "items", "overflow": overflow, "fields": []})


def test_bounded_branches_default_to_head_with_an_explicit_null_policy():
    assert Branch(length=3, value=rf.Number).overflow == Overflow.head
    assert Branch(length=3, overflow=None, value=rf.Number).overflow == Overflow.head


def test_schema_keeps_dynamic_shapes_and_normalizes_the_root_before_validation():
    schema = Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "overflow": "error",
                "fields": [
                    {
                        "type": "branch",
                        "name": "events",
                        "reduction": None,
                        "fields": [{"type": "number", "name": "amount"}],
                    }
                ],
            },
        }
    )

    assert schema.fields.length == 1
    assert schema.fields.overflow == Overflow.error
    assert schema.shapes["/events/amount"] == (1, None)
    assert schema.branch_outputs["/events"] is None
    restored = Schema.model_validate_json(schema.model_dump_json())
    assert restored.shapes == schema.shapes
    assert restored.branches["/events"].overflow is None


@pytest.mark.parametrize(
    "options",
    [{"length": 0}, {"length": -1}, {"length": True}, {"length": "3"}, {"overflow": "middle"}],
)
def test_raw_root_normalization_does_not_hide_invalid_configuration(options):
    with pytest.raises(ValueError, match="length|overflow"):
        Schema.model_validate({"d_model": 8, "fields": {**options, "fields": [{"name": "value", "type": "number"}]}})


@pytest.mark.parametrize("options", [{"length": None, "overflow": None}, {"length": 2, "overflow": "tail"}])
def test_valid_raw_root_options_normalize_to_the_singleton(options):
    schema = Schema.model_validate(
        {"d_model": 8, "fields": {**options, "fields": [{"name": "value", "type": "number"}]}}
    )

    assert schema.fields.length == 1
    assert schema.fields.overflow == Overflow.error


def test_collection_bound_mutations_preserve_the_pair_and_roll_back_failures():
    schema = Schema.from_tree(d_model=8, n_layers=1, n_heads=2, events=Branch(length=3, value=rf.Number))
    selected = rf.where("address") == "/events"
    before = schema.model_dump()

    with pytest.raises(ValueError, match="unbounded branch.*finite length"):
        schema.update(selected, length=None)
    assert schema.model_dump() == before

    schema.update(selected, length=None, overflow=None)
    assert schema.branches["/events"].length is None
    assert schema.branches["/events"].overflow is None

    with pytest.raises(ValueError, match="unbounded branch.*finite length"):
        schema.update(selected, overflow="tail")
    assert schema.branches["/events"].length is None
    assert schema.branches["/events"].overflow is None

    schema.update(selected, length=5)
    assert schema.branches["/events"].length == 5
    assert schema.branches["/events"].overflow == Overflow.head

    before = schema.model_dump()
    with pytest.raises(ValueError, match="root must remain a singleton"):
        schema.update(rf.where("address") == "/", length=None, overflow=None)
    assert schema.model_dump() == before


def test_branch_rejects_invalid_overflow_policy():
    with pytest.raises(ValueError):
        Branch(category_leaf=Category(), overflow="middle")


def test_schema_converts_leaf_instances_nested_in_branches():
    structure = Schema(
        d_model=16,
        fields={
            "type": "branch",
            "fields": [
                {"name": "branch", "type": "branch", "fields": [Category.model_validate({"name": "category_leaf"})]}
            ],
        },
    )

    request = structure.requests["/branch/category_leaf"]
    assert type(request) is rf.Category
    assert request.parent is structure.branches["/branch"]
    assert request.address == "/branch/category_leaf"


def test_raw_generic_leaves_materialize_through_the_live_registry():
    extension = rf.Extension(name="generic_materialization", types=(float,))
    try:

        @extension.register(rf.Component.Request)
        class Request(rf.RequestBase):
            type: Literal["generic_materialization"] = "generic_materialization"
            scale: float = pydantic.Field(default=3.0, gt=0)

        generic = rf.RequestBase.model_validate({"name": "value", "type": extension.name})
        assert type(generic) is rf.RequestBase
        branch = Branch.model_validate({"name": "items", "length": 2, "fields": [generic]})
        assert type(branch.fields[0]) is Request
        assert branch.fields[0].scale == 3.0
        assert branch.fields[0].parent is branch

        schema = Schema.model_validate(
            {
                "d_model": 8,
                "fields": {"fields": [{"name": "items", "type": "branch", "length": 2, "fields": [generic]}]},
            }
        )
        request = schema.requests["/items/value"]
        assert type(request) is Request
        assert request.parent is schema.branches["/items"]
        assert request.shape == (1, 2)
        assert generic.parent is None
        assert Schema.model_validate_json(schema.model_dump_json()).model_dump() == schema.model_dump()
    finally:
        rf.TENSORFIELDS.pop(extension.name, None)


def test_schema_depthwise_contains_branch_levels():
    structure = Schema.model_validate(schema_payload())
    assert structure.depthwise == [["/"], ["/branch"]]


def test_schema_string_representation_contains_tree_nodes():
    structure = Schema.model_validate(schema_payload())
    rendered = str(structure)
    assert "type='schema'" in rendered
    assert "fields=Branch(" in rendered
    assert "name='branch'" in rendered
    assert "name='category_leaf'" in rendered


def test_schema_preserves_field_and_branch_descriptions():
    structure = Schema.model_validate(schema_payload())
    assert structure.branches["/"].description == "root branch docs"
    assert structure.branches["/branch"].description == "branch docs"
    assert structure.requests["/branch/category_leaf"].description == "category docs"


def test_schema_uses_direct_branch_dropout():
    structure = Schema.model_validate(schema_payload())

    assert structure.branches["/"].dropout == 0.1
    assert structure.branches["/branch"].dropout is None
    assert structure.requests["/branch/category_leaf"].dropout is None


def test_schema_allows_missing_dropout():
    payload = schema_payload()
    payload["fields"].pop("dropout")
    structure = Schema.model_validate(payload)

    assert structure.branches["/"].dropout is None
    assert structure.branches["/branch"].dropout is None


def test_schema_preserves_direct_field_dropout():
    payload = schema_payload()
    payload["fields"]["fields"][0]["fields"][0]["dropout"] = 0.4

    structure = Schema.model_validate(payload)

    assert structure.requests["/branch/category_leaf"].dropout == 0.4


def test_schema_rejects_undeclared_branch_options():
    payload = schema_payload()
    payload["fields"]["notes"] = "Undeclared option"

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Schema.model_validate(payload)


def test_schema_allows_missing_mask():
    structure = Schema.model_validate(schema_payload())

    assert structure.fields.mask == ()
    assert structure.branches["/branch"].mask == ()
    assert structure.requests["/branch/category_leaf"].mask == ()


def test_inactive_leaf_nodes_are_kept_in_tree_but_removed_from_runtime_maps():
    payload = schema_payload()
    payload["fields"]["fields"][0]["fields"][0]["active"] = False
    payload["fields"]["fields"][0]["fields"][0]["mask"] = True
    payload["fields"]["fields"][0]["fields"][0]["embed"] = True

    structure = Schema.model_validate(payload)
    inactive = structure.select(lambda node: getattr(node, "name", None) == "category_leaf")[0]

    assert inactive.active is False
    assert inactive.address == "/branch/category_leaf"
    assert "/branch/category_leaf" in structure.requests
    assert "/branch/category_leaf" not in structure.active_requests
    assert structure.shapes["/branch/category_leaf"] == (1, 4)
    assert structure.reconstruct == []
    assert structure.embed == []
