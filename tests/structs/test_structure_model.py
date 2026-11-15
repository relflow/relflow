from datetime import timedelta
from textwrap import dedent
from typing import Literal

import pydantic
import pytest
import yaml

import relflow as rf
from relflow.structs.enums import Overflow
from relflow.structs.experiment import Schema
from relflow.structs.grain import Steps
from relflow.structs.mixer import Conv
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
    with pytest.raises(TypeError, match="tree field 'unexpected'.*must be a Branch or Leaf instance or class"):
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


def test_plain_branch_resolves_todays_sequence_settings_and_dumps_without_them():
    branch = rf.Branch(value=rf.Number)
    assert branch.grain == rf.Coordinates()
    assert branch.mixer == rf.Transformer()
    assert branch.direction is rf.Direction.bidirectional
    for mode in ("python", "json"):
        assert {"grain", "mixer", "direction"}.isdisjoint(branch.model_dump(mode=mode))


def test_branch_sequence_settings_round_trip_when_changed():
    branch = rf.Branch(mixer=None, value=rf.Number)
    payload = branch.model_dump(mode="json", round_trip=True)
    assert payload["mixer"] is None
    assert rf.Branch.model_validate(payload).mixer is None
    restored = rf.Branch.model_validate({"type": "branch", "fields": [], "grain": {"type": "steps"}})
    assert restored.grain == Steps()


def test_branch_rejects_unknown_direction():
    with pytest.raises(pydantic.ValidationError, match="bidirectional"):
        rf.Branch.model_validate({"type": "branch", "fields": [], "direction": "sideways"})


def test_child_named_items_is_still_a_child():
    branch = rf.Branch(items=rf.Branch(value=rf.Number))
    assert [child.name for child in branch.fields] == ["items"]


def test_time_index_must_name_a_child_leaf():
    with pytest.raises(ValueError, match="time_index='when'.*no child named 'when'"):
        rf.Branch(length=4, time_index="when", ts=rf.Number, value=rf.Number)


def test_time_index_cannot_name_a_child_branch():
    with pytest.raises(ValueError, match="time_index='ts'.*must name a leaf"):
        rf.Branch(length=4, time_index="ts", ts=rf.Branch(length=2, x=rf.Number), value=rf.Number)


def test_time_step_requires_time_index():
    with pytest.raises(ValueError, match="time_step.*requires time_index"):
        rf.Branch(length=4, time_step=2.0, value=rf.Number)


@pytest.mark.parametrize("step", [0.0, -1.0, timedelta(0), timedelta(seconds=-1)])
def test_time_step_must_be_positive(step):
    with pytest.raises(ValueError, match="time_step must be positive"):
        rf.Branch(length=4, time_index="ts", time_step=step, ts=rf.Number)


def test_steps_rejects_unreduced_child_branch():
    with pytest.raises(ValueError, match="child branch 'parts'.*needs a reduction"):
        rf.Branch(length=4, grain=Steps(), parts=rf.Branch(length=2, reduction=None, x=rf.Number))


def test_conv_requires_steps():
    with pytest.raises(ValueError, match="mixer=Conv.*requires grain=Steps"):
        rf.Branch(length=4, mixer=Conv(), value=rf.Number)


def test_plain_branch_dumps_without_time_settings():
    branch = rf.Branch(length=4, ts=rf.Number, value=rf.Number)
    for mode in ("python", "json"):
        assert {"time_index", "time_step"}.isdisjoint(branch.model_dump(mode=mode))


def test_child_named_time_is_still_a_child():
    branch = rf.Branch(time=rf.Number, value=rf.Number)
    assert [child.name for child in branch.fields] == ["time", "value"]


@pytest.mark.parametrize("step", [3600.0, timedelta(hours=1)])
def test_time_settings_round_trip_through_yaml(step):
    branch = rf.Branch(length=4, time_index="ts", time_step=step, ts=rf.Number, value=rf.Number)
    restored = rf.Branch.model_validate(yaml.safe_load(yaml.safe_dump(branch.model_dump(mode="json"))))
    assert restored.time_index == "ts"
    assert restored.time_step == step
    assert type(restored.time_step) is type(step)


def test_chronological_fills_defaults():
    branch = rf.Branch.chronological(length=336, value=rf.Number)
    assert type(branch) is rf.Branch
    assert branch.overflow == Overflow.tail
    assert branch.grain == rf.Steps() and branch.mixer == rf.Conv() and branch.direction == rf.Direction.causal
    assert branch.time_index is None and branch.time_step is None


def test_chronological_unbounded_keeps_overflow_none():
    assert rf.Branch.chronological(value=rf.Number).overflow is None


def test_explicit_arguments_override_defaults():
    branch = rf.Branch.chronological(
        length=8, overflow="head", mixer=None, direction=rf.Direction.bidirectional, value=rf.Number
    )
    assert branch.overflow == Overflow.head and branch.mixer is None
    assert branch.direction == rf.Direction.bidirectional


def test_chronological_with_transformer_and_coordinates_is_allowed():
    rf.Branch.chronological(length=8, grain=rf.Coordinates(), mixer=rf.Transformer(), value=rf.Number)


def test_chronological_rejects_missing_time_index_child():
    with pytest.raises(ValueError, match="no child named 'ts'"):
        rf.Branch.chronological(length=8, time_index="ts", value=rf.Number)


def test_chronological_rejects_conv_with_coordinates():
    with pytest.raises(ValueError, match="requires grain=Steps"):
        rf.Branch.chronological(length=8, grain=rf.Coordinates(), value=rf.Number)


def test_chronological_loads_from_model_yaml_as_a_plain_branch(tmp_path):
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        ett=rf.Branch.chronological(
            length=8, time_index="ts", time_step=3600.0, ts=rf.Number, value=rf.Number(mask=True)
        ),
    )
    path = tmp_path / "model.yaml"
    path.write_text(
        dedent(
            """
            d_model: 16
            n_layers: 1
            n_heads: 4
            fields:
              ett:
                type: branch
                length: 8
                overflow: tail
                grain: {type: steps}
                mixer: {type: conv}
                direction: causal
                time_index: ts
                time_step: 3600.0
                fields:
                  ts: {type: number}
                  value: {type: number, mask: true}
            """
        )
    )
    loaded = rf.Model.from_yaml(path)
    assert loaded.schema.branches["/ett"].model_dump() == model.schema.branches["/ett"].model_dump()
    assert type(loaded.schema.branches["/ett"]) is rf.Branch
    assert "chronological" not in yaml.safe_dump(model.schema.model_dump(mode="json"))


def test_chronological_docstring_warns_about_classification():
    assert "direction=rf.Direction.bidirectional" in rf.Branch.chronological.__doc__


CAUSAL = rf.Direction.causal
FORECAST = rf.Mask(select=rf.Forecast(horizon=2, targets=("value",)), reconstruct=True)


def forecast_mask(targets: tuple[str, ...], known: tuple[str, ...] = ()) -> rf.Mask:
    return rf.Mask(select=rf.Forecast(horizon=2, targets=targets, known=known), reconstruct=True)


def checked(mask: rf.Mask) -> rf.Branch:
    return rf.Branch(
        length=3,
        direction=CAUSAL,
        overflow="tail",
        mask=mask,
        time_index="ts",
        time_step=1.0,
        ts=rf.Number,
        value=rf.Number,
        load=rf.Number,
        tags=rf.Branch(length=2, name=rf.Category),
    )


def test_forecast_with_bidirectional_direction_is_rejected():
    with pytest.raises(ValueError, match=r"branch.*Forecast.*direction=bidirectional.*Direction\.causal"):
        rf.Branch(length=3, mask=FORECAST, value=rf.Number)


@pytest.mark.parametrize("overflow", [None, "head"])
def test_forecast_with_head_overflow_is_rejected(overflow):
    with pytest.raises(ValueError, match=r"branch.*Forecast.*head.*overflow=.tail."):
        rf.Branch(length=3, direction=CAUSAL, overflow=overflow, mask=FORECAST, value=rf.Number)


@pytest.mark.parametrize("overflow", ["tail", "random", "error"])
def test_forecast_keeps_compatible_explicit_overflow_policies(overflow):
    branch = rf.Branch(length=3, direction=CAUSAL, overflow=overflow, mask=FORECAST, value=rf.Number)
    assert branch.overflow == Overflow(overflow)


def test_chronological_forecast_with_bidirectional_direction_is_rejected():
    with pytest.raises(ValueError, match=r"Direction\.causal"):
        rf.Branch.chronological(length=3, direction=rf.Direction.bidirectional, mask=FORECAST, value=rf.Number)


def test_a_branch_accepts_one_forecast_only():
    second = rf.Mask(select=rf.Forecast(horizon=3, targets=("value",)), reconstruct=True)
    with pytest.raises(ValueError, match="at most one Forecast"):
        rf.Branch(length=3, direction=CAUSAL, mask=(FORECAST, second), value=rf.Number)


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (("missing",), r"Forecast target 'missing'.*not a child"),
        (("ts",), r"Forecast target 'ts'.*time_index"),
        (("tags",), r"Forecast target 'tags'.*branch"),
        (("value", "value"), r"Forecast targets.*'value'.*more than once"),
    ],
)
def test_forecast_targets_must_be_distinct_direct_leaf_children(targets, message):
    with pytest.raises(ValueError, match=message):
        checked(forecast_mask(targets))


@pytest.mark.parametrize(
    ("known", "message"),
    [
        (("missing",), r"branch.*Forecast known 'missing'.*not a child"),
        (("value",), r"branch.*Forecast known 'value'.*target"),
        (("ts",), r"branch.*Forecast known 'ts'.*time_index"),
        (("tags",), r"branch.*Forecast known 'tags'.*branch"),
        (("load", "load"), r"branch.*Forecast known.*'load'.*more than once"),
    ],
)
def test_forecast_known_must_be_distinct_direct_leaf_covariates(known, message):
    with pytest.raises(ValueError, match=message):
        checked(forecast_mask(("value",), known))


def test_forecast_accepts_known_covariates():
    assert checked(forecast_mask(("value",), ("load",))).forecast.known == ("load",)


def test_forecast_time_index_child_must_read_one_member():
    with pytest.raises(ValueError, match=r"time_index child 'ts'.*single member"):
        rf.Branch(
            length=3,
            direction=CAUSAL,
            overflow="tail",
            mask=FORECAST,
            time_index="ts",
            time_step=1.0,
            ts=rf.Number(query="meta.ts"),
            value=rf.Number,
        )


def test_capacity_adds_the_forecast_horizon_to_the_length():
    forecasting = rf.Branch(length=3, direction=CAUSAL, overflow="tail", mask=FORECAST, value=rf.Number)
    assert forecasting.capacity == 5
    assert forecasting.forecast == FORECAST.select
    assert rf.Branch(direction=CAUSAL, mask=FORECAST, value=rf.Number).capacity is None
    assert rf.Branch(length=3, value=rf.Number).capacity == 3
    assert rf.Branch(length=3, value=rf.Number).forecast is None


def test_forecast_branch_round_trips_through_yaml_schema():
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(length=3, direction=CAUSAL, overflow="tail", mask=FORECAST, value=rf.Number),
    )
    restored = rf.Schema.model_validate_json(schema.model_dump_json())
    assert restored.branches["/events"].mask == (FORECAST,)
    assert restored.branches["/events"].capacity == 5


@pytest.mark.parametrize(
    ("options", "problem"),
    [
        ({"time_index": "missing"}, "no child named 'missing'"),
        ({"time_step": 1.0}, "requires time_index"),
        ({"mixer": rf.Conv()}, "requires grain=Steps()"),
        (
            {"mask": rf.Mask(select=rf.Forecast(horizon=2, targets=("gone",)), reconstruct=True)},
            "set direction=rf.Direction.causal",
        ),
    ],
    ids=["time_index", "time_step", "conv", "forecast"],
)
def test_unbound_branch_errors_never_name_none(options, problem):
    with pytest.raises(ValueError, match=problem) as caught:
        rf.Branch(length=4, x=rf.Number, **options)
    assert "None" not in str(caught.value)
