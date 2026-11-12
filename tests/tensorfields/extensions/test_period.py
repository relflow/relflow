"""The Period constructor preserves DateParts' schema and trained state contracts."""

import inspect
from datetime import datetime

import pyarrow as pa
import pydantic
import pytest
import torch
import yaml

import relflow as rf
from relflow.tensorfields.extensions.dateparts import DatePart


@pytest.mark.parametrize("constructor", [rf.Period, rf.DateParts])
def test_period_selects_ordered_parts_with_keyword_options(constructor):
    parts = ("HourOfDay", DatePart.day_of_week, "month-of-year")
    request = constructor(*parts, pattern="%Y-%m-%d", jitter={"add": 0.1}, mask=True)
    assert request.dateparts == [DatePart.hour_of_day, DatePart.day_of_week, DatePart.month_of_year]
    assert request.pattern == "%Y-%m-%d"
    assert request.jitter == rf.Jitter(add=0.1)
    assert request.mask == (rf.Mask(skip=True, reconstruct=True),)
    assert constructor(dateparts=parts).dateparts == request.dateparts


@pytest.mark.parametrize("constructor", [rf.Period, rf.DateParts])
def test_period_signature_accepts_parts_and_keeps_options_keyword_only(constructor):
    signature = inspect.signature(constructor)
    assert signature.parameters["parts"].kind is inspect.Parameter.VAR_POSITIONAL
    for name in ("dateparts", "pattern", "jitter", "mask"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert "name" not in signature.parameters
    signature.bind("day_of_week", pattern="%Y-%m-%d")


@pytest.mark.parametrize("constructor", [rf.Period, rf.DateParts])
@pytest.mark.parametrize("options", [{}, {"dateparts": []}, {"dateparts": ()}])
def test_period_requires_an_explicit_nonempty_selection(constructor, options):
    with pytest.raises(pydantic.ValidationError, match="dateparts cannot be empty"):
        constructor(**options)


@pytest.mark.parametrize("constructor", [rf.Period, rf.DateParts])
def test_bare_period_class_requires_a_selection_in_a_schema(constructor):
    with pytest.raises(pydantic.ValidationError, match="dateparts"):
        rf.Model(preset="small", created=constructor)


@pytest.mark.parametrize(
    "parts,match",
    [
        (("day_of_week", "Day Of Week"), "dateparts must be unique"),
        (("day_of_week", DatePart.day_of_week), "dateparts must be unique"),
        (("day of wek",), "did you mean 'day_of_week'"),
        (("year",), "unknown datepart 'year'"),
        ((7,), "datepart values must be strings, got int"),
        ((None,), "datepart values must be strings, got NoneType"),
        ((["day_of_week"],), "datepart values must be strings, got list"),
    ],
)
def test_period_validates_positional_parts_through_the_existing_validators(parts, match):
    with pytest.raises(pydantic.ValidationError, match=match):
        rf.Period(*parts)


@pytest.mark.parametrize("constructor", [rf.Period, rf.DateParts])
@pytest.mark.parametrize("selection", [None, [], ["hour_of_day"]])
def test_period_rejects_mixed_selections(constructor, selection):
    with pytest.raises(TypeError, match="positionally or with dateparts="):
        constructor("day_of_week", dateparts=selection)


def test_period_preserves_rejection_of_a_null_keyword_selection():
    with pytest.raises(pydantic.ValidationError, match="dateparts"):
        rf.Period(dateparts=None)


@pytest.mark.parametrize("constructor", [rf.Period, rf.DateParts])
def test_period_preserves_json_yaml_and_legacy_request_restoration(constructor, tmp_path):
    request = constructor("day_of_week", DatePart.hour_of_day, embed=True)
    model = rf.Model(preset="small", d_model=8, events=rf.Branch(length=2, created=request))
    payload = model.schema.model_dump(mode="json")
    restored = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert restored.model_dump(mode="json") == payload
    field = restored.requests["/events/created"]
    assert field.parent is restored.branches["/events"]
    legacy = field.model_dump(mode="json")
    assert legacy["type"] == "dateparts"
    assert legacy["dateparts"] == ["day_of_week", "hour_of_day"]
    assert constructor.model_validate(legacy).model_dump(mode="json") == legacy
    assert constructor.model_validate_json(field.model_dump_json()).model_dump(mode="json") == legacy
    path = tmp_path / "period.yaml"
    path.write_text(yaml.safe_dump(payload))
    assert rf.Schema.model_validate(yaml.safe_load(path.read_text())).model_dump(mode="json") == payload
    path.write_text(
        "preset: small\nd_model: 8\nfields:\n  created:\n    type: dateparts\n"
        "    dateparts: [day_of_week, hour_of_day]\n    embed: true\n"
    )
    loaded = rf.Model.from_yaml(path)
    assert loaded.schema.model_dump() == rf.Model(preset="small", d_model=8, created=request).schema.model_dump()


@pytest.mark.parametrize("constructor", [rf.Period, rf.DateParts])
def test_period_checkpoint_preserves_parameters_and_predictions(constructor, tmp_path):
    model = rf.Model(
        preset="small",
        d_model=8,
        created=constructor("day_of_week", "hour_of_day", mask=True, embed=True),
        label=rf.Boolean(mask=True),
    )
    inputs = pa.table({"created": [datetime(2026, 1, 1, 14), None]})
    expected = model.predict(inputs)
    restored = rf.Model.load(model.save(tmp_path / "period.ckpt"))
    assert restored.schema.model_dump() == model.schema.model_dump()
    assert restored.predict(inputs).equals(expected)
    assert restored.state_dict().keys() == model.state_dict().keys()
    assert any(".dateparts.day_of_week." in name for name in model.state_dict())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
