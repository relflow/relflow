"""Schema and checkpoint contracts for Period."""

import inspect
import math
from datetime import datetime

import numpy as np
import pyarrow as pa
import pydantic
import pytest
import torch
import yaml

import relflow as rf
from relflow.tensorfields.extensions.period import Part


def test_period_selects_ordered_parts_with_keyword_options():
    parts = ("HourOfDay", Part.day_of_week, "month-of-year")
    request = rf.Period(*parts, pattern="%Y-%m-%d", jitter={"add": 0.1}, predict=True)
    assert request.parts == [Part.hour_of_day, Part.day_of_week, Part.month_of_year]
    assert request.pattern == "%Y-%m-%d"
    assert request.jitter == rf.Jitter(add=0.1)
    assert request.predict is True
    assert rf.Period(parts=parts).parts == request.parts


def test_period_signature_accepts_parts_and_keeps_options_keyword_only():
    signature = inspect.signature(rf.Period)
    assert signature.parameters["selected"].kind is inspect.Parameter.VAR_POSITIONAL
    for name in ("parts", "pattern", "jitter", "predict"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert "name" not in signature.parameters
    signature.bind("day_of_week", pattern="%Y-%m-%d")


@pytest.mark.parametrize("options", [{}, {"parts": []}, {"parts": ()}])
def test_period_requires_an_explicit_nonempty_selection(options):
    with pytest.raises(pydantic.ValidationError, match="Period parts cannot be empty"):
        rf.Period(**options)


def test_bare_period_class_requires_a_selection_in_a_schema():
    with pytest.raises(pydantic.ValidationError, match="parts"):
        rf.Model(preset="small", created=rf.Period)


@pytest.mark.parametrize(
    "parts,match",
    [
        (("day_of_week", "Day Of Week"), "Period parts must be unique"),
        (("day_of_week", Part.day_of_week), "Period parts must be unique"),
        (("day of wek",), "did you mean 'day_of_week'"),
        (("year",), "unknown Period part 'year'"),
        ((7,), "Period part values must be strings, got int"),
        ((None,), "Period part values must be strings, got NoneType"),
        ((["day_of_week"],), "Period part values must be strings, got list"),
    ],
)
def test_period_validates_positional_parts_through_the_existing_validators(parts, match):
    with pytest.raises(pydantic.ValidationError, match=match):
        rf.Period(*parts)


@pytest.mark.parametrize("selection", [None, [], ["hour_of_day"]])
def test_period_rejects_mixed_selections(selection):
    with pytest.raises(TypeError, match="positionally or with parts="):
        rf.Period("day_of_week", parts=selection)


def test_period_preserves_rejection_of_a_null_keyword_selection():
    with pytest.raises(pydantic.ValidationError, match="parts"):
        rf.Period(parts=None)


def test_period_round_trips_through_json_and_yaml(tmp_path):
    request = rf.Period("day_of_week", Part.hour_of_day, embed=True)
    model = rf.Model(preset="small", d_model=8, events=rf.Branch(length=2, created=request))
    payload = model.schema.model_dump(mode="json")
    restored = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert restored.model_dump(mode="json") == payload
    field = restored.requests["/events/created"]
    assert field.parent is restored.branches["/events"]
    dumped = field.model_dump(mode="json")
    assert dumped["type"] == "period"
    assert dumped["parts"] == ["day_of_week", "hour_of_day"]
    assert rf.Period.model_validate(dumped).model_dump(mode="json") == dumped
    assert rf.Period.model_validate_json(field.model_dump_json()).model_dump(mode="json") == dumped
    path = tmp_path / "period.yaml"
    path.write_text(yaml.safe_dump(payload))
    assert rf.Schema.model_validate(yaml.safe_load(path.read_text())).model_dump(mode="json") == payload
    path.write_text(
        "preset: small\nd_model: 8\nfields:\n  created:\n    type: period\n"
        "    parts: [day_of_week, hour_of_day]\n    embed: true\n"
    )
    loaded = rf.Model.from_yaml(path)
    assert loaded.schema.model_dump() == rf.Model(preset="small", d_model=8, created=request).schema.model_dump()


def test_period_checkpoint_preserves_parameters_and_predictions(tmp_path):
    model = rf.Model(
        preset="small",
        d_model=8,
        created=rf.Period("day_of_week", "hour_of_day", predict=True, embed=True),
        label=rf.Boolean(predict=True),
    )
    inputs = pa.table({"created": [datetime(2026, 1, 1, 14), None]})
    expected = pa.table(model.predict(inputs))
    restored = rf.Model.load(model.save(tmp_path / "period.ckpt"))
    assert restored.schema.model_dump() == model.schema.model_dump()
    assert pa.table(restored.predict(inputs)).equals(expected)
    assert restored.state_dict().keys() == model.state_dict().keys()
    assert any(".parts.day_of_week." in name for name in model.state_dict())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


def test_week_of_year_counts_sunday_start_weeks_without_aliasing():
    days = np.array(
        ["2024-01-03", "2024-01-04", "2024-01-06", "2024-01-07", "2000-01-01", "2000-01-02", "2000-12-31"],
        dtype="datetime64[D]",
    )

    sin, cos = Part.week_of_year(days)

    weeks = [round(math.atan2(s, c) / (2 * math.pi) * 54) % 54 or 54 for s, c in zip(sin, cos, strict=True)]
    assert weeks == [1, 1, 1, 2, 1, 2, 54]
