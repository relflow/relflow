"""Duration constants retain their types, precision, and format across persistence."""

from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pytest
import yaml

import relflow as rf


def restore(definition, route, tmp_path):
    if route == "json":
        return rf.Duration.model_validate_json(definition.model_dump_json())
    path = tmp_path / "duration.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "preset": "small",
                "d_model": 8,
                "fields": {"elapsed": definition.model_dump(mode="json", exclude={"name"})},
            }
        )
    )
    return rf.Model.from_yaml(path).schema.requests["/elapsed"]


def content(definition, source):
    return rf.Model(preset="small", d_model=8, elapsed=definition).encode(source)["/elapsed"].content.item()


@pytest.mark.parametrize("route", ["json", "yaml"])
@pytest.mark.parametrize("endpoint,sign", [("start", 1), ("end", -1)])
@pytest.mark.parametrize("constant,hours", [(date(2026, 1, 1), 36.0), (datetime(2026, 1, 1, 12), 24.0)])
def test_duration_native_constants_keep_their_types_under_custom_source_formats(
    route, endpoint, sign, constant, hours, tmp_path
):
    definition = rf.Duration(query="observed", pattern="%d/%m/%Y %H:%M", granularity="hours", **{endpoint: constant})
    restored = restore(definition, route, tmp_path)
    source = pa.table({"observed": ["02/01/2026 12:00"]})

    assert type(getattr(restored, endpoint)) is type(constant)
    assert content(definition, source) == content(restored, source) == sign * hours


@pytest.mark.parametrize("route", ["json", "yaml"])
@pytest.mark.parametrize("constant", [date(2500, 1, 1), datetime(2500, 1, 1, 12)])
def test_duration_native_constants_preserve_dates_outside_nanosecond_string_range(route, constant, tmp_path):
    definition = rf.Duration(query="observed", start=constant, granularity="days")
    restored = restore(definition, route, tmp_path)
    observed = datetime(2500, 1, 2, 12) if isinstance(constant, datetime) else datetime(2500, 1, 2)
    source = pa.table({"observed": pa.array([observed], type=pa.timestamp("us"))})

    assert type(restored.start) is type(constant)
    assert content(definition, source) == content(restored, source) == 1.0


@pytest.mark.parametrize("route", ["json", "yaml"])
def test_duration_string_constants_preserve_nanoseconds_without_datetime_coercion(route, tmp_path):
    start = "2026-01-01T01:00:00.000000001+01:00"
    definition = rf.Duration(query="observed", start=start, granularity="seconds")
    restored = restore(definition, route, tmp_path)
    source = pa.table({"observed": ["2026-01-01T00:00:00.000000003Z"]})

    assert restored.start == start
    assert definition.model_dump(mode="json")["start"] == start
    assert content(definition, source) == content(restored, source) == 2e-9


@pytest.mark.parametrize("route", ["json", "yaml"])
def test_duration_custom_string_constants_keep_their_original_format(route, tmp_path):
    start = "31/01/2026 12:00"
    definition = rf.Duration(query="observed", start=start, pattern="%d/%m/%Y %H:%M", granularity="hours")
    restored = restore(definition, route, tmp_path)
    source = pa.table({"observed": ["02/02/2026 00:00"]})

    assert restored.start == start
    assert content(definition, source) == content(restored, source) == 36.0


def test_duration_native_constants_serialize_with_distinct_tags_and_keep_python_types():
    start = date(2026, 1, 1)
    end = datetime(2026, 1, 2, 12, tzinfo=timezone(timedelta(hours=2)))
    definition = rf.Duration(start=start, end=end)

    assert definition.model_dump(mode="json")["start"] == {"type": "date", "value": "2026-01-01"}
    assert definition.model_dump(mode="json")["end"] == {"type": "datetime", "value": "2026-01-02T12:00:00+02:00"}
    assert definition.model_dump(mode="python")["start"] is start
    assert definition.model_dump(mode="python")["end"] is end
    restored = rf.Duration.model_validate_json(definition.model_dump_json())
    assert type(restored.start) is date
    assert restored.end == end


@pytest.mark.parametrize(
    "tag",
    [
        {},
        {"type": "date"},
        {"value": "2026-01-01"},
        {"type": "date", "value": "2026-01-01", "extra": True},
        {"type": "unknown", "value": "2026-01-01"},
        {"type": 1, "value": "2026-01-01"},
        {"type": "date", "value": None},
        {"type": "date", "value": "2026-02-30"},
        {"type": "datetime", "value": "not-a-timestamp"},
        {"type": "datetime", "value": "2026-02-30T12:00:00"},
    ],
)
def test_duration_rejects_malformed_endpoint_tags(tag):
    with pytest.raises(ValueError, match="Duration endpoint"):
        rf.Duration(query="observed", start=tag)
