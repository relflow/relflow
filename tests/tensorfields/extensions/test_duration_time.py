"""Duration's public input path preserves elapsed-time semantics and source states."""

from datetime import datetime, timezone

import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.structs.enums import Tokens


def encode(definition, source):
    model = rf.Model.xs(d_model=8, elapsed=definition)
    return model.encode(source)["/elapsed"]


def test_duration_preserves_iso_offsets_and_nanoseconds_with_a_static_start():
    field = encode(
        rf.Duration(
            query="posted_at",
            start="2026-01-01T01:00:00.000000001+01:00",
            granularity="seconds",
        ),
        pa.table({"posted_at": ["2026-01-01T00:00:00.000000003Z"]}),
    )

    assert field.content.dtype == torch.float64
    torch.testing.assert_close(field.content.flatten(), torch.tensor([2e-9], dtype=torch.float64), rtol=0, atol=0)
    assert field.state.flatten().tolist() == [Tokens.valued]


def test_duration_preserves_iso_offsets_and_nanoseconds_with_a_static_end():
    field = encode(
        rf.Duration(
            query="created_at",
            end="2026-01-01T01:00:00.000000003+01:00",
            granularity="seconds",
        ),
        pa.table({"created_at": ["2026-01-01T00:00:00.000000001Z"]}),
    )

    torch.testing.assert_close(field.content.flatten(), torch.tensor([2e-9], dtype=torch.float64), rtol=0, atol=0)


def test_duration_measures_absolute_time_at_spring_transition_and_autumn_fold():
    field = encode(
        rf.Duration(query=("start", "end"), granularity="hours"),
        pa.table(
            {
                "start": ["2026-03-08T01:30:00-05:00", "2026-11-01T01:30:00-04:00"],
                "end": ["2026-03-08T03:30:00-04:00", "2026-11-01T01:30:00-05:00"],
            }
        ),
    )

    torch.testing.assert_close(field.content.flatten(), torch.tensor([1.0, 1.0], dtype=torch.float64), rtol=0, atol=0)


def test_duration_applies_the_custom_format_to_query_and_static_strings():
    field = encode(
        rf.Duration(query="posted_at", start="31/01/2026 12:00", granularity="days", pattern="%d/%m/%Y %H:%M"),
        pa.table({"posted_at": ["02/02/2026 00:00", "31/01/2026 00:00"]}),
    )

    torch.testing.assert_close(field.content.flatten(), torch.tensor([1.5, -0.5], dtype=torch.float64), rtol=0, atol=0)


def test_duration_honors_timezone_offsets_in_a_custom_format():
    field = encode(
        rf.Duration(query=("start", "end"), granularity="hours", pattern="%Y/%m/%d %H:%M %z"),
        pa.table({"start": ["2026/03/08 01:30 -0500"], "end": ["2026/03/08 03:30 -0400"]}),
    )

    assert field.content.item() == 1.0


def test_duration_names_its_field_when_aware_and_naive_endpoints_are_mixed():
    definition = rf.Duration(query="posted_at", start=datetime(2026, 1, 1, tzinfo=timezone.utc))

    with pytest.raises(ValueError, match="Duration field at '/elapsed'.*timezone-aware and timezone-naive"):
        encode(definition, pa.table({"posted_at": [datetime(2026, 1, 2)]}))


def test_duration_explicit_null_endpoints_remain_real_null_field_slots():
    field = encode(
        rf.Duration(query=("start", "end"), granularity="days"),
        pa.table(
            {
                "start": ["2026-01-01", None, None, "2026-01-01"],
                "end": [None, "2026-01-02", None, "2026-01-01T12:00:00"],
            }
        ),
    )

    assert field.state.flatten().tolist() == [Tokens.null, Tokens.null, Tokens.null, Tokens.valued]
    assert field.present.flatten().tolist() == [True, True, True, True]
    torch.testing.assert_close(field.content.flatten(), torch.tensor([0.0, 0.0, 0.0, 0.5], dtype=torch.float64))


@pytest.mark.parametrize("values", [[None], [[]]])
def test_duration_rejects_null_or_empty_list_values_without_silently_erasing_an_axis(values):
    starts = pa.array(values, type=pa.list_(pa.timestamp("us")))
    source = pa.table({"start": starts, "end": [datetime(2026, 1, 2)]})

    with pytest.raises(ValueError, match="Duration field at '/elapsed'.*align collection axes"):
        encode(rf.Duration(query=("start", "end")), source)


@pytest.mark.parametrize("end", ["2026-02-30", "2026-04-31", "2025-02-29"])
def test_duration_custom_format_rejects_impossible_calendar_dates_with_field_context(end):
    with pytest.raises(ValueError, match="Duration field at '/elapsed'.*impossible calendar date"):
        encode(
            rf.Duration(query="end", start="2026-01-01", pattern="%Y-%m-%d"),
            pa.table({"end": [end]}),
        )


def test_duration_custom_format_preserves_nonpadded_dates_and_offsets_across_midnight():
    field = encode(
        rf.Duration(
            query="end",
            start="2026-1-1 00:30 +0100",
            granularity="hours",
            pattern="%Y-%m-%d %H:%M %z",
        ),
        pa.table({"end": ["2026-1-1 00:30 +0000"]}),
    )

    assert field.content.item() == 1.0
