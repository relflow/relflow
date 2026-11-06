"""Elapsed-time arithmetic preserves instants, fine precision, and null structure."""

from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pytest

from relflow.tensorfields.shared.temporal import difference, parse


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("nanoseconds", 129600000000000.0),
        ("microseconds", 129600000000.0),
        ("milliseconds", 129600000.0),
        ("seconds", 129600.0),
        ("minutes", 2160.0),
        ("hours", 36.0),
        ("days", 1.5),
        ("weeks", 1.5 / 7),
    ],
)
def test_difference_is_signed_and_fractional_in_fixed_units(unit, expected):
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 2, 12)

    assert difference(start, end, unit=unit).as_py() == pytest.approx(expected)
    assert difference(end, start, unit=unit).as_py() == pytest.approx(-expected)


@pytest.mark.parametrize("unit", ["months", "years", "day", "D", None, 1])
def test_difference_rejects_unsupported_or_variable_units(unit):
    with pytest.raises(ValueError, match="duration unit"):
        difference(date(2026, 1, 1), date(2026, 1, 2), unit=unit)


def test_difference_subtracts_before_casting_large_epoch_coordinates_to_floats():
    start = pa.array([1800000000000000001, 1800000000000000004], type=pa.timestamp("ns"))
    end = pa.array([1800000000000000003, 1800000000000000003], type=pa.timestamp("ns"))

    assert difference(start, end, unit="nanoseconds").to_pylist() == [2.0, -1.0]
    assert difference(start, end, unit="seconds").to_pylist() == pytest.approx([2e-9, -1e-9])


def test_difference_preserves_nanoseconds_when_parsing_iso_offset_strings():
    result = difference(
        "2026-01-01T01:00:00.000000001+01:00",
        "2026-01-01T00:00:00.000000003Z",
        unit="nanoseconds",
    )

    assert result.as_py() == 2.0


@pytest.mark.parametrize("unit,expected", [("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)])
def test_difference_preserves_typed_timestamp_resolution(unit, expected):
    result = difference(pa.array([1], type=pa.timestamp(unit)), pa.array([2], type=pa.timestamp(unit)), unit="seconds")

    assert result.to_pylist() == pytest.approx([expected])


def test_difference_aligns_different_resolutions_without_losing_the_finer_endpoint():
    start = pa.array([1800000000], type=pa.timestamp("s"))
    end = pa.array([1800000000000000001], type=pa.timestamp("ns"))

    assert difference(start, end, unit="nanoseconds").to_pylist() == [1.0]


@pytest.mark.parametrize("datatype", [pa.date32(), pa.date64()])
def test_difference_accepts_arrow_dates_and_static_dates(datatype):
    starts = pa.array([date(2026, 1, 1), date(2026, 1, 3)], type=datatype)

    assert difference(starts, date(2026, 1, 2), unit="days").to_pylist() == [1.0, -1.0]
    assert difference(date(2026, 1, 2), starts, unit="days").to_pylist() == [-1.0, 1.0]


def test_difference_measures_absolute_time_across_spring_daylight_saving_transition():
    start = datetime(2026, 3, 8, 1, 30, tzinfo=timezone(timedelta(hours=-5)))
    end = datetime(2026, 3, 8, 3, 30, tzinfo=timezone(timedelta(hours=-4)))

    assert difference(start, end, unit="hours").as_py() == 1.0
    assert difference(start, end, unit="days").as_py() == pytest.approx(1 / 24)


def test_difference_distinguishes_repeated_wall_clock_times_at_the_autumn_fold():
    start = datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=-4)))
    end = datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=-5)))

    assert difference(start, end, unit="hours").as_py() == 1.0


def test_difference_normalizes_distinct_timezone_metadata_to_the_same_instant():
    start = pa.array([datetime(2026, 1, 1, 0, tzinfo=timezone.utc)], type=pa.timestamp("us", tz="Europe/Paris"))
    end = pa.array([datetime(2026, 1, 1, 0, tzinfo=timezone.utc)])

    assert difference(start, end, unit="seconds").to_pylist() == [0.0]
    assert parse(start).type == pa.timestamp("us", tz="UTC")


@pytest.mark.parametrize("reverse", [False, True])
def test_difference_rejects_mixing_aware_and_naive_endpoints(reverse):
    endpoints = [datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=timezone.utc)]
    if reverse:
        endpoints.reverse()

    with pytest.raises(ValueError, match="timezone-aware and timezone-naive"):
        difference(*endpoints)


def test_parse_rejects_mixed_awareness_inside_one_string_column():
    values = pa.array(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00", None])

    with pytest.raises(ValueError, match="timezone-aware and timezone-naive"):
        parse(values)


def test_parse_rejects_mixed_awareness_across_arrow_chunks():
    values = pa.chunked_array([["2026-01-01T00:00:00Z"], ["2026-01-01T00:00:00"]])

    with pytest.raises(ValueError, match="timezone-aware and timezone-naive"):
        parse(values)


def test_difference_broadcasts_a_static_operand_over_chunked_values_and_preserves_nulls():
    starts = pa.chunked_array([["2026-01-01T00:00:00", None], ["2026-01-02T12:00:00"]])

    assert difference(starts, datetime(2026, 1, 3), unit="hours").to_pylist() == [48.0, None, 12.0]
    assert difference(datetime(2026, 1, 3), starts, unit="hours").to_pylist() == [-48.0, None, -12.0]


def test_difference_accepts_unequal_chunk_boundaries_for_aligned_arrays():
    starts = pa.chunked_array([[datetime(2026, 1, 1)], [datetime(2026, 1, 2), None]])
    ends = pa.chunked_array([[datetime(2026, 1, 2), datetime(2026, 1, 2, 1)], [datetime(2026, 1, 4)]])

    assert difference(starts, ends, unit="hours").to_pylist() == [24.0, 1.0, None]


@pytest.mark.parametrize("values", [pa.nulls(2), pa.array([None, None], type=pa.timestamp("us")), None])
def test_missing_only_operands_propagate_null_without_inventing_timezone_requirements(values):
    other = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = difference(values, other)

    assert result.type == pa.float64()
    if isinstance(result, pa.Scalar):
        assert result.as_py() is None
    else:
        assert result.to_pylist() == [None, None]


def test_empty_arrow_inputs_produce_typed_empty_results():
    result = difference(pa.chunked_array([], type=pa.timestamp("us")), datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert len(result) == 0
    assert result.type == pa.float64()


def test_difference_uses_custom_arrow_formats_for_strings():
    result = difference("31/01/2026 12:00", "02/02/2026 00:00", pattern="%d/%m/%Y %H:%M", unit="days")

    assert result.as_py() == 1.5


def test_custom_format_with_zone_offsets_measures_absolute_time():
    result = difference(
        "2026/03/08 01:30 -0500",
        "2026/03/08 03:30 -0400",
        pattern="%Y/%m/%d %H:%M %z",
        unit="hours",
    )

    assert result.as_py() == 1.0


def test_difference_decodes_dictionary_encoded_temporal_values():
    values = pa.array(["2026-01-01", "2026-01-02", None]).dictionary_encode()

    assert difference(values, date(2026, 1, 3), unit="days").to_pylist() == [2.0, 1.0, None]


@pytest.mark.parametrize("text", ["not-a-date", "", "2026-02-30"])
def test_parse_rejects_malformed_dates_instead_of_turning_them_into_nulls(text):
    with pytest.raises(ValueError, match="could not be parsed"):
        difference(pa.array([text, None]), date(2026, 1, 1))


def test_parse_rejects_strings_that_do_not_match_the_requested_format():
    with pytest.raises(ValueError, match="could not be parsed"):
        difference("31/01/2026", "01/02/2026", pattern="%Y-%m-%d")


def test_difference_rejects_unequal_arrays_instead_of_broadcasting_a_length_one_array():
    with pytest.raises(ValueError, match="equal lengths or be scalar"):
        difference(pa.array(["2026-01-01"]), pa.array(["2026-01-02", "2026-01-03"]))


@pytest.mark.parametrize(
    "values",
    [
        pa.array([1]),
        pa.array([True]),
        pa.array([["2026-01-01"]]),
        pa.array([None], type=pa.list_(pa.timestamp("us"))),
        pa.array([], type=pa.list_(pa.timestamp("us"))),
        pa.chunked_array([], type=pa.list_(pa.timestamp("us"))),
        pa.array([None], type=pa.struct([("time", pa.timestamp("us"))])),
        pa.array([None], type=pa.int64()),
        pa.array([], type=pa.bool_()),
        1,
        False,
    ],
)
def test_parse_rejects_numeric_or_collection_operands(values):
    with pytest.raises(TypeError, match="duration endpoint"):
        parse(values)


def test_difference_rejects_integer_overflow_instead_of_wrapping_elapsed_counts():
    start = pa.array([-9000000000000000000], type=pa.timestamp("ns"))
    end = pa.array([9000000000000000000], type=pa.timestamp("ns"))

    with pytest.raises(ValueError, match="cannot be subtracted safely"):
        difference(start, end)


def test_difference_rejects_unsafe_rescaling_of_extremely_distant_dates():
    start = pa.array([datetime(2500, 1, 1)], type=pa.timestamp("us"))
    end = pa.array([1800000000000000000], type=pa.timestamp("ns"))

    with pytest.raises(ValueError, match="cannot be subtracted safely"):
        difference(start, end)


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        ("2026-02-30", "%Y-%m-%d"),
        ("2026-04-31", "%Y-%m-%d"),
        ("2025-02-29", "%Y-%m-%d"),
        ("26/2/30", "%y/%m/%d"),
        ("2025-366", "%Y-%j"),
        ("2026-02-30", "%F"),
        ("02/30/26", "%D"),
        ("02/30/26", "%x"),
        ("Mon Feb 30 00:00:00 2026", "%c"),
        ("2026-02-30 00:30 +0100", "%Y-%m-%d %H:%M %z"),
        ("2026-04-31 23:30 -0530", "%Y-%m-%d %H:%M %z"),
    ],
)
def test_custom_formats_reject_impossible_calendar_dates_instead_of_normalizing(text, pattern):
    with pytest.raises(ValueError, match="impossible calendar date"):
        parse(pa.array([None, text]), pattern)


@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        ("2024-02-29", "%Y-%m-%d", datetime(2024, 2, 29)),
        ("2026-2-3", "%Y-%m-%d", datetime(2026, 2, 3)),
        ("2026-02- 3", "%Y-%m-%e", datetime(2026, 2, 3)),
        ("2024-366", "%Y-%j", datetime(2024, 12, 31)),
        ("24/2/29", "%y/%m/%d", datetime(2024, 2, 29)),
        ("2024-02-29", "%F", datetime(2024, 2, 29)),
        ("02/29/24", "%D", datetime(2024, 2, 29)),
        ("02/29/24", "%x", datetime(2024, 2, 29)),
        ("Thu Feb 29 00:00:00 2024", "%c", datetime(2024, 2, 29)),
        ("3 Feb 2026 12:34:56", "%d %b %Y %H:%M:%S", datetime(2026, 2, 3, 12, 34, 56)),
        ("2026-01-01 %Z", "%Y-%m-%d %%Z", datetime(2026, 1, 1)),
    ],
)
def test_custom_calendar_validation_preserves_leap_dates_padding_and_supported_formats(text, pattern, expected):
    result = parse(pa.array([text, None]), pattern)

    assert result.to_pylist() == [expected, None]
    assert result.type == pa.timestamp("ns")


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2026-01-01 00:30 +0100", "2026-01-01 00:30 +0000", 1.0),
        ("2025-12-31 23:30 -0530", "2025-12-31 23:45 -0530", 0.25),
        ("2024-02-29 00:30 +0100", "2024-02-29 01:30 +0100", 1.0),
    ],
)
def test_custom_calendar_validation_uses_each_source_offset_across_midnight(start, end, expected):
    assert difference(start, end, pattern="%Y-%m-%d %H:%M %z", unit="hours").as_py() == expected


@pytest.mark.parametrize("offset", ["+2560", "+2400", "-1260"])
def test_custom_formats_reject_invalid_numeric_timezone_offsets(offset):
    with pytest.raises(ValueError, match="numeric timezone offset"):
        parse(f"2026-01-01 00:30 {offset}", "%Y-%m-%d %H:%M %z")


def test_custom_formats_reject_named_zones_with_an_explicit_offset_or_typed_timestamp_remedy():
    with pytest.raises(ValueError, match="named %Z zones.*typed timezone-aware timestamps.*%z numeric offsets"):
        parse("2026-01-01 UTC", "%Y-%m-%d %Z")


@pytest.mark.parametrize("day", ["2026-01-01", "2026-02-30"])
@pytest.mark.parametrize("zone", ["Z", "z"])
def test_custom_numeric_offsets_do_not_silently_accept_unsupported_iso_zone_suffixes(day, zone):
    with pytest.raises(ValueError, match="could not be parsed"):
        parse(pa.array([None, f"{day} 00:30 {zone}"]), "%Y-%m-%d %H:%M %z")
