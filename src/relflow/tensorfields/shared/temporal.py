"""Absolute elapsed time from Arrow temporal values, without epoch-float subtraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Literal, TypeAlias, cast

import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["Unit", "difference", "parse"]

Temporal: TypeAlias = pa.Array | pa.ChunkedArray | pa.Scalar
Operand: TypeAlias = Temporal | date | datetime | str | None
Resolution: TypeAlias = Literal["s", "ms", "us", "ns"]
Unit: TypeAlias = Literal["nanoseconds", "microseconds", "milliseconds", "seconds", "minutes", "hours", "days", "weeks"]
SECONDS = {
    "nanoseconds": 1e-9,
    "microseconds": 1e-6,
    "milliseconds": 1e-3,
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 604800.0,
}
PRECISION = {"s": 1, "ms": 1000, "us": 1000000, "ns": 1000000000}
OFFSET = r"(?:[zZ]|[+-][0-9]{2}:?[0-9]{2}|[Tt ][^Tt ]*[+-][0-9]{2})$"

CALENDAR = {
    "Y": r"[0-9]{1,4}",
    "y": r"[0-9]{1,2}",
    "m": r"[0-9]{1,2}",
    "d": r"[0-9]{1,2}",
    "e": r" *[0-9]{1,2}",
    "j": r"[0-9]{1,3}",
    "C": r"[0-9]{1,2}",
    "G": r"[0-9]{1,4}",
    "g": r"[0-9]{1,2}",
    "U": r"[0-9]{1,2}",
    "V": r"[0-9]{1,2}",
    "W": r"[0-9]{1,2}",
    "u": r"[0-9]",
    "w": r"[0-9]",
}
COMPOSITES = {"F": "%Y-%m-%d", "D": "%m/%d/%y", "x": "%m/%d/%y", "c": "%a %b %e %H:%M:%S %Y"}


@dataclass(frozen=True)
class Format:
    """Source coordinates and a format whose numeric offsets are handled by Arrow arithmetic."""

    expression: str
    fields: tuple[tuple[str, str], ...]
    pattern: str
    bodies: tuple[str, ...]


@lru_cache(maxsize=64)
def calendar(pattern: str) -> Format:
    """Compile supplied calendar coordinates without constraining numeric padding.

    Composite expansions affect only validation; Arrow parses the original
    format. Other directives retain Arrow's semantics. Numeric coordinates
    reveal normalized impossible dates even when their text is parseable.
    """
    fragments = ["(?P<body_0>"]
    fields = []
    bodies = ["body_0"]

    def include(format: str) -> None:
        for match in re.finditer(r"%(?:[EO])?.|[^%]+|%$", format):
            token = match.group()
            if not token.startswith("%"):
                fragments.extend(r"\s*" if character.isspace() else re.escape(character) for character in token)
                continue
            code = token[-1]
            if token == "%%":
                fragments.append("%")
            elif token == "%":
                raise ValueError("duration format has an incomplete '%' directive")
            elif code == "Z":
                raise ValueError(
                    "duration formats with named %Z zones are unsupported; "
                    "supply typed timezone-aware timestamps or use %z numeric offsets"
                )
            elif code in COMPOSITES:
                include(COMPOSITES[code])
            elif code in {"n", "t"}:
                fragments.append(r"\s*")
            elif code in CALENDAR or code == "z":
                name = f"coordinate_{len(fields)}"
                fields.append((code, name))
                content = CALENDAR[code] if code != "z" else r"[+-][0-9]{2}:?[0-9]{2}"
                if code == "z":
                    body = f"body_{len(bodies)}"
                    bodies.append(body)
                    fragments.append(f")(?P<{name}>{content})(?P<{body}>")
                else:
                    fragments.append(f"(?P<{name}>{content})")
            else:
                fragments.append(".*?")

    include(pattern)
    naive = "".join(
        match.group()
        for match in re.finditer(r"%(?:[EO])?.|[^%]+|%$", pattern)
        if not (match.group().startswith("%") and match.group()[-1] == "z")
    )
    return Format("^" + "".join(fragments) + ")$", tuple(fields), naive, tuple(bodies))


def formatted(strings: pa.StringArray, pattern: str) -> pa.Array:
    """Parse source clocks and validate numeric offsets independently of platform strptime."""
    format = calendar(pattern)
    captured = pc.extract_regex(strings, format.expression)
    invalid = pc.and_(pc.is_valid(strings), pc.invert(pc.is_valid(captured)))
    if pc.any(invalid).as_py():
        raise ValueError("duration endpoint could not be parsed; supply strings matching the configured format")
    offset = pa.repeat(pa.scalar(0, type=pa.int64()), len(strings))
    have_offset = False
    for code, name in format.fields:
        if code != "z":
            continue
        text = pc.replace_substring(pc.struct_field(captured, name), ":", "")
        hours = pc.cast(pc.utf8_slice_codeunits(text, 1, 3), pa.int64())
        minutes = pc.cast(pc.utf8_slice_codeunits(text, 3, 5), pa.int64())
        invalid = pc.or_kleene(
            invalid,
            pc.or_kleene(pc.greater(hours, pa.scalar(23)), pc.greater(minutes, pa.scalar(59))),
        )
        seconds = pc.add(pc.multiply(hours, 3600), pc.multiply(minutes, 60))
        selected = pc.if_else(
            pc.equal(pc.utf8_slice_codeunits(text, 0, 1), pa.scalar("-")), pc.negate(seconds), seconds
        )
        if have_offset:
            invalid = pc.or_kleene(invalid, pc.not_equal(selected, offset))
        offset = selected
        have_offset = True
    if pc.any(invalid).as_py():
        raise ValueError(
            "duration endpoint contains an invalid numeric timezone offset; "
            "use hours 00 through 23, minutes 00 through 59, and consistent repeated offsets"
        )

    # Windows strptime does not support %z, while other platforms differ in
    # their rejection of malformed offsets. Parse the exact non-offset source
    # pieces as a naive clock, then apply the validated offset in integer ns.
    body = pc.binary_join_element_wise(
        *(pc.struct_field(captured, name) for name in format.bodies), pa.repeat(pa.scalar(""), len(strings))
    )
    values = pc.strptime(body, format=format.pattern, unit="ns", error_is_null=False)

    # Calendar validation needs whole source-clock seconds. Integer floor
    # avoids both epoch-float cancellation and truncation toward the next day
    # for negative subsecond epochs; seconds also avoid ns offset overflow.
    ticks = pc.cast(values, pa.int64())
    seconds = pc.divide(ticks, pa.scalar(1000000000, type=pa.int64()))
    remainder = pc.subtract(ticks, pc.multiply(seconds, 1000000000))
    seconds = pc.subtract(seconds, pc.cast(pc.less(remainder, pa.scalar(0)), pa.int64()))
    clock = cast(pa.TimestampArray, pc.cast(seconds, pa.timestamp("s")))
    for code, name in format.fields:
        if code == "z":
            continue
        supplied = pc.cast(pc.utf8_trim_whitespace(pc.struct_field(captured, name)), pa.int64())
        actual = pc.cast(pc.strftime(clock, format="%" + ("d" if code == "e" else code)), pa.int64())
        invalid = pc.or_kleene(invalid, pc.not_equal(supplied, actual))
    if pc.any(invalid).as_py():
        raise ValueError(
            "duration endpoint contains an impossible calendar date or numeric timezone offset; "
            "supply actual calendar coordinates matching the configured format"
        )
    if have_offset:
        utc = pc.subtract_checked(ticks, pc.multiply(offset, 1000000000))
        return pc.cast(utc, pa.timestamp("ns", tz="UTC"))
    return values


def common(values: tuple[Temporal, ...]) -> pa.TimestampType[Any, Any]:
    """Choose a lossless common resolution without inventing a missing timezone."""
    timestamps = [
        cast("pa.TimestampType[Any, Any]", value.type) for value in values if not pa.types.is_null(value.type)
    ]
    aware = {datatype.tz is not None for datatype in timestamps}
    if len(aware) > 1:
        raise ValueError(
            "duration operands mix timezone-aware and timezone-naive timestamps; "
            "supply timestamps with explicit, consistent timezone semantics"
        )
    unit = cast(Resolution, max((datatype.unit for datatype in timestamps), key=PRECISION.__getitem__, default="ns"))
    return pa.timestamp(unit, tz="UTC") if aware == {True} else pa.timestamp(unit)


def parse(values: Operand, pattern: str | None = None) -> Temporal:
    """Parse flat endpoints, preserving timestamps' instants and finest resolution.

    ISO strings use nanosecond resolution. Custom patterns follow Arrow's
    C/C++ strptime syntax. Aware values normalize to UTC; naive values retain
    their naive timeline. Missing-only operands carry no inferred timezone.
    """
    if pattern is not None and (not isinstance(pattern, str) or not pattern):
        raise ValueError("duration format must be a nonempty string or None")
    if not isinstance(values, (pa.Array, pa.ChunkedArray, pa.Scalar)):
        if values is not None and not isinstance(values, (date, datetime, str)):
            raise TypeError(
                f"duration endpoint must be an Arrow temporal value, date, datetime, or string, got {type(values).__name__}"
            )
        # A one-value collection has one physical type; the overload stubs
        # cannot distribute this union of scalar Python input types.
        scalar = pa.array([values])  # pyrefly: ignore [no-matching-overload]
        return cast(pa.Array, parse(scalar, pattern))[0]
    if isinstance(values, pa.Scalar):
        return cast(pa.Array, parse(pa.array([values], type=values.type), pattern))[0]
    datatype = values.type
    while pa.types.is_dictionary(datatype) or isinstance(datatype, pa.ExtensionType):
        datatype = (
            cast(pa.DictionaryType, datatype).value_type
            if pa.types.is_dictionary(datatype)
            else cast(pa.ExtensionType, datatype).storage_type
        )
    if not (
        pa.types.is_null(datatype)
        or pa.types.is_date(datatype)
        or pa.types.is_timestamp(datatype)
        or pa.types.is_string(datatype)
        or pa.types.is_large_string(datatype)
    ):
        raise TypeError(
            f"duration endpoint requires dates, timestamps, or strings, got Arrow {values.type}; "
            "align collection axes before calculating elapsed time"
        )
    if values.null_count == len(values):
        return pa.nulls(len(values))
    if isinstance(values, pa.ChunkedArray):
        chunks = tuple(cast(pa.Array, parse(chunk, pattern)) for chunk in values.chunks)
        datatype = common(chunks)
        return pa.chunked_array([pc.cast(chunk, datatype) for chunk in chunks], type=datatype)
    if pa.types.is_dictionary(values.type):
        dictionary = cast(pa.DictionaryArray, values)
        return parse(pc.take(dictionary.dictionary, dictionary.indices), pattern)
    if isinstance(values, pa.ExtensionArray):
        return parse(values.storage, pattern)
    if pa.types.is_timestamp(values.type):
        timestamp = cast("pa.TimestampType[Any, Any]", values.type)
        datatype = pa.timestamp(timestamp.unit, tz="UTC") if timestamp.tz is not None else pa.timestamp(timestamp.unit)
    elif pa.types.is_date32(values.type):
        datatype = pa.timestamp("s")
    elif pa.types.is_date64(values.type):
        datatype = pa.timestamp("ms")
    elif pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
        strings = cast(pa.StringArray, values)
        try:
            if pattern is not None:
                return formatted(strings, pattern)
            offsets = pc.match_substring_regex(strings, OFFSET)
            if pc.any(offsets).as_py() and pc.any(pc.invert(offsets)).as_py():
                raise ValueError(
                    "duration endpoint strings mix timezone-aware and timezone-naive timestamps; "
                    "supply an explicit timezone consistently"
                )
            datatype = pa.timestamp("ns", tz="UTC" if pc.any(offsets).as_py() else None)
            return pc.cast(values, datatype)
        except pa.ArrowException as error:
            raise ValueError(
                "duration endpoint could not be parsed at nanosecond resolution; "
                "supply valid ISO timestamps in range or strings matching the configured format"
            ) from error
    else:
        raise TypeError(
            f"duration endpoint requires dates, timestamps, or strings, got Arrow {values.type}; "
            "align collection axes before calculating elapsed time"
        )
    return pc.cast(values, datatype)


def difference(start: Operand, end: Operand, *, unit: Unit = "days", pattern: str | None = None) -> Temporal:
    """Return signed ``end - start`` in fixed units as Arrow float64 values.

    Scalars broadcast; two arrays must have equal lengths. Subtract integer
    timestamps before converting the elapsed count to float64, retaining small
    differences at large epoch offsets. Unsafe timestamp rescaling and signed
    integer overflow raise instead of wrapping or truncating. Months and years
    have no fixed elapsed length and are deliberately unsupported.
    """
    if not isinstance(unit, str) or unit not in SECONDS:
        raise ValueError(f"duration unit must be one of {', '.join(SECONDS)}, got {unit!r}")
    start = parse(start, pattern)
    end = parse(end, pattern)
    if not isinstance(start, pa.Scalar) and not isinstance(end, pa.Scalar) and len(start) != len(end):
        raise ValueError(f"duration operands must have equal lengths or be scalar, got {len(start)} and {len(end)}")
    datatype = common((start, end))
    try:
        # Arrow accepts every scalar/array combination; the overload stubs do
        # not distribute these supported unions.
        elapsed = pc.subtract_checked(pc.cast(end, datatype), pc.cast(start, datatype))  # pyrefly: ignore [no-matching-overload]
    except pa.ArrowInvalid as error:
        raise ValueError(
            f"duration endpoints cannot be subtracted safely at {datatype.unit!r} precision; "
            "supply timestamps with a common representable range and resolution"
        ) from error
    counts = pc.cast(elapsed, pa.int64())
    # Long intervals may exceed exact float64 integer precision; only elapsed
    # counts, never large epoch coordinates, enter this unavoidable conversion.
    return pc.divide(pc.cast(counts, pa.float64(), safe=False), PRECISION[datatype.unit] * SECONDS[unit])
