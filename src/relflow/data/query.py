"""Small node-relative query language compiled to Arrow operations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from typing import Generic, TypeAlias, TypeGuard, TypeVar, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

Atom: TypeAlias = str | int | bool
ListType: TypeAlias = pa.ListType | pa.LargeListType | pa.FixedSizeListType
Array = TypeVar("Array", bound=pa.Array | pa.ChunkedArray, covariant=True)


class MissingFieldError(ValueError):
    """A structural query names a member absent from its Arrow struct type."""


@dataclass(frozen=True, slots=True)
class Member:
    """A named struct member."""

    name: str
    segment: str


@dataclass(frozen=True, slots=True)
class Literal:
    """A bracket literal resolved from its parent Arrow type during binding."""

    value: Atom
    segment: str


@dataclass(frozen=True, slots=True)
class Traverse:
    """Traversal over exactly one list axis."""

    segment: str = "[*]"


@dataclass(frozen=True, slots=True)
class Index:
    """A single list position."""

    value: int
    segment: str


@dataclass(frozen=True, slots=True)
class Slice:
    """A half-open list slice with optional Python-style bounds."""

    start: int | None
    stop: int | None
    segment: str


@dataclass(frozen=True, slots=True)
class Lookup:
    """An exact literal map-key lookup."""

    value: Atom
    segment: str


Parsed: TypeAlias = Member | Literal | Traverse | Slice
Bound: TypeAlias = Member | Traverse | Index | Slice | Lookup


@dataclass(frozen=True, slots=True)
class Query:
    """A syntax-checked, Arrow-type-neutral query."""

    text: str
    steps: tuple[Parsed, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    """A query bound to one Arrow input type."""

    query: Query
    steps: tuple[Bound, ...]
    input: pa.DataType
    output: pa.DataType
    presence: pa.DataType


@dataclass(frozen=True, slots=True)
class Projection(Generic[Array]):
    """Selected Arrow values and an aligned structural-presence bitmap."""

    values: Array
    present: Array


class Parser:
    """Strict parser for RelFlow's structural path grammar."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def error(self, reason: str, position: int | None = None) -> ValueError:
        offset = self.position if position is None else position
        segment = self.text[offset:] or "<end>"
        return ValueError(f"invalid query {self.text!r} at {segment!r}: {reason}")

    def identifier(self) -> Member:
        start = self.position
        if self.position >= len(self.text) or not self.text[self.position].isalpha():
            raise self.error("expected an identifier beginning with a letter")

        self.position += 1
        while self.position < len(self.text):
            character = self.text[self.position]
            if not (character.isalpha() or character.isdigit() or character == "_"):
                break
            self.position += 1

        name = self.text[start : self.position]
        return Member(name=name, segment=name)

    def bracket(self) -> Literal | Traverse | Slice:
        start = self.position
        self.position += 1
        if self.position >= len(self.text):
            raise self.error("unterminated bracket", start)

        if self.text.startswith("*]", self.position):
            self.position += 2
            return Traverse()

        if self.text[self.position] == '"':
            try:
                value, consumed = json.JSONDecoder().raw_decode(self.text[self.position :])
            except json.JSONDecodeError as error:
                raise self.error("invalid JSON string literal", start) from error
            if not isinstance(value, str):
                raise self.error("quoted bracket values must be strings", start)

            self.position += consumed
            if self.position >= len(self.text) or self.text[self.position] != "]":
                raise self.error("expected ']' after the string literal", start)
            self.position += 1
            return Literal(value=value, segment=self.text[start : self.position])

        end = self.text.find("]", self.position)
        if end < 0:
            raise self.error("unterminated bracket", start)

        content = self.text[self.position : end]
        self.position = end + 1
        segment = self.text[start : self.position]

        if not content:
            raise self.error("flattening is not supported; use a preprocessor", start)
        if content.startswith("?"):
            raise self.error("filters are not supported; use a preprocessor", start)
        if ":" in content:
            bounds = content.split(":")
            if len(bounds) != 2:
                raise self.error("slice steps are not supported", start)
            if not all(not bound or re.fullmatch(r"-?(0|[1-9][0-9]*)", bound) for bound in bounds):
                raise self.error("slice bounds must be integers", start)
            return Slice(
                start=int(bounds[0]) if bounds[0] else None,
                stop=int(bounds[1]) if bounds[1] else None,
                segment=segment,
            )
        if content == "true":
            return Literal(value=True, segment=segment)
        if content == "false":
            return Literal(value=False, segment=segment)
        if re.fullmatch(r"-?(0|[1-9][0-9]*)", content):
            return Literal(value=int(content), segment=segment)
        if content.startswith("'"):
            raise self.error("string literals must use JSON double quotes", start)
        if "," in content:
            raise self.error("multiselects are not supported; use a preprocessor", start)
        raise self.error("unsupported bracket expression", start)

    def parse(self) -> Query:
        if not self.text or not self.text.strip():
            raise self.error("query must be a non-empty string", 0)
        if self.text != self.text.strip():
            raise self.error("whitespace is not allowed outside quoted field names", 0)
        if self.text.startswith("[*]"):
            raise self.error("queries are observation-relative and must not begin with '[*]'", 0)
        if self.text.startswith("$"):
            raise self.error("queries are observation-relative and must not begin with '$'", 0)

        if self.text.startswith("["):
            first = self.bracket()
            if not isinstance(first, Literal) or not isinstance(first.value, str):
                raise self.error("a query must begin with a field name", 0)
            steps: list[Parsed] = [first]
        else:
            steps = [self.identifier()]

        while self.position < len(self.text):
            character = self.text[self.position]
            if character == ".":
                if self.text.startswith("..", self.position):
                    raise self.error("recursive descent is not supported; use a preprocessor")
                self.position += 1
                if self.position < len(self.text) and self.text[self.position] == "*":
                    raise self.error("object wildcards are not supported; name the field explicitly")
                steps.append(self.identifier())
                continue
            if character == "[":
                steps.append(self.bracket())
                continue
            if character == "|":
                raise self.error("pipes are not supported; use a preprocessor")
            if character == "(":
                raise self.error("functions are not supported; use a preprocessor")
            if character in "<>=!&":
                raise self.error("expressions are not supported; use a preprocessor")
            raise self.error("expected '.' or '['")

        return Query(text=self.text, steps=tuple(steps))


@cache
def compile(expression: str) -> Query:
    """Parse one observation-relative structural query."""

    if not isinstance(expression, str):
        raise TypeError(f"query must be a string, got {type(expression).__name__}")
    return Parser(expression).parse()


@cache
def scope(expression: str) -> tuple[int, str]:
    """Separate explicit leaf ancestor steps from the ordinary structural path."""
    if not isinstance(expression, str):
        raise TypeError(f"query must be a string, got {type(expression).__name__}")
    depth = 0
    while expression.startswith("../"):
        depth += 1
        expression = expression[3:]
    compile(expression)
    return depth, expression


def listed(datatype: pa.DataType) -> TypeGuard[ListType]:
    """Return whether an Arrow type owns one traversable list axis."""

    return pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype)


def nest(container: ListType, child: pa.DataType) -> ListType:
    """Replace a list container's value type while preserving its list kind."""

    value = pa.field(
        container.value_field.name,
        child,
        nullable=container.value_field.nullable,
        metadata=container.value_field.metadata,
    )
    if pa.types.is_large_list(container):
        return pa.large_list(value)
    if pa.types.is_fixed_size_list(container):
        return pa.list_(value, container.list_size)
    return pa.list_(value)


def bounds(length: int, start: int | None, stop: int | None) -> tuple[int, int]:
    """Resolve static Python slice bounds for a fixed-size list type."""

    lower, upper, _ = slice(start, stop).indices(length)
    return lower, max(lower, upper)


def sliced(datatype: pa.DataType, start: int | None, stop: int | None) -> pa.DataType:
    """Return the stable Arrow type produced by a list slice."""

    if not pa.types.is_fixed_size_list(datatype):
        return datatype

    lower, upper = bounds(datatype.list_size, start, stop)
    size = upper - lower
    if size == 0:
        return pa.list_(datatype.value_field)
    return pa.list_(datatype.value_field, size)


def context(query: Query, segment: str, address: str | None, reason: str) -> ValueError:
    """Build a query error with expression, segment, and model address context."""

    location = f" for address {address!r}" if address is not None else ""
    return ValueError(f"query {query.text!r}{location} cannot apply segment {segment!r}: {reason}")


@cache
def bind(query: Query | str, source: pa.Schema | pa.DataType, *, address: str | None = None) -> Plan:
    """Resolve neutral bracket operations against an Arrow schema."""

    parsed = compile(query) if isinstance(query, str) else query
    datatype = pa.struct(list(source)) if isinstance(source, pa.Schema) else source
    input_type = datatype
    steps: list[Bound] = []
    wrappers: list[ListType] = []

    for step in parsed.steps:
        if isinstance(step, Member):
            if not pa.types.is_struct(datatype):
                raise context(parsed, step.segment, address, f"expected a struct, got {datatype}")
            indices = datatype.get_all_field_indices(step.name)
            if not indices:
                raise MissingFieldError(
                    context(parsed, step.segment, address, f"field {step.name!r} is absent from {datatype}")
                )
            if len(indices) > 1:
                raise context(parsed, step.segment, address, f"field {step.name!r} is ambiguous in {datatype}")
            datatype = datatype.field(indices[0]).type
            steps.append(step)
            continue

        if isinstance(step, Traverse):
            if not listed(datatype):
                raise context(parsed, step.segment, address, f"expected a list, got {datatype}")
            wrappers.append(datatype)
            datatype = datatype.value_type
            steps.append(step)
            continue

        if isinstance(step, Slice):
            if not listed(datatype):
                raise context(parsed, step.segment, address, f"expected a list, got {datatype}")
            datatype = sliced(datatype, step.start, step.stop)
            steps.append(step)
            continue

        if pa.types.is_struct(datatype) and isinstance(step.value, str):
            indices = datatype.get_all_field_indices(step.value)
            if not indices:
                raise MissingFieldError(
                    context(parsed, step.segment, address, f"field {step.value!r} is absent from {datatype}")
                )
            if len(indices) > 1:
                raise context(parsed, step.segment, address, f"field {step.value!r} is ambiguous in {datatype}")
            datatype = datatype.field(indices[0]).type
            steps.append(Member(name=step.value, segment=step.segment))
            continue

        if listed(datatype) and isinstance(step.value, int) and not isinstance(step.value, bool):
            datatype = datatype.value_type
            steps.append(Index(value=step.value, segment=step.segment))
            continue

        if pa.types.is_map(datatype):
            key_type = datatype.key_type
            compatible = (
                isinstance(step.value, bool)
                and pa.types.is_boolean(key_type)
                or isinstance(step.value, int)
                and not isinstance(step.value, bool)
                and pa.types.is_integer(key_type)
                or isinstance(step.value, str)
                and (pa.types.is_string(key_type) or pa.types.is_large_string(key_type))
            )
            if not compatible:
                raise context(
                    parsed,
                    step.segment,
                    address,
                    f"literal {step.value!r} is not an exact {key_type} map key",
                )
            try:
                pa.scalar(step.value, type=key_type)
            except (pa.ArrowException, OverflowError, TypeError, ValueError) as error:
                raise context(
                    parsed,
                    step.segment,
                    address,
                    f"literal {step.value!r} is not an exact {key_type} map key",
                ) from error
            datatype = datatype.item_type
            steps.append(Lookup(value=step.value, segment=step.segment))
            continue

        raise context(parsed, step.segment, address, f"bracket literal is invalid for {datatype}")

    output = datatype
    presence = pa.bool_()
    for wrapper in reversed(wrappers):
        output = nest(wrapper, output)
        presence = nest(wrapper, presence)

    return Plan(query=parsed, steps=tuple(steps), input=input_type, output=output, presence=presence)


def parts(values: pa.Array) -> tuple[np.ndarray, pa.Array, np.ndarray]:
    """Return list offsets, the underlying child array, and parent validity."""

    valid = pc.is_valid(values).to_numpy(zero_copy_only=False)
    if pa.types.is_fixed_size_list(values.type):
        values = cast(pa.FixedSizeListArray, values)
        start = values.offset * values.type.list_size
        stop = start + len(values) * values.type.list_size
        offsets = np.arange(start, stop + 1, values.type.list_size, dtype=np.int64)
        return offsets, values.values, valid

    variable = cast(pa.ListArray | pa.LargeListArray, values)
    offsets = variable.offsets.to_numpy(zero_copy_only=False)
    return offsets, variable.values, valid


def wrap(container: pa.Array, child: pa.Array, mask: pa.Array) -> pa.Array:
    """Rebuild one list container around transformed child values."""

    datatype = nest(cast(ListType, container.type), child.type)
    if pa.types.is_fixed_size_list(container.type):
        # Arrow supports explicit types; pyarrow-stubs omits this overload.
        return pa.FixedSizeListArray.from_arrays(child, type=datatype, mask=mask)  # pyrefly: ignore[no-matching-overload]

    offsets = cast(pa.ListArray | pa.LargeListArray, container).offsets
    base = offsets[0].as_py()
    # The offset and scalar dtypes agree; the stub cannot correlate their unions.
    normalized = pc.subtract(offsets, pa.scalar(base, type=offsets.type))  # pyrefly: ignore[no-matching-overload]
    if pa.types.is_large_list(container.type):
        return pa.LargeListArray.from_arrays(normalized, child, type=datatype, mask=mask)
    return pa.ListArray.from_arrays(normalized, child, type=cast(pa.ListType, datatype), mask=mask)


def index(values: pa.Array, present: pa.Array, position: int) -> Projection[pa.Array]:
    """Safely select one position from every list without bounds failures."""

    offsets, children, valid_parent = parts(values)
    lengths = np.diff(offsets)
    positions = lengths + position if position < 0 else np.full(len(values), position, dtype=np.int64)
    valid = valid_parent & (positions >= 0) & (positions < lengths)
    absolute = offsets[:-1] + np.maximum(positions, 0)
    indices = pa.array(absolute, mask=~valid, type=pa.int64())
    selected = pc.take(children, indices)
    structural = pc.and_(present, pa.array(valid, type=pa.bool_()))
    return Projection(values=selected, present=structural)


def subslice(values: pa.Array, present: pa.Array, start: int | None, stop: int | None) -> pa.Array:
    """Apply Python half-open slice bounds independently to every list."""

    offsets, children, valid = parts(values)
    lengths = np.diff(offsets)

    if start is None:
        lower = np.zeros(len(values), dtype=np.int64)
    elif start < 0:
        lower = np.maximum(lengths + start, 0)
    else:
        lower = np.minimum(start, lengths)

    if stop is None:
        upper = lengths.copy()
    elif stop < 0:
        upper = np.maximum(lengths + stop, 0)
    else:
        upper = np.minimum(stop, lengths)

    counts = np.maximum(upper - lower, 0)
    if pa.types.is_fixed_size_list(values.type):
        fixed_lower, fixed_upper = bounds(values.type.list_size, start, stop)
        counts.fill(fixed_upper - fixed_lower)
        lower.fill(fixed_lower)

    total = int(counts.sum())
    if total:
        parents = np.repeat(np.arange(len(values), dtype=np.int64), counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        positions = offsets[:-1][parents] + lower[parents] + np.arange(total, dtype=np.int64) - starts
        selected = pc.take(children, pa.array(positions, type=pa.int64()))
    else:
        selected = children.slice(0, 0)

    mask = pc.invert(pc.and_(present, pa.array(valid, type=pa.bool_())))
    datatype = sliced(values.type, start, stop)
    if pa.types.is_fixed_size_list(datatype):
        # Arrow supports explicit types; pyarrow-stubs omits this overload.
        return pa.FixedSizeListArray.from_arrays(selected, type=datatype, mask=mask)  # pyrefly: ignore[no-matching-overload]

    result_offsets = pa.array(np.concatenate(([0], np.cumsum(counts))), type=pa.int64())
    if pa.types.is_large_list(datatype):
        return pa.LargeListArray.from_arrays(result_offsets, selected, type=datatype, mask=mask)
    return pa.ListArray.from_arrays(
        cast(pa.Int32Array, pc.cast(result_offsets, pa.int32())), selected, type=cast(pa.ListType, datatype), mask=mask
    )


def lookup(values: pa.Array, present: pa.Array, step: Lookup, plan: Plan, address: str | None) -> Projection[pa.Array]:
    """Select one exact map key and reject duplicate matches."""

    matches = pc.map_lookup(values, step.value, occurrence="all")
    # Arrow accepts Python scalar arguments; the stubs require explicit Scalars.
    counts = cast(pa.Int32Array, pc.fill_null(pc.list_value_length(matches), 0))  # pyrefly: ignore[bad-specialization]
    duplicated = pc.any(pc.greater(counts, 1)).as_py()  # pyrefly: ignore[no-matching-overload]
    if duplicated:
        raise context(plan.query, step.segment, address, f"map contains duplicate key {step.value!r}")
    return index(matches, present, 0)


def select(
    values: pa.Array,
    present: pa.Array,
    steps: tuple[Bound, ...],
    plan: Plan,
    address: str | None,
) -> Projection[pa.Array]:
    """Execute bound steps against one non-chunked Arrow array."""

    if not steps:
        return Projection(values=values, present=present)

    step, *remaining = steps
    tail = tuple(remaining)
    if isinstance(step, Member):
        structural = pc.and_(present, pc.is_valid(values))
        child = pc.struct_field(values, step.name)
        return select(child, structural, tail, plan, address)
    if isinstance(step, Index):
        selected = index(values, present, step.value)
        return select(selected.values, selected.present, tail, plan, address)
    if isinstance(step, Slice):
        child = subslice(values, present, step.start, step.stop)
        return select(child, present, tail, plan, address)
    if isinstance(step, Lookup):
        selected = lookup(values, present, step, plan, address)
        return select(selected.values, selected.present, tail, plan, address)

    offsets, children, valid = parts(values)
    lengths = np.diff(offsets)
    parents = pa.array(np.repeat(np.arange(len(values), dtype=np.int64), lengths), type=pa.int64())
    child_present = pc.and_(pc.take(present, parents), pc.take(pa.array(valid), parents))
    start = int(offsets[0])
    stop = int(offsets[-1])
    selected = select(children.slice(start, stop - start), child_present, tail, plan, address)
    mask = pc.invert(pc.and_(present, pa.array(valid, type=pa.bool_())))
    return Projection(
        values=wrap(values, selected.values, mask),
        present=wrap(values, selected.present, mask),
    )


def query(
    source: pa.Table | pa.Array | pa.ChunkedArray,
    expression: Query | Plan | str,
    *,
    address: str | None = None,
) -> Projection[pa.Array | pa.ChunkedArray]:
    """Bind and execute a structural query without Python row materialization."""

    if isinstance(source, pa.Table):
        datatype = pa.struct(list(source.schema))
        # Arrow accepts nested zero-chunk types; its stubs omit that overload.
        values: pa.Array | pa.ChunkedArray = (
            source.to_struct_array() if source.num_rows else pa.chunked_array([], type=datatype)  # pyrefly: ignore[no-matching-overload]
        )
    elif isinstance(source, (pa.Array, pa.ChunkedArray)):
        datatype = source.type
        values = source
    else:
        raise TypeError(f"query source must be an Arrow Table, Array, or ChunkedArray, got {type(source).__name__}")

    if isinstance(expression, Plan):
        plan = expression
        if plan.input != datatype:
            raise TypeError(f"query plan expects Arrow input {plan.input}, got {datatype}")
    else:
        plan = bind(expression, datatype, address=address)

    if isinstance(values, pa.ChunkedArray):
        selected = [
            select(chunk, pa.repeat(pa.scalar(True), len(chunk)), plan.steps, plan, address) for chunk in values.chunks
        ]
        # Arrow accepts Array chunks with a runtime DataType; the stub omits this overload.
        return Projection(
            values=pa.chunked_array([item.values for item in selected], type=plan.output),  # pyrefly: ignore[no-matching-overload]
            present=pa.chunked_array([item.present for item in selected], type=plan.presence),  # pyrefly: ignore[no-matching-overload]
        )

    selected = select(values, pa.repeat(pa.scalar(True), len(values)), plan.steps, plan, address)
    if selected.values.type != plan.output or selected.present.type != plan.presence:
        location = f" for address {address!r}" if address is not None else ""
        raise TypeError(
            f"query {plan.query.text!r}{location} produced Arrow types "
            f"({selected.values.type}, {selected.present.type}); expected ({plan.output}, {plan.presence})"
        )
    return selected


def leaves(values: list[pa.Array], *, expressions: tuple[str, ...], address: str | None) -> pa.Array:
    """Interleave aligned scalar sources into one fixed-size channel axis.

    Args:
        values: One array per channel, all of the same length.
        expressions: The tuple query, used in messages.
        address: Field address, used in messages.

    Returns:
        A fixed-size list per row, null only when every channel is null.

    Raises:
        ValueError: If the source types have no common type.
    """
    types = [item.type for item in values]
    try:
        schemas = [pa.schema([pa.field("value", datatype)]) for datatype in types]
        common = pa.unify_schemas(schemas, promote_options="permissive").field("value").type
        converted = [pc.cast(item, common, safe=True) for item in values]
    except pa.ArrowException as error:
        raise ValueError(
            f"tuple query {expressions!r} at {address!r} cannot stack source types {types}; "
            "cast them to one type in a preprocessor"
        ) from error

    rows, width = len(values[0]), len(values)
    order = (np.arange(rows)[:, None] + rows * np.arange(width)[None, :]).ravel()
    child = pc.take(pa.concat_arrays(converted), pa.array(order, type=pa.int64()))
    any_valid = np.any([pc.is_valid(item).to_numpy(zero_copy_only=False) for item in values], axis=0)
    # Arrow accepts an explicit mask; pyarrow-stubs omits it.
    return pa.FixedSizeListArray.from_arrays(child, width, mask=pa.array(~any_valid, type=pa.bool_()))  # pyrefly: ignore[no-matching-overload]


def lists(values: list[pa.Array], *, expressions: tuple[str, ...], address: str | None) -> pa.Array:
    """Align list sources on shared offsets and stack their children.

    A null list is a missing channel: it takes the length of the valid sources.
    Valid lists must have equal lengths.

    Args:
        values: One list array per channel, all of the same length.
        expressions: The tuple query, used in messages.
        address: Field address, used in messages.

    Returns:
        One list container holding the stacked children, null when every source is null.

    Raises:
        ValueError: If valid lists differ in length at some record.
    """
    sources = [parts(item) for item in values]
    offsets = [item[0] for item in sources]
    valid = np.stack([item[2] for item in sources])
    lengths = np.stack([np.diff(item) for item in offsets])
    reference = np.where(valid, lengths, 0).max(axis=0)
    mismatch = valid & (lengths != reference[None, :])
    if mismatch.any():
        first = int(np.flatnonzero(mismatch.any(axis=0))[0])
        raise ValueError(
            f"tuple query {expressions!r} at {address!r} selects lists of different lengths at record {first}: "
            f"{lengths[:, first].tolist()}; every source must select the same rows. Pad or trim them in a preprocessor"
        )

    total = int(reference.sum())
    starts = np.cumsum(reference) - reference
    parents = np.repeat(np.arange(len(reference)), reference)
    within = np.arange(total) - np.repeat(starts, reference)
    children = [
        pc.take(source[1], pa.array(base[parents] + within, mask=~ok[parents], type=pa.int64()))
        for source, base, ok in zip(sources, offsets, valid, strict=True)
    ]
    inner = channels(children, expressions=expressions, address=address)

    large = any(pa.types.is_large_list(item.type) for item in values)
    boundaries = pa.array(np.append(starts, total), type=pa.int64() if large else pa.int32())
    container = pa.LargeListArray if large else pa.ListArray
    mask = pa.array(np.asarray(~valid.any(axis=0), dtype=bool))
    # Arrow accepts an explicit mask; pyarrow-stubs omits it.
    return container.from_arrays(boundaries, inner, mask=mask)  # pyrefly: ignore[no-matching-overload]


def channels(values: list[pa.Array], *, expressions: tuple[str, ...], address: str | None) -> pa.Array:
    """Stack aligned sources level by level until the scalar leaves.

    A source whose Arrow type is ``null`` has no value anywhere; it takes the
    type of its siblings so it reads as a missing channel.

    Raises:
        ValueError: If only some sources are lists.
    """
    typed = next((item.type for item in values if not pa.types.is_null(item.type)), None)
    if typed is not None:
        values = [pc.cast(item, typed) if pa.types.is_null(item.type) else item for item in values]
    nested = [listed(item.type) for item in values]
    if not any(nested):
        return leaves(values, expressions=expressions, address=address)
    if not all(nested):
        raise ValueError(
            f"tuple query {expressions!r} at {address!r} selects values at different nesting: "
            f"{[str(item.type) for item in values]}; select the same list depth for every source"
        )
    return lists(values, expressions=expressions, address=address)


def observed(present: list[pa.Array]) -> pa.Array:
    """Combine source presences: a record is present when any source is present at it."""
    flags = [
        item if item.type == pa.bool_() else pc.is_valid(item)  # traversed sources carry list presence
        for item in present
    ]
    return pa.array(np.any([item.fill_null(False).to_numpy(zero_copy_only=False) for item in flags], axis=0))


def stack(
    source: pa.Table | pa.Array | pa.ChunkedArray, expressions: tuple[str, ...], *, address: str | None = None
) -> Projection[pa.Array | pa.ChunkedArray]:
    """Bind several queries and stack the selected values into a trailing channel axis.

    Channel ``c`` reads ``expressions[c]``. Validity is kept per channel: a
    source that is null at a row only makes its channel null there.

    Args:
        source: Records, as for ``query``.
        expressions: One query per channel.
        address: Field address, used in messages.

    Returns:
        Values of ``fixed_size_list<N>`` below any shared list levels, and one
        Boolean presence per record: true when any source is present.

    Raises:
        ValueError: If the sources select different nesting, different list
            lengths, or types without a common type.
    """
    selected = [query(source, text, address=address) for text in expressions]
    chunked = isinstance(selected[0].values, pa.ChunkedArray)
    groups = (
        list(zip(*(cast(pa.ChunkedArray, item.values).chunks for item in selected), strict=True))
        if chunked
        else [tuple(cast(pa.Array, item.values) for item in selected)]
    )
    marks = (
        list(zip(*(cast(pa.ChunkedArray, item.present).chunks for item in selected), strict=True))
        if chunked
        else [tuple(cast(pa.Array, item.present) for item in selected)]
    )
    # A zero-chunk input still needs the stacked type, so stack one empty group.
    empty = chunked and not groups
    if empty:
        groups = [tuple(pa.array([], type=item.values.type) for item in selected)]
        marks = [tuple(pa.array([], type=item.present.type) for item in selected)]
    stacked = [channels(list(group), expressions=expressions, address=address) for group in groups]
    present = [observed(list(mark)) for mark in marks]
    if not chunked:
        return Projection(values=stacked[0], present=present[0])
    return Projection(
        values=pa.chunked_array([] if empty else stacked, type=stacked[0].type),
        present=pa.chunked_array([] if empty else present, type=pa.bool_()),
    )


__all__ = ["Plan", "Projection", "Query", "bind", "compile", "query", "scope", "stack"]
