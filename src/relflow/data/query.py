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
from lark import Lark, UnexpectedInput

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


def member(name: str) -> str:
    """Render a schema name as one native query member."""

    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) else f"[{json.dumps(name)}]"


GRAMMAR = r"""
?start: absolute
      | relative
      | path

absolute: "/" branch* path
relative: RELATIVE path
branch: BRANCH "/"

path: first step*
?first: NAME                       -> member
      | "[" STRING "]"             -> string

?step: "." NAME                    -> member
     | "[" STRING "]"              -> string
     | "[" BOOL "]"                -> boolean
     | "[" INT "]"                 -> integer
     | "[" "*" "]"                 -> traverse
     | "[" [INT] ":" [INT] "]"    -> slice

RELATIVE: "../"+

// Priority keeps slash-terminated branches ahead of overlapping field names.
BRANCH.2: /[\w-]+(?=\/)/u

// Validate Python identifier characters and JSON string escapes after lexing.
NAME: /[^\W\d_][\w]*/u
STRING: /"(?:[^"\\\x00-\x1f]|\\.)*"/
INT: /-?(0|[1-9][0-9]*)/
BOOL: "true" | "false"
"""

parser = Lark(GRAMMAR, parser="lalr", propagate_positions=True)


def error(expression: str, offset: int, reason: str) -> ValueError:
    segment = expression[offset:] or "<end>"
    return ValueError(f"invalid query {expression!r} at {segment!r}: {reason}")


def diagnostic(expression: str, failure: UnexpectedInput) -> ValueError:
    """Translate common unsupported constructs, then use a generic syntax error."""

    if not expression or not expression.strip():
        return error(expression, 0, "query must be a non-empty string")
    if expression != expression.strip():
        return error(expression, 0, "whitespace is not allowed outside quoted field names")
    if expression.startswith("[*]"):
        return error(expression, 0, "queries are observation-relative and must not begin with '[*]'")
    if expression.startswith("$"):
        return error(expression, 0, "queries are observation-relative and must not begin with '$'")

    offset = max(0, failure.pos_in_stream or 0)
    remaining = expression[offset:]
    expected = getattr(failure, "expected", None) or getattr(failure, "allowed", None) or set()
    if remaining.startswith("$") and "NAME" in expected and "LSQB" in expected:
        return error(expression, offset, "queries are observation-relative and must not begin with '$'")
    if expression.startswith("/") and "/" in remaining.split("[", 1)[0]:
        return ValueError(f"invalid absolute query {expression!r}: branch names must be schema names")
    if "NAME" in expected and "LSQB" in expected:
        if remaining and remaining[0].isspace():
            return error(expression, offset, "whitespace is not allowed outside quoted field names")
        return error(expression, offset, "expected an identifier beginning with a letter")
    bracket = expression.rfind("[", 0, offset + 1)
    if bracket >= 0 and "]" not in expression[bracket:offset]:
        content = expression[bracket + 1 :]
        if content.startswith("*]") and "STRING" in expected:
            return error(expression, bracket, "queries are observation-relative and must not begin with '[*]'")
        if content.startswith('"'):
            try:
                _, consumed = json.JSONDecoder().raw_decode(content)
            except json.JSONDecodeError:
                return error(expression, bracket, "invalid JSON string literal")
            if content[consumed : consumed + 1] != "]":
                return error(expression, bracket, "expected ']' after the string literal")
        end = content.find("]")
        if end < 0:
            return error(expression, bracket, "unterminated bracket")
        content = content[:end]
        if not content:
            return error(expression, bracket, "flattening is not supported; use a preprocessor")
        if content.startswith("?"):
            return error(expression, bracket, "filters are not supported; use a preprocessor")
        if ":" in content:
            bounds = content.split(":")
            if len(bounds) != 2:
                return error(expression, bracket, "slice steps are not supported")
            if not all(not bound or re.fullmatch(r"-?(0|[1-9][0-9]*)", bound) for bound in bounds):
                return error(expression, bracket, "slice bounds must be integers")
        if "STRING" in expected and (
            content in {"true", "false"} or re.fullmatch(r"-?(0|[1-9][0-9]*)", content) or ":" in content
        ):
            return error(expression, bracket, "a query must begin with a field name")
        if content.startswith("'"):
            return error(expression, bracket, "string literals must use JSON double quotes")
        if "," in content:
            return error(expression, bracket, "multiselects are not supported; use a preprocessor")
        return error(expression, bracket, "unsupported bracket expression")
    if remaining.startswith("..") or (remaining.startswith(".") and offset and expression[offset - 1] == "."):
        return error(expression, max(0, offset - 1), "recursive descent is not supported; use a preprocessor")
    if remaining.startswith("*") and offset and expression[offset - 1] == ".":
        return error(expression, offset, "object wildcards are not supported; name the field explicitly")
    if remaining.startswith("|"):
        return error(expression, offset, "pipes are not supported; use a preprocessor")
    if remaining.startswith("("):
        return error(expression, offset, "functions are not supported; use a preprocessor")
    if remaining and remaining[0] in "<>=!&":
        return error(expression, offset, "expressions are not supported; use a preprocessor")
    if "NAME" in expected:
        return error(expression, offset, "expected an identifier beginning with a letter")
    if "DOT" in expected or "LSQB" in expected:
        return error(expression, offset, "expected '.' or '['")
    return error(expression, offset, "expected a structural path member or bracket selector")


@cache
def scope(expression: str) -> tuple[int | tuple[str, ...], Query]:
    """Parse a branch scope and its Arrow-neutral structural query together."""

    if not isinstance(expression, str):
        raise TypeError(f"query must be a string, got {type(expression).__name__}")
    try:
        tree = parser.parse(expression)
    except UnexpectedInput as failure:
        raise diagnostic(expression, failure) from failure

    owner: int | tuple[str, ...]
    if tree.data == "absolute":
        owner = tuple(str(branch.children[0]) for branch in tree.children[:-1])
        path = tree.children[-1]
    elif tree.data == "relative":
        owner = len(str(tree.children[0])) // 3
        path = tree.children[-1]
    else:
        owner, path = 0, tree

    steps: list[Parsed] = []
    for node in path.children:
        segment = expression[node.meta.start_pos : node.meta.end_pos]
        match node.data:
            case "member":
                token = node.children[0]
                name = str(token)
                start = node.meta.end_pos - len(name)
                if not name[0].isalpha():
                    raise error(expression, start, "expected an identifier beginning with a letter")
                for index, char in enumerate(name[1:], start=1):
                    if not (char.isalpha() or char.isdigit() or char == "_"):
                        raise error(expression, start + index, "expected '.' or '['")
                steps.append(Member(name=name, segment=name))
            case "string":
                try:
                    value = json.loads(str(node.children[0]))
                except json.JSONDecodeError as failure:
                    raise error(expression, node.meta.start_pos, "invalid JSON string literal") from failure
                steps.append(Literal(value=value, segment=segment))
            case "boolean":
                steps.append(Literal(value=str(node.children[0]) == "true", segment=segment))
            case "integer":
                steps.append(Literal(value=int(str(node.children[0])), segment=segment))
            case "slice":
                lower, upper = node.children
                steps.append(
                    Slice(
                        start=None if lower is None else int(str(lower)),
                        stop=None if upper is None else int(str(upper)),
                        segment=segment,
                    )
                )
            case "traverse":
                steps.append(Traverse())
            case _:
                raise AssertionError(node.data)
    local = expression[path.meta.start_pos : path.meta.end_pos]
    return owner, Query(text=local, steps=tuple(steps))


def compile(expression: str) -> Query:
    """Parse one observation-relative structural query."""

    if isinstance(expression, str) and (expression.startswith("/") or expression.startswith("../")):
        raise error(expression, 0, "expected an observation-relative structural path")
    _, query = scope(expression)
    return query


def origin(expression: str, *, address: str) -> tuple[int, Query]:
    """Resolve relative and absolute branch scopes against a bound leaf."""
    owner, path = scope(expression)
    branches = tuple(address.strip("/").split("/")[:-1])
    if isinstance(owner, tuple):
        if branches[: len(owner)] != owner:
            raise ValueError(
                f"absolute source query {expression!r} at '{address}' must name the current branch or an ancestor"
            )
        return len(branches) - len(owner), path
    if owner > len(branches):
        raise ValueError(
            f"source query {expression!r} at '{address}' ascends {owner} owner branches, "
            f"but only {len(branches)} ancestor branches exist; keep the source within the model root"
        )
    return owner, path


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


def gather(
    source: pa.Array, expression: Query | str, indices: pa.Array, *, address: str | None = None
) -> Projection[pa.Array]:
    """Select query inputs at particular rows without gathering unrelated struct members.

    Follow struct members before gathering, then execute remaining list or map
    operations on the selected rows. Null indices and null enclosing structs
    remain structurally absent.
    """
    plan = bind(expression, source.type, address=address)
    steps = plan.steps
    present = pc.is_valid(indices)
    while steps and isinstance(steps[0], Member):
        if source.null_count:
            present = pc.and_kleene(present, pc.take(pc.is_valid(source), indices))
        source = pc.struct_field(source, steps[0].name)
        steps = steps[1:]
    return select(pc.take(source, indices), present, steps, plan, address)


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


__all__ = ["Plan", "Projection", "Query", "bind", "compile", "query", "scope"]
