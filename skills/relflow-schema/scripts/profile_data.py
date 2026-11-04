"""Profile Arrow structure and sampled value patterns without importing relflow."""

from __future__ import annotations

import argparse
import datetime
import decimal
import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from observations import MAX_BYTES, read


def summary(values: pa.Array) -> dict:
    """Summarize a numeric array with Arrow kernels, retaining exact extrema."""
    if not len(values) or values.null_count == len(values):
        return {}
    result = pc.min_max(values).as_py()
    result["quantiles"] = pc.quantile(values, q=[0.5, 0.95, 0.99]).to_pylist()
    return result


def field(values: pa.Array, *, examples: int, max_values: int, depth: int, remaining: list[int]) -> dict:
    """Recurse over Arrow containers, never over individual data records."""
    if remaining[0] == 0:
        raise ValueError("profile exceeds --max-fields; select columns or raise the explicit budget")
    remaining[0] -= 1
    total = len(values)
    values = values.slice(0, max_values)
    datatype = values.type
    result = {
        "arrow_type": str(datatype),
        "available_values": total,
        "sampled_values": len(values),
        "truncated": total > len(values),
        "nulls": values.null_count,
    }
    if depth == 0:
        result["details"] = "depth limit reached"
        return result
    if pa.types.is_dictionary(datatype):
        values = values.dictionary_decode()
        datatype = values.type
    if pa.types.is_struct(datatype):
        if len(set(datatype.names)) != len(datatype):
            raise ValueError("struct field names must be unique; rename duplicate children")
        result["kind"] = "record"
        result["fields"] = {
            child.name: field(array, examples=examples, max_values=max_values, depth=depth - 1, remaining=remaining)
            for child, array in zip(datatype, values.flatten(), strict=True)
        }
    elif pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype):
        result.update(kind="list", lengths=summary(pc.list_value_length(values)))
        result["element"] = field(
            pc.list_flatten(values),
            examples=examples,
            max_values=max_values,
            depth=depth - 1,
            remaining=remaining,
        )
    elif pa.types.is_union(datatype) or pa.types.is_map(datatype) or isinstance(datatype, pa.ExtensionType):
        result.update(kind="extension-or-variant", details="inspect with the owning datatype or source adapter")
    else:
        result["kind"] = "scalar"
        valid = pc.drop_null(values)
        result["distinct"] = pc.count_distinct(valid).as_py() if len(valid) else 0
        if examples:
            result["examples"] = [
                value[:160]
                if isinstance(value, str)
                else str(value)
                if isinstance(value, float) and not math.isfinite(value)
                else value
                for value in (pc.unique(valid).slice(0, examples).to_pylist() if len(valid) else [])
            ]
        if pa.types.is_string(datatype) or pa.types.is_large_string(datatype):
            result["characters"] = summary(pc.utf8_length(valid))
            result["patterns"] = {
                name: pc.sum(pc.cast(pc.match_substring_regex(valid, pattern), pa.int64())).as_py() or 0
                for name, pattern in {
                    "integer_text": r"^[+-]?[0-9]+$",
                    "leading_zero_integer_text": r"^0[0-9]+$",
                    "iso_date_prefix": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}($|[T ])",
                }.items()
            }
        elif pa.types.is_floating(datatype):
            result["nonfinite"] = pc.sum(pc.cast(pc.invert(pc.is_finite(valid)), pa.int64())).as_py() or 0
            result["numeric"] = summary(pc.filter(valid, pc.is_finite(valid)))
        elif pa.types.is_integer(datatype) or pa.types.is_decimal(datatype):
            result["numeric"] = summary(valid)
    return result


def profile(
    table: pa.Table, *, examples: int = 3, max_values: int = 100_000, depth: int = 8, max_fields: int = 2048
) -> dict:
    if examples < 0 or min(max_values, depth, max_fields) < 1:
        raise ValueError("examples must be nonnegative; max_values, depth, and max_fields must be positive")
    remaining = [max_fields]
    return {
        "scope": "sampled Arrow structure and values; semantic choices remain open",
        "fields": {
            name: field(
                table[name].combine_chunks(),
                examples=examples,
                max_values=max_values,
                depth=depth,
                remaining=remaining,
            )
            for name in table.column_names
        },
    }


def json_value(value: object) -> str:
    """Render non-JSON Arrow atoms only at the report boundary."""
    if isinstance(value, (datetime.date, datetime.time, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, datetime.timedelta):
        return str(value)
    if isinstance(value, bytes):
        return value[:80].hex()
    raise TypeError(f"profile cannot render report value of type {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--max-values", type=int, default=100_000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--max-fields", type=int, default=2048)
    args = parser.parse_args()
    try:
        table, source = read(args.source, rows=args.rows, max_bytes=args.max_bytes)
        result = profile(
            table, examples=args.examples, max_values=args.max_values, depth=args.depth, max_fields=args.max_fields
        )
        print(json.dumps({"source": source, **result}, indent=2, default=json_value, allow_nan=False))
    except (OSError, ValueError, TypeError, pa.ArrowException) as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
