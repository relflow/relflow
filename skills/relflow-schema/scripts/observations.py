"""Bounded local observation ingress shared by profiling and validation.

Limits constrain retained Arrow data and JSON input bytes, not peak allocation
inside third-party parsers. This is a local inspection tool, not a parser sandbox.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as parquet

MAX_BYTES = 32 * 1024 * 1024


def records(values: object) -> pa.Table:
    """Align the fields of bounded JSON records before their one Arrow conversion."""
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
        raise ValueError("JSON observations must be an object or an array of objects; extract records first")
    names = dict.fromkeys(name for row in values for name in row)
    if values and not names:
        raise ValueError("JSON observations contain no fields; provide records with named fields")
    return pa.Table.from_pylist([{name: row.get(name) for name in names} for row in values])


def read(path: Path, *, rows: int = 2048, max_bytes: int = MAX_BYTES) -> tuple[pa.Table, dict]:
    """Read a prefix without loading an entire Parquet or Arrow dataset."""
    if rows < 1 or max_bytes < 1:
        raise ValueError("observation rows and max_bytes must be positive integers")
    path = path.resolve(strict=True)
    stat = path.stat()
    metadata = {"path": str(path), "file_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    suffix = path.suffix.lower()
    batches = []
    count = 0
    size = 0

    if suffix in {".json", ".jsonl", ".ndjson"}:
        metadata["format"] = "json" if suffix == ".json" else "jsonl"
        if suffix == ".json":
            if stat.st_size > max_bytes:
                raise ValueError(
                    f"JSON source has {stat.st_size} bytes, above max_bytes={max_bytes}; use JSONL or Arrow"
                )
            with path.open(encoding="utf-8") as source:
                values = json.load(source)
            if isinstance(values, list):
                metadata["total_rows"] = len(values)
                values = values[:rows]
            table = records(values)
        else:
            values = []
            with path.open("rb") as source:
                while len(values) < rows:
                    line = source.readline(max_bytes - size + 1)
                    if not line:
                        break
                    size += len(line)
                    if size > max_bytes:
                        raise ValueError(f"JSONL prefix exceeds max_bytes={max_bytes}; lower --rows")
                    if line.strip():
                        values.append(json.loads(line))
            table = records(values)
    else:
        with path.open("rb") as source:
            if suffix == ".parquet":
                reader = parquet.ParquetFile(source)
                metadata.update(format="parquet", total_rows=reader.metadata.num_rows)
                schema = reader.schema_arrow
                iterator = reader.iter_batches(batch_size=min(rows, 256))
            elif suffix == ".csv":
                options = csv.ReadOptions(block_size=min(max_bytes, 1024 * 1024))
                names = csv.open_csv(source, read_options=options).schema.names
                source.seek(0)
                # Preserve identifiers, leading zeros, and literal null-like strings.
                reader = csv.open_csv(
                    source,
                    read_options=options,
                    convert_options=csv.ConvertOptions(column_types=dict.fromkeys(names, pa.string())),
                )
                metadata.update(format="csv", values="strings; conversions require explicit interpretation")
                schema = reader.schema
                iterator = iter(reader)
            elif suffix in {".arrow", ".ipc", ".feather"}:
                metadata["format"] = "arrow-ipc"
                try:
                    reader = pa.ipc.open_file(source)
                except pa.ArrowInvalid:
                    source.seek(0)
                    reader = pa.ipc.open_stream(source)
                    iterator = iter(reader)
                else:
                    iterator = (reader.get_batch(index) for index in range(reader.num_record_batches))
                schema = reader.schema
            else:
                raise ValueError(
                    f"unsupported observation suffix {suffix!r}; use JSON, JSONL, CSV, Parquet, or Arrow IPC"
                )
            for batch in iterator:
                batch = batch.slice(0, rows - count)
                size += batch.nbytes
                if size > max_bytes:
                    raise ValueError(
                        f"Arrow prefix exceeds max_bytes={max_bytes}; lower --rows or prepare smaller records"
                    )
                batches.append(batch)
                count += batch.num_rows
                if count == rows:
                    break
            table = pa.Table.from_batches(batches, schema=schema)

    if table.nbytes > max_bytes:
        raise ValueError(f"Arrow prefix has {table.nbytes} bytes, above max_bytes={max_bytes}; lower --rows")
    if len(set(table.column_names)) != table.num_columns:
        raise ValueError("observation column names must be unique; rename duplicate columns before profiling")
    table = table.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    metadata.update(
        sample_rows=table.num_rows,
        sample_bytes=table.nbytes,
        sample_sha256=hashlib.sha256(sink.getvalue()).hexdigest(),
        sampling="prefix; statistics do not establish population constraints",
    )
    return table, metadata
