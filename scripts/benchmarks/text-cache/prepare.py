"""Build the saved text-only user-badge fixture from a local relbench cache.

Requires duckdb and pyarrow. This reads cached Parquet files and never trains
or downloads a model. Keep one generated fixture for every compared checkout.
"""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def project(array, fields):
    children = []
    for name, definition in fields.items():
        child = array.field(name)
        if definition["type"] == "branch":
            child = pa.ListArray.from_arrays(
                child.offsets, project(child.values, definition["fields"]), mask=pc.is_null(child)
            )
        children.append(child)
    return pa.StructArray.from_arrays(children, names=list(fields), mask=pc.is_null(array))


def counts(table, fields, prefix=""):
    output = {}
    for name, definition in fields.items():
        path = prefix + "/" + name
        array = table[name].combine_chunks()
        if definition["type"] == "branch":
            child = pa.Table.from_struct_array(pc.list_flatten(array))
            output.update(counts(child, definition["fields"], path))
        else:
            output[path] = len(array.drop_null())
    return output


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    files = Path(__file__).parent
    args.output.mkdir(parents=True, exist_ok=True)
    fields = json.loads((files / "user-badge.fields.json").read_text())
    with duckdb.connect() as connection:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='8GB'")
        for path in sorted((args.database / "db").glob("*.parquet")):
            quoted = str(path.resolve()).replace("'", "''")
            connection.execute(
                f'CREATE VIEW "{path.stem}" AS SELECT *, file_row_number AS "__source_row_id" '
                f"FROM read_parquet('{quoted}', file_row_number=true)"
            )
        source = args.database / "tasks/user-badge/train.parquet"
        anchors = connection.execute(
            "WITH sample AS (SELECT UserId, timestamp, hash(UserId, timestamp, ?) AS ordinal "
            "FROM read_parquet(?) ORDER BY ordinal, UserId, timestamp LIMIT ?) "
            'SELECT row_number() OVER (ORDER BY ordinal, UserId, timestamp)-1 AS "__row_position", '
            'UserId AS "__entity_key", timestamp AS "__as_of" FROM sample',
            [args.seed, str(source), args.rows],
        ).to_arrow_table()
        pq.write_table(anchors, args.output / "anchors.parquet")
        table = connection.execute(
            (files / "user-badge.sql").read_text(),
            {"anchors": str(args.output / "anchors.parquet"), "start": 0, "stop": args.rows},
        ).to_arrow_table()
    assert len(table) == args.rows
    root = pa.StructArray.from_arrays(
        [table[name].combine_chunks() for name in table.column_names], names=table.column_names
    )
    table = pa.Table.from_struct_array(project(root, fields))
    path = args.output / "observations.arrow"
    with ipc.new_file(str(path), table.schema) as writer:
        writer.write_table(table)
    (args.output / "fields.json").write_text(json.dumps(fields, indent=2))
    text_counts = counts(table, fields)
    receipt = dict(
        task="rel-stack/user-badge",
        rows=len(table),
        seed=args.seed,
        source_rows=pq.ParquetFile(source).metadata.num_rows,
        sampling="deterministic hash(UserId, timestamp, seed), without replacement; ordered by hash",
        scope="Saved temporally censored nested recipe, retaining only text leaves. No labels or model training.",
        text_values=sum(text_counts.values()),
        text_fields=len(text_counts),
        text_counts=text_counts,
        observations_sha256=digest(path),
        sql_sha256=digest(files / "user-badge.sql"),
        fields_sha256=digest(args.output / "fields.json"),
        arrow_bytes=table.nbytes,
        duckdb=duckdb.__version__,
    )
    (args.output / "observe.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path.home() / ".cache/relbench/rel-stack")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7302)
    prepare(parser.parse_args())
