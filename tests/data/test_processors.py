"""Polars processors return surviving columns to their source Arrow fields."""

from __future__ import annotations

import polars as pl
import pyarrow as pa

import relflow as rf
from relflow.data.datasets.arrow import process
from relflow.data.processors import apply, equivalent
from relflow.structs.enums import Strata


def envelope() -> pa.Table:
    item = pa.struct([pa.field("a", pa.float32()), pa.field("state", pa.int8())])
    items = pa.array(
        [None, [], [{"a": 1.0, "state": 0}, None], [{"a": 2.0, "state": 1}]],
        type=pa.list_(pa.field("element", item)),
    )
    embedding = pa.list_(pa.field("element", pa.float32()), 4)
    representations = pa.array(
        [[{"embedding": [1.0, 2.0, 3.0, 4.0]}], None, [], [{"embedding": None}]],
        type=pa.list_(pa.field("element", pa.struct([pa.field("embedding", embedding)]))),
    )
    inputs = pa.StructArray.from_arrays(
        [pa.array(["x", "y", None, "z"]).dictionary_encode(), pa.array(["p", "q", "r", "s"])],
        names=["merchant", "note"],
    )
    predictions = pa.StructArray.from_arrays([items], names=["items"])
    schema = pa.schema(
        [
            pa.field("predictions", predictions.type),
            pa.field("representations", representations.type),
            pa.field("inputs", inputs.type, metadata={"source": "request"}),
        ]
    )
    return pa.Table.from_arrays([predictions, representations, inputs], schema=schema)


@rf.postprocess
def identity(frame: pl.DataFrame) -> pl.DataFrame:
    return frame


@rf.postprocess
def widen(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.lit("new").alias("label"))


@rf.postprocess
def promote(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("count").cast(pl.Float64))


@rf.preprocess
def double(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns((pl.col("amount") * 2).alias("double"))


def test_identity_postprocessor_preserves_the_arrow_output_schema():
    table = envelope()

    result = apply(table, [identity])

    assert result.schema.equals(table.schema, check_metadata=True)
    assert result.to_pylist() == table.to_pylist()


def test_new_columns_keep_polars_types_while_source_columns_return():
    table = envelope()

    result = apply(table, [widen])

    assert result.schema.field("label").type == pa.large_string()
    assert result.select(table.column_names).schema.equals(table.schema, check_metadata=True)


def test_intentional_type_changes_survive():
    table = pa.table({"count": pa.array([1, 2], type=pa.int64())})

    assert apply(table, [promote]).schema.field("count").type == pa.float64()


def test_preprocessor_output_returns_source_columns():
    events = pa.array([[{"x": 1}], []], type=pa.list_(pa.field("element", pa.struct([pa.field("x", pa.int64())]))))
    source = pa.table({"amount": [1.0, 2.0], "events": events, "name": pa.array(["a", "b"]).dictionary_encode()})

    [processed] = list(
        process(
            (source,),
            preprocessor=double,
            strata=Strata.train,
            schema=None,
            encoding_context={},
        )
    )

    assert processed.schema.field("events").equals(source.schema.field("events"), check_metadata=True)
    assert processed.schema.field("name").type == pa.dictionary(pa.int32(), pa.string())
    assert processed.schema.field("double").type == pa.float64()
    assert processed["events"].to_pylist() == source["events"].to_pylist()


def test_equivalent_separates_polars_renderings_from_real_changes():
    assert equivalent(pa.list_(pa.field("element", pa.string())), pa.large_list(pa.field("item", pa.large_string())))
    assert equivalent(pa.dictionary(pa.int8(), pa.string()), pa.dictionary(pa.uint32(), pa.large_string()))
    assert equivalent(pa.binary(), pa.large_binary())
    assert not equivalent(pa.list_(pa.int64()), pa.list_(pa.float64()))
    assert not equivalent(
        pa.list_(pa.field("element", pa.float32()), 4), pa.list_(pa.field("element", pa.float32()), 3)
    )
    assert not equivalent(pa.int64(), pa.float64())
    assert not equivalent(pa.struct([pa.field("a", pa.int64())]), pa.struct([pa.field("b", pa.int64())]))
