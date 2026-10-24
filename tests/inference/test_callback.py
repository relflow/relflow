from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import relflow as rf
from relflow.inference.callback import Writer


def write(writer: Writer, output: object, *, rank: int = 0, batch_idx: int = 0) -> None:
    writer.write_on_batch_end(
        trainer=SimpleNamespace(global_rank=rank),
        pl_module=SimpleNamespace(),
        output=output,
        batch_indices=None,
        batch=None,
        batch_idx=batch_idx,
        dataloader_idx=0,
    )


def test_writer_persists_the_written_arrow_table(tmp_path):
    output = pa.table(
        {
            "inputs": pa.array([{"request_id": "a"}, {"request_id": "b"}]),
            "predictions": pa.array([{"score": 0.25}, {"score": 0.75}]),
        }
    )
    writer = Writer(tmp_path)

    write(writer, output, rank=7)
    writer.on_predict_end(SimpleNamespace(), SimpleNamespace())

    path = tmp_path / "rank-7.parquet"
    assert path.exists()
    assert not (tmp_path / "rank-0.parquet").exists()
    assert pq.read_table(path).equals(output)
    assert writer.writer is None


def test_writer_applies_polars_postprocessors_in_order_before_persistence(tmp_path):
    calls = []

    @rf.postprocess
    def compact(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("compact")
        return frame.select(
            request_id=pl.col("inputs").struct.field("request_id"),
            score=pl.col("predictions").struct.field("score"),
        )

    @rf.postprocess
    def decide(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("decide")
        return frame.select("request_id", review=pl.col("score") >= 0.5)

    output = pa.table(
        {
            "inputs": pa.array([{"request_id": "a"}]),
            "predictions": pa.array([{"score": 0.5}]),
        }
    )
    writer = Writer(tmp_path, postprocessor=[compact, decide])

    write(writer, output)
    writer.close()

    assert calls == ["compact", "decide"]
    assert writer.postprocessors == (compact, decide)
    table = pq.read_table(tmp_path / "rank-0.parquet")
    assert table.column_names == ["request_id", "review"]
    assert table.to_pylist() == [{"request_id": "a", "review": True}]


def test_writer_accepts_postprocessor_row_changes(tmp_path):
    @rf.postprocess
    def reorder(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.sort("score", descending=True).head(1)

    writer = Writer(tmp_path, postprocessor=reorder)
    write(writer, pa.table({"score": [0.25, 0.75]}))
    writer.close()

    assert pq.read_table(tmp_path / "rank-0.parquet").to_pydict() == {"score": [0.75]}


def test_writer_locks_the_exact_first_batch_schema_and_discards_on_drift(tmp_path):
    writer = Writer(tmp_path)
    write(writer, pa.table({"score": pa.array([1], type=pa.int64())}), batch_idx=0)
    assert (tmp_path / "rank-0.parquet.partial").exists()

    with pytest.raises(ValueError, match="schema differs from the first batch"):
        write(writer, pa.table({"score": pa.array([1.0], type=pa.float64())}), batch_idx=1)

    assert writer.writer is None
    assert writer.schema is None
    assert list(tmp_path.iterdir()) == []


def test_writer_locks_prediction_metadata_and_discards_on_provenance_drift(tmp_path):
    writer = Writer(tmp_path)
    output = pa.table({"score": [0.25]}).replace_schema_metadata({b"relflow.model_version": b"first"})
    write(writer, output)

    with pytest.raises(ValueError, match="schema differs from the first batch"):
        write(writer, output.replace_schema_metadata({b"relflow.model_version": b"second"}), batch_idx=1)

    assert writer.writer is None
    assert list(tmp_path.iterdir()) == []


def test_writer_preserves_metadata_through_postprocessing(tmp_path):
    @rf.postprocess
    def select(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select("score")

    output = pa.table({"score": [0.25], "other": [1]}).replace_schema_metadata({b"relflow.model_version": b"example"})
    writer = Writer(tmp_path, postprocessor=select)
    write(writer, output)
    writer.close()

    restored = pq.read_table(tmp_path / "rank-0.parquet")
    assert restored.column_names == ["score"]
    assert restored.schema.metadata == output.schema.metadata


@pytest.mark.parametrize("layout", ["null_fixed_list", "null_struct", "null_struct_in_list"])
def test_writer_rejects_null_fixed_size_list_coordinates_before_creating_a_shard(tmp_path, layout):
    vector = pa.list_(pa.field("element", pa.float32()), 4)
    embedded = pa.struct([("embedding", vector)])
    if layout == "null_fixed_list":
        column = pa.array([None], type=vector)
    elif layout == "null_struct":
        column = pa.array([None], type=embedded)
    else:
        column = pa.array([[None]], type=pa.list_(pa.field("element", embedded)))
    writer = Writer(tmp_path)

    with pytest.raises(ValueError, match="fixed-size lists.*use Arrow IPC"):
        write(writer, pa.table({"predictions": column}))

    assert writer.writer is None
    assert not (tmp_path / "rank-0.parquet").exists()


def test_writer_accepts_null_collection_and_null_struct_with_scalar_descendants(tmp_path):
    result = pa.struct([("value", pa.float32())])
    output = pa.table(
        {
            "predictions": pa.array(
                [None, [], [None], [{"value": 0.5}, None]], type=pa.list_(pa.field("element", result))
            )
        }
    )
    writer = Writer(tmp_path)
    write(writer, output)
    writer.close()

    assert pq.read_table(tmp_path / "rank-0.parquet").equals(output)


def test_writer_accepts_sliced_fixed_size_embeddings_in_nonnull_representation_rows(tmp_path):
    vector = pa.list_(pa.field("element", pa.float32()), 4)
    output = pa.table(
        {
            "representations": pa.array(
                [None, [], [{"embedding": [1.0, 2.0, 3.0, 4.0]}]],
                type=pa.list_(pa.field("element", pa.struct([("embedding", vector)]))),
            )
        }
    ).slice(1)
    writer = Writer(tmp_path)
    write(writer, output)
    writer.close()

    assert pq.read_table(tmp_path / "rank-0.parquet").equals(output)


def test_writer_treats_identity_as_an_ordinary_user_column(tmp_path):
    writer = Writer(tmp_path)

    write(writer, pa.table({"identity": ["user-owned"]}))
    writer.close()

    assert pq.read_table(tmp_path / "rank-0.parquet").to_pydict() == {"identity": ["user-owned"]}


def test_writer_requires_predict_step_to_return_arrow_table(tmp_path):
    writer = Writer(tmp_path)

    with pytest.raises(TypeError, match="predict_step.*pyarrow.Table"):
        write(writer, [{"score": 1.0}])

    assert writer.writer is None


def test_writer_renames_the_partial_shard_when_prediction_ends(tmp_path):
    writer = Writer(tmp_path)
    write(writer, pa.table({"score": [0.5]}))

    assert (tmp_path / "rank-0.parquet.partial").exists()
    assert not (tmp_path / "rank-0.parquet").exists()

    writer.on_predict_end(SimpleNamespace(), SimpleNamespace())

    assert list(tmp_path.iterdir()) == [tmp_path / "rank-0.parquet"]
    assert pq.read_table(tmp_path / "rank-0.parquet").to_pydict() == {"score": [0.5]}


def test_writer_discards_the_partial_shard_on_exception_and_accepts_a_rerun(tmp_path):
    writer = Writer(tmp_path)
    write(writer, pa.table({"score": pa.array([1], type=pa.int64())}))

    writer.on_exception(SimpleNamespace(), SimpleNamespace(), RuntimeError("interrupted"))

    assert list(tmp_path.iterdir()) == []
    assert writer.schema is None

    write(writer, pa.table({"score": pa.array([0.5], type=pa.float64())}))
    writer.on_predict_end(SimpleNamespace(), SimpleNamespace())

    restored = pq.read_table(tmp_path / "rank-0.parquet")
    assert restored["score"].type == pa.float64()
    assert restored.to_pydict() == {"score": [0.5]}


def test_writer_buffers_batches_into_whole_row_groups(tmp_path):
    writer = Writer(tmp_path, row_group_size=4)

    for start, pending in ((0, 3), (3, 2), (6, 1)):
        write(writer, pa.table({"score": list(range(start, start + 3))}), batch_idx=start)
        assert writer.rows == pending
    writer.close()

    shard = pq.ParquetFile(tmp_path / "rank-0.parquet")
    assert [shard.metadata.row_group(index).num_rows for index in range(shard.num_row_groups)] == [4, 4, 1]
    assert shard.read().to_pydict() == {"score": list(range(9))}


def test_writer_rejects_a_non_positive_row_group_size(tmp_path):
    with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
        Writer(tmp_path, row_group_size=0)
