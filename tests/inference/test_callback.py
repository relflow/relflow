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


def test_writer_locks_the_exact_first_batch_schema_and_closes_on_drift(tmp_path):
    writer = Writer(tmp_path)
    write(writer, pa.table({"score": pa.array([1], type=pa.int64())}), batch_idx=0)

    with pytest.raises(ValueError, match="schema differs from the first batch"):
        write(writer, pa.table({"score": pa.array([1.0], type=pa.float64())}), batch_idx=1)

    assert writer.writer is None
    table = pq.read_table(tmp_path / "rank-0.parquet")
    assert table["score"].type == pa.int64()
    assert table["score"].to_pylist() == [1]


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
