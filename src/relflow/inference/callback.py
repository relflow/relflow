"""Arrow-native prediction persistence."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lightning.pytorch import callbacks

from relflow.data.processors import Postprocessor, PostprocessorInput, apply


def check_parquet(array: pa.Array | pa.ChunkedArray, address: str, valid: np.ndarray | None = None) -> None:
    """Reject null fixed-width lists that Arrow's Parquet reader cannot restore."""

    if isinstance(array, pa.ChunkedArray):
        for chunk in array.chunks:
            check_parquet(chunk, address)
        return

    if not isinstance(
        array,
        (pa.FixedSizeListArray, pa.StructArray, pa.ListArray, pa.LargeListArray, pa.MapArray, pa.ExtensionArray),
    ):
        return
    current = array.is_valid().to_numpy(zero_copy_only=False)
    if valid is not None:
        current = current & valid
    if isinstance(array, pa.FixedSizeListArray):
        if not current.all():
            raise ValueError(
                f"Parquet cannot round-trip fixed-size lists at {address!r} beneath a null container "
                "or null list value; use Arrow IPC for this output"
            )
        width = array.type.list_size
        children = array.values.slice(array.offset * width, len(array) * width)
        check_parquet(children, f"{address}[]", np.repeat(current, width))
    elif isinstance(array, pa.StructArray):
        for field in array.type:
            check_parquet(array.field(field.name), f"{address}/{field.name}", current)
    elif isinstance(array, (pa.ListArray, pa.LargeListArray, pa.MapArray)):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        start = int(offsets[0])
        count = int(offsets[-1]) - start
        children_valid = np.repeat(current, np.diff(offsets))
        if isinstance(array, pa.MapArray):
            check_parquet(array.keys.slice(start, count), f"{address}/keys[]", children_valid)
            check_parquet(array.items.slice(start, count), f"{address}/values[]", children_valid)
        else:
            check_parquet(array.values.slice(start, count), f"{address}[]", children_valid)
    elif isinstance(array, pa.ExtensionArray):
        check_parquet(array.storage, address, current)


class Writer(callbacks.BasePredictionWriter):
    """Persist each rank's written prediction tables as one Arrow Parquet shard.

    ``Model.predict_step`` has already converted tensors into the canonical
    Arrow prediction envelope. Configured Polars postprocessors run in order,
    and the first result locks the exact Parquet schema for the rank. Batches
    accumulate until ``row_group_size`` rows are pending and are then written
    as whole row groups, so downstream readers see few large groups instead of
    one per prediction batch. The shard grows as ``rank-N.parquet.partial`` and
    takes its final name only when prediction ends; an interrupted or failed
    run removes the partial file instead of leaving a complete-looking shard.
    Arrow IPC is required when null containers contain fixed-size-list
    descendants.
    """

    def __init__(
        self,
        path: os.PathLike[str] | str,
        postprocessor: PostprocessorInput = (),
        row_group_size: int = 65536,
    ) -> None:
        super().__init__(write_interval="batch")
        if isinstance(row_group_size, bool) or not isinstance(row_group_size, int) or row_group_size <= 0:
            raise ValueError(f"Writer row_group_size must be a positive integer, got {row_group_size!r}")
        self.path = Path(path)
        self.postprocessors = Postprocessor.normalize(postprocessor)
        self.row_group_size = row_group_size
        self.schema: pa.Schema | None = None
        self.writer: pq.ParquetWriter | None = None
        self.shard: Path | None = None
        self.pending: list[pa.Table] = []
        self.rows = 0

    # Keep the public ``output`` keyword; Lightning calls this hook positionally.
    def write_on_batch_end(  # pyrefly: ignore[bad-override-param-name]
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        output: pa.Table,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Stage the Arrow table returned by one prediction step."""

        try:
            if not isinstance(output, pa.Table):
                raise TypeError(f"Writer requires predict_step to return a pyarrow.Table, got {type(output).__name__}")

            table = apply(output, self.postprocessors).replace_schema_metadata(output.schema.metadata)
            for name in table.column_names:
                check_parquet(table[name], f"/{name}")

            if self.writer is None:
                self.path.mkdir(parents=True, exist_ok=True)
                self.schema = table.schema
                self.shard = self.path / f"rank-{trainer.global_rank}.parquet"
                self.writer = pq.ParquetWriter(where=self.partial(self.shard), schema=table.schema)
            elif self.schema is None or not table.schema.equals(self.schema, check_metadata=True):
                raise ValueError(
                    f"prediction batch schema differs from the first batch; expected {self.schema}, got {table.schema}"
                )

            self.pending.append(table)
            self.rows += table.num_rows
            if self.rows >= self.row_group_size:
                self.flush(final=False)
        except BaseException:
            self.discard()
            raise

    @staticmethod
    def partial(shard: Path) -> Path:
        """Name the in-progress file beside its final shard."""

        return shard.with_name(f"{shard.name}.partial")

    def flush(self, *, final: bool) -> None:
        """Write whole row groups from the pending tables; ``final`` also writes the remainder."""

        assert self.writer is not None
        if not self.pending:
            return
        table = pa.concat_tables(self.pending)
        rows = table.num_rows if final else table.num_rows - table.num_rows % self.row_group_size
        if rows:
            self.writer.write_table(table.slice(0, rows), row_group_size=self.row_group_size)
        remainder = table.slice(rows)
        self.pending = [remainder] if remainder.num_rows else []
        self.rows = remainder.num_rows

    def close(self) -> None:
        """Flush pending rows, finalize the shard under its final name, and unlock the schema."""

        if self.writer is None:
            return
        try:
            self.flush(final=True)
            self.writer.close()
            assert self.shard is not None
            self.partial(self.shard).replace(self.shard)
        except BaseException:
            self.discard()
            raise
        self.reset()

    def discard(self) -> None:
        """Drop the partial shard so a failed run leaves no complete-looking output."""

        if self.writer is not None:
            try:
                self.writer.close()
            finally:
                assert self.shard is not None
                self.partial(self.shard).unlink(missing_ok=True)
        self.reset()

    def reset(self) -> None:
        self.writer = None
        self.schema = None
        self.shard = None
        self.pending = []
        self.rows = 0

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.close()

    def on_exception(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        exception: BaseException,
    ) -> None:
        self.discard()


__all__ = ["Writer"]
