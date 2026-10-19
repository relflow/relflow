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
    """Persist each written prediction table as one Arrow Parquet shard.

    ``Model.predict_step`` has already converted tensors into the canonical
    Arrow prediction envelope. Configured Polars postprocessors run in order,
    and the first result locks the exact Parquet schema for the rank. Arrow IPC
    is required when null containers contain fixed-size-list descendants.
    """

    def __init__(
        self,
        path: os.PathLike[str] | str,
        postprocessor: PostprocessorInput = (),
    ) -> None:
        super().__init__(write_interval="batch")
        self.path = Path(path)
        self.postprocessors = Postprocessor.normalize(postprocessor)
        self.schema: pa.Schema | None = None
        self.writer: pq.ParquetWriter | None = None

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
        """Write the Arrow table returned by one prediction step."""

        try:
            if not isinstance(output, pa.Table):
                raise TypeError(f"Writer requires predict_step to return a pyarrow.Table, got {type(output).__name__}")

            table = apply(output, self.postprocessors).replace_schema_metadata(output.schema.metadata)
            for name in table.column_names:
                check_parquet(table[name], f"/{name}")

            if self.writer is None:
                self.path.mkdir(parents=True, exist_ok=True)
                self.schema = table.schema
                self.writer = pq.ParquetWriter(
                    where=self.path / f"rank-{trainer.global_rank}.parquet",
                    schema=table.schema,
                )
            elif self.schema is None or not table.schema.equals(self.schema, check_metadata=True):
                raise ValueError(
                    f"prediction batch schema differs from the first batch; expected {self.schema}, got {table.schema}"
                )

            self.writer.write_table(table)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the active Parquet shard, if any."""

        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.close()

    def on_exception(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        exception: BaseException,
    ) -> None:
        self.close()


__all__ = ["Writer"]
