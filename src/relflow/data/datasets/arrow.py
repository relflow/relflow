"""The canonical Arrow-backed Lightning data module."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, IterableDataset

import relflow
from relflow.data.datasets.base import InterprocessEncodingContext
from relflow.data.iterables import encode
from relflow.data.processors import Preprocessor, PreprocessorInput, arrow, polars
from relflow.distributed import rank, world_size
from relflow.structs.enums import Strata

ArrowUnit: TypeAlias = pa.Table | pa.RecordBatch
ArrowStream: TypeAlias = pa.RecordBatchReader | Iterable[ArrowUnit]
ArrowSource: TypeAlias = ArrowUnit | ds.Dataset | Callable[[], ArrowStream]
Retain: TypeAlias = tuple[str, ...] | Literal["*"]


@dataclass(slots=True)
class Schemas:
    """Persist exact source and processed schemas for one configured split."""

    source: pa.Schema | None = None
    processed: pa.Schema | None = None


def lock(schemas: Schemas, stage: Literal["source", "processed"], actual: pa.Schema, *, context: str) -> None:
    """Record one exact Arrow schema or reject drift from an earlier pass."""

    expected = getattr(schemas, stage)
    if expected is None:
        setattr(schemas, stage, actual)
        return
    if not expected.equals(actual, check_metadata=True):
        raise TypeError(f"{context}: expected {expected}, got {actual}")


def passthrough(value: Any) -> Any:
    """Keep values intact when Lightning's DataLoader has batching disabled."""

    return value


def splits(
    *,
    train: Any = None,
    validate: Any = None,
    test: Any = None,
    predict: Any = None,
) -> dict[Strata, Any]:
    """Collect explicitly named, non-null data splits."""

    values = {
        Strata.train: train,
        Strata.validate: validate,
        Strata.test: test,
        Strata.predict: predict,
    }
    configured = {strata: value for strata, value in values.items() if value is not None}
    if not configured:
        raise ValueError("at least one named data split is required")
    return configured


def expand(value: Any, *, default: Any) -> dict[Strata, Any]:
    """Expand a scalar or named stratum overrides to all strata."""

    return Strata.expand(value, default=default)


def accept(source: Any, *, strata: Strata) -> ArrowSource:
    """Validate the restartable Arrow source boundary."""

    if isinstance(source, (pa.Table, pa.RecordBatch, ds.Dataset)) or callable(source):
        return source
    if isinstance(source, ds.Scanner):
        raise TypeError(f"{strata} source is a configured Scanner; pass its Dataset so RelFlow can plan each scan")
    if isinstance(source, pa.RecordBatchReader):
        raise TypeError(f"{strata} source is a one-shot RecordBatchReader; pass a callable that creates a fresh reader")
    if isinstance(source, Mapping):
        raise TypeError(f"{strata} source is a mapping; use CustomDataModule or SyntheticDataModule")
    if isinstance(source, (str, os.PathLike)):
        raise TypeError(f"{strata} source is a path; build a pyarrow Dataset before using ArrowDataModule")
    raise TypeError(
        f"{strata} source must be a pyarrow Table, pyarrow RecordBatch, Dataset, "
        f"or restartable Arrow factory; got {type(source).__name__}"
    )


def convert(unit: ArrowUnit) -> pa.Table:
    """Normalize one Arrow unit to a table."""

    if isinstance(unit, pa.RecordBatch):
        table = pa.Table.from_batches([unit])
    elif isinstance(unit, pa.Table):
        table = unit
    else:
        raise TypeError(f"Arrow factories must yield Table or RecordBatch; got {type(unit).__name__}")
    if table.num_columns == 0:
        raise ValueError("Arrow observations must contain at least one column")
    return table


def scan(source: ArrowSource, *, schemas: Schemas | None = None) -> Iterator[pa.Table]:
    """Read an in-memory unit or restartable Arrow factory without row conversion."""

    schemas = Schemas() if schemas is None else schemas
    if isinstance(source, (pa.Table, pa.RecordBatch)):
        current = convert(source)
        lock(schemas, "source", current.schema, context="Arrow source schema changed")
        yield current
        return

    stream = source.scanner().to_reader() if isinstance(source, ds.Dataset) else source()
    if isinstance(stream, (pa.Table, pa.RecordBatch)):
        raise TypeError("an Arrow source factory must return a reader or iterable, not one Arrow unit")
    if not isinstance(stream, (pa.RecordBatchReader, Iterable)):
        raise TypeError(
            "an Arrow source factory must return a RecordBatchReader or iterable of Arrow units; "
            f"got {type(stream).__name__}"
        )

    declared = stream.schema if isinstance(stream, pa.RecordBatchReader) else None
    if declared is not None:
        lock(schemas, "source", declared, context="Arrow source schema changed")

    emitted = False
    for unit in stream:
        current = convert(unit)
        lock(schemas, "source", current.schema, context="Arrow source schema changed")
        emitted = True
        yield current
    if emitted:
        return
    if declared is None:
        raise ValueError("an empty Arrow factory must yield an empty Arrow unit carrying its schema")

    empty = pa.Table.from_batches([], schema=declared)
    yield convert(empty)


def merge(batches: Iterable[pa.Table]) -> pa.Table | None:
    """Concatenate Arrow tables with one exact schema."""

    items = list(batches)
    if not items:
        return None

    schema = items[0].schema
    for item in items[1:]:
        if not schema.equals(item.schema, check_metadata=True):
            raise TypeError(f"Arrow batch schema changed: expected {schema}, got {item.schema}")

    return pa.concat_tables(items)


def stage(
    frames: Iterable[pl.DataFrame],
    *,
    preprocessor: Preprocessor,
    strata: Strata,
    schema: Any,
    encoding_context: InterprocessEncodingContext,
) -> Iterator[pl.DataFrame]:
    """Apply one preprocessor to every frame emitted by the preceding stage."""

    for item in frames:
        yield from preprocessor.run(
            item,
            strata=strata,
            schema=schema,
            encoding_context=encoding_context,
        )


def process(
    batches: Iterable[pa.Table],
    *,
    preprocessor: PreprocessorInput = (),
    strata: Strata,
    schema: Any,
    encoding_context: InterprocessEncodingContext,
    schemas: Schemas | None = None,
) -> Iterator[pa.Table]:
    """Apply an ordered preprocessor pipeline and enforce its final schema."""

    pipeline = Preprocessor.normalize(preprocessor)
    schemas = Schemas() if schemas is None else schemas
    if not pipeline:
        for table in batches:
            lock(
                schemas,
                "processed",
                table.schema,
                context=f"processed schema changed in {strata}",
            )
            yield table
        return

    current: Iterable[pl.DataFrame] = (polars(batch, context="preprocessor input") for batch in batches)
    for processor in pipeline:
        if processor.scope == "dataset":
            frames = list(current)
            current = () if not frames else (pl.concat(frames, how="vertical"),)
        current = stage(
            current,
            preprocessor=processor,
            strata=strata,
            schema=schema,
            encoding_context=encoding_context,
        )

    for output in current:
        table = arrow(output, context="preprocessor output")
        lock(
            schemas,
            "processed",
            table.schema,
            context=f"preprocessor schema changed in {strata}",
        )
        yield table


def randomizer(seed: int, *, strata: Strata, epoch: int, operation: str) -> np.random.Generator:
    """Create one operation-isolated deterministic random generator."""

    payload = f"{seed}:{strata}:{epoch}:{operation}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def arrange(table: pa.Table, *, random: np.random.Generator) -> pa.Table:
    """Randomly permute one bounded table."""

    indices = random.permutation(table.num_rows)
    return table.take(pa.array(indices, type=pa.int64()))


def sample(
    batches: Iterable[pa.Table],
    *,
    rate: float,
    random: np.random.Generator,
) -> Iterator[pa.Table]:
    """Select observations with one seeded positional random stream."""

    if rate >= 1.0:
        yield from batches
        return
    for item in batches:
        selected = item.filter(pa.array(random.random(item.num_rows) < rate))
        if selected.num_rows:
            yield selected


def limit(batches: Iterable[pa.Table], *, size: int | None) -> Iterator[pa.Table]:
    """Stop after a fixed number of logical observations."""

    if size is None:
        yield from batches
        return

    remaining = size
    iterator = iter(batches)
    while remaining:
        try:
            item = next(iterator)
        except StopIteration:
            return
        selected = item.slice(0, min(remaining, item.num_rows))
        remaining -= selected.num_rows
        if selected.num_rows:
            yield selected


def shuffle(
    batches: Iterable[pa.Table],
    *,
    rows: int,
    random: np.random.Generator,
) -> Iterator[pa.Table]:
    """Mix Arrow batches in a bounded row buffer."""

    if rows == 1:
        yield from batches
        return

    held: pa.Table | None = None
    capacity = rows * 2
    for item in batches:
        offset = 0
        while offset < item.num_rows:
            available = capacity - (held.num_rows if held is not None else 0)
            selected = item.slice(offset, min(available, item.num_rows - offset))
            held = merge((held, selected)) if held is not None else selected
            offset += selected.num_rows
            if held is not None and held.num_rows == capacity:
                ordered = arrange(held, random=random)
                yield ordered.slice(0, rows)
                held = ordered.slice(rows)

    if held is not None and held.num_rows:
        yield arrange(held, random=random)


def rebatch(batches: Iterable[pa.Table], *, size: int, drop_last: bool) -> Iterator[pa.Table]:
    """Form exact model batches by slicing and concatenating Arrow buffers."""

    held: pa.Table | None = None
    for item in batches:
        held = merge((held, item)) if held is not None else item
        while held is not None and held.num_rows >= size:
            yield held.slice(0, size)
            held = held.slice(size)

    if held is not None and held.num_rows and not drop_last:
        yield held


def distribute(
    batches: Iterable[pa.Table],
    *,
    size: int,
    global_rank: int,
    world_size: int,
    drop_last: bool,
) -> Iterator[pa.Table]:
    """Give every distributed rank disjoint batches with an equal step count."""

    global_size = size * world_size
    for item in rebatch(batches, size=global_size, drop_last=False):
        if item.num_rows < global_size and drop_last:
            return

        usable = item.num_rows - item.num_rows % world_size
        if not usable:
            return
        indices = pa.array(np.arange(global_rank, usable, world_size), type=pa.int64())
        yield item.take(indices)


class ArrowDataset(IterableDataset):
    """Feed one Arrow source through the shared model-input pipeline."""

    def __init__(
        self,
        *,
        source: ArrowSource,
        schema: Any,
        preprocessors: tuple[Preprocessor, ...],
        encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        seed: int,
        shuffle_data: bool,
        sample_rate: float,
        epoch_size: int | None,
        shuffle_rows: int,
        drop_last: bool,
        retain: Retain,
        schemas: Schemas,
        epochs: dict[Strata, int],
    ):
        super().__init__()
        self.source = source
        self.schema = schema
        self.preprocessors = preprocessors
        self.encoding_context = encoding_context
        self.batch_size = batch_size
        self.strata = strata
        self.seed = seed
        self.shuffle_data = shuffle_data
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size
        self.shuffle_rows = shuffle_rows
        self.drop_last = drop_last
        self.retain = retain
        self.schemas = schemas
        self.epochs = epochs

    def __iter__(self):
        epoch = self.epochs[self.strata] if self.strata == Strata.train else 0
        if self.strata == Strata.train:
            self.epochs[self.strata] += 1
        global_rank = rank()
        replicas = world_size()
        for field_context in self.encoding_context.values():
            configure = getattr(field_context, "configure_distributed", None)
            if callable(configure):
                configure(global_rank=global_rank, world_size=replicas)

        scanned: Iterable[pa.Table] = scan(
            self.source,
            schemas=self.schemas,
        )
        batches: Iterable[pa.Table] = process(
            scanned,
            preprocessor=self.preprocessors,
            strata=self.strata,
            schema=self.schema,
            encoding_context=self.encoding_context,
            schemas=self.schemas,
        )
        batches = sample(
            batches,
            rate=self.sample_rate,
            random=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="sample"),
        )

        if isinstance(self.source, (pa.Table, pa.RecordBatch)):
            materialized = merge(batches)
            batches = () if materialized is None else (materialized,)
            if self.shuffle_data and materialized is not None:
                materialized = arrange(
                    materialized,
                    random=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="shuffle"),
                )
                batches = (materialized,)
            batches = limit(batches, size=self.epoch_size)
        else:
            if callable(self.source):
                batches = limit(batches, size=self.epoch_size)
            if self.shuffle_data:
                batches = shuffle(
                    batches,
                    rows=self.shuffle_rows,
                    random=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="shuffle"),
                )
            batches = limit(batches, size=self.epoch_size)

        for item in distribute(
            batches,
            size=self.batch_size,
            global_rank=global_rank,
            world_size=replicas,
            drop_last=self.drop_last,
        ):
            yield encode(
                batch=item,
                schema=self.schema,
                strata=self.strata,
                interprocess_encoding_context=self.encoding_context,
                seed=self.seed,
                epoch=epoch,
                retain=self.retain,
            )


def loader(
    *,
    source: ArrowSource,
    schema: Any,
    preprocessors: tuple[Preprocessor, ...],
    encoding_context: InterprocessEncodingContext,
    batch_size: int,
    strata: Strata,
    seed: int,
    shuffle_data: bool,
    sample_rate: float,
    epoch_size: int | None,
    shuffle_rows: int,
    drop_last: bool,
    retain: Retain,
    pin_memory: bool,
    schemas: Schemas,
    epochs: dict[Strata, int],
) -> DataLoader:
    """Build the sole Lightning DataLoader used by all four data modules."""

    return DataLoader(
        dataset=ArrowDataset(
            source=source,
            schema=schema,
            preprocessors=preprocessors,
            encoding_context=encoding_context,
            batch_size=batch_size,
            strata=strata,
            seed=seed,
            shuffle_data=shuffle_data,
            sample_rate=sample_rate,
            epoch_size=epoch_size,
            shuffle_rows=shuffle_rows,
            drop_last=drop_last,
            retain=retain,
            schemas=schemas,
            epochs=epochs,
        ),
        batch_size=None,
        collate_fn=passthrough,
        num_workers=0,
        persistent_workers=False,
        pin_memory=pin_memory and strata != Strata.predict and torch.cuda.is_available(),
    )


class ArrowDataModule(lit.LightningDataModule):
    """Lightning data module for native in-memory or restartable Arrow sources.

    Tables and record batches stay columnar through preprocessing, selection,
    shuffling, and model rebatching. A callable source must create a fresh
    ``RecordBatchReader`` or Arrow-unit iterable for every iteration.

    Arrow Datasets are scanned afresh for each iteration. Distributed ranks
    replay the deterministic global stream and receive disjoint rows from equal
    global superbatches. Projection pushdown, source-native rank scans,
    coordinated caches, and DataLoader workers remain deferred.
    """

    def __init__(
        self,
        model: relflow.Model,
        *,
        train: ArrowSource | None = None,
        validate: ArrowSource | None = None,
        test: ArrowSource | None = None,
        predict: ArrowSource | None = None,
        preprocessor: PreprocessorInput | Mapping[Strata | str, PreprocessorInput] = (),
        seed: int = 0,
        shuffle: bool | None | Mapping[Strata | str, bool | None] = None,
        sample: float | Mapping[Strata | str, float] = 1.0,
        replacement: bool | Mapping[Strata | str, bool] = False,
        epoch_size: int | None | Mapping[Strata | str, int | None] = None,
        shuffle_rows: int | None | Mapping[Strata | str, int | None] = None,
        drop_last: bool | Mapping[Strata | str, bool] = False,
        num_workers: int | Mapping[Strata | str, int] = 0,
        persistent_workers: bool | Mapping[Strata | str, bool] = False,
        pin_memory: bool | Mapping[Strata | str, bool] = False,
        retain: Retain | Mapping[Strata | str, Retain] = (),
    ):
        super().__init__()
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")

        self.model = model
        self.sources = {
            strata: accept(source, strata=strata)
            for strata, source in splits(train=train, validate=validate, test=test, predict=predict).items()
        }
        self.schemas = {strata: Schemas() for strata in self.sources}
        self.epochs = {strata: 0 for strata in self.sources}
        if isinstance(retain, Mapping):
            normalized = {Strata.normalize(key): value for key, value in retain.items()}
            if set(normalized) != set(self.sources):
                missing = set(self.sources) - set(normalized)
                extra = set(normalized) - set(self.sources)
                details = []
                if missing:
                    details.append("missing " + ", ".join(sorted(str(item) for item in missing)))
                if extra:
                    details.append("extra " + ", ".join(sorted(str(item) for item in extra)))
                raise ValueError("retain mapping must match configured splits exactly: " + "; ".join(details))
            self.retain = normalized
        else:
            self.retain = {strata: retain for strata in self.sources}
        for strata, names in self.retain.items():
            if names != "*" and (
                not isinstance(names, tuple)
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(f"retain for {strata} must be '*', or a tuple of unique non-empty column names")
        configured_preprocessors = expand(preprocessor, default=())
        self.preprocessors = {
            strata: Preprocessor.normalize(processors) for strata, processors in configured_preprocessors.items()
        }
        for strata, processors in self.preprocessors.items():
            if any(processor.scope == "dataset" for processor in processors) and callable(self.sources.get(strata)):
                raise NotImplementedError(
                    "dataset-scoped preprocessing of an Arrow factory requires the deferred coordinated cache"
                )

        configured_shuffle = expand(shuffle, default=None)
        self.shuffle = {
            strata: strata == Strata.train if enabled is None else enabled
            for strata, enabled in configured_shuffle.items()
        }
        if any(not isinstance(value, bool) for value in self.shuffle.values()):
            raise TypeError("shuffle must contain booleans or None")

        configured_sample = expand(sample, default=1.0)
        if any(isinstance(rate, bool) or not isinstance(rate, (int, float)) for rate in configured_sample.values()):
            raise TypeError("sample must contain numeric probabilities")
        self.sample = {strata: float(rate) for strata, rate in configured_sample.items()}
        if any(not 0.0 < rate <= 1.0 for rate in self.sample.values()):
            raise ValueError("sample must be greater than zero and at most one")

        self.replacement = expand(replacement, default=False)
        if any(not isinstance(value, bool) for value in self.replacement.values()):
            raise TypeError("replacement must contain booleans")
        if any(self.replacement.values()):
            raise NotImplementedError("replacement sampling is not implemented")

        self.epoch_size = expand(epoch_size, default=None)
        if any(
            size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1)
            for size in self.epoch_size.values()
        ):
            raise ValueError("epoch_size must contain positive integers or None")

        configured_rows = expand(shuffle_rows, default=None)
        self.shuffle_rows = {
            strata: max(model.batch_size * 32, 4096) if rows is None else rows
            for strata, rows in configured_rows.items()
        }
        if any(not isinstance(rows, int) or isinstance(rows, bool) or rows < 1 for rows in self.shuffle_rows.values()):
            raise ValueError("shuffle_rows must contain positive integers or None")

        self.drop_last = expand(drop_last, default=False)
        if any(not isinstance(value, bool) for value in self.drop_last.values()):
            raise TypeError("drop_last must contain booleans")

        self.num_workers = expand(num_workers, default=0)
        if any(
            not isinstance(workers, int) or isinstance(workers, bool) or workers < 0
            for workers in self.num_workers.values()
        ):
            raise ValueError("num_workers must contain non-negative integers")
        if any(self.num_workers.values()):
            raise NotImplementedError(
                "Arrow DataLoader workers are deferred until deterministic worker ownership and merge are implemented"
            )

        self.persistent_workers = expand(persistent_workers, default=False)
        self.pin_memory = expand(pin_memory, default=False)
        if any(not isinstance(value, bool) for value in (*self.persistent_workers.values(), *self.pin_memory.values())):
            raise TypeError("persistent_workers and pin_memory must contain booleans")
        if any(self.persistent_workers.values()):
            raise ValueError("persistent_workers requires num_workers > 0; Arrow workers are not implemented yet")
        self.seed = seed

    @property
    def schema(self):
        return self.model.schema

    @property
    def batch_size(self) -> int:
        return self.model.batch_size

    @property
    def encoding_context(self) -> InterprocessEncodingContext:
        return self.model.interprocess_encoding_context

    def dataloader(self, strata: Strata | str, required: bool = True) -> DataLoader | None:
        """Create the shared loader for one configured split."""

        normalized = Strata.normalize(strata)
        if normalized not in self.sources:
            if not required:
                return None
            raise ValueError(f"no source configured for strata: {normalized}")
        return loader(
            source=self.sources[normalized],
            schema=self.schema,
            preprocessors=self.preprocessors[normalized],
            encoding_context=self.encoding_context,
            batch_size=self.batch_size,
            strata=normalized,
            seed=self.seed,
            shuffle_data=self.shuffle[normalized],
            sample_rate=self.sample[normalized],
            epoch_size=self.epoch_size[normalized],
            shuffle_rows=self.shuffle_rows[normalized],
            drop_last=self.drop_last[normalized],
            retain=self.retain[normalized],
            pin_memory=self.pin_memory[normalized],
            schemas=self.schemas[normalized],
            epochs=self.epochs,
        )

    def train_dataloader(self) -> DataLoader | None:
        return self.dataloader(Strata.train, required=False)

    def val_dataloader(self) -> DataLoader | None:
        return self.dataloader(Strata.validate, required=False)

    def test_dataloader(self) -> DataLoader | None:
        return self.dataloader(Strata.test, required=False)

    def predict_dataloader(self) -> DataLoader | None:
        return self.dataloader(Strata.predict, required=False)


__all__ = ["ArrowDataModule"]
