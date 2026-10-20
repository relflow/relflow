"""The canonical Arrow-backed Lightning data module."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeAlias, TypeVar, overload

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import relflow
from relflow.data.arrow import Encoded, Retain, retention
from relflow.data.datasets.base import InterprocessEncodingContext, StratumConfig
from relflow.data.iterables import encode
from relflow.data.processors import Preprocessor, PreprocessorInput, arrow, polars
from relflow.data.sources import Source
from relflow.data.sources import source as file_source
from relflow.distributed import broadcast_object, rank, world_size
from relflow.structs.enums import Strata
from relflow.structs.experiment import Schema

Value = TypeVar("Value")
Default = TypeVar("Default")

ArrowUnit: TypeAlias = pa.Table | pa.RecordBatch
ArrowStream: TypeAlias = pa.RecordBatchReader | Iterable[ArrowUnit]


ArrowSource: TypeAlias = ArrowUnit | ds.Dataset | Source | Callable[[], ArrowStream]
ArrowInput: TypeAlias = ArrowSource | str | os.PathLike[str]


@dataclass(slots=True)
class Schemas:
    """Persist exact source and processed schemas for one configured split."""

    source: pa.Schema | None = None
    processed: pa.Schema | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Options:
    """Validated settings for one split, shared by its module and workers."""

    preprocessors: tuple[Preprocessor, ...]
    shuffle: bool
    sample: float
    epoch_size: int | None
    shuffle_rows: int | None
    drop_last: bool
    num_workers: int
    persistent_workers: bool
    pin_memory: bool
    prefetch_factor: int
    retain: Retain


def lock(schemas: Schemas, stage: Literal["source", "processed"], actual: pa.Schema, *, context: str) -> None:
    """Record one exact Arrow schema or reject drift from an earlier pass."""

    expected = getattr(schemas, stage)
    if expected is None:
        setattr(schemas, stage, actual)
        return
    if not expected.equals(actual, check_metadata=True):
        raise TypeError(f"{context}: expected {expected}, got {actual}")


def passthrough(value: Value) -> Value:
    """Keep values intact when Lightning's DataLoader has batching disabled."""

    return value


def splits(
    *,
    train: Value | None = None,
    validate: Value | None = None,
    test: Value | None = None,
    predict: Value | None = None,
) -> dict[Strata, Value]:
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


def expand(value: StratumConfig[Value], *, default: Default) -> dict[Strata, Value | Default]:
    """Expand a scalar or named stratum overrides to all strata."""

    return Strata.expand(value, default=default)


def accept(source: ArrowInput, *, strata: Strata) -> ArrowSource:
    """Validate the restartable Arrow source boundary."""

    if isinstance(source, (str, os.PathLike)):
        source = file_source(source)
    if isinstance(source, Source):
        return replace(source, context=str(strata))
    if isinstance(source, (pa.Table, pa.RecordBatch, ds.Dataset)) or callable(source):
        return source
    if isinstance(source, ds.Scanner):
        raise TypeError(f"{strata} source is a configured Scanner; pass its Dataset so RelFlow can plan each scan")
    if isinstance(source, pa.RecordBatchReader):
        raise TypeError(f"{strata} source is a one-shot RecordBatchReader; pass a callable that creates a fresh reader")
    if isinstance(source, Mapping):
        raise TypeError(f"{strata} source is a mapping; use CustomDataModule or SyntheticDataModule")
    raise TypeError(
        f"{strata} source must be a file path, relflow.source, pyarrow Table, pyarrow RecordBatch, Dataset, "
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


def scan(source: ArrowSource | ArrowStream, *, schemas: Schemas | None = None) -> Iterator[pa.Table]:
    """Read an in-memory unit or restartable Arrow factory without row conversion."""

    schemas = Schemas() if schemas is None else schemas
    if isinstance(source, (pa.Table, pa.RecordBatch)):
        current = convert(source)
        lock(schemas, "source", current.schema, context="Arrow source schema changed")
        yield current
        return

    declared = None
    if isinstance(source, ds.Dataset):
        declared = source.schema
        # RecordBatchReader can block JSON scans through Python filesystems.
        stream = source.scanner().to_batches()
    elif isinstance(source, pa.RecordBatchReader):
        stream = source
    elif callable(source):
        stream = source()
    else:
        stream = source
    if isinstance(stream, (pa.Table, pa.RecordBatch)):
        raise TypeError("an Arrow source factory must return a reader or iterable, not one Arrow unit")
    if not isinstance(stream, (pa.RecordBatchReader, Iterable)):
        raise TypeError(
            "an Arrow source factory must return a RecordBatchReader or iterable of Arrow units; "
            f"got {type(stream).__name__}"
        )

    iterator = iter(stream)
    try:
        declared = stream.schema if isinstance(stream, pa.RecordBatchReader) else declared
        if declared is not None:
            lock(schemas, "source", declared, context="Arrow source schema changed")

        emitted = False
        for unit in iterator:
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
    finally:
        for resource in (stream,) if iterator is stream else (stream, iterator):
            if isinstance(resource, (pa.RecordBatchReader, Generator)):
                resource.close()


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


class ArrowDataset(IterableDataset[Encoded]):
    """Feed one Arrow source through the shared model-input pipeline."""

    def __init__(
        self,
        *,
        source: ArrowSource,
        schema: Schema,
        options: Options,
        encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        seed: int,
        schemas: Schemas,
        epochs: dict[Strata, int],
        distributed_rank: int,
        distributed_world_size: int,
        epoch: Any = None,
        epoch_snapshot: Any = None,
        epoch_barrier: Any = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.schema = schema
        self.options = options
        self.encoding_context = encoding_context
        self.batch_size = batch_size
        self.strata = strata
        self.seed = seed
        self.schemas = schemas
        self.epochs = epochs
        self.distributed_rank = distributed_rank
        self.distributed_world_size = distributed_world_size
        self.epoch = epoch
        self.epoch_snapshot = epoch_snapshot
        self.epoch_barrier = epoch_barrier

    def iteration(self) -> tuple[int, int, int]:
        """Resolve one deterministic epoch and local worker identity."""

        worker = get_worker_info()
        if worker is None:
            epoch = self.epochs[self.strata] if self.strata == Strata.train else 0
            if self.strata == Strata.train:
                self.epochs[self.strata] += 1
            return epoch, 0, 1

        if worker.num_workers != self.options.num_workers:
            raise RuntimeError(f"expected {self.options.num_workers} Arrow workers, got {worker.num_workers}")

        if self.strata == Strata.train:
            if self.epoch is None or self.epoch_snapshot is None or self.epoch_barrier is None:
                raise RuntimeError("training Arrow workers require shared epoch state")
            self.epoch_barrier.wait()
            if worker.id == 0:
                with self.epoch.get_lock():
                    self.epoch_snapshot.value = self.epoch.value
                    self.epoch.value += 1
            self.epoch_barrier.wait()
            epoch = self.epoch_snapshot.value
        else:
            epoch = 0

        return epoch, worker.id, worker.num_workers

    def __iter__(self) -> Iterator[Encoded]:
        epoch, worker, workers = self.iteration()
        scanned: Iterable[pa.Table] = scan(self.source, schemas=self.schemas)
        try:
            batches: Iterable[pa.Table] = process(
                scanned,
                preprocessor=self.options.preprocessors,
                strata=self.strata,
                schema=self.schema,
                encoding_context=self.encoding_context,
                schemas=self.schemas,
            )
            batches = sample(
                batches,
                rate=self.options.sample,
                random=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="sample"),
            )

            if isinstance(self.source, (pa.Table, pa.RecordBatch)):
                materialized = merge(batches)
                batches = () if materialized is None else (materialized,)
                if self.options.shuffle and materialized is not None:
                    materialized = arrange(
                        materialized,
                        random=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="shuffle"),
                    )
                    batches = (materialized,)
                batches = limit(batches, size=self.options.epoch_size)
            else:
                if callable(self.source):
                    batches = limit(batches, size=self.options.epoch_size)
                if self.options.shuffle:
                    assert self.options.shuffle_rows is not None
                    batches = shuffle(
                        batches,
                        rows=self.options.shuffle_rows,
                        random=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="shuffle"),
                    )
                batches = limit(batches, size=self.options.epoch_size)

            distributed = distribute(
                batches,
                size=self.batch_size,
                global_rank=self.distributed_rank,
                world_size=self.distributed_world_size,
                drop_last=self.options.drop_last,
            )
            # Only distributed ranks require equal steps. Local workers own complete
            # rank batches, including a partial tail that would otherwise be lost.
            assigned = (batch for index, batch in enumerate(distributed) if index % workers == worker)
            yield from self.encode(assigned, epoch=epoch)
        finally:
            # Downstream failures and epoch limits must release suspended scans,
            # even when callers keep the loader or exception traceback alive.
            if isinstance(scanned, Generator):
                scanned.close()

    def encode(self, batches: Iterable[pa.Table], *, epoch: int) -> Iterator[Encoded]:
        """Encode one consumer's final model batches."""

        for item in batches:
            yield encode(
                batch=item,
                schema=self.schema,
                strata=self.strata,
                interprocess_encoding_context=self.encoding_context,
                seed=self.seed,
                epoch=epoch,
                retain=self.options.retain,
            )


def loader(
    *,
    source: ArrowSource,
    schema: Schema,
    options: Options,
    encoding_context: InterprocessEncodingContext,
    batch_size: int,
    strata: Strata,
    seed: int,
    schemas: Schemas,
    epochs: dict[Strata, int],
    multiprocessing_context: Any,
    epoch: Any = None,
) -> DataLoader[Encoded]:
    """Build the shared Lightning DataLoader used by every data module."""

    distributed_rank = rank()
    distributed_world_size = world_size()
    if isinstance(source, Source):
        # Rank zero freezes selection once; failures are broadcast before any
        # rank starts workers. Fork users plan in a disposable spawned process.
        manifest = None
        if distributed_rank == 0:
            try:
                manifest = source.manifest
                if manifest is None:
                    if options.num_workers and multiprocessing_context.get_start_method() == "fork":
                        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as planner:
                            manifest = planner.submit(source.discover).result()
                    else:
                        manifest = source.discover()
            except Exception as error:
                manifest = error
        manifest = broadcast_object(manifest)
        if isinstance(manifest, Exception):
            raise manifest
        source.manifest = manifest
    epoch_snapshot = multiprocessing_context.Value("q", 0) if options.num_workers else None
    epoch_barrier = multiprocessing_context.Barrier(options.num_workers) if options.num_workers else None

    return DataLoader(
        dataset=ArrowDataset(
            source=source,
            schema=schema,
            options=options,
            encoding_context=encoding_context,
            batch_size=batch_size,
            strata=strata,
            seed=seed,
            schemas=schemas,
            epochs=epochs,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
            epoch=epoch,
            epoch_snapshot=epoch_snapshot,
            epoch_barrier=epoch_barrier,
        ),
        batch_size=None,
        collate_fn=passthrough,
        num_workers=options.num_workers,
        persistent_workers=options.persistent_workers,
        pin_memory=options.pin_memory and strata != Strata.predict and torch.cuda.is_available(),
        multiprocessing_context=multiprocessing_context if options.num_workers else None,
        prefetch_factor=options.prefetch_factor if options.num_workers else None,
    )


class ArrowDataModule(lit.LightningDataModule):
    """Lightning data module for native in-memory or restartable Arrow sources.

    Tables and record batches stay columnar through preprocessing, selection,
    shuffling, and model rebatching. A callable source must create a fresh
    ``RecordBatchReader`` or Arrow-unit iterable for every iteration.

    Arrow Datasets are scanned afresh for each iteration. Distributed ranks and
    data workers replay the same global stream. Ranks receive disjoint rows from
    equal global superbatches; local workers take turns processing complete rank
    batches. File sources freeze their manifest before workers start, then open
    independent Arrow readers in each consuming process.

    Pass at least one of ``train``, ``validate``, ``test`` or ``predict``.
    Configuration accepts one value for every split or a mapping keyed by
    ``"train"``, ``"validate"``, ``"test"`` and ``"predict"``. Shuffling defaults
    to training only; ``sample`` is a probability, ``epoch_size`` caps rows, and
    ``shuffle_rows`` bounds the streaming shuffle buffer. ``retain`` preserves
    processed columns for prediction output and requires exact split keys when
    given as a mapping. Data loaders yield :class:`~relflow.data.arrow.Encoded`
    batches with Arrow source metadata and a TensorDict payload.

    ``model`` may be omitted until a model loop or Lightning Trainer binds this
    module. Direct data-loader access requires a bound model.
    """

    def __init__(
        self,
        model: relflow.Model | None = None,
        *,
        train: ArrowInput | None = None,
        validate: ArrowInput | None = None,
        test: ArrowInput | None = None,
        predict: ArrowInput | None = None,
        preprocessor: StratumConfig[PreprocessorInput] = (),
        seed: int = 0,
        shuffle: StratumConfig[bool | None] = None,
        sample: StratumConfig[float] = 1.0,
        epoch_size: StratumConfig[int | None] = None,
        shuffle_rows: StratumConfig[int | None] = None,
        drop_last: StratumConfig[bool] = False,
        num_workers: StratumConfig[int] = 0,
        persistent_workers: StratumConfig[bool] = False,
        pin_memory: StratumConfig[bool] = False,
        prefetch_factor: StratumConfig[int] = 2,
        multiprocessing_context: str | None = None,
        retain: StratumConfig[Retain] = (),
    ) -> None:
        super().__init__()
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")

        self.model: relflow.Model | None = None
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
            retained = normalized
        else:
            retained = {strata: retain for strata in self.sources}
        retained = {strata: retention(names, context=f"retain for {strata}") for strata, names in retained.items()}
        configured_preprocessors = expand(preprocessor, default=())
        preprocessors = {
            strata: Preprocessor.normalize(processors) for strata, processors in configured_preprocessors.items()
        }
        for strata, processors in preprocessors.items():
            if any(processor.scope == "dataset" for processor in processors) and callable(self.sources.get(strata)):
                raise NotImplementedError(
                    "dataset-scoped preprocessing of an Arrow factory requires the deferred coordinated cache"
                )

        configured_shuffle = expand(shuffle, default=None)
        shuffled = {
            strata: strata == Strata.train if enabled is None else enabled
            for strata, enabled in configured_shuffle.items()
        }
        if any(not isinstance(value, bool) for value in shuffled.values()):
            raise TypeError("shuffle must contain booleans or None")

        configured_sample = expand(sample, default=1.0)
        if any(isinstance(rate, bool) or not isinstance(rate, (int, float)) for rate in configured_sample.values()):
            raise TypeError("sample must contain numeric probabilities")
        rates = {strata: float(rate) for strata, rate in configured_sample.items()}
        if any(not 0.0 < rate <= 1.0 for rate in rates.values()):
            raise ValueError("sample must be greater than zero and at most one")

        sizes = expand(epoch_size, default=None)
        if any(
            size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1)
            for size in sizes.values()
        ):
            raise ValueError("epoch_size must contain positive integers or None")

        configured_rows = expand(shuffle_rows, default=None)
        if any(
            rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows < 1)
            for rows in configured_rows.values()
        ):
            raise ValueError("shuffle_rows must contain positive integers or None")

        dropped = expand(drop_last, default=False)
        if any(not isinstance(value, bool) for value in dropped.values()):
            raise TypeError("drop_last must contain booleans")

        workers = expand(num_workers, default=0)
        if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in workers.values()):
            raise ValueError("num_workers must contain non-negative integers")
        persistent = expand(persistent_workers, default=False)
        pinned = expand(pin_memory, default=False)
        if any(not isinstance(value, bool) for value in (*persistent.values(), *pinned.values())):
            raise TypeError("persistent_workers and pin_memory must contain booleans")
        invalid_persistent = [strata for strata, enabled in persistent.items() if enabled and not workers[strata]]
        if invalid_persistent:
            names = ", ".join(map(str, invalid_persistent))
            raise ValueError(f"persistent_workers requires num_workers > 0 for: {names}")

        prefetch = expand(prefetch_factor, default=2)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in prefetch.values()):
            raise ValueError("prefetch_factor must contain positive integers")

        if multiprocessing_context is not None and not isinstance(multiprocessing_context, str):
            raise TypeError("multiprocessing_context must be a start-method string or None")
        self.options = {
            strata: Options(
                preprocessors=preprocessors[strata],
                shuffle=shuffled[strata],
                sample=rates[strata],
                epoch_size=sizes[strata],
                shuffle_rows=configured_rows[strata],
                drop_last=dropped[strata],
                num_workers=workers[strata],
                persistent_workers=persistent[strata],
                pin_memory=pinned[strata],
                prefetch_factor=prefetch[strata],
                retain=retained.get(strata, ()),
            )
            for strata in Strata
        }
        self.worker_context = None
        self.worker_epochs: dict[Strata, Any] = {}
        if any(options.num_workers for options in self.options.values()):
            method = "spawn" if multiprocessing_context is None else multiprocessing_context
            try:
                self.worker_context = mp.get_context(method)
            except ValueError as error:
                available = ", ".join(mp.get_all_start_methods())
                raise ValueError(f"unsupported multiprocessing_context {method!r}; choose from: {available}") from error
            self.worker_epochs = {
                strata: self.worker_context.Value("q", 0)
                for strata, options in self.options.items()
                if options.num_workers
            }
        self.seed = seed
        if model is not None:
            self.bind(model)

    def bind(self, model: relflow.Model) -> None:
        """Bind once to the model that owns encoding, schema, and batch size."""

        if not isinstance(model, relflow.Model):
            raise TypeError(f"data module requires a relflow.Model, got {type(model).__name__}")
        if self.model is model:
            return
        if self.model is not None:
            raise ValueError("data module belongs to a different model; construct a new data module for this model")
        self.model = model
        self.options = {
            strata: replace(options, shuffle_rows=max(model.batch_size * 32, 4096))
            if options.shuffle_rows is None
            else options
            for strata, options in self.options.items()
        }

    @property
    def bound_model(self) -> relflow.Model:
        """Require ownership before accessing model-dependent data settings."""

        if self.model is None:
            raise RuntimeError(
                "data module is not bound to a model; pass model=, call bind(model), or pass it to "
                "model.fit(), model.validate(), model.test(), or model.predict()"
            )
        return self.model

    def setup(self, stage: str | None = None) -> None:
        """Bind native Lightning loops to their attached RelFlow model."""

        if self.trainer is not None:
            model = self.trainer.lightning_module
            if not isinstance(model, relflow.Model):
                raise TypeError(f"data module requires a relflow.Model, got {type(model).__name__}")
            self.bind(model)

    @property
    def schema(self) -> Schema:
        """The schema of the model whose batches this module prepares."""
        return self.bound_model.schema

    @property
    def batch_size(self) -> int:
        return self.bound_model.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        """Let Lightning's batch-size finder update the model-owned batch size."""
        self.bound_model.batch_size = value

    @property
    def encoding_context(self) -> InterprocessEncodingContext:
        return self.bound_model.interprocess_encoding_context

    @overload
    def dataloader(self, strata: Strata | str, required: Literal[True] = True) -> DataLoader[Encoded]: ...

    @overload
    def dataloader(self, strata: Strata | str, required: Literal[False]) -> DataLoader[Encoded] | None: ...

    @overload
    def dataloader(self, strata: Strata | str, required: bool) -> DataLoader[Encoded] | None: ...

    def dataloader(self, strata: Strata | str, required: bool = True) -> DataLoader[Encoded] | None:
        """Create one split's loader; missing splits raise unless ``required=False``."""

        normalized = Strata.normalize(strata)
        if normalized not in self.sources:
            if not required:
                return None
            raise ValueError(f"no source configured for strata: {normalized}")
        options = self.options[normalized]
        encoding_context = self.encoding_context
        if options.num_workers:
            for context in encoding_context.values():
                share = getattr(context, "share", None)
                if callable(share):
                    share()

        return loader(
            source=self.sources[normalized],
            schema=self.schema,
            options=options,
            encoding_context=encoding_context,
            batch_size=self.batch_size,
            strata=normalized,
            seed=self.seed,
            schemas=self.schemas[normalized],
            epochs=self.epochs,
            multiprocessing_context=self.worker_context,
            epoch=self.worker_epochs.get(normalized),
        )

    def train_dataloader(self) -> DataLoader[Encoded] | None:
        return self.dataloader(Strata.train, required=False)

    def val_dataloader(self) -> DataLoader[Encoded] | None:
        return self.dataloader(Strata.validate, required=False)

    def test_dataloader(self) -> DataLoader[Encoded] | None:
        return self.dataloader(Strata.test, required=False)

    def predict_dataloader(self) -> DataLoader[Encoded] | None:
        return self.dataloader(Strata.predict, required=False)


__all__ = ["ArrowDataModule", "ArrowInput", "ArrowSource", "ArrowStream", "ArrowUnit", "Retain"]
