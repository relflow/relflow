"""Bounded Python-mapping ingress for the canonical Arrow data module."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
from torch.utils.data import IterableDataset

import relflow
from relflow.data.arrow import Retain, mappings
from relflow.data.datasets.arrow import ArrowDataModule
from relflow.data.datasets.base import StratumConfig
from relflow.data.processors import PreprocessorInput
from relflow.structs.enums import Strata


def schemas(
    value: pa.Schema | Mapping[Strata | str, pa.Schema] | None,
    *,
    configured: set[Strata],
) -> dict[Strata, pa.Schema | None]:
    """Normalize optional explicit schemas for configured mapping sources."""

    if value is None or isinstance(value, pa.Schema):
        return {strata: value for strata in configured}
    if not isinstance(value, Mapping):
        raise TypeError("arrow_schema must be a pyarrow.Schema, named schema mapping, or None")

    normalized: dict[Strata, pa.Schema | None] = {Strata.normalize(key): schema for key, schema in value.items()}
    if set(normalized) != configured:
        missing = sorted(str(item) for item in configured - set(normalized))
        extra = sorted(str(item) for item in set(normalized) - configured)
        raise ValueError(f"arrow_schema keys must exactly match configured splits; missing={missing}, extra={extra}")
    if any(not isinstance(schema, pa.Schema) for schema in normalized.values()):
        raise TypeError("every arrow_schema mapping value must be a pyarrow.Schema")
    return normalized


@dataclass(slots=True)
class Adapter:
    """Convert bounded mapping groups into restartable Arrow record batches.

    The inferred schema is instance state, so every pass in one process agrees;
    a spawned worker infers from the first group of its own fresh pass.
    """

    source: Callable[[], Iterable[Mapping[str, Any]]]
    schema: pa.Schema | None
    rows: int
    name: str
    previous: object | None = field(default=None, init=False, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        # The previous iterator only guards freshness within one process.
        return {"source": self.source, "schema": self.schema, "rows": self.rows, "name": self.name}

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "previous", None)

    def __call__(self) -> Iterator[pa.RecordBatch]:
        stream = self.source()
        if stream is self.previous:
            raise TypeError(f"{self.name} source must return a fresh iterator for every pass")
        self.previous = stream
        iterator = iter(stream)

        emitted = False
        while True:
            values: list[Mapping[str, Any]] = []
            for _ in range(self.rows):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                if not isinstance(item, Mapping):
                    raise TypeError(f"{self.name} source must yield mappings, got {type(item).__name__}")
                if any(not isinstance(key, str) for key in item):
                    raise TypeError(f"{self.name} source mapping keys must be strings")
                values.append(item)

            if not values:
                if not emitted and self.schema is None:
                    raise ValueError(f"{self.name} source is empty; provide arrow_schema")
                if not emitted and self.schema is not None:
                    yield pa.RecordBatch.from_pylist([], schema=self.schema)
                return

            emitted = True
            if self.schema is None:
                record_batch = pa.RecordBatch.from_pylist(mappings(values, context=f"{self.name} source"))
                ambiguous = [field.name for field in record_batch.schema if pa.types.is_null(field.type)]
                if ambiguous:
                    names = ", ".join(repr(field) for field in ambiguous)
                    raise TypeError(f"{self.name} source has all-null field(s) {names}; provide arrow_schema")
                self.schema = record_batch.schema
            else:
                try:
                    record_batch = pa.RecordBatch.from_pylist(
                        mappings(values, schema=self.schema, context=f"{self.name} source"),
                        schema=self.schema,
                    )
                except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as error:
                    raise TypeError(f"{self.name} source does not match arrow_schema: {error}") from error

            yield record_batch
            if len(values) < self.rows:
                return


class CustomDataModule(ArrowDataModule):
    """Adapt restartable mapping ``IterableDataset`` splits to Arrow once per chunk."""

    def __init__(
        self,
        model: relflow.Model | None = None,
        *,
        train: IterableDataset[Mapping[str, Any]] | None = None,
        validate: IterableDataset[Mapping[str, Any]] | None = None,
        test: IterableDataset[Mapping[str, Any]] | None = None,
        predict: IterableDataset[Mapping[str, Any]] | None = None,
        arrow_schema: pa.Schema | Mapping[Strata | str, pa.Schema] | None = None,
        ingress_rows: int = 4096,
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
        if not isinstance(ingress_rows, int) or isinstance(ingress_rows, bool) or ingress_rows < 1:
            raise ValueError("ingress_rows must be a positive integer")

        datasets = {
            Strata.train: train,
            Strata.validate: validate,
            Strata.test: test,
            Strata.predict: predict,
        }
        configured = {strata for strata, dataset in datasets.items() if dataset is not None}
        if not configured:
            raise ValueError("at least one named data split is required")
        resolved_schemas = schemas(arrow_schema, configured=configured)

        sources: dict[Strata, Adapter] = {}
        for strata, dataset in datasets.items():
            if dataset is None:
                continue
            if not isinstance(dataset, IterableDataset):
                raise TypeError(f"{strata} must be a torch IterableDataset, got {type(dataset).__name__}")
            sources[strata] = Adapter(dataset.__iter__, resolved_schemas[strata], ingress_rows, str(strata))

        super().__init__(
            model=model,
            train=sources.get(Strata.train),
            validate=sources.get(Strata.validate),
            test=sources.get(Strata.test),
            predict=sources.get(Strata.predict),
            preprocessor=preprocessor,
            seed=seed,
            shuffle=shuffle,
            sample=sample,
            epoch_size=epoch_size,
            shuffle_rows=shuffle_rows,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
            retain=retain,
        )


__all__ = ["Adapter", "CustomDataModule"]
