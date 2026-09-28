"""DuckDB query ingress for the canonical Arrow data module."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import pyarrow as pa

import relflow
from relflow.data.datasets.arrow import ArrowDataModule, ArrowUnit, Retain, splits
from relflow.data.datasets.base import StratumConfig
from relflow.data.processors import PreprocessorInput
from relflow.structs.enums import Strata

Parameters: TypeAlias = Sequence[Any] | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Query:
    """Serializable query configuration; connections live only inside a scan."""

    database: str
    sql: str
    parameters: tuple[Any, ...] | dict[str, Any] | None
    config: dict[str, Any]
    rows: int
    strata: Strata

    def __call__(self) -> Iterator[ArrowUnit]:
        try:
            import duckdb
        except ModuleNotFoundError as error:
            if error.name != "duckdb":
                raise
            raise ModuleNotFoundError(
                "DuckDBDataModule requires DuckDB; install it with `pip install 'relflow[duckdb]'`.",
                name="duckdb",
            ) from error

        try:
            with duckdb.connect(
                database=self.database, read_only=self.database != ":memory:", config=self.config
            ) as connection:
                statements = connection.extract_statements(self.sql)
                if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                    raise ValueError(f"{self.strata} DuckDB source requires exactly one SELECT query, including a CTE")
                with connection.execute(self.sql, self.parameters).to_arrow_reader(self.rows) as reader:
                    emitted = False
                    for batch in reader:
                        emitted = True
                        yield batch
                    if not emitted:
                        yield pa.Table.from_batches([], schema=reader.schema)
        except duckdb.Error as error:
            error.add_note(f"While reading the {self.strata} DuckDB source from {self.database!r}.")
            raise


class DuckDBDataModule(ArrowDataModule):
    """Stream named SQL queries through the shared Arrow pipeline.

    File databases are opened read-only. The default ``:memory:`` database is
    useful for queries over files, such as ``read_parquet(...)``. Each pass opens
    and closes its own connection and reader, including in spawned workers.

    Splits accept one SELECT statement each. ``parameters`` binds values by split;
    ``ingress_rows`` bounds Arrow result batches independently of model batches.
    ``config`` forwards DuckDB connection settings and defaults to one SQL thread.

    Queries must return the same ordered rows to all consumers during each pass;
    use ORDER BY with a unique key. Workers and distributed ranks execute the
    same query and the Arrow pipeline partitions its rows, without SQL pushdown.
    """

    def __init__(
        self,
        model: relflow.Model,
        *,
        database: str | os.PathLike[str] = ":memory:",
        train: str | None = None,
        validate: str | None = None,
        test: str | None = None,
        predict: str | None = None,
        parameters: Mapping[Strata | str, Parameters] | None = None,
        config: Mapping[str, Any] | None = None,
        ingress_rows: int = 4096,
        preprocessor: StratumConfig[PreprocessorInput] = (),
        seed: int = 0,
        shuffle: StratumConfig[bool | None] = None,
        sample: StratumConfig[float] = 1.0,
        replacement: StratumConfig[bool] = False,
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
        if not isinstance(database, (str, os.PathLike)):
            raise TypeError(
                "DuckDB database must be a path or ':memory:'; live connections and relations are not supported"
            )
        location = os.fspath(database)
        if not isinstance(location, str) or not location.strip():
            raise ValueError("DuckDB database must be a non-empty text path or ':memory:'")
        if location != ":memory:":
            location = str(Path(location).expanduser().absolute())
        if not isinstance(ingress_rows, int) or isinstance(ingress_rows, bool) or ingress_rows < 1:
            raise ValueError("ingress_rows must be a positive integer")
        if config is not None and (
            not isinstance(config, Mapping) or any(not isinstance(key, str) or not key for key in config)
        ):
            raise TypeError("DuckDB config must be a mapping with non-empty string keys")
        settings = {"threads": 1, **({} if config is None else config)}
        queries = splits(train=train, validate=validate, test=test, predict=predict)
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("DuckDB parameters must be a mapping keyed by configured split names")
        bindings: dict[Strata, tuple[Any, ...] | dict[str, Any]] = {}
        for name, values in ({} if parameters is None else parameters).items():
            strata = Strata.normalize(name)
            if strata not in queries:
                raise ValueError(f"DuckDB parameters configure {strata}, but no query is configured for that split")
            if strata in bindings:
                raise ValueError(f"DuckDB parameters configure {strata} more than once")
            if isinstance(values, Mapping):
                if any(not isinstance(key, str) or not key for key in values):
                    raise TypeError(f"{strata} DuckDB named parameters require non-empty string keys")
                bindings[strata] = dict(values)
            elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                bindings[strata] = tuple(values)
            else:
                raise TypeError(f"{strata} DuckDB parameters must be a positional sequence or named mapping")

        sources = {}
        for strata, sql in queries.items():
            if not isinstance(sql, str) or not sql.strip():
                raise TypeError(f"{strata} DuckDB source must be a non-empty SQL string; got {type(sql).__name__}")
            sources[strata] = Query(location, sql, bindings.get(strata), settings.copy(), ingress_rows, strata)

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
            replacement=replacement,
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


__all__ = ["DuckDBDataModule"]
