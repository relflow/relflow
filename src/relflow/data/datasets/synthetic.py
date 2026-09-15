"""Synthetic mapping ingress for the canonical Arrow data module."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

import pyarrow as pa

import relflow
from relflow.data.datasets.arrow import ArrowDataModule, ArrowStream, Retain
from relflow.data.datasets.base import StratumConfig
from relflow.data.datasets.custom import adapt, schemas
from relflow.data.processors import PreprocessorInput
from relflow.structs.enums import Strata

Generator: TypeAlias = Callable[[], Iterable[Mapping[str, Any]]]


class SyntheticDataModule(ArrowDataModule):
    """Adapt restartable mapping factories to bounded Arrow record batches.

    Each split is a zero-argument callable returning a fresh iterable of records.
    ``ingress_rows`` bounds each Arrow conversion; ``arrow_schema`` can declare
    field types explicitly, including empty or all-null sources.
    """

    def __init__(
        self,
        model: relflow.Model,
        *,
        train: Generator | None = None,
        validate: Generator | None = None,
        test: Generator | None = None,
        predict: Generator | None = None,
        arrow_schema: pa.Schema | Mapping[Strata | str, pa.Schema] | None = None,
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
        if not isinstance(ingress_rows, int) or isinstance(ingress_rows, bool) or ingress_rows < 1:
            raise ValueError("ingress_rows must be a positive integer")

        generators = {
            Strata.train: train,
            Strata.validate: validate,
            Strata.test: test,
            Strata.predict: predict,
        }
        configured = {strata for strata, generator in generators.items() if generator is not None}
        if not configured:
            raise ValueError("at least one named data split is required")
        resolved_schemas = schemas(arrow_schema, configured=configured)

        sources: dict[Strata, Callable[[], ArrowStream]] = {}
        for strata, generator in generators.items():
            if generator is None:
                continue
            if not callable(generator):
                raise TypeError(f"{strata} generator must be callable, got {type(generator).__name__}")
            sources[strata] = adapt(
                generator,
                schema=resolved_schemas[strata],
                rows=ingress_rows,
                name=str(strata),
            )

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


__all__ = ["Generator", "SyntheticDataModule"]
