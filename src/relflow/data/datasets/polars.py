"""Polars ingress for the canonical Arrow data module."""

from __future__ import annotations

import polars as pl
import pyarrow as pa

import relflow
from relflow.data.datasets.arrow import ArrowDataModule, Retain
from relflow.data.datasets.base import StratumConfig
from relflow.data.processors import PreprocessorInput


class PolarsDataModule(ArrowDataModule):
    """Convert in-memory Polars frames once, then use the Arrow pipeline.

    ``LazyFrame`` inputs are intentionally rejected. Collect them explicitly so
    users can see where query execution and materialization happen.
    """

    def __init__(
        self,
        model: relflow.Model,
        *,
        train: pl.DataFrame | None = None,
        validate: pl.DataFrame | None = None,
        test: pl.DataFrame | None = None,
        predict: pl.DataFrame | None = None,
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
        frames = {
            "train": train,
            "validate": validate,
            "test": test,
            "predict": predict,
        }
        converted: dict[str, pa.Table | None] = {}
        for name, frame in frames.items():
            if frame is None:
                converted[name] = None
                continue
            if isinstance(frame, pl.LazyFrame):
                raise TypeError(f"{name} must be a collected polars.DataFrame, not LazyFrame")
            if not isinstance(frame, pl.DataFrame):
                raise TypeError(f"{name} must be a polars.DataFrame, got {type(frame).__name__}")
            converted[name] = frame.to_arrow()

        super().__init__(
            model=model,
            train=converted["train"],
            validate=converted["validate"],
            test=converted["test"],
            predict=converted["predict"],
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


__all__ = ["PolarsDataModule"]
