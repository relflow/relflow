"""Public data modules over one Arrow-backed loader."""

from __future__ import annotations

from relflow.data.datasets.arrow import ArrowDataModule, ArrowInput, ArrowSource, ArrowStream, ArrowUnit, Retain
from relflow.data.datasets.base import StratumConfig
from relflow.data.datasets.custom import CustomDataModule
from relflow.data.datasets.duckdb import DuckDBDataModule
from relflow.data.datasets.polars import PolarsDataModule
from relflow.data.datasets.synthetic import SyntheticDataModule

__all__ = [
    "ArrowDataModule",
    "ArrowInput",
    "ArrowSource",
    "ArrowStream",
    "ArrowUnit",
    "CustomDataModule",
    "DuckDBDataModule",
    "PolarsDataModule",
    "Retain",
    "StratumConfig",
    "SyntheticDataModule",
]
