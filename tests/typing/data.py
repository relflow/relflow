"""Editor-facing contracts for data ingress, processors and prediction writing."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, assert_type

import polars as pl
import pyarrow as pa
from torch.utils.data import DataLoader, IterableDataset

import relflow as rf
from relflow.data.arrow import Encoded
from relflow.data.datasets import StratumConfig
from relflow.data.processors import Postprocessor, Preprocessor


@rf.preprocess
def increase(frame: pl.DataFrame, *, amount: float = 1.0) -> pl.DataFrame:
    return frame.with_columns((pl.col("amount") + amount).alias("amount"))


@rf.preprocess(scope="dataset")
def partitions(frame: pl.DataFrame) -> Iterator[pl.DataFrame]:
    yield frame


@rf.preprocess()
def discard(frame: pl.DataFrame) -> None:
    return None


@rf.postprocess()
def prediction_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select("predictions")


def processors(frame: pl.DataFrame) -> None:
    assert_type(increase, Preprocessor[pl.DataFrame])
    assert_type(increase.partial(amount=2.0), Preprocessor[pl.DataFrame])
    assert_type(increase(frame), pl.DataFrame)
    assert_type(partitions(frame), Iterator[pl.DataFrame])
    assert_type(discard(frame), None)
    assert_type(prediction_columns, Postprocessor)
    assert_type(prediction_columns(frame), pl.DataFrame)


def modules(model: rf.Model, table: pa.Table, frame: pl.DataFrame) -> None:
    workers: StratumConfig[int] = {"train": 2, "validate": 0}
    data = rf.ArrowDataModule(model, train=table, validate=table, num_workers=workers, preprocessor=increase)
    assert_type(data.model, rf.Model)
    assert_type(data.dataloader("train"), DataLoader[Encoded])
    assert_type(data.dataloader("test", required=False), DataLoader[Encoded] | None)
    assert_type(data.train_dataloader(), DataLoader[Encoded] | None)
    rf.PolarsDataModule(model, train=frame, preprocessor=[increase])
    writer = rf.Writer(Path("predictions"), postprocessor=prediction_columns)
    assert_type(writer.path, Path)
    assert_type(writer.close(), None)


class Records(IterableDataset[Mapping[str, Any]]):
    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        yield {"amount": 1.0}


def adapters(model: rf.Model) -> None:
    rf.CustomDataModule(model, train=Records())
    rf.SyntheticDataModule(model, train=lambda: [{"amount": 1.0}])


class Service(rf.Deployment):
    def label(self) -> str:
        return "service"


def serving() -> None:
    deployment = Service(checkpoint="model.ckpt")
    assert_type(deployment.preprocess(increase), Service)
    assert_type(deployment.postprocess(prediction_columns), Service)
    assert_type(deployment.forge(), Service)
