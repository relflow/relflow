"""Static contracts for the public model facade; checked without training."""

from pathlib import Path
from typing import Any, assert_type

import pyarrow as pa
import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.runtime import Output
from relflow.structs.packages import Prediction


class Classifier(rf.Model):
    compact = rf.presets.factory(rf.presets.SM)

    def label(self) -> str:
        return "classifier"


def model_api(table: pa.Table, batch: TensorDict, loss: torch.Tensor) -> None:
    model = Classifier(
        d_model=32,
        n_layers=1,
        n_heads=4,
        amount=rf.Number,
        label=rf.Category(mask=True),
        optimizer=rf.adamw(1e-3),
    )
    assert_type(model.compile(encoders=True, pools=False), Classifier)
    assert_type(model.save("model.ckpt"), str)
    assert_type(model.save(Path("model.ckpt")), Path)
    assert_type(Classifier.load("model.ckpt"), Classifier)
    assert_type(Classifier.xs(amount=rf.Number, label=rf.Category(mask=True)), Classifier)
    assert_type(Classifier.sm(amount=rf.Number, n_heads=2), Classifier)
    assert_type(Classifier.md(amount=rf.Number, reduction=None), Classifier)
    assert_type(Classifier.lg(amount=rf.Number, attention=None), Classifier)
    assert_type(Classifier.xl(amount=rf.Number, dropout=None, optimizer=rf.adamw(1e-3)), Classifier)
    assert_type(rf.Model.md(fields={"size": rf.Number}, scheduler=schedule), rf.Model)
    assert_type(Classifier.compact(amount=rf.Number, n_layers=2), Classifier)
    assert_type(model.track(("train", "loss"), loss), torch.Tensor)
    assert_type(model.encode(table), TensorDict)
    assert_type(rf.Enum.vocabulary(model, rf.Address("label")), tuple[bool | int | float | str | bytes, ...])
    assert_type(
        rf.Enum.vocabulary(model.interprocess_encoding_context, "/label"),
        tuple[bool | int | float | str | bytes, ...],
    )
    assert_type(rf.Enum.counts(model, rf.Address("label")), dict[bool | int | float | str | bytes, int])
    assert_type(rf.Quantile.normalization(model, rf.Address("amount")), dict[str, int | float | None])
    assert_type(
        rf.Quantile.normalization(model.interprocess_encoding_context, "/amount"), dict[str, int | float | None]
    )
    assert_type(model(batch, strata="train"), list[Prediction])
    assert_type(model.training_step(batch, 0), Output | None)
    assert_type(model.validation_step(batch, 0), Output)
    assert_type(model.predict(table), pa.Table)
    assert_type(model.predict([{"amount": 1.0}]), pa.Table)
    assert_type(model.write([], source=table), pa.Table)
    rf.Model(model.schema, optimizer=rf.adamw(1e-3))
    rf.Model(schema=model.schema, optimizer=rf.adamw(1e-3))
    rf.Model(d_model=32, n_heads=4, n_layers=1, fields={"name": rf.Number, "batch_size": rf.Boolean})
    rf.Model(d_model=32, n_heads=4, n_layers=1, name=rf.Category())
    model.extend(rf.where("address") == "/", risk_score=rf.Number)
    model.extend(fields={"fields": rf.Number})


def schedule(model: rf.Model, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)


def schedulers(schema: rf.Schema, optimizer: torch.optim.Optimizer) -> None:
    rf.Model(schema, optimizer=optimizer, scheduler=schedule)
    rf.Model(schema, optimizer=optimizer, scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1))
    rf.Model(
        schema,
        optimizer=optimizer,
        scheduler={"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1), "interval": "step"},
    )


def wrapper(field: rf.TreeFieldInput, mask: rf.MaskInput, attention: rf.AttentionInput) -> rf.Model:
    return rf.Model(d_model=32, n_layers=1, n_heads=4, field=field, mask=mask, attention=attention)


class CustomScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


def custom_scheduler(schema: rf.Schema, optimizer: torch.optim.Optimizer) -> None:
    scheduler = CustomScheduler(optimizer)
    rf.Model(schema, optimizer=optimizer, scheduler=scheduler)
    rf.Model(schema, optimizer=optimizer, scheduler={"scheduler": scheduler, "interval": "step"})


def lightning_scheduler(schema: rf.Schema, scheduler: LRSchedulerConfigType) -> rf.Model:
    return rf.Model(schema, scheduler=scheduler)
