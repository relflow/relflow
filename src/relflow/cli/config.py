"""Strict, versioned run settings for file-backed CLI workflows."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field

from relflow.architecture.template import Loader
from relflow.data.processors import Postprocessor, Preprocessor

__all__ = ["Run", "read", "processor", "location"]

Positive = Annotated[int, Field(gt=0)]
Nonnegative = Annotated[int, Field(ge=0)]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class LoaderOptions(Options):
    num_workers: Nonnegative = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    prefetch_factor: Positive = 2
    shuffle_rows: Positive | None = None
    epoch_size: Positive | None = None


class Fit(Options):
    batch_size: Positive | None = None
    learning_rate: Annotated[float, Field(gt=0)] | None = None
    weight_decay: Annotated[float, Field(ge=0)] | None = None
    max_epochs: Positive | None = None
    max_steps: Positive | None = None
    accelerator: str = "auto"
    precision: str | int | None = None
    early_stopping: bool = True
    rollback: bool = True
    monitor: str = "loss/validate"
    mode: Literal["min", "max"] = "min"
    patience: Nonnegative = 10
    min_delta: Annotated[float, Field(ge=0)] = 0.0
    limit_train_batches: Positive | Annotated[float, Field(gt=0, le=1)] | None = None
    limit_val_batches: Nonnegative | Annotated[float, Field(ge=0, le=1)] | None = None
    num_sanity_val_steps: Nonnegative = 2


class Predict(Options):
    batch_size: Positive = 256
    accelerator: str = "auto"
    precision: str | int | None = None
    format: Literal["parquet", "ipc"] = "parquet"
    retain: list[str] = Field(default_factory=list)
    row_group_size: Positive = 65536


class Serve(Options):
    host: str | None = None
    port: Annotated[int, Field(ge=1, le=65535)] | None = None
    accelerator: str | None = None
    max_batch_size: Positive | None = None
    batch_timeout: Annotated[float, Field(ge=0)] | None = None
    max_queue_size: Positive | None = None
    retain: list[str] | None = None
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] | None = None


class Run(Options):
    version: Annotated[int, Field(ge=1, le=1)] = 1
    seed: Annotated[int, Field(ge=0, le=4294967295)] = 0
    imports: list[str] = Field(default_factory=list)
    preprocessor: str | None = None
    postprocessor: str | None = None
    data: dict[Literal["train", "validate", "test", "predict"], str | list[str]] = Field(default_factory=dict)
    loader: LoaderOptions = Field(default_factory=LoaderOptions)
    fit: Fit = Field(default_factory=Fit)
    predict: Predict = Field(default_factory=Predict)
    serve: Serve = Field(default_factory=Serve)

    def check_training(self) -> None:
        """Check whether the configured selection policy can observe its monitor."""
        if self.fit.monitor == "loss/validate" and (self.fit.early_stopping or self.fit.rollback):
            if "validate" not in self.data or self.fit.limit_val_batches == 0:
                raise ValueError(
                    "training selection requires validation data; supply --validate or disable early_stopping and rollback"
                )


def location(value: str, base: Path) -> str:
    """Resolve local locations without rewriting filesystem URIs."""
    if not value:
        raise ValueError("data location must be a nonempty path or filesystem URI")
    if "://" in value:
        uri = urlsplit(value)
        if uri.username or uri.password or uri.query:
            raise ValueError(
                "data URI must omit credentials and query strings; configure authentication in the filesystem backend"
            )
    return value if "://" in value else str((base / Path(value).expanduser()).resolve())


def read(pathname: str | Path | None) -> Run:
    if pathname is None:
        return Run()
    path = Path(pathname).resolve()
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
    if not isinstance(payload, dict):
        raise ValueError(f"run configuration {path}: expected one mapping")
    if "version" not in payload:
        raise ValueError(f"run configuration {path}: declare version: 1")
    run = Run.model_validate(payload)
    run.data = {
        split: [location(item, path.parent) for item in value]
        if isinstance(value, list)
        else location(value, path.parent)
        for split, value in run.data.items()
    }
    return run


def processor(reference: str | None, kind: type[Preprocessor] | type[Postprocessor]) -> Any:
    """Resolve an application-owned, already configured processor object."""
    if reference is None:
        return ()
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"processor {reference!r}: expected an installed module:object reference")
    value = importlib.import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    if not isinstance(value, kind):
        raise TypeError(f"processor {reference!r}: expected a configured {kind.__name__} object")
    return value
