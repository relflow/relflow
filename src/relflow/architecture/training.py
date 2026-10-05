"""Data-module loop configuration and built-in training policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

import lightning.pytorch as lit
import torch
from lightning.pytorch.accelerators import (
    Accelerator,
    AcceleratorRegistry,
    CPUAccelerator,
    CUDAAccelerator,
    MPSAccelerator,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.plugins.precision import Precision

from relflow.architecture.checkpoint import RollbackCheckpoint
from relflow.structs.enums import Strata

if TYPE_CHECKING:
    from relflow.architecture.root import Model

__all__ = ["trainer", "policies"]


@runtime_checkable
class ModelData(Protocol):
    """Data modules whose encoding is bound to one model."""

    model: Model


@runtime_checkable
class SplitData(Protocol):
    """Data modules declaring their available split sources before setup."""

    sources: Mapping[Strata, object]


def hardware(options: dict[str, Any]) -> None:
    """Resolve automatic hardware settings before choosing strategy and precision."""
    accelerator = options.get("accelerator", "auto")
    if accelerator in ("auto", "gpu"):
        if MPSAccelerator.is_available():
            accelerator = "mps"
        elif CUDAAccelerator.is_available():
            accelerator = "cuda"
        elif accelerator == "auto":
            accelerator = "cpu"
        else:
            raise ValueError("accelerator='gpu' requires an available CUDA or MPS device")
    engine = accelerator if isinstance(accelerator, Accelerator) else AcceleratorRegistry.get(accelerator)
    devices = options.get("devices", "auto")
    if devices == "auto":
        devices = engine.auto_device_count()
    selected = engine.get_parallel_devices(engine.parse_devices(devices))
    options["accelerator"] = accelerator
    options["devices"] = devices
    if options.get("strategy", "auto") == "auto":
        options["strategy"] = (
            "ddp_find_unused_parameters_true" if len(selected) * options.get("num_nodes", 1) > 1 else "auto"
        )
    if options.get("precision", "auto") != "auto":
        return
    plugins = options.get("plugins") or ()
    if isinstance(plugins, Precision) or (
        isinstance(plugins, Sequence) and any(isinstance(plugin, Precision) for plugin in plugins)
    ):
        options["precision"] = None
        return
    supported = False
    if isinstance(engine, CPUAccelerator):
        supported = torch.amp.autocast_mode.is_autocast_available("cpu")
    elif isinstance(engine, MPSAccelerator):
        supported = torch.amp.autocast_mode.is_autocast_available("mps") and torch.backends.mps.is_macos_or_newer(14, 0)
    elif isinstance(engine, CUDAAccelerator):
        supported = bool(torch.version.hip) or all(
            torch.cuda.get_device_capability(device)[0] >= 8 for device in selected
        )
    options["precision"] = "bf16-mixed" if supported else "32-true"


def trainer(
    model: Model,
    datamodule: lit.LightningDataModule,
    *,
    strata: Strata,
    callbacks: Sequence[lit.Callback] | None,
    options: dict[str, Any],
) -> lit.Trainer:
    """Validate loop ownership and compose callbacks and framework settings."""
    if not isinstance(datamodule, lit.LightningDataModule):
        raise TypeError(f"{strata.value} requires a LightningDataModule, got {type(datamodule).__name__}")
    if isinstance(datamodule, ModelData) and datamodule.model is not model:
        raise ValueError(f"{strata.value} data module belongs to a different model; construct it with model=self")
    if isinstance(datamodule, SplitData):
        if strata not in datamodule.sources:
            raise ValueError(f"{strata.value} requires a {strata.value} split in the data module")
        if strata == Strata.train and Strata.validate not in datamodule.sources:
            options.setdefault("limit_val_batches", 0)
    batch_size = options.pop("batch_size", None)
    if batch_size is not None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"model batch_size must be a positive integer, got {batch_size!r}")
        model.batch_size = batch_size
    configured = list(callbacks or ())
    if not all(isinstance(callback, lit.Callback) for callback in configured):
        raise TypeError("callbacks must contain Lightning Callback instances")
    defaults: dict[str, Any] = {"logger": False, "enable_model_summary": False, "log_every_n_steps": 50}
    if strata == Strata.train and not any(
        options.get(key) is not None for key in ("max_epochs", "max_steps", "max_time")
    ):
        defaults["max_epochs"] = 50
    defaults.update(options)
    hardware(defaults)
    defaults.setdefault("enable_checkpointing", any(isinstance(callback, ModelCheckpoint) for callback in configured))
    return lit.Trainer(callbacks=configured, **defaults)


def policies(
    callbacks: Sequence[lit.Callback] | None,
    *,
    early_stopping: bool | EarlyStopping,
    rollback: bool | RollbackCheckpoint,
    monitor: str,
    mode: str,
    patience: int,
    min_delta: float,
    checkpoint_dir: str | Path | None,
    options: dict[str, Any],
) -> list[lit.Callback]:
    """Compose one automatic stopping/rollback policy with arbitrary callbacks."""
    configured = list(callbacks or ())
    for policy, kind, name in (
        (early_stopping, EarlyStopping, "early_stopping"),
        (rollback, RollbackCheckpoint, "rollback"),
    ):
        if not isinstance(policy, (bool, kind)):
            raise TypeError(f"{name} must be a bool or {kind.__name__}, got {type(policy).__name__}")
        existing = [callback for callback in configured if isinstance(callback, kind)]
        if isinstance(policy, kind):
            if existing and policy not in existing:
                raise ValueError(f"{name} is configured both as an argument and in callbacks; choose one location")
            if policy not in existing:
                configured.append(policy)
        elif policy and not existing and not options.get("fast_dev_run"):
            if kind is EarlyStopping:
                configured.append(EarlyStopping(monitor=monitor, mode=mode, patience=patience, min_delta=min_delta))
            else:
                path = (
                    checkpoint_dir or Path(options.get("default_root_dir") or Path.cwd()) / "checkpoints" / uuid4().hex
                )
                configured.append(
                    RollbackCheckpoint(
                        dirpath=path,
                        monitor=monitor,
                        mode=mode,
                        save_top_k=1,
                        save_last=True,
                        filename="epoch-{epoch:03d}-step-{step}",
                        auto_insert_metric_name=False,
                    )
                )
    if (
        any(isinstance(callback, ModelCheckpoint) for callback in configured)
        and options.get("enable_checkpointing") is False
    ):
        raise ValueError(
            "checkpoint callbacks require enable_checkpointing=True; disable rollback to turn off checkpointing"
        )
    return configured


class Selection(lit.Callback):
    """Require validation before training when automatic policies monitor it."""

    def __init__(self, monitors: tuple[str, ...]) -> None:
        self.monitors = monitors

    def on_train_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        if (not trainer.enable_validation or not sum(trainer.num_val_batches)) and "loss/validate" in self.monitors:
            raise ValueError(
                "automatic early stopping and rollback monitor validation; supply a validate split, "
                "choose a training monitor, or set early_stopping=False and rollback=False"
            )
