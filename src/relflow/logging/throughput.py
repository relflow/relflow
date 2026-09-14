from __future__ import annotations

import datetime
from collections import defaultdict
from functools import partialmethod
from typing import TYPE_CHECKING, cast

import torch
from lightning import Callback, LightningModule, Trainer

from relflow.structs.enums import Metric, Strata

if TYPE_CHECKING:
    from relflow.architecture.root import Model


class ThroughputLogger(Callback):
    """Measure completed batches over a RelFlow loop's elapsed wall time."""

    def __init__(self) -> None:
        super().__init__()

        self.timestamp: dict[Strata, datetime.datetime] = defaultdict(lambda: datetime.datetime.now())
        self.batches: dict[Strata, int] = defaultdict(int)
        self.throughput: dict[Strata, float] = {}

    def start(self, trainer: Trainer, pl_module: LightningModule, strata: Strata) -> None:
        self.timestamp[strata] = datetime.datetime.now()
        self.batches[strata] = 0

    def count(
        self, trainer: Trainer, pl_module: LightningModule, *args: object, strata: Strata, **kwargs: object
    ) -> None:
        self.batches[strata] += 1

    def end(self, trainer: Trainer, pl_module: LightningModule, strata: Strata) -> None:
        # RelFlow installs this callback on models exposing batch_size and track.
        module = cast("Model", pl_module)
        now = datetime.datetime.now()
        then = self.timestamp[strata]
        elapsed = (now - then).total_seconds()
        observations = self.batches[strata] * module.batch_size
        throughput = observations / elapsed if elapsed > 0.0 else 0.0
        self.throughput[strata] = throughput

        # Lightning does not register a result collection for prediction hooks,
        # so LightningModule.log()/Model.track() raises at predict epoch end.
        # Send the scalar straight to an attached logger and retain it on this
        # callback for runtimes without a logger.
        if strata == Strata.predict:
            logger = getattr(trainer, "logger", None)
            if logger is not None and getattr(trainer, "is_global_zero", True):
                logger.log_metrics(
                    {f"{Metric.throughput.value}/{strata.value}": throughput},
                    step=getattr(trainer, "global_step", None),
                )
            return

        device = getattr(pl_module, "device", None)

        module.track(
            (Metric.throughput, strata),
            value=torch.tensor(throughput, device=device) if device is not None else torch.tensor(throughput),
        )

    # Preserve Lightning hook signatures for the runtime partialmethod descriptors.
    if TYPE_CHECKING:
        on_train_epoch_start = Callback.on_train_epoch_start
        on_validation_epoch_start = Callback.on_validation_epoch_start
        on_test_epoch_start = Callback.on_test_epoch_start
        on_predict_epoch_start = Callback.on_predict_epoch_start
        on_train_batch_end = Callback.on_train_batch_end
        on_validation_batch_end = Callback.on_validation_batch_end
        on_test_batch_end = Callback.on_test_batch_end
        on_predict_batch_end = Callback.on_predict_batch_end
        on_train_epoch_end = Callback.on_train_epoch_end
        on_validation_epoch_end = Callback.on_validation_epoch_end
        on_test_epoch_end = Callback.on_test_epoch_end
        on_predict_epoch_end = Callback.on_predict_epoch_end
    else:
        on_train_epoch_start = partialmethod(start, strata=Strata.train)
        on_validation_epoch_start = partialmethod(start, strata=Strata.validate)
        on_test_epoch_start = partialmethod(start, strata=Strata.test)
        on_predict_epoch_start = partialmethod(start, strata=Strata.predict)

        on_train_batch_end = partialmethod(count, strata=Strata.train)
        on_validation_batch_end = partialmethod(count, strata=Strata.validate)
        on_test_batch_end = partialmethod(count, strata=Strata.test)
        on_predict_batch_end = partialmethod(count, strata=Strata.predict)

        on_train_epoch_end = partialmethod(end, strata=Strata.train)
        on_validation_epoch_end = partialmethod(end, strata=Strata.validate)
        on_test_epoch_end = partialmethod(end, strata=Strata.test)
        on_predict_epoch_end = partialmethod(end, strata=Strata.predict)
