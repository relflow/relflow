from __future__ import annotations

from functools import partialmethod
from time import perf_counter
from typing import TYPE_CHECKING, cast

import torch
from lightning import Callback, LightningModule, Trainer

from relflow.structs.enums import Metric, Strata

if TYPE_CHECKING:
    from relflow.architecture.root import Model


class ThroughputLogger(Callback):
    """Log each batch's estimated per-rank processing rate as a step metric."""

    def __init__(self) -> None:
        super().__init__()

        self.timestamp: float | None = None
        self.throughput: dict[Strata, float] = {}

    def start(self, trainer: Trainer, pl_module: LightningModule, *args: object, **kwargs: object) -> None:
        self.timestamp = perf_counter()

    def end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Strata,
    ) -> None:
        module = cast("Model", pl_module)
        assert self.timestamp is not None
        elapsed = perf_counter() - self.timestamp
        self.timestamp = None
        throughput = module.batch_size / elapsed if elapsed > 0.0 else 0.0
        self.throughput[strata] = throughput

        # Lightning does not register a result collection for prediction hooks,
        # so send prediction step metrics directly to the attached loggers.
        if strata == Strata.predict:
            if trainer.is_global_zero:
                for logger in trainer.loggers:
                    logger.log_metrics({f"{Metric.throughput.value}/{strata.value}": throughput}, step=batch_idx)
            return

        module.track(
            (Metric.throughput, strata),
            value=torch.tensor(throughput, device=module.device),
            on_step=True,
            on_epoch=False,
        )

    # Preserve Lightning hook signatures for the runtime partialmethod descriptors.
    if TYPE_CHECKING:
        on_train_batch_start = Callback.on_train_batch_start
        on_validation_batch_start = Callback.on_validation_batch_start
        on_test_batch_start = Callback.on_test_batch_start
        on_predict_batch_start = Callback.on_predict_batch_start
        on_train_batch_end = Callback.on_train_batch_end
        on_validation_batch_end = Callback.on_validation_batch_end
        on_test_batch_end = Callback.on_test_batch_end
        on_predict_batch_end = Callback.on_predict_batch_end
    else:
        on_train_batch_start = start
        on_validation_batch_start = start
        on_test_batch_start = start
        on_predict_batch_start = start

        on_train_batch_end = partialmethod(end, strata=Strata.train)
        on_validation_batch_end = partialmethod(end, strata=Strata.validate)
        on_test_batch_end = partialmethod(end, strata=Strata.test)
        on_predict_batch_end = partialmethod(end, strata=Strata.predict)
