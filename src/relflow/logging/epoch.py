from __future__ import annotations

from functools import partialmethod
from typing import TYPE_CHECKING, Literal

from lightning import Callback, LightningModule, Trainer

from relflow.logging.config import logger
from relflow.structs.enums import Strata


class EpochLifecycleLogger(Callback):
    """Record each Lightning epoch boundary with its rank and loop context."""

    def info(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        strata: Strata,
        hook: Literal["start", "end"],
    ) -> None:
        logger.bind(
            component="epoch",
            source="lightning",
            rank=pl_module.global_rank,
            epoch=pl_module.current_epoch,
            step=pl_module.global_step,
            hook=hook,
            strata=str(strata),
        ).info(f"{hook}ing {strata} epoch {pl_module.current_epoch}")

    # Preserve Lightning hook signatures for the runtime partialmethod descriptors.
    if TYPE_CHECKING:
        on_train_epoch_start = Callback.on_train_epoch_start
        on_train_epoch_end = Callback.on_train_epoch_end
        on_validation_epoch_start = Callback.on_validation_epoch_start
        on_validation_epoch_end = Callback.on_validation_epoch_end
        on_test_epoch_start = Callback.on_test_epoch_start
        on_test_epoch_end = Callback.on_test_epoch_end
        on_predict_epoch_start = Callback.on_predict_epoch_start
        on_predict_epoch_end = Callback.on_predict_epoch_end
    else:
        on_train_epoch_start = partialmethod(info, strata=Strata.train, hook="start")
        on_train_epoch_end = partialmethod(info, strata=Strata.train, hook="end")

        on_validation_epoch_start = partialmethod(info, strata=Strata.validate, hook="start")
        on_validation_epoch_end = partialmethod(info, strata=Strata.validate, hook="end")

        on_test_epoch_start = partialmethod(info, strata=Strata.test, hook="start")
        on_test_epoch_end = partialmethod(info, strata=Strata.test, hook="end")

        on_predict_epoch_start = partialmethod(info, strata=Strata.predict, hook="start")
        on_predict_epoch_end = partialmethod(info, strata=Strata.predict, hook="end")
