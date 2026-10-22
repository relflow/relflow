"""Parameter-only exponential averaging for Lightning training and checkpoint selection."""

from __future__ import annotations

from math import isfinite
from typing import Any

import lightning.pytorch as lit
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor, nn
from torch.nn.parameter import UninitializedParameter
from torch.utils.hooks import RemovableHandle

__all__ = ["EMA"]


def parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    """Exclude frozen resources and parameters that have not consumed an input yet."""
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and not isinstance(parameter, UninitializedParameter)
    }


class EMA(lit.Callback):
    """Evaluate, select checkpoints, and finish fitting with averaged learned parameters.

    Pass ``callbacks=[rf.EMA(decay=0.99)]`` to ``model.fit``. Averaging starts
    after the first optimizer update, never on accumulated microbatches. The
    effective decay is capped at ``updates / (updates + 1)`` while warming up,
    so short runs do not retain a dominant first snapshot. Frozen parameters,
    normalizers, counters, and vocabulary metadata keep their current values.

    Full Lightning checkpoints export averaged inference weights and retain
    live parameters separately for optimizer-consistent ``ckpt_path`` resumes.
    Replacing a parameter or changing its shape starts a new average for that
    tensor; unchanged parameters keep their history. No row ordering is assumed.
    """

    def __init__(self, decay: float = 0.99) -> None:
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not isfinite(decay) or not 0 <= decay < 1:
            raise ValueError(f"EMA decay must be finite and in [0, 1), got {decay!r}")
        self.decay = decay
        self.updates = 0
        self.average: dict[str, Tensor] = {}
        self.live: dict[str, Tensor] = {}
        self.live_parameters: dict[str, nn.Parameter] = {}
        self.pending: dict[str, Tensor] = {}
        self.tracked: dict[str, nn.Parameter] = {}
        self.handles: list[RemovableHandle] = []
        self.active = False
        self.fitting = False
        self.finished = False

    def setup(self, trainer: lit.Trainer, pl_module: lit.LightningModule, stage: str) -> None:
        self.detach()
        self.updates = 0
        self.average = {}
        self.live = {}
        self.live_parameters = {}
        self.pending = {}
        self.tracked = {}
        self.active = False
        self.fitting = stage == "fit"
        self.finished = False
        if self.fitting and any(
            isinstance(callback, ModelCheckpoint) and callback.save_weights_only
            for callback in getattr(trainer, "callbacks", ())
        ):
            raise ValueError("EMA requires full checkpoint callbacks; set save_weights_only=False")

    @torch.no_grad()
    def update(self, module: lit.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        """Run from the optimizer's post-step hook, including manual optimization."""
        # Fused FP16 optimizers receive overflow state instead of skipping step().
        overflow = getattr(optimizer, "found_inf", None)
        if isinstance(overflow, Tensor) and bool(overflow.item()):
            return
        if self.active:
            raise RuntimeError("EMA evaluation weights are active during an optimizer step")
        current = self.synchronize(module)
        decay = min(self.decay, self.updates / (self.updates + 1))
        averaged = {
            name: self.average[name].to(parameter).clone().lerp_(parameter.detach(), 1 - decay)
            if name in self.average
            else parameter.detach().clone()
            for name, parameter in current.items()
        }
        self.average = averaged
        self.tracked = current
        self.updates += 1

    @torch.no_grad()
    def synchronize(self, module: lit.LightningModule) -> dict[str, nn.Parameter]:
        """Reset replaced tensors without guessing how an extension remapped its rows."""
        current = parameters(module)
        if self.average:
            self.average = {
                name: self.average[name]
                if name in self.average
                and self.tracked.get(name) is parameter
                and self.average[name].shape == parameter.shape
                else parameter.detach().clone()
                for name, parameter in current.items()
            }
        self.tracked = current
        return current

    @torch.no_grad()
    def copy(self, module: lit.LightningModule, values: dict[str, Tensor]) -> None:
        current = parameters(module)
        aligned = {}
        for name, value in values.items():
            if name not in current:
                raise ValueError(f"EMA parameter {name!r} is absent from the current model; start a new fit")
            if value.shape != current[name].shape:
                raise ValueError(
                    f"EMA parameter {name!r} must match its snapshot shape {tuple(value.shape)}, "
                    f"got {tuple(current[name].shape)}; start a new fit after changing the architecture"
                )
            aligned[name] = value.to(current[name])
        for name, value in aligned.items():
            current[name].copy_(value)

    @torch.no_grad()
    def begin(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        if not self.fitting or self.active or not self.average:
            return
        current = self.synchronize(pl_module)
        self.live = {name: parameter.detach().clone() for name, parameter in current.items()}
        self.live_parameters = current
        self.copy(pl_module, self.average)
        self.active = True

    @torch.no_grad()
    def restore(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        if not self.active:
            return
        current = parameters(pl_module)
        self.copy(
            pl_module,
            {
                name: value
                for name, value in self.live.items()
                if self.live_parameters.get(name) is current.get(name) and value.shape == current[name].shape
            },
        )
        self.live = {}
        self.live_parameters = {}
        self.active = False
        self.synchronize(pl_module)

    def on_train_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        # Sanity validation may have run with exported EMA weights. Optimizer
        # moments belong to the live snapshot, restored before any training.
        if self.pending:
            self.copy(pl_module, self.pending)
            self.pending = {}
        self.tracked = parameters(pl_module)
        self.handles = [
            optimizer.register_step_post_hook(lambda optimizer, args, kwargs: self.update(pl_module, optimizer))
            for optimizer in trainer.optimizers
        ]

    def on_train_batch_start(
        self, trainer: lit.Trainer, pl_module: lit.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self.restore(trainer, pl_module)

    def on_validation_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.begin(trainer, pl_module)

    def on_validation_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.restore(trainer, pl_module)

    def on_test_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.begin(trainer, pl_module)

    def on_test_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.restore(trainer, pl_module)

    def on_predict_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.begin(trainer, pl_module)

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.restore(trainer, pl_module)

    def on_train_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.detach()
        self.begin(trainer, pl_module)

    def on_fit_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        # ModelCheckpoint callbacks run last: RollbackCheckpoint may now replace
        # the graph with its selected averaged checkpoint.
        self.active = False
        self.fitting = False
        self.finished = True

    def teardown(self, trainer: lit.Trainer, pl_module: lit.LightningModule, stage: str) -> None:
        self.detach()
        current = parameters(pl_module)
        if stage == "fit" and any(current.get(name) is not value for name, value in self.tracked.items()):
            self.average = {}
            self.live = {}
            self.live_parameters = {}
            self.tracked = {}
            self.updates = 0

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def on_exception(self, trainer: lit.Trainer, pl_module: lit.LightningModule, exception: BaseException) -> None:
        try:
            self.restore(trainer, pl_module)
        finally:
            self.live = {}
            self.live_parameters = {}
            self.active = False
            self.detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "average": {name: value.detach().clone() for name, value in self.average.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.updates = state_dict["updates"]
        self.average = {name: value.detach().clone() for name, value in state_dict["average"].items()}
        self.pending = {name: value.detach().clone() for name, value in state_dict.get("live", {}).items()}

    def on_save_checkpoint(
        self, trainer: lit.Trainer, pl_module: lit.LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        if not self.average:
            return
        current = self.synchronize(pl_module)
        saved = checkpoint["callbacks"][self.state_key]
        saved["average"] = {name: value.detach().clone() for name, value in self.average.items()}
        saved["live"] = {
            name: (
                self.live[name]
                if (self.active or self.finished)
                and self.live_parameters.get(name) is parameter
                and self.live[name].shape == parameter.shape
                else parameter
            )
            .detach()
            .clone()
            for name, parameter in current.items()
        }
        checkpoint["state_dict"].update(saved["average"])
