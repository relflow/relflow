"""Explicit, extension-owned epoch statistics with identical DDP collectives."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torchmetrics import Metric

from relflow.structs.enums import Strata
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.architecture.root import Model


class Totals(Metric):
    """Integer denominators and accelerator-portable floating-point sums."""

    full_state_update = False

    def __init__(self, counts: int, sums: int):
        super().__init__()
        self.counts: torch.Tensor
        self.sums: torch.Tensor
        self.add_state("counts", default=torch.zeros(counts, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("sums", default=torch.zeros(sums, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, counts: torch.Tensor, sums: torch.Tensor) -> None:
        self.counts += counts.detach().to(self.counts)
        self.sums += sums.detach().to(self.sums)


class Reconstruction(Totals):
    """Categorical quality conditional on coverage, alongside all-target accuracy."""

    def __init__(self, topk: Sequence[int] = ()):
        super().__init__(counts=3 + len(topk), sums=2)
        self.topk = tuple(topk)

    def compute(self) -> dict[str, torch.Tensor]:
        known, valued, correct, *topk = self.counts.unbind()
        objective, nll = self.sums.unbind()
        # Empty denominators deliberately produce NaN, not perfect accuracy
        # or zero reconstruction error. Counts explain the missing evidence.
        return {
            "targets.known": known,
            "targets.unavailable": valued - known,
            "coverage.content": known / valued,
            "accuracy.content": correct / known,
            "accuracy.all": correct / valued,
            "loss.content": objective / known,
            "nll.content": nll / known,
            **{f"accuracy.top{k}": value / known for k, value in zip(self.topk, topk, strict=True)},
        }


class Scores(torch.nn.ModuleDict):
    """Register independent statistics for each epoch stage at one field address."""

    def __init__(self, address: Address, factory: Callable[[], Totals]):
        super().__init__(
            {f"{stage.value}_metrics": factory() for stage in (Strata.train, Strata.validate, Strata.test)}
        )
        self.address = address

    def compute(self, strata: Strata) -> dict[str, torch.Tensor]:
        metric = cast(Totals, self[f"{strata.value}_metrics"])
        # Every rank participates, even if its extension loss was never called.
        metric.update(torch.zeros_like(metric.counts), torch.zeros_like(metric.sums))
        return metric.compute()


class Metrics(Callback):
    """Reduce registered Scores on every rank, including objective-empty ranks."""

    def reset(self, pl_module: LightningModule, strata: Strata) -> None:
        for component in pl_module.modules():
            if isinstance(component, Scores):
                # Reset also detaches CPU spawn workers from parent-owned
                # shared metric storage before any local accumulation.
                cast(Totals, component[f"{strata.value}_metrics"]).reset()

    def record(self, pl_module: LightningModule, strata: Strata) -> None:
        module = cast("Model", pl_module)
        for component in module.modules():
            if isinstance(component, Scores):
                for name, value in component.compute(strata).items():
                    module.track((component.address, strata, *name.split(".")), value=value.float())

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.reset(pl_module, Strata.train)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.reset(pl_module, Strata.validate)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.reset(pl_module, Strata.test)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.record(pl_module, Strata.train)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.record(pl_module, Strata.validate)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.record(pl_module, Strata.test)
