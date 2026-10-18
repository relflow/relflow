"""Group-owned membership refinement from its conditional identity likelihood."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.trainer.states import TrainerFn

from relflow.data.arrow import Encoded
from relflow.distributed import (
    all_gather_object,
    all_reduce_sum,
    broadcast_object,
    synchronize_epoch_metrics,
    world_size,
)
from relflow.logging import logger
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.tree import Address
from relflow.tensorfields.base import TensorFieldBase
from relflow.tensorfields.membership import propose

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import EncodedInput

INTERVAL = 200
TRAIN_ROWS = 4096
VALIDATION_ROWS = 1024
TRAIN_BYTES = 64 * 1024**2
VALIDATION_BYTES = 16 * 1024**2


@dataclass
class Calibration:
    """Own bounded canonical observations without retaining loader buffers."""

    limit: int
    byte_limit: int
    tables: list[pa.Table] = field(default_factory=list)
    rows: int = 0
    size: int = 0

    def add(self, source: pa.Table) -> None:
        count = min(source.num_rows, self.limit - self.rows)
        while count and source.slice(0, count).nbytes > self.byte_limit - self.size:
            count //= 2
        while count:
            table = source.slice(0, count).take(pa.array(range(count), type=pa.int64()))
            if table.nbytes <= self.byte_limit - self.size:
                self.tables.append(table)
                self.rows += count
                self.size += table.nbytes
                return
            count //= 2

    def gather(self, limit: int, byte_limit: int) -> pa.Table | None:
        result = Calibration(limit, byte_limit)
        for rank in all_gather_object(self.tables):
            for table in rank:
                result.add(table)
        return pa.concat_tables(result.tables, promote_options="permissive") if result.tables else None


def calibrate(module: Model, table: pa.Table) -> list[EncodedInput]:
    """Encode canonical observations once, without learning or preprocessing again."""
    result: list[EncodedInput] = []
    error = None
    try:
        size = min(module.batch_size, 256)
        result = [
            module.encode(table.slice(start, size), strata=Strata.validate).cpu()
            for start in range(0, table.num_rows, size)
        ]
    except Exception as cause:
        error = cause
    failures = all_gather_object(None if error is None else f"{type(error).__name__}: {error}")
    if any(failures):
        raise ValueError(f"Group calibration encoding failed across ranks: {failures}") from error
    return result


def evidence(
    module: Model, address: Address, batches: list[EncodedInput]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read only Group's own selected identities, context predictions, and NLL."""
    from relflow.architecture.runtime import ModelRuntime
    from relflow.tensorfields.extensions.group import resolve_group_embedder, vocabulary_logits

    embedder = resolve_group_embedder(module, address)
    identities, predictions, losses = [], [], []
    error = None
    try:
        for cached in batches:
            inputs = cached.to(module.device)
            batch = cast(TensorFieldBase, inputs[address])
            target = cast(torch.Tensor, batch.targets[TensorKey.content]).reshape(-1)
            states = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(-1)
            known = (
                batch.trainable.reshape(-1)
                & states.eq(Tokens.valued)
                & target.ge(0)
                & target.lt(len(embedder.vocab.master))
            )
            decoded = next(
                prediction
                for prediction in ModelRuntime.forward(module, inputs, strata=Strata.validate, participating={address})
                if prediction.address == address
            )
            logits = cast(torch.Tensor, decoded.payload[TensorKey.group]).reshape(-1, embedder.size)
            context = embedder.support(logits).float().log_softmax(-1)[known]
            if not torch.isfinite(context.logsumexp(-1)).all():
                raise ValueError(f"Group {address} has no finite available-group probability mass")
            target = target[known]
            errors = torch.nn.functional.cross_entropy(
                vocabulary_logits(embedder, context.exp()), target, reduction="none"
            )
            if not torch.isfinite(errors).all():
                raise ValueError(f"Group {address} has nonfinite identity reconstruction errors")
            identities.append(target)
            predictions.append(context)
            losses.append(errors)
    except Exception as cause:
        error = cause
    failures = all_gather_object(None if error is None else f"{type(error).__name__}: {error}")
    if any(failures):
        raise ValueError(f"Group {address} calibration failed across ranks: {failures}") from error
    return torch.cat(identities), torch.cat(predictions), torch.cat(losses)


def average(values: torch.Tensor) -> torch.Tensor:
    """Reduce Group's identity loss and its own target count on every rank."""
    dtype = torch.float32 if values.device.type == "mps" else torch.float64
    totals = all_reduce_sum(torch.stack((values.to(dtype).sum(), values.new_tensor(values.numel(), dtype=dtype))))
    return totals[0] / totals[1]


@torch.no_grad()
def refine(
    module: Model,
    address: Address | str,
    train: pa.Table,
    validate: pa.Table,
    optimizers: Sequence[torch.optim.Optimizer] = (),
) -> dict[str, Any]:
    """Accept a membership proposal only when Group's validation NLL improves.

    Group owns the objective, targets, proposal, and acceptance. Other field
    losses are never called. Their ordinary backpropagation can still train
    Group embeddings and memberships through the model's existing graph.
    All DDP ranks must participate with the same canonical Arrow tables.
    """
    from relflow.tensorfields.extensions.group import reset_history, resolve_group_embedder

    address = Address(str(address))
    embedder = resolve_group_embedder(module, address)
    weight = cast(torch.Tensor, embedder.embeddings[TensorKey.group.name].weight)
    event: dict[str, Any] = {"accepted": False, "changed_ids": 0, "objective": "group_identity_nll"}
    eligible = (
        address in module.schema.reconstruct
        and weight.requires_grad
        and not embedder._override_depth
        and bool(torch.isfinite(weight).all())
    )
    usable = eligible and len(embedder.vocab.master) > 0 and train.num_rows > 0 and validate.num_rows > 0
    configurations = all_gather_object((usable, len(embedder.vocab.master), tuple(embedder.retired.tolist())))
    if not all(item[0] for item in configurations):
        return {
            **event,
            "reason": "requires a trainable Group objective, known identities, and train/validation observations",
        }
    if any(item != configurations[0] for item in configurations):
        return {**event, "reason": "inconsistent Group vocabulary or support across ranks"}

    modes = {layer: layer.training for layer in module.modules()}
    saved = weight.detach().clone()
    size = len(embedder.vocab.master)
    accepted = False
    device = module.device
    mps_rng = torch.mps.get_rng_state() if device.type == "mps" else None
    trainer = getattr(module, "_trainer", None)
    precision = trainer.precision_plugin.forward_context() if trainer is not None else nullcontext()
    try:
        module.eval()
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []), precision:
            training, validation = (calibrate(module, table) for table in (train, validate))
            validation_ids, _, baseline = evidence(module, address, validation)
            before = average(baseline)
            if not torch.isfinite(before):
                return {**event, "reason": "no finite known Group validation targets"}
            identities, context, _ = evidence(module, address, training)
            replacement, rows = propose(saved[:size], context, identities, ~embedder.retired)
            if not rows.numel():
                return {**event, "reason": "no supported membership proposal"}
            weight[rows] = replacement[rows]
            candidate_ids, _, candidate_error = evidence(module, address, validation)
            if not all(all_gather_object(torch.equal(validation_ids, candidate_ids))):
                return {**event, "reason": "proposal changed Group validation target coverage"}
            after = average(candidate_error)
            decision = bool(broadcast_object(bool(torch.isfinite(after) and after < before)))
            event.update(
                accepted=decision,
                before=float(before),
                proposed=float(after),
                training_targets=identities.numel(),
                validation_targets=validation_ids.numel(),
                changed_ids=int((replacement[rows].argmax(-1) != saved[rows].argmax(-1)).sum()) if decision else 0,
            )
            if decision:
                weight[:size].copy_(broadcast_object(weight[:size].cpu()).to(weight))
                for index in rows.tolist():
                    reset_history(weight, 0, index, optimizers)
                accepted = True
            return event
    except ValueError as error:
        return {**event, "reason": str(error)}
    finally:
        if not accepted:
            weight.copy_(saved)
        for layer, training_mode in modes.items():
            layer.training = training_mode
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


class Refinement(Callback):
    """Refine Group before ordinary validation and checkpoint selection."""

    def __init__(self) -> None:
        self.train = Calibration(TRAIN_ROWS, TRAIN_BYTES)
        self.validate = Calibration(VALIDATION_ROWS, VALIDATION_BYTES)
        self.addresses: tuple[Address, ...] = ()
        self.enabled = False
        self.last_step = 0
        self.events: list[dict[str, Any]] = []

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step, "events": self.events}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.last_step = state_dict["last_step"]
        self.events = state_dict["events"][-16:]

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from relflow.tensorfields.extensions.group import Request

        module = cast("Model", pl_module)
        self.addresses = tuple(
            address
            for address in sorted(module.schema.reconstruct)
            if isinstance(module.schema.requests[address], Request)
        )
        self.enabled = bool(self.addresses) and trainer.enable_validation
        count = world_size() if self.enabled else 1
        self.train = Calibration(math.ceil(TRAIN_ROWS / count), math.ceil(TRAIN_BYTES / count))
        self.validate = Calibration(math.ceil(VALIDATION_ROWS / count), math.ceil(VALIDATION_BYTES / count))

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.addresses and not trainer.enable_validation:
            logger.bind(component="group").warning("Group refinement requires validation; using ordinary training only")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.enabled and isinstance(batch, Encoded):
            self.train.add(batch.source)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self.enabled and dataloader_idx == 0 and isinstance(batch, Encoded):
            self.validate.add(batch.source)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (
            not self.enabled
            or trainer.state.fn != TrainerFn.FITTING
            or trainer.sanity_checking
            or trainer.global_step - self.last_step < INTERVAL
        ):
            return
        # Validation can interrupt a group of accumulated microbatches. Wait
        # until Lightning has completed their optimizer step before replacing rows.
        if trainer.fit_loop.epoch_loop._should_accumulate():
            return
        module = cast("Model", pl_module)
        synchronize_epoch_metrics(trainer)
        train = self.train.gather(TRAIN_ROWS, TRAIN_BYTES)
        validate = self.validate.gather(VALIDATION_ROWS, VALIDATION_BYTES)
        if train is None or validate is None:
            logger.bind(component="group").warning("Group refinement requires train and validation observations")
            return
        for address in self.addresses:
            event = refine(module, address, train, validate, trainer.optimizers)
            event.update(address=str(address), step=trainer.global_step)
            self.events = [*self.events, event][-16:]
            logger.bind(component="group", **event).info("Group identity refinement evaluated")
        self.last_step = trainer.global_step
