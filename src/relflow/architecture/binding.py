"""Bind worker-local field encodings before Lightning enters model forward."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as distributed
from lightning.pytorch.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel

from relflow.data.arrow import Encoded
from relflow.helpers.resize import Resize
from relflow.structs.enums import Component, Strata
from relflow.tensorfields.base import TENSORFIELDS, Bind, TensorFieldBase

if TYPE_CHECKING:
    from relflow.architecture.root import Model


def optimizers(module: Model) -> list[torch.optim.Optimizer]:
    """Find optimizers that own the live model, including interactive training."""
    trainer = getattr(module, "_trainer", None)
    active = list(getattr(trainer, "optimizers", ()))
    if not active and isinstance(module.optimizer, torch.optim.Optimizer):
        active.append(module.optimizer)
    return active


@torch.no_grad()
def synchronize_gradients(module: torch.nn.Module) -> None:
    """Preserve unfinished accumulation before discarding DDP's usage tracking.

    A parameter used only on another rank must acquire its accumulated gradient
    here, even if no later microbatch uses it. Globally unused parameters retain
    ``None`` so weight decay and optimizer step counts keep their semantics.
    Subsequent DDP averaging preserves this already-global contribution.
    """
    if not distributed.is_initialized() or distributed.get_world_size() == 1:
        return
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        return
    presence = torch.tensor(
        [
            [parameter.grad is not None, parameter.grad is not None and parameter.grad.layout != torch.strided]
            for parameter in parameters
        ],
        dtype=torch.int64,
        device=parameters[0].device,
    )
    distributed.all_reduce(presence, op=distributed.ReduceOp.MAX)
    if presence[:, 1].any():
        raise RuntimeError("DDP resource growth during accumulation requires dense gradients")
    active = presence[:, 0].cpu().tolist()
    for parameter, used in zip(parameters, active, strict=True):
        if not used:
            continue
        gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        distributed.all_reduce(gradient)
        gradient.div_(distributed.get_world_size())
        parameter.grad = gradient


def bind(module: Model, batch: Encoded, strata: Strata) -> Encoded:
    """Resolve every binding once, then install the prepared resource changes."""
    if not batch.bindings:
        return batch
    trainer = getattr(module, "_trainer", None)
    resize = Resize(optimizers(module))
    tensors = batch.tensors.clone(recurse=False)
    observations = dict(batch.observations)
    for address, binding in sorted(batch.bindings.items()):
        extension = TENSORFIELDS[module.schema.requests[address].type]
        field = cast(TensorFieldBase, batch.tensors[address].clone())
        observation = observations.get(address)
        if observation is not None:
            observation = observation.clone()
            observations[address] = observation
        cast(Bind, extension.component(Component.bind))(
            module, field, observation, binding, address=address, strata=strata, resize=resize
        )
        tensors[address] = field

    strategy = getattr(trainer, "strategy", None)
    wrapped = getattr(strategy, "model", None)
    if resize.parameters and isinstance(wrapped, DistributedDataParallel):
        assert trainer is not None
        if not isinstance(strategy, DDPStrategy) or not module.automatic_optimization:
            raise RuntimeError("automatic resource growth requires Lightning DDP with automatic optimization")
        if wrapped.static_graph or getattr(wrapped, "_use_python_reducer", False):
            raise RuntimeError("automatic resource growth requires eager DDP with static_graph=False")
        if getattr(wrapped, "mixed_precision", None) is not None or getattr(wrapped, "_comm_hooks", ()):
            raise RuntimeError(
                "automatic resource growth does not support DDP-native mixed precision or communication hooks"
            )
        if getattr(wrapped, "_delay_all_reduce_params", ()):
            raise RuntimeError("automatic resource growth does not support delayed DDP all-reduce")
        ready = trainer.fit_loop.epoch_loop.batch_progress.current.ready
        if ready % trainer.accumulate_grad_batches:
            synchronize_gradients(module)
        if not strategy._ddp_kwargs.get("init_sync", True):
            # Disabling initial DDP synchronization cannot leave new rows rank-local.
            for _, parameter in resize.parameters:
                distributed.broadcast(parameter.detach(), src=0)
        resize.commit()
        wrapped._remove_autograd_hooks()
        strategy.model = module
        del wrapped
        # Rewrapping must not overwrite rank-local normalizer/counter updates.
        buffers = [(buffer, buffer.clone()) for buffer in module.buffers()]
        strategy.configure_ddp()
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
    else:
        if resize.parameters and trainer is not None and trainer.world_size > 1:
            raise RuntimeError("automatic resource growth currently supports single-device training and eager DDP")
        resize.commit()
    return replace(batch, tensors=tensors, observations=observations, bindings={})


__all__ = ["bind", "synchronize_gradients"]
