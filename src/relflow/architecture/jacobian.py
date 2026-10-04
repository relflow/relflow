"""Optional, globally coordinated Jacobian descent with Lightning-owned updates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

from relflow.architecture.compiler import Region
from relflow.distributed import all_reduce_sum, is_distributed, world_size

try:
    from torchjd.aggregation import UPGrad  # ty:ignore[unresolved-import]
    from torchjd.autojac import Incidence, jac_to_grad, jd_backward  # ty:ignore[unresolved-import]
except ImportError as error:
    raise ImportError('jacobian=True requires TorchJD; install it with pip install "relflow[torchjd]"') from error

if TYPE_CHECKING:
    from relflow.architecture.root import Model

__all__ = ["combine", "validate"]


def validate(module: Model) -> None:
    """Check optional solver availability and the supported parameter layout."""

    try:
        UPGradWeighting()
    except ImportError as error:
        raise ImportError('jacobian=True requires the solver extra; install "relflow[torchjd]"') from error
    trainer = getattr(module, "_trainer", None)
    if trainer is not None and not isinstance(trainer.strategy, (SingleDeviceStrategy, DDPStrategy)):
        raise ValueError("jacobian=True supports single-device training and DDP with unsharded parameters")
    if any(
        isinstance(region := node.__dict__.get("compute"), Region) and not region.retain_graph
        for node in module.modules()
    ):
        raise ValueError(
            "jacobian=True was enabled after compilation; call model.compile(...) again before fitting "
            "so compiled backward preserves tensors for objective gradients"
        )


def accumulate(result: torch.Tensor, matrices: Sequence[torch.Tensor], scale: float, task: int | None) -> None:
    """Average task gradients before adding their exact inner products."""
    bucket = torch.cat(tuple(matrices), dim=1)
    all_reduce_sum(bucket)
    bucket.div_(world_size() * scale)
    if task is None:
        result.addmm_(bucket, bucket.T)
    else:
        result[task, task].add_(torch.dot(bucket.flatten(), bucket.flatten()))


def gramian(
    rows: Sequence[Sequence[torch.Tensor | None]], parameters: Sequence[torch.Tensor], scale: float
) -> torch.Tensor:
    """Avoid zero-filled rows for parameters with one globally active objective."""
    dtype = torch.float64 if parameters[0].dtype == torch.float64 else torch.float32
    result = torch.zeros(len(rows), len(rows), device=parameters[0].device, dtype=dtype)
    columns = tuple(zip(*rows, strict=True))
    support = [[value is not None for value in column] for column in columns]
    # A locally private parameter can be shared by other objectives on peers.
    if is_distributed():
        active = torch.tensor(support, device=result.device, dtype=torch.int32)
        all_reduce_sum(active)
        support = active.bool().tolist()
    buckets: dict[int | None, list[torch.Tensor]] = {}
    elements: dict[int | None, int] = {}
    for parameter, values, active in zip(parameters, columns, support, strict=True):
        count = sum(active)
        if not count:
            continue
        task = active.index(True) if count == 1 else None
        if task is None:
            if any(value is None for value in values):
                zero = parameter.new_zeros(()).expand_as(parameter)
                values = tuple(zero if value is None else value for value in values)
            matrix = torch.stack(cast(tuple[torch.Tensor, ...], values))
        else:
            value = values[task]
            matrix = torch.zeros_like(parameter) if value is None else value
        matrix = matrix.reshape(len(rows) if task is None else 1, -1).to(dtype=dtype)
        buckets.setdefault(task, []).append(matrix)
        elements[task] = elements.get(task, 0) + matrix.numel()
        if elements[task] >= 1_048_576:
            accumulate(result, buckets.pop(task), scale, task)
            elements.pop(task)
    for task, matrices in buckets.items():
        accumulate(result, matrices, scale, task)
    return result


def combine(module: Model, losses: Sequence[torch.Tensor], anchor: torch.Tensor):
    """Weight objective gradients while retaining automatic backward and stepping."""

    if len(losses) == 1:
        return losses[0] + anchor

    trainer = getattr(module, "_trainer", None)
    optimizers = trainer.optimizers if trainer is not None else ()
    if not optimizers and isinstance(module.optimizer, torch.optim.Optimizer):
        optimizers = (module.optimizer,)
    owned = {parameter for optimizer in optimizers for group in optimizer.param_groups for parameter in group["params"]}
    parameters = tuple(
        parameter for parameter in module.parameters() if parameter.requires_grad and (not owned or parameter in owned)
    )
    if not parameters:
        raise ValueError("jacobian=True requires at least one trainable parameter owned by the optimizer")

    aggregator = UPGrad()

    if not isinstance(module.loss_incidence, Incidence):
        module.loss_incidence = Incidence.build_incidence(
            losses=losses,
            params=parameters
        )

    jd_backward(
        losses,
        parameters,
        incidence=module.loss_incidence,
    )

    jac_to_grad(parameters, aggregator)

    return