"""Optimizer factories for relflow models."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn import Parameter

OptimizerFactory = Callable[[torch.nn.Module], torch.optim.Optimizer]

__all__ = ["OptimizerFactory", "adamw"]


# Only linear and convolution matrices are decayed: embedding tables, learned
# queries, and per-head gains are 2-D too, but shrinking them shifts lookups and
# attention temperature rather than regularizing a projection.
MATRICES = (torch.nn.Linear, torch.nn.Bilinear, torch.nn.modules.conv._ConvNd)


def uses_weight_decay(
    owner: torch.nn.Module,
    local: str,
    name: str,
    parameter: Parameter,
    *,
    decay_bias: bool,
    decay_1d: bool,
    no_decay_name_fragments: tuple[str, ...],
) -> bool:
    # Schema addresses are module keys starting with "/"; a field called
    # "normal" must not opt its layers out of decay.
    path = ".".join(part for part in name.lower().split(".") if not part.startswith("/"))
    if any(fragment in path for fragment in no_decay_name_fragments):
        return False
    if local == "bias":
        return decay_bias
    if parameter.ndim <= 1:
        return decay_1d
    return local == "weight" and isinstance(owner, MATRICES)


def adamw(
    learning_rate: float,
    *,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool | None = True,
    decay_bias: bool = False,
    decay_1d: bool = False,
    no_decay_name_fragments: tuple[str, ...] = ("norm",),
) -> OptimizerFactory:
    """Return a factory for custom training hooks or ``Model.fit(optimizer=...)``.

    Ordinary fitting configures these options directly on ``Model.fit``.

    The factory collects trainable parameters when the model configures its
    optimizer. Weight decay applies to the weight matrices of linear and
    convolution layers. Biases follow ``decay_bias`` and other one-dimensional
    parameters follow ``decay_1d``; embedding tables, learned queries, and
    gains are never decayed. Names containing a ``no_decay_name_fragments``
    entry outside schema addresses receive no decay. Set ``fused=False`` for
    devices that do not support Torch's fused AdamW.
    """

    fragments = tuple(fragment.lower() for fragment in no_decay_name_fragments if fragment)

    return partial(
        build,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=fused,
        decay_bias=decay_bias,
        decay_1d=decay_1d,
        fragments=fragments,
    )


def build(
    module: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    fused: bool | None,
    decay_bias: bool,
    decay_1d: bool,
    fragments: tuple[str, ...],
) -> torch.optim.Optimizer:
    """Bind AdamW groups to the current graph; keep the factory spawn-picklable."""
    decay_parameters: list[Parameter] = []
    no_decay_parameters: list[Parameter] = []

    seen: set[int] = set()
    for path, owner in module.named_modules():
        for local, parameter in owner.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if uses_weight_decay(
                owner,
                local,
                f"{path}.{local}" if path else local,
                parameter,
                decay_bias=decay_bias,
                decay_1d=decay_1d,
                no_decay_name_fragments=fragments,
            ):
                decay_parameters.append(parameter)
            else:
                no_decay_parameters.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if decay_parameters:
        parameter_groups.append({"params": decay_parameters, "weight_decay": weight_decay})
    if no_decay_parameters:
        parameter_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

    return torch.optim.AdamW(
        params=parameter_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        fused=fused,
        weight_decay=weight_decay,
    )
