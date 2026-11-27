"""Sketched isotropic Gaussian regularization of routed representations (LeJEPA SIGReg)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, cast

import pydantic
import torch

from relflow.distributed import all_reduce_sum, is_distributed
from relflow.structs.packages import Parcel
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.architecture.runtime import ExecutionPlan

__all__ = ["SIGReg", "regularize", "statistic"]


class SIGReg(pydantic.BaseModel):
    """Regularize every leaf embedding and branch summary toward an isotropic Gaussian.

    Each training and evaluation step projects the present tokens routed by
    every leaf embedder and branch encoder onto ``projections`` random unit
    directions and penalizes the Epps-Pulley distance between each slice's
    empirical characteristic function and the standard normal's, evaluated at
    ``knots`` points on ``[0, bound]``. The mean statistic over addresses is
    added to the reconstruction loss with ``weight``. Memory grows with
    tokens x ``projections`` x ``knots`` per address.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    weight: Annotated[float, pydantic.Field(gt=0.0, description="Scale of the statistic in the loss.")] = 0.05
    projections: Annotated[pydantic.StrictInt, pydantic.Field(gt=0, description="Random slices per step.")] = 1024
    knots: Annotated[pydantic.StrictInt, pydantic.Field(ge=2, description="Quadrature points.")] = 17
    bound: Annotated[float, pydantic.Field(gt=0.0, description="Largest characteristic-function argument.")] = 3.0

    @pydantic.field_validator("weight", "bound", mode="before")
    @classmethod
    def check_number(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("SIGReg.weight and SIGReg.bound must be numbers, not booleans")
        return value

    def __str__(self) -> str:
        return repr(self)


class Total(torch.autograd.Function):
    """Sum a tensor across ranks; its gradient is the cross-rank sum of cotangents."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
        return all_reduce_sum(tensor.clone())

    @staticmethod
    def backward(ctx: Any, *gradients: Any) -> Any:
        return all_reduce_sum(gradients[0].clone())


def statistic(embeddings: torch.Tensor, directions: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Average Epps-Pulley statistic of ``embeddings`` (n, C) along unit ``directions`` (C, M).

    Characteristic-function sums reduce across distributed ranks with a
    gradient-carrying collective and row counts with a plain one, so every
    rank evaluates the statistic of the global batch. An empty local batch
    still joins both collectives.
    """

    projected = embeddings @ directions
    arguments = projected.unsqueeze(-1) * knots
    sums = torch.stack((arguments.cos().sum(dim=0), arguments.sin().sum(dim=0)))
    count = float(all_reduce_sum(embeddings.new_tensor(float(embeddings.shape[0]))))
    if is_distributed():
        sums = cast(torch.Tensor, Total.apply(sums))
    if count < 2:
        # Too few observations to describe a distribution; keep the graph connected.
        return sums.sum() * 0.0

    real, imaginary = sums / count
    target = torch.exp(-0.5 * knots.square())
    error = ((real - target).square() + imaginary.square()) * target
    return torch.trapezoid(error, knots, dim=-1).mul(count).mean()


def regularize(
    module: Model,
    outgoing: Mapping[Address, Parcel],
    plan: ExecutionPlan,
    *,
    batch_idx: int,
) -> torch.Tensor:
    """Mean SIGReg statistic over the present tokens of every routed representation.

    Embedded leaves, then branches from the deepest, follow schema order, which
    is identical on every rank; skipped targets never carry a token. Directions
    come from a generator seeded by step, batch, and address position, never
    from process-wide random state, so ranks agree and seeded runs repeat.
    """

    config = module.sigreg
    assert config is not None
    order = (
        *(address for address in plan.requests if module.schema.requests[address].embedded),
        *(address for depth in plan.depthwise for address in depth),
    )
    values: list[torch.Tensor] = []
    with torch.autocast(device_type=module.device.type, enabled=False):
        for index, address in enumerate(order):
            parcel = outgoing.get(address)
            if parcel is None:
                continue
            width = parcel.payload.shape[-1]
            embeddings = parcel.payload.reshape(-1, width)[parcel.present.reshape(-1)].float()
            generator = torch.Generator().manual_seed(
                hash((module.global_step, batch_idx, index)) & 0x7FFF_FFFF_FFFF_FFFF
            )
            directions = torch.randn(width, config.projections, generator=generator)
            directions = (directions / directions.norm(dim=0, keepdim=True)).to(embeddings.device)
            knots = torch.linspace(0.0, config.bound, config.knots, device=embeddings.device)
            values.append(statistic(embeddings, directions, knots))
    if not values:
        return torch.zeros((), device=module.device)
    return torch.stack(values).mean()
