from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Annotated, Literal, Unpack

import pydantic
import torch
from pydantic import StrictInt

from relflow.structs.enums import Component, Overflow
from relflow.structs.tree import FieldOptions
from relflow.tensorfields.base import Extension, RequestBase
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Metrics

series: Extension = Extension(name="series", types=(int | float,), callbacks=(Metrics,))


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean error over every selected step and channel."""
        if self == Objective.l1:
            return torch.nn.functional.l1_loss(input=inputs, target=targets)

        return torch.nn.functional.mse_loss(input=inputs, target=targets)


@series.register(Component.Request)
class Request(RequestBase):
    """One multichannel time series per record.

    Each record holds ``channels`` numeric values per step, observed at
    consecutive positions; steps are ordered but not timestamped. ``length``
    caps the retained steps with an ``overflow`` policy as ``Branch`` does;
    ``None`` keeps the whole series and pads each batch to its longest.
    ``n_blocks`` sets the number of dilated convolution blocks. As an input, the
    series embeds as one pooled token. With ``mask=True`` the series is a
    forecast target and ``horizon`` gives the number of steps to predict;
    ``horizon`` is required there and rejected elsewhere. ``jitter`` adds
    training-only noise to input steps, and ``objective`` selects mean ``"l1"``
    or ``"l2"`` error over predicted steps. Predictions contain ``horizon``
    steps of ``channels`` values in source units.
    """

    type: Literal["series"] = "series"
    channels: Annotated[int, pydantic.Field(gt=0)]
    length: Annotated[StrictInt | None, pydantic.Field(gt=0)] = None
    overflow: Overflow | None = None
    horizon: Annotated[StrictInt | None, pydantic.Field(gt=0)] = None
    n_blocks: Annotated[int, pydantic.Field(gt=0)] = 6
    jitter: Jitter = pydantic.Field(default_factory=Jitter)
    objective: Objective = Objective.l2

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            channels: int,
            length: int | None = None,
            overflow: Overflow | Literal["head", "tail", "random", "error"] | None = None,
            horizon: int | None = None,
            n_blocks: int = 6,
            jitter: Jitter = ...,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            type: Literal["series"] = "series",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @pydantic.model_validator(mode="after")
    def check_bounds(self):
        if self.length is None and self.overflow is not None:
            raise ValueError(
                f"unbounded series field cannot configure overflow={self.overflow!r}; "
                "set a finite length or remove the overflow policy"
            )
        if self.length is not None and self.overflow is None:
            self.overflow = Overflow.head

        return self
